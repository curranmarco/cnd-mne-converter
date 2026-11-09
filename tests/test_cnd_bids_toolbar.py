"""Open and save CND/BIDS from MNE's Qt data viewer."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import mne
import numpy as np
import pytest

from cnd_mne import read_cnd, read_cnd_mne
from cnd_mne.bids import compare_cnd, read_cnd_bids, write_cnd_bids
from cnd_mne.gui import add_cnd_bids_toolbar, launch_browser

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("mne_qt_browser")
mne_bids = pytest.importorskip("mne_bids")
QtWidgets = pytest.importorskip("qtpy.QtWidgets")
QtCore = pytest.importorskip("qtpy.QtCore")

FIXTURE = Path(__file__).parent / "data" / "minimal-cnd"


@pytest.fixture
def qt_app(tmp_path, monkeypatch):
    monkeypatch.setenv("MNE_BROWSER_PRECOMPUTE", "false")
    previous_format = QtCore.QSettings.defaultFormat()
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path)
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    for window in app.topLevelWidgets():
        window.close()
    app.processEvents()
    QtCore.QSettings.setDefaultFormat(previous_format)


def _viewer(raw):
    with mne.viz.use_browser_backend("qt"):
        return raw.plot(show=False, block=False, precompute=False)


def _cnd_viewer():
    recording = read_cnd_mne(FIXTURE, subject=1)
    browser = _viewer(recording.raws[0])
    browser.cnd_recording = recording
    return add_cnd_bids_toolbar(browser), recording


def _native_bids(root: Path) -> Path:
    raw = mne.io.RawArray(
        np.random.default_rng(0).standard_normal((3, 1000)) * 1e-5,
        mne.create_info(["Fz", "Cz", "Pz"], 100.0, "eeg"),
        verbose=False,
    )
    raw.info["line_freq"] = 50
    path = mne_bids.write_raw_bids(
        raw,
        mne_bids.BIDSPath(subject="01", task="rest", datatype="eeg", root=root),
        format="BrainVision",
        allow_preload=True,
        verbose=False,
    )
    return Path(path.fpath)


def test_toolbar_has_four_actions(qt_app) -> None:
    toolbar, _ = _cnd_viewer()
    assert add_cnd_bids_toolbar(toolbar.browser) is toolbar
    bars = toolbar.browser.findChildren(QtWidgets.QToolBar, "cndImportToolbar")
    assert len(bars) == 1
    assert [a.text() for a in bars[0].actions()] == [
        "Open CND…",
        "Open BIDS…",
        "Save as CND…",
        "Save as BIDS…",
    ]


def test_cnd_viewer_saves_bids_and_cnd(qt_app, tmp_path) -> None:
    toolbar, _ = _cnd_viewer()
    toolbar.save_bids(
        tmp_path / "bids",
        subject="1",
        task="listening",
        external_unit="uV",
        line_freq=50,
    )
    back = read_cnd_bids(tmp_path / "bids", subject="1", task="listening")
    assert compare_cnd(read_cnd(FIXTURE, subject=1), back)["passed"]
    paths = toolbar.save_cnd(tmp_path / "cnd", subject=3)
    assert paths.neural.name == "dataSub3.mat"
    assert read_cnd(tmp_path / "cnd", subject=3).stimulus is not None


def test_open_bids_from_cnd_export_is_whole_recording(qt_app, tmp_path) -> None:
    write_cnd_bids(
        read_cnd(FIXTURE, subject=1), tmp_path, subject=1, task="x", external_unit="uV"
    )
    toolbar, _ = _cnd_viewer()
    browser = toolbar.open_bids(
        tmp_path / "sub-1" / "eeg" / "sub-1_task-x_run-2_eeg.vhdr"
    )
    assert len(browser.cnd_recording.raws) == 2
    assert browser.cnd_trial_index == 1
    assert browser.findChild(QtWidgets.QAction, "saveBIDSAction") is not None


def test_plain_bids_keeps_viewer_bad_channels_and_annotations(qt_app, tmp_path) -> None:
    source = _native_bids(tmp_path / "original")
    toolbar, _ = _cnd_viewer()
    browser = toolbar.open_bids(source)
    assert getattr(browser, "cnd_recording", None) is None
    browser.mne.info["bads"].append("Cz")  # what clicking a channel does
    browser.mne.inst.annotations.append(2.0, 0.0, "blink")
    plain = browser._cnd_bids_toolbar
    plain.save_bids(tmp_path / "final", subject="01", task="rest")
    eeg_dir = tmp_path / "final" / "sub-01" / "eeg"
    with (eeg_dir / "sub-01_task-rest_channels.tsv").open() as handle:
        status = {
            row["name"]: row["status"] for row in csv.DictReader(handle, delimiter="\t")
        }
    assert status["Cz"] == "bad"
    assert "blink" in (eeg_dir / "sub-01_task-rest_events.tsv").read_text()
    paths = plain.save_cnd(tmp_path / "cnd", subject=1, unit="uV")
    cnd = read_cnd(tmp_path / "cnd", subject=1)
    assert paths.neural.exists() and cnd.neural is not None
    assert cnd.neural.channel_names == ("Fz", "Cz", "Pz")
    assert cnd.neural.data_unit == "uV"


def test_save_dialogs_and_overwrite(qt_app, tmp_path, monkeypatch) -> None:
    toolbar, _ = _cnd_viewer()
    dialogs = QtWidgets.QInputDialog
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )
    texts = iter(["7", "listening", "7", "listening"])
    monkeypatch.setattr(dialogs, "getText", lambda *a, **k: (next(texts), True))
    items = iter(["60", "uV", "60", "uV"])
    monkeypatch.setattr(dialogs, "getItem", lambda *a, **k: (next(items), True))
    shown = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *a: shown.append(a[-1])
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    errors = []
    monkeypatch.setattr(toolbar, "_error", errors.append)
    toolbar._ask_save_bids()
    toolbar._ask_save_bids()  # second time: files exist, overwrite accepted
    assert not errors, errors
    assert len(shown) == 2
    assert (tmp_path / "sub-7" / "eeg" / "sub-7_task-listening_run-1_eeg.vhdr").exists()


def test_launch_browser_opens_cnd_and_bids(qt_app, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(QtWidgets.QApplication, "exec", lambda self: 0)
    launch_browser(FIXTURE, subject="1")
    source = _native_bids(tmp_path)
    launch_browser(tmp_path, subject="01", task="rest")
    launch_browser(source)
    titles = [
        w for w in qt_app.topLevelWidgets() if isinstance(w, QtWidgets.QMainWindow)
    ]
    assert titles


def test_untyped_channels_need_confirmation(qt_app, tmp_path, monkeypatch) -> None:
    source = _native_bids(tmp_path / "original")
    channels = source.with_name(source.name.replace("_eeg.vhdr", "_channels.tsv"))
    lines = channels.read_text().splitlines()
    header = lines[0].split("\t")
    type_column = header.index("type")
    rows = [line.split("\t") for line in lines[1:]]
    for row in rows:
        row[type_column] = "n/a"  # as in OpenNeuro ds004574
    channels.write_text(
        "\n".join(["\t".join(header), *("\t".join(r) for r in rows)]) + "\n"
    )
    toolbar, _ = _cnd_viewer()
    plain = toolbar.open_bids(source)._cnd_bids_toolbar
    assert plain.needs_eeg_confirmation()
    with pytest.raises(ValueError, match="typed as EEG"):
        plain.save_cnd(tmp_path / "cnd")
    plain.save_cnd(tmp_path / "cnd", treat_misc_as_eeg=True)
    assert read_cnd(tmp_path / "cnd", subject=1).neural.n_channels == 3
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a: QtWidgets.QMessageBox.StandardButton.No,
    )
    assert plain._confirm_eeg() is None


def test_save_bids_dialog_uses_eeg_confirmation(qt_app, tmp_path, monkeypatch) -> None:
    source = _native_bids(tmp_path / "original")
    channels = source.with_name(source.name.replace("_eeg.vhdr", "_channels.tsv"))
    channels.write_text(channels.read_text().replace("\tEEG\t", "\tn/a\t"))
    toolbar, _ = _cnd_viewer()
    plain = toolbar.open_bids(source)._cnd_bids_toolbar
    assert plain.needs_eeg_confirmation()
    out = tmp_path / "out"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", lambda *a, **k: str(out)
    )
    texts = iter(["01", "rest"])
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *a, **k: (next(texts), True)
    )
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *a, **k: ("50", True))
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a: None)
    errors = []
    monkeypatch.setattr(plain, "_error", errors.append)
    plain._ask_save_bids()
    assert not errors, errors
    with (out / "sub-01" / "eeg" / "sub-01_task-rest_channels.tsv").open() as handle:
        types = {row["type"] for row in csv.DictReader(handle, delimiter="\t")}
    assert types == {"EEG"}
