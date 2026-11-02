"""CND -> MNE -> BIDS -> MNE -> CND round trips."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from cnd_mne import read_cnd
from cnd_mne.bids import compare_cnd, read_cnd_bids, write_cnd_bids
from cnd_mne.cli import main
from cnd_mne.exceptions import CNDUnsupportedError

pytest.importorskip("mne_bids")
pytest.importorskip("pybv")

FIXTURE = Path(__file__).parent / "data" / "minimal-cnd"


def _export(tmp_path: Path, **kwargs: object) -> Path:
    root = tmp_path / "bids"
    write_cnd_bids(
        read_cnd(FIXTURE, subject=1),
        root,
        subject=1,
        task="listening",
        external_unit="uV",
        external_channel_types="eog",
        line_freq=50,
        **kwargs,  # type: ignore[arg-type]
    )
    return root


def test_round_trip_preserves_cnd(tmp_path: Path) -> None:
    original = read_cnd(FIXTURE, subject=1)
    root = _export(tmp_path)
    restored = read_cnd_bids(root, subject=1, task="listening")
    report = compare_cnd(original, restored)
    assert report["passed"], report
    assert report["max_relative_neural_error"] < 1e-6
    assert restored.neural is not None
    assert restored.neural.external_description == "Synthetic EOG"


def test_layout_is_bids(tmp_path: Path) -> None:
    root = _export(tmp_path, montage="eeglab", coordinate_scale_to_meters=0.095)
    eeg_dir = root / "sub-1" / "eeg"
    assert (root / "dataset_description.json").exists()
    assert "*_cnd.json" in (root / ".bidsignore").read_text()
    for run in (1, 2):
        stem = f"sub-1_task-listening_run-{run}"
        assert (eeg_dir / f"{stem}_eeg.vhdr").exists()
        assert (eeg_dir / f"{stem}_recording-SpeechEnvelope_stim.tsv.gz").exists()
    assert list(eeg_dir.glob("*_electrodes.tsv"))
    sidecar = json.loads((eeg_dir / "sub-1_task-listening_run-1_eeg.json").read_text())
    assert sidecar["PowerLineFrequency"] == 50
    assert sidecar["Manufacturer"] == "n/a"
    assert not any(key.startswith("CND") for key in sidecar)


def test_stimulus_sidecar_and_events(tmp_path: Path) -> None:
    root = _export(tmp_path)
    eeg_dir = root / "sub-1" / "eeg"
    stem = "sub-1_task-listening_run-1"
    sidecar = json.loads(
        (eeg_dir / f"{stem}_recording-WordOnsets_stim.json").read_text()
    )
    assert sidecar["SamplingFrequency"] == 100.0
    assert sidecar["CNDFeatureName"] == "Word Onsets"
    with gzip.open(
        eeg_dir / f"{stem}_recording-WordOnsets_stim.tsv.gz", "rt"
    ) as handle:
        values = np.loadtxt(handle)
    original = read_cnd(FIXTURE, subject=1).stimulus
    assert original is not None
    np.testing.assert_array_equal(values, np.ravel(original.feature("Word Onsets")[0]))
    events = (eeg_dir / f"{stem}_events.tsv").read_text().splitlines()
    assert sum("Word Onsets" in line for line in events) == int(values.sum())
    assert not any("Speech Envelope" in line for line in events)


def test_external_channels_need_a_unit(tmp_path: Path) -> None:
    with pytest.warns(RuntimeWarning, match="extChan"):
        write_cnd_bids(
            read_cnd(FIXTURE, subject=1), tmp_path / "bids", subject=1, task="x"
        )
    restored = read_cnd_bids(tmp_path / "bids", subject=1, task="x")
    assert restored.neural is not None and restored.neural.external_trials is None


def test_existing_files_need_overwrite(tmp_path: Path) -> None:
    root = _export(tmp_path)
    with pytest.raises(FileExistsError):
        write_cnd_bids(
            read_cnd(FIXTURE, subject=1),
            root,
            subject=1,
            task="listening",
            external_unit="uV",
        )


def test_rejects_bad_labels_and_non_eeg(tmp_path: Path) -> None:
    recording = read_cnd(FIXTURE, subject=1)
    with pytest.raises(ValueError, match="letters/digits"):
        write_cnd_bids(recording, tmp_path, subject="sub_1", task="x")
    assert recording.neural is not None
    recording.neural.data_type = "MEG"
    with pytest.raises(CNDUnsupportedError, match="EEG only"):
        write_cnd_bids(recording, tmp_path, subject=1, task="x")


def test_cli_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "bids"
    code = main(
        [
            "to-bids",
            str(FIXTURE),
            "--subject",
            "1",
            "--bids-root",
            str(root),
            "--task",
            "listening",
            "--external-unit",
            "uV",
            "--verify",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["round_trip"]["passed"]
    out = tmp_path / "dataCND"
    code = main(
        [
            "from-bids",
            str(root),
            "--subject",
            "1",
            "--task",
            "listening",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    back = read_cnd(out, subject=1)
    assert back.neural is not None and back.neural.n_trials == 2
