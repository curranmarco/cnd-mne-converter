"""Any BIDS EEG dataset -> CND (read_any_bids / from-bids)."""

from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np
import pytest

from cnd_mne.bids import bids_to_cnd, read_any_bids, write_cnd_bids
from cnd_mne.cli import main
from cnd_mne.exceptions import CNDValidationError
from cnd_mne.io import read_cnd

mne_bids = pytest.importorskip("mne_bids")
pytest.importorskip("pybv")

SFREQ = 250.0


def _write_run(root: Path, run: int, *, eog: bool = True, sfreq: float = SFREQ) -> None:
    rng = np.random.default_rng(run)
    names = ["Fz", "Cz", "Pz"] + (["VEOG"] if eog else [])
    types = ["eeg"] * 3 + (["eog"] if eog else [])
    raw = mne.io.RawArray(
        rng.standard_normal((len(names), 2500)) * 1e-5,
        mne.create_info(names, sfreq, types),
        verbose=False,
    )
    onsets = np.array([1.0, 2.0, 3.0, 4.0]) + 0.1 * run
    raw.set_annotations(
        mne.Annotations(onsets, 0.0, ["tone", "word", "tone", "boundary"])
    )
    raw.info["line_freq"] = 50
    mne_bids.write_raw_bids(
        raw,
        mne_bids.BIDSPath(
            subject="01", task="listen", run=run, datatype="eeg", root=root
        ),
        format="BrainVision",
        allow_preload=True,
        event_id={"tone": 1, "word": 2, "boundary": 3},
        verbose=False,
    )


def _dataset(root: Path) -> Path:
    for run in (1, 2):
        _write_run(root, run)
    return root


def test_runs_become_trials_and_events_become_features(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "bids")
    recording = read_any_bids(root, subject="01", task="listen")

    neural = recording.neural
    assert neural.n_trials == 2
    assert neural.channel_names == ("Fz", "Cz", "Pz")
    assert neural.data_unit == "uV"
    assert neural.original_trial_positions == (1, 2)
    original = mne_bids.read_raw_bids(
        mne_bids.BIDSPath(
            subject="01", task="listen", run=1, datatype="eeg", root=root
        ),
        verbose=False,
    )
    np.testing.assert_allclose(
        neural.trials[0], original.get_data(picks=["Fz", "Cz", "Pz"]).T * 1e6, rtol=1e-6
    )
    assert neural.external_trials is not None
    assert neural.external_trials[0].shape == (2500, 1)

    stimulus = recording.stimulus
    assert stimulus is not None
    assert stimulus.names == ("tone", "word")  # boundary skipped
    assert stimulus.sfreq == SFREQ
    tone_run2 = stimulus.features[0][1]
    assert np.flatnonzero(tone_run2).tolist() == [
        round(1.2 * SFREQ),
        round(3.2 * SFREQ),
    ]


def test_dispatcher_prefers_cnd_sidecar(tmp_path: Path) -> None:
    generic = read_any_bids(_dataset(tmp_path / "bids"), subject="01", task="listen")
    write_cnd_bids(
        generic, tmp_path / "ours", subject="01", task="listen", line_freq=50
    )
    restored = bids_to_cnd(tmp_path / "ours", subject="01", task="listen")
    assert restored.stimulus is not None
    assert restored.stimulus.names == generic.stimulus.names  # type: ignore[union-attr]
    assert restored.neural.data_unit == "uV"


def test_value_column_and_ignore(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "bids")
    recording = read_any_bids(
        root, subject="01", task="listen", event_column="value", ignore_events=()
    )
    assert recording.stimulus is not None
    assert len(recording.stimulus.names) == 3


def test_misc_channels_need_confirmation(tmp_path: Path) -> None:
    root = tmp_path / "bids"
    _write_run(root, 1, eog=False)
    channels = next(root.rglob("*_channels.tsv"))
    with channels.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["type"] = "n/a"
    with channels.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(CNDValidationError, match="treat-misc-as-eeg"):
        read_any_bids(root, subject="01", task="listen")
    recording = read_any_bids(root, subject="01", task="listen", treat_misc_as_eeg=True)
    assert recording.neural.n_channels == 3


def test_errors(tmp_path: Path) -> None:
    root = tmp_path / "bids"
    _write_run(root, 1)
    _write_run(root, 2, sfreq=500.0)
    with pytest.raises(CNDValidationError, match="sampling rates"):
        read_any_bids(root, subject="01", task="listen")
    single = tmp_path / "single"
    _write_run(single, 1)
    with pytest.raises(CNDValidationError, match="no column"):
        read_any_bids(single, subject="01", task="listen", event_column="nope")
    with pytest.raises(FileNotFoundError):
        read_any_bids(root, subject="02", task="listen")


def test_cli_from_bids_then_compare(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _dataset(tmp_path / "bids")
    out = tmp_path / "dataCND"
    assert (
        main(
            [
                "from-bids",
                str(root),
                "--subject",
                "01",
                "--task",
                "listen",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert read_cnd(out, subject=1).neural.n_trials == 2

    final = tmp_path / "final"
    assert (
        main(
            [
                "to-bids",
                str(out),
                "--subject",
                "1",
                "--task",
                "listen",
                "--bids-root",
                str(final),
                "--line-freq",
                "50",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "compare-bids",
                str(root),
                str(final),
                "--subject",
                "01",
                "--task",
                "listen",
                "--final-subject",
                "1",
                "--run",
                "1",
                "--event-column",
                "trial_type",
            ]
        )
        == 0
    )
    report = capsys.readouterr().out
    assert '"passed": true' in report
    assert '"tone"' in report

    assert (
        main(
            [
                "from-bids",
                str(root),
                "--subject",
                "09",
                "--task",
                "listen",
                "--output",
                str(out),
            ]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err


def _session_dataset(root: Path, scale: float = 1e-5) -> None:
    raw = mne.io.RawArray(
        np.random.default_rng(3).standard_normal((3, 2000)) * scale,
        mne.create_info(["Fz", "Cz", "Pz"], SFREQ, "eeg"),
        verbose=False,
    )
    raw.set_annotations(mne.Annotations([1.0, 2.0, 3.0], 0.0, ["go", "stop", "go"]))
    mne_bids.write_raw_bids(
        raw,
        mne_bids.BIDSPath(
            subject="01", session="01", task="t", datatype="eeg", root=root
        ),
        format="BrainVision",
        allow_preload=True,
        event_id={"go": 1, "stop": 2},
        verbose=False,
    )


def test_sessions_round_trip(tmp_path: Path) -> None:
    from cnd_mne.bids import compare_bids

    _session_dataset(tmp_path / "original")
    cnd = read_any_bids(tmp_path / "original", subject="01", task="t", session="01")
    write_cnd_bids(cnd, tmp_path / "final", subject=1, task="t")  # no session
    report = compare_bids(
        tmp_path / "original",
        tmp_path / "final",
        subject="01",
        task="t",
        session="01",
        final_subject="1",
        event_column="trial_type",
    )
    assert report["passed"]
    assert report["events"]["go"]["n_final"] == 2


def test_combined_event_labels_match() -> None:
    from cnd_mne.bids import _event_matches

    row = {"trial_type": "response", "value": "2"}
    assert _event_matches(row, "trial_type", "response")
    assert _event_matches(row, "trial_type", "response/2")
    assert not _event_matches(row, "trial_type", "response/4")
    assert not _event_matches({"value": "S  1"}, "value", "S  2")


def test_impossible_amplitude_warns(tmp_path: Path) -> None:
    _session_dataset(tmp_path, scale=5.0)  # volts: impossible for EEG
    with pytest.warns(RuntimeWarning, match="above 1 V"):
        read_any_bids(tmp_path, subject="01", task="t", session="01")
