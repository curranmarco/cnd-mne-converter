"""BIDS (native) -> CND -> BIDS, checked with compare_bids."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pytest

from cnd_mne import CNDStimulus, from_mne
from cnd_mne.bids import compare_bids, event_map_from_recipe, write_cnd_bids
from cnd_mne.cli import main

mne_bids = pytest.importorskip("mne_bids")
pytest.importorskip("pybv")


def _native_bids(root: Path) -> mne.io.BaseRaw:
    """A BIDS dataset written directly by MNE-BIDS, as on OpenNeuro."""
    rng = np.random.default_rng(0)
    sfreq = 250.0
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((4, 2500)) * 1e-5, info, verbose=False)
    onsets = np.arange(1.0, 9.0, 0.5)
    raw.set_annotations(
        mne.Annotations(onsets, [0.0] * len(onsets), ["S  1", "S  2"] * 8)
    )
    raw.info["line_freq"] = 50
    mne_bids.write_raw_bids(
        raw,
        mne_bids.BIDSPath(subject="01", task="oddball", datatype="eeg", root=root),
        format="BrainVision",
        allow_preload=True,
        event_id={"S  1": 1, "S  2": 2},
        verbose=False,
    )
    return raw


def _bids_to_cnd_to_bids(tmp_path: Path, shift: float = 0.0) -> tuple[Path, Path]:
    original_root = tmp_path / "original"
    raw = _native_bids(original_root)
    # Stand-in for the pipeline step: one impulse track per event type.
    tracks = []
    for code in ("S  1", "S  2"):
        track = np.zeros(raw.n_times)
        onsets = raw.annotations.onset[raw.annotations.description == code] + shift
        track[np.rint(onsets * raw.info["sfreq"]).astype(int)] = 1.0
        tracks.append((track,))
    stimulus = CNDStimulus(
        names=("precue", "go"), features=tuple(tracks), sfreq=raw.info["sfreq"]
    )
    raw.set_annotations(None)
    cnd = from_mne([raw], stimulus=stimulus, output_unit="uV")
    final_root = tmp_path / "final"
    write_cnd_bids(cnd, final_root, subject="01", task="oddball", line_freq=50)
    return original_root, final_root


EVENT_MAP = {"precue": ("trial_type", "S  1"), "go": ("trial_type", "S  2")}


def test_chain_matches_original(tmp_path: Path) -> None:
    original, final = _bids_to_cnd_to_bids(tmp_path)
    report = compare_bids(
        original, final, subject="01", task="oddball", event_map=EVENT_MAP
    )
    assert report["passed"], report
    assert report["eeg"]["max_relative_error"] < 1e-6
    assert report["events"]["precue"]["n_original"] == 8
    assert report["events"]["go"]["max_onset_difference_s"] == 0.0


def test_chain_detects_shifted_events(tmp_path: Path) -> None:
    original, final = _bids_to_cnd_to_bids(tmp_path, shift=0.1)
    report = compare_bids(
        original, final, subject="01", task="oddball", event_map=EVENT_MAP
    )
    assert report["eeg"]["passed"]
    assert not report["events"]["precue"]["passed"]
    assert not report["passed"]


def test_event_map_from_recipe(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "name": "precue",
                        "kind": "bids_event_impulse",
                        "source_column": "value",
                        "source_value": "S  1",
                    },
                    {
                        "name": "go",
                        "kind": "bids_event_impulse",
                        "source_column": "trial_type",
                        "source_values": ["S  2"],
                    },
                    {"name": "envelope", "kind": "audio_envelope"},
                ]
            }
        )
    )
    assert event_map_from_recipe(recipe) == {
        "precue": ("value", "S  1"),
        "go": ("trial_type", "S  2"),
    }


def test_compare_bids_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    original, final = _bids_to_cnd_to_bids(tmp_path)
    code = main(
        [
            "compare-bids",
            str(original),
            str(final),
            "--subject",
            "01",
            "--task",
            "oddball",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["metadata"]["electrode_positions"] == {
        "original": False,
        "final": False,
    }


def test_compare_bids_needs_one_run(tmp_path: Path) -> None:
    original, final = _bids_to_cnd_to_bids(tmp_path)
    with pytest.raises(ValueError, match="Expected one original"):
        compare_bids(original, final, subject="99", task="oddball")
