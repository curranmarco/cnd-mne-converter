"""Numerical and metadata checks for MNE and MATLAB round trips."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import mne
import numpy as np

from .exceptions import CNDValidationError
from .io import read_cnd, write_cnd
from .mne import MNECNDRecording
from .model import CNDRecording


def _raw_values_match(
    trials: tuple[np.ndarray, ...], raws: tuple[mne.io.RawArray, ...]
) -> bool:
    if len(trials) != len(raws):
        return False
    for trial, raw in zip(trials, raws, strict=True):
        expected = np.asarray(trial)
        if expected.ndim == 1:
            expected = expected[:, None]
        actual = raw.get_data()
        if actual.shape != expected.T.shape or not np.array_equal(actual, expected.T):
            return False
    return True


def _check_memory_round_trip(
    recording: CNDRecording, converted: MNECNDRecording
) -> tuple[CNDRecording, float, bool, bool]:
    """Check values and template metadata after conversion back from MNE."""
    neural = recording.neural
    if neural is None:
        raise CNDValidationError("MNE round trip requires neural data")
    converted_back = converted.to_cnd(on_unsupported_metadata="raise")
    if converted_back.neural is None:
        raise CNDValidationError("MNE round trip lost neural data")
    round_trip_error = max(
        _max_abs_error(before, after)
        for before, after in zip(
            neural.trials, converted_back.neural.trials, strict=True
        )
    )
    metadata_preserved = (
        converted_back.stimulus is recording.stimulus
        and _metadata_equal(
            converted_back.additional_variables,
            recording.additional_variables,
        )
        and converted_back.neural.extra_fields is neural.extra_fields
        and converted_back.neural.external_fields is neural.external_fields
        and converted_back.neural.channel_locations is neural.channel_locations
        and converted_back.neural.channel_locations_raw is neural.channel_locations_raw
        and converted_back.neural.original_trial_positions
        == neural.original_trial_positions
        and converted_back.neural.external_layout == neural.external_layout
        and converted_back.neural.external_group_names == neural.external_group_names
        and converted_back.neural.external_group_channel_counts
        == neural.external_group_channel_counts
        and converted_back.neural.external_group_fields is neural.external_group_fields
    )
    numerically_equivalent = all(
        np.allclose(before, after, rtol=1e-12, atol=1e-12)
        for before, after in zip(
            neural.trials, converted_back.neural.trials, strict=True
        )
    )
    round_trip_verified = numerically_equivalent and metadata_preserved
    return converted_back, round_trip_error, metadata_preserved, round_trip_verified


def _mne_psd_is_finite(raw: mne.io.BaseRaw) -> bool:
    duration = min(10.0, raw.times[-1])
    segment = raw.copy().crop(tmin=0.0, tmax=duration).pick([0])
    n_fft = min(1024, segment.n_times)
    if n_fft < 2:
        return False
    nyquist = float(segment.info["sfreq"]) / 2
    spectrum = segment.compute_psd(
        method="welch",
        fmin=0.0,
        fmax=nyquist,
        n_fft=n_fft,
        verbose="ERROR",
    )
    return bool(np.isfinite(spectrum.get_data()).all())


def _serialized_round_trip(
    recording: CNDRecording, *, mat_version: Literal["5", "7.3"]
) -> tuple[bool, float]:
    """Exercise the real MATLAB writer and reader in an isolated directory."""
    with tempfile.TemporaryDirectory(prefix="cnd-mne-verify-") as temporary:
        destination = Path(temporary) / "dataCND"
        write_cnd(recording, destination, mat_version=mat_version)
        reloaded = read_cnd(
            destination, subject=1 if recording.neural is not None else None
        )

    expected_neural = recording.neural
    actual_neural = reloaded.neural
    if expected_neural is None or actual_neural is None:
        return expected_neural is actual_neural, 0.0
    if len(expected_neural.trials) != len(actual_neural.trials):
        return False, float("inf")
    errors = [
        _max_abs_error(expected, actual)
        for expected, actual in zip(
            expected_neural.trials, actual_neural.trials, strict=True
        )
    ]
    neural_equal = all(
        np.asarray(expected).shape == np.asarray(actual).shape
        and np.allclose(expected, actual, rtol=1e-12, atol=1e-12)
        for expected, actual in zip(
            expected_neural.trials, actual_neural.trials, strict=True
        )
    )
    stimulus_equal = _stimulus_values_equal(recording, reloaded)
    external_equal = _optional_trial_values_equal(
        expected_neural.external_trials, actual_neural.external_trials
    )
    essentials_equal = (
        actual_neural.sfreq == expected_neural.sfreq
        and actual_neural.data_type == expected_neural.data_type
        and actual_neural.device_name == expected_neural.device_name
        and actual_neural.data_unit == expected_neural.data_unit
        and actual_neural.original_trial_positions
        == expected_neural.original_trial_positions
        and actual_neural.channel_names == expected_neural.channel_names
        and _metadata_equal(actual_neural.rereference, expected_neural.rereference)
        and _metadata_equal(
            actual_neural.padding_start_sample,
            expected_neural.padding_start_sample,
        )
        and actual_neural.cnd_version == expected_neural.cnd_version
        and actual_neural.external_layout == expected_neural.external_layout
        and actual_neural.external_group_names == expected_neural.external_group_names
        and actual_neural.external_group_channel_counts
        == expected_neural.external_group_channel_counts
        and _metadata_equal(
            actual_neural.external_group_fields,
            expected_neural.external_group_fields,
        )
        and _metadata_equal(
            reloaded.additional_variables,
            recording.additional_variables,
        )
    )
    return (
        neural_equal and stimulus_equal and external_equal and essentials_equal,
        max(errors, default=0.0),
    )


def _stimulus_values_equal(expected: CNDRecording, actual: CNDRecording) -> bool:
    expected_stimulus = expected.stimulus
    actual_stimulus = actual.stimulus
    if expected_stimulus is None or actual_stimulus is None:
        return expected_stimulus is actual_stimulus
    if (
        expected_stimulus.names != actual_stimulus.names
        or expected_stimulus.sfreq != actual_stimulus.sfreq
        or expected_stimulus.stimulus_indices != actual_stimulus.stimulus_indices
        or expected_stimulus.condition_indices != actual_stimulus.condition_indices
        or expected_stimulus.condition_names != actual_stimulus.condition_names
        or expected_stimulus.cnd_version != actual_stimulus.cnd_version
        or len(expected_stimulus.features) != len(actual_stimulus.features)
    ):
        return False
    return all(
        np.asarray(before).shape == np.asarray(after).shape
        and np.array_equal(before, after)
        for before_feature, after_feature in zip(
            expected_stimulus.features, actual_stimulus.features, strict=True
        )
        for before, after in zip(before_feature, after_feature, strict=True)
    )


def _optional_trial_values_equal(
    expected: tuple[np.ndarray, ...] | None,
    actual: tuple[np.ndarray, ...] | None,
) -> bool:
    if expected is None or actual is None:
        return expected is actual
    return len(expected) == len(actual) and all(
        np.asarray(before).shape == np.asarray(after).shape
        and np.array_equal(before, after)
        for before, after in zip(expected, actual, strict=True)
    )


def _metadata_equal(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, dict) or isinstance(actual, dict):
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return False
        return expected.keys() == actual.keys() and all(
            _metadata_equal(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        if not isinstance(expected, (list, tuple)) or not isinstance(
            actual, (list, tuple)
        ):
            return False
        return len(expected) == len(actual) and all(
            _metadata_equal(before, after)
            for before, after in zip(expected, actual, strict=True)
        )
    try:
        result = expected == actual
    except (TypeError, ValueError):
        return False
    return bool(np.all(result))


def _max_abs_error(expected: Any, actual: Any) -> float:
    difference = np.abs(
        np.asarray(expected, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    )
    return float(np.max(difference, initial=0.0))
