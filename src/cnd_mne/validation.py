"""Checks on CND recordings."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .exceptions import CNDValidationError
from .model import CNDNeural, CNDRecording, CNDStimulus

Severity = Literal["error", "warning"]


@dataclass(slots=True, frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    path: str
    message: str


@dataclass(slots=True, frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if not self.errors:
            return
        summary = "; ".join(f"{issue.path}: {issue.message}" for issue in self.errors)
        raise CNDValidationError(summary)


def validate_cnd(
    recording: CNDRecording,
    *,
    duration_tolerance_seconds: float | None = None,
    strict_spec: bool = False,
) -> ValidationReport:
    """Check layout and paired durations; distinct sampling rates stay warnings."""
    if duration_tolerance_seconds is not None and duration_tolerance_seconds < 0:
        raise ValueError("duration_tolerance_seconds must be non-negative")
    return ValidationReport(
        tuple(_recording_issues(recording, strict_spec, duration_tolerance_seconds))
    )


def _recording_issues(
    recording: CNDRecording, strict_spec: bool, duration_tolerance_seconds: float | None
) -> Iterator[ValidationIssue]:
    neural, stimulus = recording.neural, recording.stimulus
    if neural is None and stimulus is None:
        yield _error("empty_recording", "$", "No neural or stimulus data")
    if neural is not None and neural.variable_name in recording.additional_variables:
        yield _error(
            "duplicate_file_variable",
            "additional_variables",
            f"Contains selected neural variable {neural.variable_name!r}",
        )
    if "stim" in recording.additional_variables:
        yield _error(
            "duplicate_stimulus_variable",
            "additional_variables",
            "The reserved 'stim' variable must use recording.stimulus",
        )
    if neural is not None:
        yield from _neural_issues(neural, strict_spec)
        yield from _version_issues(neural.cnd_version, "neural", strict_spec)
        yield from _external_issues(neural)
    if stimulus is not None:
        yield from _stimulus_issues(stimulus, strict_spec)
        yield from _version_issues(stimulus.cnd_version, "stimulus", strict_spec)
    if neural is not None and stimulus is not None:
        yield from _pairing_issues(
            neural, stimulus, strict_spec, duration_tolerance_seconds
        )


def _neural_issues(neural: CNDNeural, strict_spec: bool) -> Iterator[ValidationIssue]:
    if not np.isfinite(neural.sfreq) or neural.sfreq <= 0:
        yield _error("invalid_sfreq", "neural.sfreq", "Must be positive")
    if not neural.trials:
        yield _error("missing_trials", "neural.trials", "No neural trials")
    expected_channels: int | None = None
    for index, trial in enumerate(neural.trials):
        array = np.asarray(trial)
        path = f"neural.trials[{index}]"
        if array.ndim != 2:
            yield _error("invalid_neural_shape", path, "Expected time x channels")
            continue
        if array.shape[0] == 0 or array.shape[1] == 0:
            yield _spec_issue(
                strict_spec,
                "empty_neural_trial",
                path,
                "Trial is empty; retained as a zero-sample trial",
            )
        if expected_channels is None:
            expected_channels = int(array.shape[1])
        elif array.shape[1] != expected_channels:
            yield _error(
                "channel_count_mismatch",
                path,
                f"Expected {expected_channels} channels, got {array.shape[1]}",
            )
    if (
        neural.original_trial_positions is not None
        and len(neural.original_trial_positions) != neural.n_trials
    ):
        yield _error(
            "trial_position_count",
            "neural.original_trial_positions",
            "Length does not match neural trial count",
        )
    elif neural.original_trial_positions is not None:
        expected_positions = set(range(1, neural.n_trials + 1))
        if set(neural.original_trial_positions) != expected_positions:
            yield _spec_issue(
                strict_spec,
                "invalid_trial_positions",
                "neural.original_trial_positions",
                "Expected a one-based permutation of all trial positions",
            )
    if (
        neural.channel_locations is not None
        and expected_channels is not None
        and (len(neural.channel_locations) != expected_channels)
    ):
        yield _error(
            "channel_location_count",
            "neural.channel_locations",
            "Length does not match neural channel count",
        )
    if neural.channel_locations is None:
        yield _warning(
            "missing_channel_locations",
            "neural.channel_locations",
            "MNE conversion will use generated channel names and no montage",
        )
    if neural.signal_types is not None:
        counts = neural.channels_per_signal_type
        if counts is None or len(counts) != len(neural.signal_types):
            yield _error(
                "signal_type_count",
                "neural.channels_per_signal_type",
                "Must match neural.signal_types",
            )
        elif expected_channels is not None and sum(counts) != expected_channels:
            yield _error(
                "signal_channel_count",
                "neural.channels_per_signal_type",
                "Sum does not match neural channel count",
            )
    if neural.channel_names is not None and len(set(neural.channel_names)) != len(
        neural.channel_names
    ):
        yield _error(
            "duplicate_channel_names",
            "neural.channel_locations",
            "Channel labels must be unique for MNE conversion",
        )
    if neural.data_unit is None:
        yield _warning(
            "missing_data_unit",
            "neural.data_unit",
            "MNE conversion requires an explicit physical unit",
        )


def _external_issues(neural: CNDNeural) -> Iterator[ValidationIssue]:
    if neural.external_trials is not None:
        if neural.external_layout in {"named_fields", "struct_array"}:
            names = neural.external_group_names
            counts = neural.external_group_channel_counts
            if (
                names is None
                or counts is None
                or len(names) != len(counts)
                or (not neural.external_trials)
                or (sum(counts) != np.asarray(neural.external_trials[0]).shape[1])
            ):
                yield _error(
                    "external_group_metadata",
                    "neural.external_group_channel_counts",
                    "Group names/counts must describe every external channel",
                )
        if len(neural.external_trials) != neural.n_trials:
            yield _error(
                "external_trial_count",
                "neural.external_trials",
                "Length does not match neural trial count",
            )
        else:
            for index, (trial, external) in enumerate(
                zip(neural.trials, neural.external_trials, strict=True)
            ):
                if np.asarray(external).ndim != 2:
                    yield _error(
                        "invalid_external_shape",
                        f"neural.external_trials[{index}]",
                        "Expected time x external channels",
                    )
                elif np.asarray(external).shape[0] != np.asarray(trial).shape[0]:
                    yield _warning(
                        "external_length_mismatch",
                        f"neural.external_trials[{index}]",
                        "External and neural sample counts differ",
                    )


def _stimulus_issues(
    stimulus: CNDStimulus, strict_spec: bool
) -> Iterator[ValidationIssue]:
    if not np.isfinite(stimulus.sfreq) or stimulus.sfreq <= 0:
        yield _error("invalid_sfreq", "stimulus.sfreq", "Must be positive")
    if len(stimulus.names) != len(stimulus.features):
        yield _error(
            "feature_name_count",
            "stimulus.names",
            "Name count does not match feature count",
        )
    if len(set(stimulus.names)) != len(stimulus.names):
        yield _spec_issue(
            strict_spec,
            "duplicate_feature_names",
            "stimulus.names",
            "Stimulus feature-set names must be unique",
        )
    expected_trials = stimulus.n_trials
    if stimulus.stimulus_indices is None:
        yield _warning(
            "missing_stimulus_indices",
            "stimulus.stimulus_indices",
            "Using ordinal one-based trial indices in memory",
        )
    elif len(stimulus.stimulus_indices) != expected_trials:
        yield _error(
            "stimulus_index_count",
            "stimulus.stimulus_indices",
            "Length does not match stimulus trial count",
        )
    for feature_index, trials in enumerate(stimulus.features):
        if len(trials) != expected_trials:
            yield _error(
                "stimulus_trial_count",
                f"stimulus.features[{feature_index}]",
                f"Expected {expected_trials} trials, got {len(trials)}",
            )
        for trial_index, trial in enumerate(trials):
            array = np.asarray(trial)
            if array.ndim not in {1, 2}:
                yield _error(
                    "invalid_stimulus_shape",
                    f"stimulus.features[{feature_index}][{trial_index}]",
                    "Expected time or time x feature-dimension data",
                )
            elif array.shape[0] == 0:
                yield _spec_issue(
                    strict_spec,
                    "empty_stimulus_trial",
                    f"stimulus.features[{feature_index}][{trial_index}]",
                    "Stimulus trial is empty",
                )
    if (
        stimulus.condition_indices is not None
        and len(stimulus.condition_indices) != expected_trials
    ):
        yield _error(
            "condition_index_count",
            "stimulus.condition_indices",
            "Length does not match stimulus trial count",
        )
    for trial_index in range(expected_trials):
        lengths = [
            np.asarray(feature[trial_index]).shape[0]
            for feature in stimulus.features
            if len(feature) > trial_index
        ]
        if lengths and len(set(lengths)) != 1:
            yield _spec_issue(
                strict_spec,
                "feature_length_mismatch",
                f"stimulus.trial[{trial_index}]",
                f"Feature lengths differ: {lengths}",
            )


def _version_issues(
    version: object, path: str, strict_spec: bool
) -> Iterator[ValidationIssue]:
    if version is None:
        yield _warning(
            "missing_cnd_version",
            f"{path}.cnd_version",
            "Legacy file does not declare the CND specification version",
        )
    elif not isinstance(version, (int, float, np.integer, np.floating)):
        yield _spec_issue(
            strict_spec,
            "non_numeric_cnd_version",
            f"{path}.cnd_version",
            "CND 1.0 specifies cndVersion as a numeric scalar",
        )


def _pairing_issues(
    neural: CNDNeural,
    stimulus: CNDStimulus,
    strict_spec: bool,
    duration_tolerance_seconds: float | None,
) -> Iterator[ValidationIssue]:
    if neural.n_trials != stimulus.n_trials:
        yield _spec_issue(
            strict_spec,
            "paired_trial_count",
            "$",
            f"Neural has {neural.n_trials} trials; stimulus has {stimulus.n_trials}",
        )
    elif (
        np.isfinite(neural.sfreq)
        and neural.sfreq > 0
        and np.isfinite(stimulus.sfreq)
        and (stimulus.sfreq > 0)
    ):
        if not np.isclose(neural.sfreq, stimulus.sfreq, rtol=0, atol=0):
            yield _warning(
                "sampling_frequency_mismatch",
                "$",
                "Neural and stimulus sampling rates differ; preserve both "
                "stored clocks and align/resample during analysis. Found "
                f"{neural.sfreq:g} and {stimulus.sfreq:g} Hz",
            )
        if duration_tolerance_seconds is None:
            duration_tolerance_seconds = max(1 / neural.sfreq, 1 / stimulus.sfreq)
        for index in range(neural.n_trials):
            neural_duration = np.asarray(neural.trials[index]).shape[0] / neural.sfreq
            if not stimulus.features:
                continue
            stimulus_duration = (
                np.asarray(stimulus.features[0][index]).shape[0] / stimulus.sfreq
            )
            difference = abs(neural_duration - stimulus_duration)
            if difference > duration_tolerance_seconds:
                yield _warning(
                    "duration_mismatch",
                    f"trial[{index}]",
                    f"Neural/stimulus durations differ by {difference:.6f} s",
                )


def _error(code: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue("error", code, path, message)


def _warning(code: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue("warning", code, path, message)


def _spec_issue(
    strict_spec: bool, code: str, path: str, message: str
) -> ValidationIssue:
    return (_error if strict_spec else _warning)(code, path, message)
