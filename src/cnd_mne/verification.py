"""Check a folder of CND files with ``verify-dataset``."""

from __future__ import annotations

import platform
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import mne
import numpy as np

from ._channels import _unit_scale
from ._paths import _subject_from_filename
from ._round_trip import (
    _check_memory_round_trip,
    _max_abs_error,
    _mne_psd_is_finite,
    _raw_values_match,
    _serialized_round_trip,
)
from ._verification_report import DatasetVerification as DatasetVerification
from ._verification_report import SubjectVerification as SubjectVerification
from .exceptions import CNDValidationError
from .io import read_cnd
from .mne import MNECNDRecording, to_mne
from .model import CNDRecording
from .validation import ValidationIssue, ValidationReport, validate_cnd


def verify_dataset(
    path: str | Path,
    *,
    dataset_name: str | None = None,
    neural_unit: str | None = None,
    strict_spec: bool = False,
    round_trip: bool = True,
    serialized_round_trip: bool = False,
    serialized_mat_version: Literal["5", "7.3"] = "5",
    mne_smoke_test: bool = True,
) -> DatasetVerification:
    """Verify every ``dataSub*.mat`` file in a local CND dataset.

    ``neural_unit`` is an explicit testing assumption for legacy datasets. It
    is recorded in the report and is never inferred. When omitted, parsing and
    validation run but MNE and round-trip checks are skipped.

    ``serialized_round_trip`` additionally writes the converted recording to a
    temporary MATLAB file and reads it back. It is opt-in because serializing
    every subject in a multi-gigabyte public collection can require substantial
    time and temporary disk space. Both supported writer formats can be tested
    with ``serialized_mat_version="5"`` or ``"7.3"``.
    """
    if serialized_mat_version not in {"5", "7.3"}:
        raise ValueError("serialized_mat_version must be '5' or '7.3'")
    if serialized_round_trip and not round_trip:
        raise ValueError("serialized_round_trip requires round_trip=True")
    supplied_path = Path(path).expanduser()
    source = supplied_path.resolve()
    data_directory = source / "dataCND" if (source / "dataCND").is_dir() else source
    discovered_files = sorted(
        {
            *data_directory.glob("dataSub*.mat"),
            *data_directory.glob("pre_dataSub*.mat"),
            *data_directory.glob("dataParticipant_*.mat"),
        },
        key=_subject_sort_key,
    )
    if not discovered_files:
        raise FileNotFoundError(
            f"No CND subject files found in {data_directory}; expected "
            "dataSub*.mat, pre_dataSub*.mat, or dataParticipant_*.mat"
        )
    skipped_empty_files = tuple(
        file.name for file in discovered_files if file.stat().st_size == 0
    )
    files = [file for file in discovered_files if file.stat().st_size > 0]
    if not files:
        raise FileNotFoundError(
            f"No non-empty CND subject files found in {data_directory}"
        )
    subjects = tuple(
        _verify_subject(
            file,
            neural_unit=neural_unit,
            strict_spec=strict_spec,
            round_trip=round_trip,
            serialized_round_trip=serialized_round_trip,
            serialized_mat_version=serialized_mat_version,
            mne_smoke_test=mne_smoke_test,
        )
        for file in files
    )
    return DatasetVerification(
        schema_version=4,
        dataset_name=dataset_name or source.name,
        dataset_path=(
            source.name if supplied_path.is_absolute() else str(supplied_path)
        ),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        versions={
            "cnd_mne": _package_version(),
            "mne": mne.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        neural_unit_assumption=neural_unit,
        strict_spec=strict_spec,
        round_trip_requested=round_trip and neural_unit is not None,
        serialized_round_trip_requested=(
            serialized_round_trip and neural_unit is not None
        ),
        serialized_mat_version=(
            serialized_mat_version
            if serialized_round_trip and neural_unit is not None
            else None
        ),
        mne_smoke_test_requested=mne_smoke_test and neural_unit is not None,
        skipped_empty_files=skipped_empty_files,
        subjects=subjects,
    )


def _verify_subject(
    neural_file: Path,
    *,
    neural_unit: str | None,
    strict_spec: bool,
    round_trip: bool,
    serialized_round_trip: bool,
    serialized_mat_version: Literal["5", "7.3"],
    mne_smoke_test: bool,
) -> SubjectVerification:
    started = time.monotonic()
    subject = _subject_label(neural_file)
    stage = "read"
    result = asdict(
        SubjectVerification(
            subject=subject, outcome="source_read_failure", neural_file=neural_file.name
        )
    )
    del result["elapsed_seconds"]  # Set once the check finishes.
    try:
        recording = read_cnd(neural_file)
        stage = "validate"
        report = validate_cnd(recording, strict_spec=strict_spec)
        neural = recording.neural
        if neural is None:
            raise CNDValidationError("Subject file did not produce neural data")
        result.update(_structural_results(recording, report))
        if report.errors:
            result["outcome"] = "validation_failure"
            return SubjectVerification(
                **result, elapsed_seconds=round(time.monotonic() - started, 6)
            )
        if neural_unit is None:
            result["outcome"] = (
                "empty_neural_data"
                if result["n_neural_samples"] == 0
                else "structural_pass"
            )
            return SubjectVerification(
                **result, elapsed_seconds=round(time.monotonic() - started, 6)
            )

        stage = "convert"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            converted = to_mne(recording, neural_unit=neural_unit)
        checks, passed = _conversion_results(
            recording,
            converted,
            neural_unit,
            round_trip,
            serialized_round_trip,
            serialized_mat_version,
            mne_smoke_test,
        )
        checks["conversion_warnings"] = tuple(str(item.message) for item in caught)
        result.update(checks)
        if result["n_neural_samples"] == 0:
            result["outcome"] = "empty_neural_data"
        elif passed:
            result["outcome"] = "complete_pass"
        else:
            result["outcome"] = "verification_failure"
    except Exception as error:  # report each subject without aborting the matrix
        result["failure"] = _safe_failure_message(error, neural_file)
        if stage == "read" and isinstance(error, CNDValidationError):
            result["outcome"] = "validation_failure"
        else:
            result["outcome"] = {
                "read": "source_read_failure",
                "validate": "validation_failure",
                "convert": "conversion_failure",
            }[stage]
    return SubjectVerification(
        **result, elapsed_seconds=round(time.monotonic() - started, 6)
    )


def _max_duration_difference(recording: CNDRecording) -> float | None:
    differences = [
        abs(metadata.neural_duration_seconds - metadata.stimulus_duration_seconds)
        for metadata in recording.trial_metadata
        if metadata.neural_duration_seconds is not None
        and metadata.stimulus_duration_seconds is not None
    ]
    return max(differences) if differences else None


def _issue_dict(issue: ValidationIssue) -> dict[str, str]:
    return {"code": issue.code, "path": issue.path, "message": issue.message}


def _subject_sort_key(path: Path) -> tuple[int, str]:
    label = _subject_label(path)
    return (int(label), label) if label.isdigit() else (2**31 - 1, label)


def _subject_label(path: Path) -> str:
    label = _subject_from_filename(path)
    return path.stem if label is None else label


def _safe_failure_message(error: Exception, neural_file: Path) -> str:
    message = str(error).replace(str(neural_file), neural_file.name)
    temporary = Path(tempfile.gettempdir())
    for prefix in {str(temporary), str(temporary.resolve())}:
        message = message.replace(prefix, "<temporary>")
    return f"{type(error).__name__}: {message}"


def _package_version() -> str:
    try:
        return version("cnd-mne-converter")
    except PackageNotFoundError:
        return "unknown"


def _structural_results(
    recording: CNDRecording, report: ValidationReport
) -> dict[str, Any]:
    """Describe the source and its validation issues before conversion."""
    neural, stimulus = recording.neural, recording.stimulus
    if neural is None:
        raise CNDValidationError("Subject file did not produce neural data")
    return {
        "stimulus_file": (
            stimulus.source_path.name
            if stimulus is not None and stimulus.source_path is not None
            else None
        ),
        "n_trials": neural.n_trials,
        "n_channels": neural.n_channels,
        "n_neural_samples": sum(np.asarray(trial).shape[0] for trial in neural.trials),
        "neural_sfreq": neural.sfreq,
        "stimulus_sfreq": stimulus.sfreq if stimulus is not None else None,
        "n_stimulus_features": (stimulus.n_features if stimulus is not None else 0),
        "warning_counts": dict(
            sorted(Counter(issue.code for issue in report.warnings).items())
        ),
        "validation_errors": tuple(_issue_dict(issue) for issue in report.errors),
        "max_duration_difference_seconds": _max_duration_difference(recording),
    }


def _conversion_results(
    recording: CNDRecording,
    converted: MNECNDRecording,
    neural_unit: str,
    round_trip: bool,
    serialized_round_trip: bool,
    serialized_mat_version: Literal["5", "7.3"],
    mne_smoke_test: bool,
) -> tuple[dict[str, Any], bool]:
    """Run requested checks and return their results together."""
    neural, stimulus = recording.neural, recording.stimulus
    if neural is None:
        raise CNDValidationError("Subject file did not produce neural data")
    scale = _unit_scale(neural_unit, neural.data_type.strip().lower())
    shape_verified = True
    mne_error = 0.0
    for source_trial, raw in zip(neural.trials, converted.raws, strict=True):
        expected = np.asarray(source_trial)
        actual_source_units = raw.get_data().T / scale
        shape_verified &= actual_source_units.shape == expected.shape
        if actual_source_units.shape == expected.shape:
            mne_error = max(mne_error, _max_abs_error(expected, actual_source_units))

    psd_finite = None
    if mne_smoke_test:
        nonempty = next((raw for raw in converted.raws if raw.n_times), None)
        psd_finite = _mne_psd_is_finite(nonempty) if nonempty is not None else False

    stimulus_views_verified = None
    if stimulus is not None:
        stimulus_views_verified = all(
            _raw_values_match(trials, converted.stimulus_raws(index))
            for index, trials in enumerate(stimulus.features)
        )
    external_views_verified = None
    if neural.external_trials is not None:
        # Identity scaling tests values without claiming physiological units.
        external_views_verified = _raw_values_match(
            neural.external_trials,
            converted.external_raws(unit="V", channel_types="misc"),
        )

    round_trip_verified = False
    round_trip_error = None
    metadata_preserved = None
    serialized_verified = None
    serialized_error = None
    if round_trip:
        (
            converted_back,
            round_trip_error,
            metadata_preserved,
            round_trip_verified,
        ) = _check_memory_round_trip(recording, converted)
        if serialized_round_trip:
            serialized_verified, serialized_error = _serialized_round_trip(
                converted_back,
                mat_version=serialized_mat_version,
            )

    return (
        {
            "mne_created": True,
            "mne_shape_verified": shape_verified,
            "mne_max_abs_error_source_units": mne_error,
            "stimulus_mne_views_verified": stimulus_views_verified,
            "external_mne_views_verified": external_views_verified,
            "mne_psd_finite": psd_finite,
            "round_trip_verified": round_trip_verified,
            "round_trip_max_abs_error_source_units": round_trip_error,
            "cnd_metadata_preserved": metadata_preserved,
            "serialized_round_trip_verified": serialized_verified,
            "serialized_round_trip_max_abs_error_source_units": (serialized_error),
        },
        shape_verified
        and stimulus_views_verified is not False
        and external_views_verified is not False
        and (not round_trip or round_trip_verified)
        and (not serialized_round_trip or serialized_verified is True)
        and (not mne_smoke_test or psd_finite is True),
    )
