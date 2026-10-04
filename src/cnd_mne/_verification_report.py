"""Dataset report types and summary aggregation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class SubjectVerification:
    """Result for one subject file."""

    subject: str
    outcome: str
    neural_file: str
    stimulus_file: str | None = None
    n_trials: int = 0
    n_channels: int = 0
    n_neural_samples: int = 0
    neural_sfreq: float = 0.0
    stimulus_sfreq: float | None = None
    n_stimulus_features: int = 0
    warning_counts: dict[str, int] = field(default_factory=dict)
    validation_errors: tuple[dict[str, str], ...] = ()
    max_duration_difference_seconds: float | None = None
    mne_created: bool = False
    mne_shape_verified: bool = False
    mne_max_abs_error_source_units: float | None = None
    stimulus_mne_views_verified: bool | None = None
    external_mne_views_verified: bool | None = None
    mne_psd_finite: bool | None = None
    round_trip_verified: bool = False
    round_trip_max_abs_error_source_units: float | None = None
    cnd_metadata_preserved: bool | None = None
    serialized_round_trip_verified: bool | None = None
    serialized_round_trip_max_abs_error_source_units: float | None = None
    conversion_warnings: tuple[str, ...] = ()
    failure: str | None = None
    elapsed_seconds: float = 0.0


@dataclass(slots=True, frozen=True)
class DatasetVerification:
    """Report for a whole CND dataset."""

    schema_version: int
    dataset_name: str
    dataset_path: str
    generated_at_utc: str
    versions: dict[str, str]
    neural_unit_assumption: str | None
    strict_spec: bool
    round_trip_requested: bool
    serialized_round_trip_requested: bool
    serialized_mat_version: str | None
    mne_smoke_test_requested: bool
    skipped_empty_files: tuple[str, ...]
    subjects: tuple[SubjectVerification, ...]

    @property
    def passed(self) -> bool:
        """True if no file failed to open or convert.

        Empty-but-valid files count as a pass here (``empty_neural_data``).
        """
        return all(
            subject.outcome in {"complete_pass", "structural_pass", "empty_neural_data"}
            for subject in self.subjects
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON report with totals."""
        result = asdict(self)
        warning_counts: Counter[str] = Counter()
        for subject in self.subjects:
            warning_counts.update(subject.warning_counts)
        max_mne_error = _optional_max(
            subject.mne_max_abs_error_source_units for subject in self.subjects
        )
        max_round_trip_error = _optional_max(
            subject.round_trip_max_abs_error_source_units for subject in self.subjects
        )
        outcome_counts = Counter(subject.outcome for subject in self.subjects)
        result["summary"] = {
            "passed": self.passed,
            "n_discovered_subject_files": len(self.subjects)
            + len(self.skipped_empty_files),
            "n_skipped_empty_files": len(self.skipped_empty_files),
            "n_subjects": len(self.subjects),
            "n_failed_subjects": sum(
                count
                for outcome, count in outcome_counts.items()
                if outcome
                not in {"complete_pass", "structural_pass", "empty_neural_data"}
            ),
            "n_empty_neural_files": outcome_counts["empty_neural_data"],
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "n_trials": sum(subject.n_trials for subject in self.subjects),
            "n_neural_samples": sum(
                subject.n_neural_samples for subject in self.subjects
            ),
            "warning_counts": dict(sorted(warning_counts.items())),
            "max_mne_abs_error_source_units": max_mne_error,
            "max_round_trip_abs_error_source_units": max_round_trip_error,
        }
        return result


def _optional_max(values: Any) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None
