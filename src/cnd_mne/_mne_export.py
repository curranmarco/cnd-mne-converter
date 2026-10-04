"""Validate MNE inputs and build CND recordings."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import mne
import numpy as np

from ._channels import (
    _canonical_unit,
    _channel_spec,
    _extract_channel_locations,
    _unit_scale,
)
from .exceptions import CNDAmbiguousUnitError, CNDUnsupportedError, CNDValidationError
from .model import CNDNeural, CNDRecording, CNDStimulus, CNDVersion
from .validation import validate_cnd

UnsupportedMetadataPolicy = Literal["warn", "raise", "ignore"]


def from_mne(
    raws: mne.io.BaseRaw | Sequence[mne.io.BaseRaw],
    *,
    stimulus: CNDStimulus | None = None,
    external_raws: mne.io.BaseRaw | Sequence[mne.io.BaseRaw] | None = None,
    external_unit: str | None = None,
    external_description: str | None = None,
    cnsp_external_cells: bool = False,
    output_unit: str = "V",
    device_name: str | None = None,
    original_trial_positions: Sequence[int] | None = None,
    cnd_version: CNDVersion | None = 1.0,
    template: CNDRecording | None = None,
    on_unsupported_metadata: UnsupportedMetadataPolicy = "warn",
) -> CNDRecording:
    """Build CND from one or more MNE ``Raw`` objects.

    MNE stores EEG in volts. ``output_unit`` is what gets written to CND.
    Pass stimulus data yourself; MNE cannot invent envelopes. Auxiliary channels
    such as EOG can be passed separately through ``external_raws``. Use
    ``template`` (or :meth:`MNECNDRecording.to_cnd`) to keep leftover CND fields.
    New external inputs use MATLAB cell groups. Set ``cnsp_external_cells=True``
    to also normalize a legacy template single-group struct to that layout;
    the default preserves the template layout.
    """
    raw_trials = (raws,) if isinstance(raws, mne.io.BaseRaw) else tuple(raws)
    if not raw_trials:
        raise CNDValidationError("At least one MNE Raw object is required")
    if on_unsupported_metadata not in {"warn", "raise", "ignore"}:
        raise ValueError("on_unsupported_metadata must be 'warn', 'raise', or 'ignore'")
    _handle_unsupported_mne_metadata(raw_trials, on_unsupported_metadata)

    external_trials_raw = (
        ()
        if external_raws is None
        else (external_raws,)
        if isinstance(external_raws, mne.io.BaseRaw)
        else tuple(external_raws)
    )
    if external_trials_raw and external_unit is None:
        raise CNDAmbiguousUnitError(
            "external_unit is required when exporting external_raws"
        )
    _handle_unsupported_mne_metadata(external_trials_raw, on_unsupported_metadata)

    first = raw_trials[0]
    sfreq = float(first.info["sfreq"])
    template_neural = template.neural if template is not None else None
    if template is not None and template_neural is None:
        raise CNDValidationError("CND template has no neural data")
    template_data_type = (
        template_neural.data_type.strip().lower()
        if template_neural is not None
        else "eeg"
    )
    _check_neural_trials(raw_trials, template_data_type)

    external_trials = (
        _convert_external_trials(raw_trials, external_trials_raw, str(external_unit))
        if external_trials_raw
        else None
    )
    if template_neural is not None:
        _check_template(raw_trials, template_neural)

    mne_units_per_output_unit = _unit_scale(output_unit, template_data_type)
    trials = tuple(raw.get_data().T / mne_units_per_output_unit for raw in raw_trials)
    if original_trial_positions is None and template_neural is not None:
        original = template_neural.original_trial_positions
    elif original_trial_positions is None:
        original = tuple(range(1, len(raw_trials) + 1))
    else:
        original = tuple(int(value) for value in original_trial_positions)
    if original is not None and len(original) != len(raw_trials):
        raise CNDValidationError(
            "original_trial_positions length does not match MNE trial count"
        )

    if template_neural is not None:
        if template is None:
            raise CNDValidationError("Internal template state is inconsistent")
        neural = replace(
            template_neural,
            trials=trials,
            sfreq=sfreq,
            device_name=device_name or template_neural.device_name,
            original_trial_positions=original,
            data_unit=_canonical_unit(output_unit, template_data_type),
            source_path=None,
        )
        resolved_stimulus = stimulus if stimulus is not None else template.stimulus
    else:
        neural = _new_neural(
            first,
            trials,
            original,
            output_unit,
            device_name,
            external_description,
            cnd_version,
        )
        resolved_stimulus = stimulus
    neural = _attach_external(
        neural,
        external_trials,
        external_trials_raw,
        external_unit,
        external_description,
        cnsp_external_cells,
    )
    recording = CNDRecording(
        neural,
        resolved_stimulus,
        dict(template.additional_variables) if template is not None else {},
    )
    validate_cnd(recording).raise_for_errors()
    return recording


def _convert_external_trials(
    raws: Sequence[mne.io.BaseRaw], external_raws: Sequence[mne.io.BaseRaw], unit: str
) -> tuple[np.ndarray, ...]:
    """Check channel consistency and shared time origins before scaling."""
    sfreq = float(raws[0].info["sfreq"])
    if len(external_raws) != len(raws):
        raise CNDValidationError(
            "external_raws trial count does not match neural trial count"
        )
    external_names = tuple(external_raws[0].ch_names)
    external_types = tuple(external_raws[0].get_channel_types())
    external_scale = _unit_scale(unit, "eeg")
    converted_external: list[np.ndarray] = []
    for index, (raw, external) in enumerate(zip(raws, external_raws, strict=True)):
        if not np.isclose(float(external.info["sfreq"]), sfreq, rtol=0, atol=1e-12):
            raise CNDValidationError(
                f"MNE external trial {index} has a different sampling rate"
            )
        if (
            external.first_samp != raw.first_samp
            or external.info["meas_date"] != raw.info["meas_date"]
        ):
            raise CNDValidationError(
                f"MNE external trial {index} has a different time origin"
            )
        if external.n_times != raw.n_times:
            raise CNDValidationError(
                f"MNE external trial {index} has a different sample count"
            )
        if tuple(external.ch_names) != external_names:
            raise CNDValidationError(
                f"MNE external trial {index} has different channel names/order"
            )
        if tuple(external.get_channel_types()) != external_types:
            raise CNDValidationError(
                f"MNE external trial {index} has different channel types"
            )
        converted_external.append(external.get_data().T / external_scale)
    return tuple(converted_external)


def _check_template(raws: Sequence[mne.io.BaseRaw], neural: CNDNeural) -> None:
    """Reject edits that would break the template's channel or trial alignment."""
    if len(raws) != neural.n_trials:
        raise CNDValidationError("MNE trial count does not match the CND template")
    template_names, template_types = _channel_spec(neural)
    if tuple(raws[0].ch_names) != template_names:
        raise CNDValidationError(
            "MNE channel names/order do not match the CND template"
        )
    if tuple(raws[0].get_channel_types()) != template_types:
        raise CNDValidationError("MNE channel types do not match the CND template")
    for index, (raw, source_trial) in enumerate(zip(raws, neural.trials, strict=True)):
        if raw.n_times != np.asarray(source_trial).shape[0]:
            raise CNDValidationError(
                f"MNE trial {index} length does not match the CND template; "
                "supply synchronized replacement stimulus/external data through "
                "a new CND recording instead"
            )


def _handle_unsupported_mne_metadata(
    raws: Sequence[mne.io.BaseRaw], policy: UnsupportedMetadataPolicy
) -> None:
    if policy == "ignore":
        return
    issues: list[str] = []
    for index, raw in enumerate(raws):
        if len(raw.annotations):
            issues.append(f"trial {index}: annotations")
        if raw.info["bads"]:
            issues.append(f"trial {index}: bad-channel list")
        if raw.info["projs"]:
            issues.append(f"trial {index}: projection metadata")
        if raw.first_samp != 0:
            issues.append(f"trial {index}: non-zero first sample")
    if not issues:
        return
    message = (
        "MNE metadata has no standardized CND mapping and will not be exported: "
        + ", ".join(issues)
    )
    if policy == "raise":
        raise CNDUnsupportedError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _check_neural_trials(raw_trials: Sequence[mne.io.BaseRaw], data_type: str) -> None:
    """Require one channel layout and sampling rate across neural trials."""
    sfreq = float(raw_trials[0].info["sfreq"])
    ch_names = tuple(raw_trials[0].ch_names)
    ch_types = tuple(raw_trials[0].get_channel_types())
    allowed_types = (
        {"hbo", "hbr", "misc"} if data_type in {"fnirs", "nirs"} else {"eeg"}
    )
    if any(channel_type not in allowed_types for channel_type in ch_types):
        raise CNDUnsupportedError(
            f"Exporter for {data_type!r} does not support channel types "
            f"{sorted(set(ch_types) - allowed_types)!r}"
        )
    for index, raw in enumerate(raw_trials[1:], start=1):
        if not np.isclose(float(raw.info["sfreq"]), sfreq, rtol=0, atol=1e-12):
            raise CNDValidationError(f"MNE trial {index} has a different sampling rate")
        if tuple(raw.ch_names) != ch_names:
            raise CNDValidationError(
                f"MNE trial {index} has different channel names/order"
            )
        if tuple(raw.get_channel_types()) != ch_types:
            raise CNDValidationError(f"MNE trial {index} has different channel types")


def _new_neural(
    first: mne.io.BaseRaw,
    trials: tuple[np.ndarray, ...],
    original: tuple[int, ...] | None,
    output_unit: str,
    device_name: str | None,
    external_description: str | None,
    cnd_version: CNDVersion | None,
) -> CNDNeural:
    """Describe a new EEG recording, including explicit coordinate units."""
    sfreq = float(first.info["sfreq"])
    ch_names = tuple(first.ch_names)
    positions = _extract_channel_locations(first)
    extras: dict[str, object] = {}
    if positions is not None:
        extras["coordUnit"] = "m"
        extras["coordTransform"] = "MNE-head-to-EEGLAB-axis"
    return CNDNeural(
        trials=trials,
        sfreq=sfreq,
        data_type="EEG",
        device_name=device_name,
        original_trial_positions=original,
        channel_locations=positions or tuple({"labels": name} for name in ch_names),
        external_description=external_description,
        cnd_version=cnd_version,
        data_unit=_canonical_unit(output_unit, "eeg"),
        extra_fields=extras,
        variable_name="eeg",
    )


def _attach_external(
    neural: CNDNeural,
    external_trials: tuple[np.ndarray, ...] | None,
    external_trials_raw: Sequence[mne.io.BaseRaw],
    external_unit: str | None,
    external_description: str | None,
    cnsp_external_cells: bool,
) -> CNDNeural:
    """Attach external metadata and preserve or normalize the MATLAB layout."""
    if external_trials is not None:
        neural = replace(
            neural,
            external_trials=external_trials,
            external_description=external_description,
            external_fields={
                "channelNames": np.asarray(
                    external_trials_raw[0].ch_names, dtype=object
                ),
                "channelTypes": np.asarray(
                    external_trials_raw[0].get_channel_types(), dtype=object
                ),
                "dataUnit": _canonical_unit(str(external_unit), "eeg"),
            },
            external_layout="single_struct",
        )
    if (external_trials is not None or cnsp_external_cells) and (
        neural.external_trials is not None and neural.external_layout == "single_struct"
    ):
        # scipy simplifies a singleton MATLAB cell to its contained struct.
        # Canonicalise single-group MNE exports so CNSP brace indexing survives.
        neural = replace(
            neural,
            external_layout="struct_array",
            external_group_names=(neural.external_description or "External channels",),
            external_group_channel_counts=(int(neural.external_trials[0].shape[1]),),
            external_group_fields=(dict(neural.external_fields),),
        )
    return neural
