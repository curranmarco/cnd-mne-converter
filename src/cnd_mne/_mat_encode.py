"""Encode CND models as MATLAB structures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .exceptions import CNDReadError
from .model import CNDNeural, CNDStimulus


def _neural_to_mat(neural: CNDNeural) -> dict[str, Any]:
    result = dict(neural.extra_fields)
    result.update(
        {
            "dataType": neural.data_type,
            "fs": neural.sfreq,
            "data": _neural_data_to_mat(neural),
        }
    )
    if neural.device_name is not None:
        result["deviceName"] = neural.device_name
    if neural.original_trial_positions is not None:
        result["origTrialPosition"] = np.asarray(neural.original_trial_positions)
    if neural.channel_locations is not None:
        result["chanlocs"] = (
            neural.channel_locations_raw
            if neural.channel_locations_raw is not None
            else _struct_array(neural.channel_locations)
        )
    if neural.external_trials is not None:
        result["extChan"] = _external_to_mat(neural)
    elif neural.external_fields:
        result["extChan"] = dict(neural.external_fields)
    if neural.rereference is not None:
        result["reRef"] = neural.rereference
    if neural.padding_start_sample is not None:
        result["paddingStartSample"] = neural.padding_start_sample
    if neural.cnd_version is not None:
        result["cndVersion"] = neural.cnd_version
    if neural.data_unit is not None:
        result["dataUnit"] = neural.data_unit
    if neural.signal_types is not None:
        result["datatype"] = _cell_row(neural.signal_types)
    return _without_none(result)


def _external_to_mat(neural: CNDNeural) -> Any:
    trials = neural.external_trials
    if trials is None:
        raise CNDReadError("Cannot serialize absent external-channel trials")
    layout = neural.external_layout or "single_struct"
    if layout == "single_struct":
        external = dict(neural.external_fields)
        external.update(
            {
                "data": _cell_row(trials),
                "description": neural.external_description or "External channels",
            }
        )
        return external

    names = neural.external_group_names
    counts = neural.external_group_channel_counts
    if names is None or counts is None or len(names) != len(counts):
        raise CNDReadError(
            f"external_layout={layout!r} requires matching group names and counts"
        )
    if sum(counts) != int(np.asarray(trials[0]).shape[1]):
        raise CNDReadError("external group channel counts do not match external data")
    stops = np.cumsum((0, *counts))
    split_trials = tuple(
        tuple(np.asarray(trial)[:, stops[index] : stops[index + 1]] for trial in trials)
        for index in range(len(counts))
    )
    if layout == "named_fields":
        return {
            name: _cell_row(group)
            for name, group in zip(names, split_trials, strict=True)
        }
    if layout == "struct_array":
        fields = neural.external_group_fields or tuple({} for _ in names)
        if len(fields) != len(names):
            raise CNDReadError("external group metadata count does not match groups")
        groups = np.empty(len(names), dtype=object)
        for index, (name, group, metadata) in enumerate(
            zip(names, split_trials, fields, strict=True)
        ):
            item = dict(metadata)
            item["data"] = _cell_row(group)
            if name:
                item["description"] = name
            groups[index] = item
        return groups
    raise CNDReadError(f"Unsupported external channel layout {layout!r}")


def _neural_data_to_mat(neural: CNDNeural) -> np.ndarray:
    if neural.signal_types is None:
        return _cell_row(neural.trials)
    counts = neural.channels_per_signal_type
    if counts is None or len(counts) != len(neural.signal_types):
        raise CNDReadError("signal_types require matching channels_per_signal_type")
    data = np.empty((len(counts), neural.n_trials), dtype=object)
    stops = np.cumsum((0, *counts))
    for signal_index in range(len(counts)):
        for trial_index, trial in enumerate(neural.trials):
            data[signal_index, trial_index] = np.asarray(trial)[
                :, stops[signal_index] : stops[signal_index + 1]
            ]
    return data


def _stimulus_to_mat(stimulus: CNDStimulus) -> dict[str, Any]:
    result = dict(stimulus.extra_fields)
    data = np.empty((stimulus.n_features, stimulus.n_trials), dtype=object)
    for feature_index, trials in enumerate(stimulus.features):
        for trial_index, trial in enumerate(trials):
            array = np.asarray(trial)
            data[feature_index, trial_index] = (
                array[:, None] if array.ndim == 1 else array
            )
    result.update(
        {
            "names": _cell_row(stimulus.names),
            "data": data,
            "fs": stimulus.sfreq,
        }
    )
    if stimulus.stimulus_indices is not None:
        result["stimIdxs"] = _sequence_to_mat(stimulus.stimulus_indices)
    if stimulus.condition_indices is not None:
        result["condIdxs"] = _sequence_to_mat(stimulus.condition_indices)
    if stimulus.condition_names is not None:
        result["condNames"] = _cell_row(stimulus.condition_names)
    if stimulus.cnd_version is not None:
        result["cndVersion"] = stimulus.cnd_version
    return _without_none(result)


def _cell_row(values: Sequence[Any]) -> np.ndarray:
    cell = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        cell[0, index] = value
    return cell


def _sequence_to_mat(values: Sequence[Any]) -> np.ndarray:
    if any(isinstance(value, (str, np.str_)) for value in values):
        return _cell_row(values)
    return np.asarray(values)


def _struct_array(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    fields = sorted({key for record in records for key in record})
    dtype = [(field, object) for field in fields]
    output = np.empty((1, len(records)), dtype=dtype)
    for index, record in enumerate(records):
        for field in fields:
            output[field][0, index] = _matlab_value(record.get(field))
    return output


def _matlab_value(value: Any) -> Any:
    if value is None:
        return np.empty((0, 0))
    return value


def _without_none(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _matlab_value(value) for key, value in mapping.items() if value is not None
    }
