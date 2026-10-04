"""Decode MATLAB structures into CND models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .exceptions import CNDReadError
from .model import CNDNeural, CNDStimulus, ExternalLayout

_NEURAL_RESERVED = {
    "data",
    "dataType",
    "deviceName",
    "origTrialPosition",
    "chanlocs",
    "extChan",
    "reRef",
    "paddingStartSample",
    "fs",
    "cndVersion",
    "dataUnit",
    "unit",
    "units",
    "datatype",
}


_STIMULUS_RESERVED = {
    "data",
    "names",
    "stimIdxs",
    "condIdxs",
    "condNames",
    "fs",
    "cndVersion",
}


def _parse_neural(value: Any, variable_name: str, source: Path) -> CNDNeural:
    if not isinstance(value, Mapping):
        raise CNDReadError(f"{source}:{variable_name} is not a MATLAB structure")
    channel_value = value.get("chanlocs")
    channel_locations = _parse_channel_locations(channel_value)
    channel_locations_raw = channel_value if _is_topomap_layout(channel_value) else None
    try:
        trials, signal_types, channels_per_signal_type = _parse_neural_trials(
            value,
            channel_count_hint=(
                len(channel_locations) if channel_locations is not None else None
            ),
        )
        sfreq = float(_scalar(value["fs"]))
    except KeyError as error:
        raise CNDReadError(
            f"{source}:{variable_name} is missing {error.args[0]!r}"
        ) from error

    external = _parse_external(value.get("extChan"), len(trials), source, variable_name)

    original = value.get("origTrialPosition")
    original_positions = (
        tuple(int(item) for item in np.atleast_1d(original).ravel())
        if original is not None and np.asarray(original).size
        else None
    )
    data_unit = next(
        (
            str(_scalar(value[key]))
            for key in ("dataUnit", "unit", "units")
            if key in value and value[key] is not None
        ),
        None,
    )
    extras = {key: item for key, item in value.items() if key not in _NEURAL_RESERVED}
    return CNDNeural(
        trials=trials,
        sfreq=sfreq,
        data_type=_string_or_default(value.get("dataType"), variable_name.upper()),
        device_name=_optional_string(value.get("deviceName")),
        original_trial_positions=original_positions,
        channel_locations=channel_locations,
        channel_locations_raw=channel_locations_raw,
        external_trials=external.external_trials,
        external_description=external.external_description,
        external_fields=external.external_fields,
        external_layout=external.external_layout,
        external_group_names=external.external_group_names,
        external_group_channel_counts=external.external_group_channel_counts,
        external_group_fields=external.external_group_fields,
        rereference=value.get("reRef"),
        padding_start_sample=value.get("paddingStartSample"),
        cnd_version=_optional_scalar(value.get("cndVersion")),
        data_unit=data_unit,
        signal_types=signal_types,
        channels_per_signal_type=channels_per_signal_type,
        extra_fields=extras,
        variable_name=variable_name,
        source_path=source,
    )


def _parse_named_external_groups(
    external: Mapping[str, Any], n_trials: int
) -> tuple[tuple[str, ...], list[tuple[np.ndarray, ...]], tuple[int, ...]] | None:
    """Parse CND 1.0's ``extChan.<type> = trial-cell-array`` layout."""
    names: list[str] = []
    groups: list[tuple[np.ndarray, ...]] = []
    counts: list[int] = []
    # MATLAB v5 preserves struct-field insertion order while HDF5-backed v7.3
    # readers commonly return fields lexicographically. Canonicalize here so
    # the combined channel order is stable across MAT encodings.
    for name in sorted(external):
        value = external[name]
        try:
            trials = _as_matrix_trial_tuple(value)
        except (CNDReadError, TypeError, ValueError):
            return None
        if len(trials) != n_trials or any(trial.ndim != 2 for trial in trials):
            return None
        channel_counts = {int(trial.shape[1]) for trial in trials}
        if len(channel_counts) != 1:
            return None
        names.append(str(name))
        groups.append(trials)
        counts.append(channel_counts.pop())
    if not groups:
        return None
    return tuple(names), groups, tuple(counts)


def _combine_external_groups(
    group_trials: Sequence[tuple[np.ndarray, ...]],
    source: Path,
    variable_name: str,
) -> tuple[np.ndarray, ...]:
    if len({len(trials) for trials in group_trials}) != 1:
        raise CNDReadError(
            f"{source}:{variable_name}.extChan groups have unequal trial counts"
        )
    for trial_index in range(len(group_trials[0])):
        lengths = {int(trials[trial_index].shape[0]) for trials in group_trials}
        if len(lengths) != 1:
            raise CNDReadError(
                f"{source}:{variable_name}.extChan groups have unequal sample "
                f"counts in trial {trial_index}"
            )
    return tuple(
        np.concatenate([trials[index] for trials in group_trials], axis=1)
        for index in range(len(group_trials[0]))
    )


def _parse_neural_trials(
    value: Mapping[str, Any],
    channel_count_hint: int | None = None,
) -> tuple[tuple[np.ndarray, ...], tuple[str, ...] | None, tuple[int, ...] | None]:
    """Normalize ordinary trials and fNIRS signal-type x trial cell grids."""
    data = value["data"]
    data_type = _string_or_default(value.get("dataType"), "").strip().lower()
    datatype = value.get("datatype")
    signal_types = (
        tuple(str(item) for item in np.atleast_1d(datatype).ravel())
        if datatype is not None
        else None
    )
    array = np.asarray(data, dtype=object)
    if (
        data_type in {"fnirs", "nirs"}
        and signal_types
        and array.dtype == object
        and array.ndim == 2
        and array.shape[0] == len(signal_types)
    ):
        block_counts: list[int] = []
        for signal_index in range(len(signal_types)):
            first = np.asarray(array[signal_index, 0])
            first = first[:, np.newaxis] if first.ndim == 1 else first
            if first.ndim != 2:
                raise CNDReadError("fNIRS blocks must be time x channels")
            block_counts.append(int(first.shape[1]))
        trials: list[np.ndarray] = []
        for trial_index in range(array.shape[1]):
            blocks: list[np.ndarray] = []
            samples: int | None = None
            for signal_index, expected_channels in enumerate(block_counts):
                block = np.asarray(array[signal_index, trial_index])
                block = block[:, np.newaxis] if block.ndim == 1 else block
                if block.ndim != 2 or block.shape[1] != expected_channels:
                    raise CNDReadError("fNIRS signal blocks have inconsistent shapes")
                if samples is not None and block.shape[0] != samples:
                    raise CNDReadError("fNIRS signal blocks have unequal sample counts")
                samples = int(block.shape[0])
                blocks.append(block)
            trials.append(np.concatenate(blocks, axis=1))
        return tuple(trials), signal_types, tuple(block_counts)
    return _as_matrix_trial_tuple(data, channel_count_hint), None, None


def _parse_stimulus(value: Any, source: Path) -> CNDStimulus:
    if not isinstance(value, Mapping):
        raise CNDReadError(f"{source}:stim is not a MATLAB structure")
    required = {"data", "names", "fs"}
    missing = required - set(value)
    if missing:
        raise CNDReadError(f"{source}:stim is missing {sorted(missing)!r}")
    names = tuple(str(item) for item in np.atleast_1d(value["names"]).ravel())
    features = _as_feature_tuple(value["data"], len(names))
    stimulus_indices_value = value.get("stimIdxs")
    stimulus_indices = (
        tuple(
            _python_scalar(item)
            for item in np.atleast_1d(stimulus_indices_value).ravel()
        )
        if stimulus_indices_value is not None
        else None
    )
    condition_indices = value.get("condIdxs")
    condition_names = value.get("condNames")
    extras = {key: item for key, item in value.items() if key not in _STIMULUS_RESERVED}
    return CNDStimulus(
        names=names,
        features=features,
        sfreq=float(_scalar(value["fs"])),
        stimulus_indices=stimulus_indices,
        condition_indices=(
            tuple(
                _python_scalar(item)
                for item in np.atleast_1d(condition_indices).ravel()
            )
            if condition_indices is not None
            else None
        ),
        condition_names=(
            tuple(str(item) for item in np.atleast_1d(condition_names).ravel())
            if condition_names is not None
            else None
        ),
        cnd_version=_optional_scalar(value.get("cndVersion")),
        extra_fields=extras,
        source_path=source,
    )


def _as_trial_tuple(value: Any) -> tuple[np.ndarray, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(np.asarray(item) for item in value)
    array = np.asarray(value)
    if array.dtype == object:
        return tuple(np.asarray(item) for item in array.ravel())
    if array.ndim <= 2:
        return (array,)
    if array.ndim == 3:
        return tuple(np.asarray(array[index]) for index in range(array.shape[0]))
    raise CNDReadError(f"Cannot interpret neural data with shape {array.shape}")


def _as_matrix_trial_tuple(
    value: Any, channel_count_hint: int | None = None
) -> tuple[np.ndarray, ...]:
    """Read neural-style trials and restore squeezed one-channel matrices."""
    arrays = _as_trial_tuple(value)
    observed_counts = {
        int(array.shape[1])
        for array in arrays
        if array.ndim == 2 and array.shape[1] > 0
    }
    inferred_channels = (
        next(iter(observed_counts)) if len(observed_counts) == 1 else channel_count_hint
    )
    output: list[np.ndarray] = []
    for array in arrays:
        if array.ndim == 1 and array.size == 0 and inferred_channels is not None:
            output.append(np.empty((0, inferred_channels), dtype=array.dtype))
        elif (
            array.ndim == 1
            and inferred_channels is not None
            and inferred_channels > 1
            and array.size == inferred_channels
        ):
            output.append(array[np.newaxis, :])
        elif array.ndim == 1:
            output.append(array[:, np.newaxis])
        else:
            output.append(array)
    return tuple(output)


def _as_feature_tuple(
    value: Any, n_features: int
) -> tuple[tuple[np.ndarray, ...], ...]:
    if n_features == 0:
        return ()
    if isinstance(value, (list, tuple)):
        if n_features == 1 and (not value or not isinstance(value[0], (list, tuple))):
            return (tuple(np.asarray(item) for item in value),)
        if len(value) == n_features:
            return tuple(_as_trial_tuple(feature) for feature in value)

    array = np.asarray(value)
    if array.dtype == object:
        if array.ndim >= 2 and array.shape[0] == n_features:
            return tuple(
                tuple(
                    np.asarray(array[index, trial]) for trial in range(array.shape[1])
                )
                for index in range(n_features)
            )
        if n_features == 1:
            return (tuple(np.asarray(item) for item in array.ravel()),)
        if array.ndim == 1 and array.shape[0] == n_features:
            return tuple(_as_trial_tuple(item) for item in array)
    if n_features == 1:
        return ((array,),)
    raise CNDReadError(
        f"Cannot interpret stimulus data shape {array.shape} for {n_features} features"
    )


def _parse_channel_locations(value: Any) -> tuple[dict[str, Any], ...] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if _is_topomap_layout(value):
            labels = np.atleast_1d(value["label"]).ravel()
            positions = np.asarray(value["pos"])
            locations: list[dict[str, Any]] = []
            for index, label in enumerate(labels):
                name = str(label)
                if name.upper() in {"COMNT", "SCALE"}:
                    continue
                location: dict[str, Any] = {
                    "labels": name,
                    "pos": np.asarray(positions[index]),
                }
                for key in ("width", "height"):
                    optional_value = value.get(key)
                    if optional_value is None:
                        continue
                    values = np.atleast_1d(optional_value).ravel()
                    if values.size == labels.size:
                        location[key] = values[index]
                locations.append(location)
            return tuple(locations)
        lengths = [
            np.atleast_1d(item).size
            for item in value.values()
            if np.atleast_1d(item).size > 1
        ]
        if not lengths:
            return (dict(value),)
        count = max(lengths)
        columnar_locations: list[dict[str, Any]] = []
        for index in range(count):
            columnar_location: dict[str, Any] = {}
            for key, item in value.items():
                values = np.atleast_1d(item).ravel()
                columnar_location[key] = values[index] if values.size > 1 else values[0]
            columnar_locations.append(columnar_location)
        return tuple(columnar_locations)
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, Mapping) for item in value):
            return tuple(dict(item) for item in value)
    # Preserve a NumPy structured dtype long enough to inspect its field names.
    # Casting to object here makes ``dtype.names`` disappear.
    array = np.asarray(value)
    if array.dtype.names:
        return tuple(
            {name: item[name] for name in array.dtype.names} for item in array.ravel()
        )
    if all(isinstance(item, Mapping) for item in array.ravel()):
        return tuple(dict(item) for item in array.ravel())
    raise CNDReadError("Could not interpret chanlocs structure")


def _is_topomap_layout(value: Any) -> bool:
    if not isinstance(value, Mapping) or not {"label", "pos"} <= set(value):
        return False
    labels = np.atleast_1d(value["label"]).ravel()
    positions = np.asarray(value["pos"])
    return positions.ndim == 2 and positions.shape[0] == labels.size


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.size == 1 else value


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(_scalar(value))


def _optional_scalar(value: Any) -> Any:
    if value is None:
        return None
    return _python_scalar(_scalar(value))


def _string_or_default(value: Any, default: str) -> str:
    scalar = _scalar(value) if value is not None else None
    if isinstance(scalar, (str, np.str_)):
        return str(scalar)
    return default


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


class _ExternalData(NamedTuple):
    external_trials: tuple[np.ndarray, ...] | None
    external_description: str | None
    external_fields: dict[str, Any]
    external_layout: ExternalLayout | None
    external_group_names: tuple[str, ...] | None
    external_group_channel_counts: tuple[int, ...] | None
    external_group_fields: tuple[dict[str, Any], ...] | None


def _parse_external(
    external: Any, n_trials: int, source: Path, variable_name: str
) -> _ExternalData:
    """Normalize supported external layouts and retain unrecognized fields."""
    external_trials: tuple[np.ndarray, ...] | None = None
    external_description: str | None = None
    external_fields: dict[str, Any] = {}
    external_layout: ExternalLayout | None = None
    external_group_names: tuple[str, ...] | None = None
    external_group_channel_counts: tuple[int, ...] | None = None
    external_group_fields: tuple[dict[str, Any], ...] | None = None
    if isinstance(external, Mapping) and "data" in external:
        external_trials = _as_matrix_trial_tuple(external["data"])
        external_layout = "single_struct"
        if external.get("description") is not None:
            external_description = str(_scalar(external["description"]))
        external_fields = {
            key: item
            for key, item in external.items()
            if key not in {"data", "description"}
        }
    elif isinstance(external, Mapping):
        named_groups = _parse_named_external_groups(external, n_trials)
        if named_groups is None:
            external_fields = dict(external)
        else:
            names, group_trials, counts = named_groups
            external_trials = _combine_external_groups(
                group_trials, source, variable_name
            )
            external_description = "; ".join(names)
            external_layout = "named_fields"
            external_group_names = names
            external_group_channel_counts = counts
            external_group_fields = tuple({} for _ in names)
    elif external is not None:
        groups = [
            item
            for item in np.atleast_1d(external).ravel()
            if isinstance(item, Mapping) and "data" in item
        ]
        if groups and len(groups) == np.atleast_1d(external).size:
            group_trials = [_as_matrix_trial_tuple(group["data"]) for group in groups]
            external_trials = _combine_external_groups(
                group_trials, source, variable_name
            )
            descriptions = tuple(
                (
                    str(_scalar(group["description"]))
                    if group.get("description") is not None
                    else ""
                )
                for group in groups
            )
            external_description = "; ".join(filter(None, descriptions)) or None
            external_layout = "struct_array"
            external_group_names = descriptions
            external_group_channel_counts = tuple(
                int(trials_for_group[0].shape[1]) for trials_for_group in group_trials
            )
            external_group_fields = tuple(
                {
                    key: item
                    for key, item in group.items()
                    if key not in {"data", "description"}
                }
                for group in groups
            )
        else:
            external_fields = {"unparsedValue": external}

    return _ExternalData(
        external_trials,
        external_description,
        external_fields,
        external_layout,
        external_group_names,
        external_group_channel_counts,
        external_group_fields,
    )
