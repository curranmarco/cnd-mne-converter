"""Read and write Continuous-event Neural Data MATLAB files."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import hdf5storage
import numpy as np
from scipy.io import loadmat, savemat
from scipy.io.matlab import MatReadError

from ._mat_decode import _parse_neural, _parse_stimulus
from ._mat_encode import _neural_to_mat, _stimulus_to_mat
from ._paths import (
    _canonical_subject_label,
    _cnd_output_filename,
    _resolve_stimulus_file,
    _resolve_subject_file,
    _subject_from_filename,
)
from .exceptions import CNDReadError
from .mat73 import is_mat73, load_mat73
from .model import CNDNeural, CNDPaths, CNDRecording, CNDStimulus
from .validation import validate_cnd

_MATLAB_METADATA = {"__header__", "__version__", "__globals__"}


def read_cnd(
    path: str | Path,
    *,
    stimulus_path: str | Path | None = None,
    subject: str | int | None = None,
    load_stimulus: bool = True,
    neural_variable: str | None = None,
) -> CNDRecording:
    """Read a CND directory or individual ``.mat`` file.

    When ``path`` is a subject file, a sibling ``dataStim<subject>.mat`` or
    ``dataStim.mat`` is inferred if ``load_stimulus`` is true. When ``path`` is
    a directory containing multiple subject files, ``subject`` is required.
    """
    source = Path(path).expanduser()
    if source.is_dir():
        data_directory = source / "dataCND" if (source / "dataCND").is_dir() else source
        neural_path = _resolve_subject_file(data_directory, subject)
        stimulus_subject = subject
        if stimulus_subject is None and neural_path is not None:
            stimulus_subject = _subject_from_filename(neural_path)
        inferred_stimulus = _resolve_stimulus_file(
            data_directory, stimulus_subject, required=False
        )
        if neural_path is not None:
            neural, additional_variables = _read_neural_file(
                neural_path, neural_variable
            )
        else:
            neural, additional_variables = None, {}
        selected_stimulus = stimulus_path or inferred_stimulus
        stimulus = (
            read_cnd_stimulus(selected_stimulus)
            if load_stimulus and selected_stimulus is not None
            else None
        )
        recording = CNDRecording(neural, stimulus, additional_variables)
        validate_cnd(recording).raise_for_errors()
        return recording

    if not source.exists():
        raise FileNotFoundError(source)

    variables = _load_mat(source)
    neural_key = _select_neural_key(variables, neural_variable, source)
    if neural_key is not None:
        neural = _parse_neural(variables[neural_key], neural_key, source)
        resolved_stimulus: Path | None = None
        if stimulus_path is not None:
            resolved_stimulus = Path(stimulus_path).expanduser()
        elif load_stimulus:
            resolved_stimulus = _resolve_stimulus_file(
                source.parent, _subject_from_filename(source), required=False
            )
        if "stim" in variables:
            stimulus = _parse_stimulus(variables["stim"], source)
        else:
            stimulus = (
                read_cnd_stimulus(resolved_stimulus)
                if resolved_stimulus is not None
                else None
            )
        additional_variables = {
            key: value
            for key, value in variables.items()
            if key not in {neural_key, "stim"}
        }
        recording = CNDRecording(neural, stimulus, additional_variables)
    elif "stim" in variables:
        recording = CNDRecording(stimulus=_parse_stimulus(variables["stim"], source))
    else:
        raise CNDReadError(f"{source} contains neither neural nor 'stim' data")

    validate_cnd(recording).raise_for_errors()
    return recording


def read_cnd_neural(path: str | Path, *, variable_name: str | None = None) -> CNDNeural:
    """Read one subject-specific CND neural file."""
    source = Path(path).expanduser()
    neural, _ = _read_neural_file(source, variable_name)
    return neural


def available_neural_variables(path: str | Path) -> tuple[str, ...]:
    """List CND recording-modality variables available in a MATLAB file."""
    source = Path(path).expanduser()
    return _find_neural_keys(_load_mat(source))


def read_cnd_stimulus(path: str | Path) -> CNDStimulus:
    """Read one CND stimulus file."""
    source = Path(path).expanduser()
    variables = _load_mat(source)
    if "stim" not in variables:
        raise CNDReadError(f"{source} does not contain a 'stim' structure")
    return _parse_stimulus(variables["stim"], source)


def write_cnd(
    recording: CNDRecording,
    destination: str | Path,
    *,
    subject: str | int = 1,
    neural_filename: str | None = None,
    stimulus_filename: str | None = None,
    overwrite: bool = False,
    compression: bool = True,
    mat_version: Literal["5", "7.3"] = "5",
) -> CNDPaths:
    """Write ``dataSubN.mat`` and/or ``dataStim.mat``.

    Both files land together, or neither does. Existing files are left alone
    unless ``overwrite=True``. Use ``mat_version="7.3"`` for big HDF5 files.
    """
    if mat_version not in {"5", "7.3"}:
        raise ValueError("mat_version must be '5' or '7.3'")
    report = validate_cnd(recording)
    report.raise_for_errors()
    output_dir = Path(destination).expanduser()
    subject_label = _canonical_subject_label(subject)
    if (
        neural_filename is None
        and recording.neural is not None
        and recording.neural.source_path is not None
        and recording.neural.source_path.name.startswith(
            ("dataParticipant_", "pre_dataSub")
        )
    ):
        neural_filename = recording.neural.source_path.name
    if (
        stimulus_filename is None
        and recording.stimulus is not None
        and recording.stimulus.source_path is not None
        and recording.stimulus.source_path.name.startswith("dataStim")
        and recording.stimulus.source_path.name != "dataStim.mat"
        and recording.stimulus.source_path.suffix == ".mat"
    ):
        stimulus_filename = recording.stimulus.source_path.name

    neural_path = (
        output_dir
        / _cnd_output_filename(
            neural_filename,
            default=f"dataSub{subject_label}.mat",
            pattern=r"(?:dataSub.+|pre_dataSub.+|dataParticipant_.+)\.mat",
            kind="neural",
        )
        if recording.neural is not None
        else None
    )
    stimulus_path = (
        output_dir
        / _cnd_output_filename(
            stimulus_filename,
            default="dataStim.mat",
            pattern=r"dataStim.*\.mat",
            kind="stimulus",
        )
        if recording.stimulus is not None
        else None
    )
    planned_paths = [path for path in (neural_path, stimulus_path) if path is not None]
    if not overwrite:
        existing = [path for path in planned_paths if path.exists()]
        if existing:
            formatted = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Refusing to overwrite {formatted}")

    output_dir.mkdir(parents=True, exist_ok=True)
    planned_outputs: list[tuple[Path, dict[str, Any]]] = []
    if recording.neural is not None:
        if neural_path is None:
            raise RuntimeError("Neural output path was not planned")
        neural_payload = dict(recording.additional_variables)
        neural_payload[recording.neural.variable_name] = _neural_to_mat(
            recording.neural
        )
        planned_outputs.append(
            (
                neural_path,
                neural_payload,
            )
        )
    if recording.stimulus is not None:
        if stimulus_path is None:
            raise RuntimeError("Stimulus output path was not planned")
        planned_outputs.append(
            (stimulus_path, {"stim": _stimulus_to_mat(recording.stimulus)})
        )
    _atomic_save_many(
        planned_outputs,
        overwrite=overwrite,
        compression=compression,
        mat_version=mat_version,
    )
    return CNDPaths(neural_path, stimulus_path)


def _load_mat(path: Path) -> dict[str, Any]:
    try:
        data = loadmat(path, simplify_cells=True)
    except NotImplementedError:
        try:
            return load_mat73(path)
        except (OSError, TypeError, ValueError) as hdf_error:
            raise CNDReadError(
                f"Could not read MATLAB v7.3/HDF5 file {path}: {hdf_error}"
            ) from hdf_error
    except (OSError, ValueError, TypeError, MatReadError) as error:
        if is_mat73(path):
            try:
                return load_mat73(path)
            except (OSError, TypeError, ValueError) as hdf_error:
                raise CNDReadError(
                    f"Could not read MATLAB v7.3/HDF5 file {path}: {hdf_error}"
                ) from hdf_error
        raise CNDReadError(f"Could not read MATLAB file {path}: {error}") from error
    workspace = data.pop("__function_workspace__", None)
    if workspace is not None:
        data = _decode_mcos_strings(data, workspace)
    return {key: value for key, value in data.items() if key not in _MATLAB_METADATA}


def _decode_mcos_strings(value: Any, workspace: Any) -> Any:
    """Resolve MATLAB MCOS string handles using the embedded v5 workspace.

    MATLAB's modern ``string`` class is stored as an opaque object plus a
    ``__function_workspace__`` byte array. SciPy intentionally exposes the
    handle. For the observed CND files, the fifth metadata word is a one-based
    index into the UTF-16LE strings in that workspace.
    """
    raw = np.asarray(workspace, dtype=np.uint8).tobytes()
    strings = tuple(
        match.group().decode("utf-16le")
        for match in re.finditer(rb"(?:[\x20-\x7e]\x00){2,}", raw)
    )
    if not strings:
        return value

    def resolve(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: resolve(child) for key, child in item.items()}
        if isinstance(item, list):
            return [resolve(child) for child in item]
        if isinstance(item, tuple):
            return tuple(resolve(child) for child in item)
        array = np.asarray(item) if isinstance(item, np.ndarray) else None
        if array is not None and array.dtype.names:
            fields = set(array.dtype.names)
            if {"_Class", "_ObjectMetadata"} <= fields:
                class_name = str(np.asarray(array["_Class"]).ravel()[0])
                metadata_value = np.asarray(array["_ObjectMetadata"]).ravel()[0]
                metadata = np.asarray(metadata_value).ravel()
                if class_name == "string" and metadata.size >= 5:
                    index = int(metadata[4]) - 1
                    if 0 <= index < len(strings):
                        return strings[index]
            return item
        if array is not None and array.dtype == object:
            output = np.empty(array.shape, dtype=object)
            for object_index in np.ndindex(array.shape):
                output[object_index] = resolve(array[object_index])
            return output
        return item

    return resolve(value)


def _find_neural_keys(variables: Mapping[str, Any]) -> tuple[str, ...]:
    preferred = ("eeg", "meg", "nirs", "fnirs", "ieeg", "neural")
    found: list[str] = []
    for key in preferred:
        value = variables.get(key)
        if isinstance(value, Mapping) and "data" in value and "fs" in value:
            found.append(key)
    for key, value in variables.items():
        if (
            key not in found
            and key != "stim"
            and isinstance(value, Mapping)
            and {"data", "fs"} <= set(value)
        ):
            found.append(key)
    return tuple(found)


def _select_neural_key(
    variables: Mapping[str, Any], variable_name: str | None, source: Path
) -> str | None:
    keys = _find_neural_keys(variables)
    if variable_name is None:
        return keys[0] if keys else None
    if variable_name not in keys:
        available = ", ".join(keys) if keys else "none"
        raise CNDReadError(
            f"{source} has no neural variable {variable_name!r}; available: {available}"
        )
    return variable_name


def _read_neural_file(
    source: Path, variable_name: str | None
) -> tuple[CNDNeural, dict[str, Any]]:
    variables = _load_mat(source)
    key = _select_neural_key(variables, variable_name, source)
    if key is None:
        raise CNDReadError(f"{source} does not contain a recognizable neural structure")
    neural = _parse_neural(variables[key], key, source)
    additional = {
        name: value for name, value in variables.items() if name not in {key, "stim"}
    }
    return neural, additional


def _prepare_mat(
    path: Path,
    payload: dict[str, Any],
    compression: bool,
    *,
    mat_version: Literal["5", "7.3"],
) -> Path:
    """Serialize one MAT payload beside its destination without publishing it."""
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".mat", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        if mat_version == "5":
            savemat(
                temporary,
                payload,
                appendmat=False,
                do_compression=compression,
                long_field_names=True,
            )
        else:
            hdf5storage.savemat(
                temporary,
                payload,
                appendmat=False,
                fmt="7.3",
                oned_as="row",
                store_python_metadata=False,
                truncate_existing=True,
                compress=compression,
            )
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_save_many(
    outputs: Sequence[tuple[Path, dict[str, Any]]],
    *,
    overwrite: bool,
    compression: bool,
    mat_version: Literal["5", "7.3"],
) -> None:
    """Publish a neural/stimulus file set together, rolling back on failure."""
    if not outputs:
        return
    if not overwrite:
        existing = [path for path, _ in outputs if path.exists()]
        if existing:
            formatted = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Refusing to overwrite {formatted}")

    prepared: list[tuple[Path, Path]] = []
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        # Complete every potentially expensive serialization before touching
        # any existing destination file.
        for path, payload in outputs:
            prepared.append(
                (
                    path,
                    _prepare_mat(path, payload, compression, mat_version=mat_version),
                )
            )
        for path, temporary in prepared:
            if path.exists():
                backup_handle = tempfile.NamedTemporaryFile(
                    prefix=f".{path.stem}.",
                    suffix=".backup",
                    dir=path.parent,
                    delete=False,
                )
                backup = Path(backup_handle.name)
                backup_handle.close()
                backup.unlink()
                os.replace(path, backup)
                backups.append((path, backup))
            os.replace(temporary, path)
            published.append(path)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        for path, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, path)
        raise
    finally:
        for _, temporary in prepared:
            temporary.unlink(missing_ok=True)
        for _, backup in backups:
            backup.unlink(missing_ok=True)
