"""CND recordings as MNE objects, and back."""

from __future__ import annotations

import warnings
from pathlib import Path

import mne
import numpy as np

from ._channels import (
    MontagePolicy as MontagePolicy,
)
from ._channels import (
    _canonical_unit,
    _channel_spec,
    _description,
    _make_montage,
    _unit_scale,
)
from ._mne_export import UnsupportedMetadataPolicy as UnsupportedMetadataPolicy
from ._mne_export import from_mne as from_mne
from ._mne_recording import MNECNDRecording as MNECNDRecording
from .exceptions import CNDAmbiguousUnitError, CNDUnsupportedError, CNDValidationError
from .model import (
    CNDRecording,
)
from .validation import validate_cnd


def read_cnd_mne(
    path: str | Path,
    *,
    stimulus_path: str | Path | None = None,
    subject: str | int | None = None,
    load_stimulus: bool = True,
    neural_variable: str | None = None,
    neural_unit: str | None = None,
    montage: MontagePolicy = "none",
    coordinate_scale_to_meters: float | None = None,
) -> MNECNDRecording:
    """Read CND MATLAB files into one MNE ``Raw`` per trial."""
    from .io import read_cnd

    recording = read_cnd(
        path,
        stimulus_path=stimulus_path,
        subject=subject,
        load_stimulus=load_stimulus,
        neural_variable=neural_variable,
    )
    return to_mne(
        recording,
        neural_unit=neural_unit,
        montage=montage,
        coordinate_scale_to_meters=coordinate_scale_to_meters,
    )


def to_mne(
    recording: CNDRecording,
    *,
    neural_unit: str | None = None,
    montage: MontagePolicy = "none",
    coordinate_scale_to_meters: float | None = None,
) -> MNECNDRecording:
    """Create one MNE ``RawArray`` per variable-length CND neural trial.

    Parameters
    ----------
    recording
        CND data.
    neural_unit
        Physical unit of the numerical CND EEG values. Required unless the CND
        file declares ``dataUnit``, ``unit``, or ``units``. MNE stores EEG in
        volts.
    montage
        ``"none"`` (default) avoids guessing coordinate semantics.
        ``"eeglab"`` applies the axis mapping used by existing CND importers:
        MNE XYZ = ``(-CND.Y, CND.X, CND.Z)``.
    coordinate_scale_to_meters
        Multiplier that converts stored coordinates to metres. Required when
        ``montage="eeglab"``; for example, ``0.01`` for centimetres.
    """
    report = validate_cnd(recording)
    report.raise_for_errors()
    neural = recording.neural
    if neural is None:
        raise CNDValidationError("CND recording has no neural data")
    data_type = neural.data_type.strip().lower()
    if data_type not in {"eeg", "fnirs", "nirs"}:
        raise CNDUnsupportedError(
            f"MNE adapter supports EEG and fNIRS, not {neural.data_type!r}"
        )

    unit = neural_unit or neural.data_unit
    if unit is None:
        raise CNDAmbiguousUnitError(
            "CND does not declare the neural unit; pass neural_unit explicitly"
        )
    scale = _unit_scale(unit, data_type)
    ch_names, ch_types = _channel_spec(neural)
    if neural.channel_names is None:
        warnings.warn(
            "CND has no complete channel labels; generated stable names "
            "and omitted montage",
            RuntimeWarning,
            stacklevel=2,
        )
        montage = "none"

    if len(set(ch_names)) != len(ch_names):
        raise CNDValidationError("CND channel labels are not unique")
    info = mne.create_info(list(ch_names), neural.sfreq, ch_types=list(ch_types))
    info["description"] = _description(neural)

    dig_montage = _make_montage(
        neural,
        ch_names,
        montage=montage,
        coordinate_scale_to_meters=coordinate_scale_to_meters,
    )
    raws: list[mne.io.RawArray] = []
    for trial in neural.trials:
        data_mne_units = np.asarray(trial, dtype=np.float64).T * scale
        raw = mne.io.RawArray(data_mne_units, info.copy(), verbose="ERROR")
        if dig_montage is not None:
            raw.set_montage(dig_montage, on_missing="raise", verbose="ERROR")
        raws.append(raw)
    return MNECNDRecording(tuple(raws), recording, _canonical_unit(unit, data_type))
