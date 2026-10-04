"""Channel names, physical units, and electrode coordinate transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import mne
import numpy as np

from .exceptions import CNDUnsupportedError, CNDValidationError
from .model import CNDNeural

MontagePolicy = Literal["none", "eeglab"]


_UNIT_TO_VOLTS = {
    "v": 1.0,
    "volt": 1.0,
    "volts": 1.0,
    "mv": 1e-3,
    "millivolt": 1e-3,
    "millivolts": 1e-3,
    "uv": 1e-6,
    "microvolt": 1e-6,
    "microvolts": 1e-6,
    "nv": 1e-9,
    "nanovolt": 1e-9,
    "nanovolts": 1e-9,
}


_UNIT_TO_MOLAR = {
    "m": 1.0,
    "molar": 1.0,
    "mol/l": 1.0,
    "mm": 1e-3,
    "mmolar": 1e-3,
    "mmol/l": 1e-3,
    "um": 1e-6,
    "umolar": 1e-6,
    "umol/l": 1e-6,
    "nm": 1e-9,
    "nmolar": 1e-9,
    "nmol/l": 1e-9,
}


def _channel_spec(neural: CNDNeural) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Channel names and MNE types for this CND signal."""
    data_type = neural.data_type.strip().lower()
    if data_type == "eeg":
        eeg_names = neural.channel_names or tuple(
            f"EEG{index:03d}" for index in range(1, neural.n_channels + 1)
        )
        return eeg_names, ("eeg",) * neural.n_channels
    if data_type not in {"fnirs", "nirs"}:
        raise CNDUnsupportedError(f"Unsupported neural type {neural.data_type!r}")
    signal_types = neural.signal_types
    counts = neural.channels_per_signal_type
    if signal_types is None or counts is None:
        raise CNDValidationError(
            "fNIRS conversion requires signal_types and channels_per_signal_type"
        )
    fnirs_names: list[str] = []
    channel_types: list[str] = []
    for signal_type, count in zip(signal_types, counts, strict=True):
        normalized = _normalize_unit(signal_type)
        if normalized in {"hbo", "oxy", "oxyhemoglobin"}:
            mne_type = "hbo"
            prefix = "HbO"
        elif normalized in {"hbr", "deoxy", "deoxyhemoglobin"}:
            mne_type = "hbr"
            prefix = "HbR"
        else:
            # MNE has no HbT channel type. Keep it visible as misc while its
            # molar numerical scale and original CND datatype are preserved.
            mne_type = "misc"
            prefix = str(signal_type).strip() or "fNIRS"
        width = max(2, len(str(count)))
        fnirs_names.extend(
            f"{prefix}{index:0{width}d}" for index in range(1, count + 1)
        )
        channel_types.extend([mne_type] * count)
    return tuple(fnirs_names), tuple(channel_types)


def _make_montage(
    neural: CNDNeural,
    ch_names: Sequence[str],
    *,
    montage: MontagePolicy,
    coordinate_scale_to_meters: float | None,
) -> mne.channels.DigMontage | None:
    if montage == "none":
        return None
    if montage != "eeglab":
        raise ValueError(f"Unknown montage policy {montage!r}")
    if coordinate_scale_to_meters is None or coordinate_scale_to_meters <= 0:
        raise CNDValidationError(
            "montage='eeglab' requires a positive coordinate_scale_to_meters"
        )
    if neural.channel_locations is None:
        return None
    positions: dict[str, np.ndarray] = {}
    for name, location in zip(ch_names, neural.channel_locations, strict=True):
        try:
            x = float(location["X"])
            y = float(location["Y"])
            z = float(location["Z"])
        except (KeyError, TypeError, ValueError):
            raise CNDValidationError(
                f"Channel {name!r} lacks numeric X/Y/Z coordinates"
            ) from None
        position = np.array([-y, x, z], dtype=float) * coordinate_scale_to_meters
        if not np.all(np.isfinite(position)):
            raise CNDValidationError(f"Channel {name!r} has non-finite coordinates")
        positions[name] = position

    biosemi_name = _matching_biosemi_montage(neural.device_name, ch_names)
    if biosemi_name is not None:
        median_radius = float(
            np.median([np.linalg.norm(position) for position in positions.values()])
        )
        if not 0.05 <= median_radius <= 0.11:
            raise CNDValidationError(
                "BioSemi coordinates have an implausible median head radius of "
                f"{median_radius:.6g} m; check coordinate_scale_to_meters"
            )
        template = mne.channels.make_standard_montage(
            biosemi_name, head_size=median_radius
        )
        native_head_t = mne.channels.compute_native_head_t(template, on_missing="raise")
        head_positions = {
            name: mne.transforms.apply_trans(native_head_t, position)
            for name, position in positions.items()
        }
        template_positions = template.get_positions()
        fiducials = {
            name: mne.transforms.apply_trans(native_head_t, template_positions[name])
            for name in ("nasion", "lpa", "rpa")
        }
        return mne.channels.make_dig_montage(
            ch_pos=head_positions,
            nasion=fiducials["nasion"],
            lpa=fiducials["lpa"],
            rpa=fiducials["rpa"],
            coord_frame="head",
        )
    return mne.channels.make_dig_montage(ch_pos=positions, coord_frame="head")


def _matching_biosemi_montage(
    device_name: str | None, ch_names: Sequence[str]
) -> str | None:
    """Return an exact MNE BioSemi template match for declared BioSemi data."""
    if "biosemi" not in (device_name or "").lower():
        return None
    for candidate in (
        "biosemi16",
        "biosemi32",
        "biosemi64",
        "biosemi128",
        "biosemi160",
        "biosemi256",
    ):
        template = mne.channels.make_standard_montage(candidate, head_size=0.095)
        if set(ch_names) == set(template.ch_names):
            return candidate
    return None


def _extract_channel_locations(
    raw: mne.io.BaseRaw,
) -> tuple[dict[str, object], ...] | None:
    montage = raw.get_montage()
    if montage is None:
        return None
    ch_pos = montage.get_positions().get("ch_pos") or {}
    if not all(
        name in ch_pos and np.all(np.isfinite(np.asarray(ch_pos[name], dtype=float)))
        for name in raw.ch_names
    ):
        return None
    locations = []
    for index, name in enumerate(raw.ch_names, start=1):
        x_mne, y_mne, z_mne = np.asarray(ch_pos[name], dtype=float)
        azimuth = float(np.degrees(np.arctan2(-x_mne, y_mne)))
        elevation = float(np.degrees(np.arctan2(z_mne, np.hypot(x_mne, y_mne))))
        locations.append(
            {
                "labels": name,
                "X": y_mne,
                "Y": -x_mne,
                "Z": z_mne,
                "sph_theta": azimuth,
                "sph_phi": elevation,
                "sph_radius": float(np.linalg.norm([x_mne, y_mne, z_mne])),
                "theta": -azimuth,
                "radius": 0.5 - elevation / 180.0,
                "urchan": index,
            }
        )
    return tuple(locations)


def _unit_scale(unit: str, data_type: str = "eeg") -> float:
    key = _normalize_unit(unit)
    unit_map = _UNIT_TO_MOLAR if data_type in {"fnirs", "nirs"} else _UNIT_TO_VOLTS
    try:
        return unit_map[key]
    except KeyError as error:
        label = "fNIRS" if data_type in {"fnirs", "nirs"} else "EEG"
        raise ValueError(f"Unsupported {label} unit {unit!r}") from error


def _normalize_unit(unit: str) -> str:
    return unit.strip().lower().replace("µ", "u").replace("μ", "u").replace(" ", "")


def _canonical_unit(unit: str, data_type: str = "eeg") -> str:
    scale = _unit_scale(unit, data_type)
    if data_type in {"fnirs", "nirs"}:
        return {1.0: "M", 1e-3: "mM", 1e-6: "uM", 1e-9: "nM"}[scale]
    return {1.0: "V", 1e-3: "mV", 1e-6: "uV", 1e-9: "nV"}[scale]


def _description(neural: CNDNeural) -> str:
    parts = ["Imported from CND"]
    if neural.device_name:
        parts.append(f"device={neural.device_name}")
    if neural.cnd_version:
        parts.append(f"cndVersion={neural.cnd_version}")
    return "; ".join(parts)
