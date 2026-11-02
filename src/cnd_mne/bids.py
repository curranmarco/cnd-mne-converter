"""Write CND to BIDS and read it back, with MNE-BIDS as the bridge.

CND -> MNE -> BIDS:  every CND trial becomes one BIDS run of one subject.
                     EEG goes to BrainVision files written by MNE-BIDS.
                     Every stimulus feature goes to a BIDS continuous
                     recording (``_recording-<feature>_stim.tsv.gz`` plus a
                     ``_stim.json`` sidecar) on its own clock.
                     Sparse one-dimensional features (word onsets, ...) are
                     also listed in ``events.tsv``, so BIDS tools see them.
BIDS -> MNE -> CND:  the runs, channels and stimulus files written above are
                     read back into a :class:`~cnd_mne.CNDRecording`.

CND details with no BIDS field (unit, original trial position, device name,
external-channel names) go in a ``*_cnd.json`` file per run, listed in
``.bidsignore``; stimulus and condition indices go in the ``_stim.json``
sidecars. Requires the optional
dependencies in the ``bids`` extra (``mne-bids`` and ``pybv``).
"""

from __future__ import annotations

import gzip
import json
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np

from ._channels import _canonical_unit
from ._mne_export import from_mne
from .exceptions import CNDUnsupportedError, CNDValidationError
from .mne import MontagePolicy, to_mne
from .model import CNDRecording, CNDStimulus, _resolve_condition_name

_FEATURE_LABEL = re.compile(r"[^A-Za-z0-9]")
_SPARSE_FRACTION = 0.1


@dataclass(slots=True, frozen=True)
class BIDSExport:
    """Files written by :func:`write_cnd_bids`."""

    bids_root: Path
    neural_paths: tuple[Path, ...]
    stimulus_paths: tuple[Path, ...]


def write_cnd_bids(
    recording: CNDRecording,
    bids_root: str | Path,
    *,
    subject: str | int,
    task: str,
    session: str | None = None,
    neural_unit: str | None = None,
    external_unit: str | None = None,
    external_channel_types: str | Sequence[str] | None = None,
    montage: MontagePolicy = "none",
    coordinate_scale_to_meters: float | None = None,
    line_freq: float | None = None,
    dataset_name: str | None = None,
    events_from_sparse_features: bool = True,
    overwrite: bool = False,
) -> BIDSExport:
    """Write one CND subject (neural + stimulus) as a BIDS EEG dataset.

    Parameters
    ----------
    recording
        CND data, e.g. from :func:`cnd_mne.read_cnd`.
    bids_root
        BIDS dataset folder. Created if needed; other subjects can be added
        to the same folder by further calls.
    subject, task, session
        BIDS entities. Each CND trial becomes ``run-1``, ``run-2``, ...
    neural_unit
        Physical unit of the CND values; required when the file does not
        declare ``dataUnit``.
    external_unit, external_channel_types
        Unit (and optional MNE channel types, e.g. ``"eog"``) of CND
        ``extChan`` signals. Without a unit they are left out, with a warning.
    montage, coordinate_scale_to_meters
        As in :func:`cnd_mne.to_mne`; ``montage="eeglab"`` writes
        ``electrodes.tsv`` and ``coordsystem.json``.
    line_freq
        Mains frequency in Hz (50 or 60). BIDS recommends it; ``None`` writes
        ``n/a``.
    events_from_sparse_features
        Also list onsets of sparse 1-D features in ``events.tsv``.
    """
    mne_bids = _import_mne_bids()
    neural = recording.neural
    if neural is None:
        raise CNDValidationError("CND recording has no neural data")
    if neural.data_type.strip().lower() != "eeg":
        raise CNDUnsupportedError(
            f"BIDS export supports EEG only, not {neural.data_type!r}"
        )
    empty = [i + 1 for i, trial in enumerate(neural.trials) if np.size(trial) == 0]
    if empty:
        raise CNDUnsupportedError(
            f"CND trials {empty} have no samples; BIDS cannot store empty runs"
        )

    root = Path(bids_root).expanduser()
    subject_label = _entity(subject, "subject")
    task_label = _entity(task, "task")
    session_label = _entity(session, "session") if session is not None else None

    mne_recording = to_mne(
        recording,
        neural_unit=neural_unit,
        montage=montage,
        coordinate_scale_to_meters=coordinate_scale_to_meters,
    )
    source_unit = mne_recording.neural_unit

    external_names: tuple[str, ...] = ()
    external_raws: tuple[mne.io.RawArray, ...] | None = None
    if neural.external_trials is not None:
        if external_unit is None:
            warnings.warn(
                "CND extChan signals were left out of BIDS: pass external_unit "
                "to include them",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            external_raws = mne_recording.external_raws(
                unit=external_unit, channel_types=external_channel_types
            )
            external_names = tuple(external_raws[0].ch_names)
            clash = set(external_names) & set(mne_recording.raws[0].ch_names)
            if clash:
                raise CNDValidationError(
                    f"External channel names repeat EEG channel names: {sorted(clash)}"
                )

    root.mkdir(parents=True, exist_ok=True)
    _ensure_bidsignore(root)
    if not (root / "dataset_description.json").exists():
        mne_bids.make_dataset_description(
            path=root,
            name=dataset_name or f"{task_label} (converted from CND)",
            generated_by=[
                {
                    "Name": "cnd-mne-converter",
                    "Description": "CND -> MNE -> BIDS export",
                }
            ],
            overwrite=False,
            verbose=False,
        )

    stimulus = recording.stimulus
    feature_labels = _feature_labels(stimulus.names) if stimulus is not None else ()
    neural_paths: list[Path] = []
    stimulus_paths: list[Path] = []

    for index, raw in enumerate(mne_recording.raws):
        run = index + 1
        run_raw = raw.copy()
        if external_raws is not None:
            run_raw.add_channels([external_raws[index].copy()], force_update_info=True)
        run_raw.info["line_freq"] = line_freq
        event_id: dict[str, int] | None = None
        if stimulus is not None and events_from_sparse_features:
            annotations, event_id = _sparse_feature_annotations(
                stimulus, index, run_raw.times[-1] if run_raw.n_times else 0.0
            )
            run_raw.set_annotations(annotations)

        bids_path = mne_bids.BIDSPath(
            subject=subject_label,
            session=session_label,
            task=task_label,
            run=run,
            datatype="eeg",
            root=root,
        )
        written = mne_bids.write_raw_bids(
            run_raw,
            bids_path,
            event_id=event_id or None,
            format="BrainVision",
            allow_preload=True,
            overwrite=overwrite,
            verbose=False,
        )
        neural_paths.append(Path(written.fpath))

        # MNE-BIDS infers "Brain Products" from the BrainVision file format;
        # the original amplifier is not known from CND.
        _update_json(_sidecar(written, "eeg"), {"Manufacturer": "n/a"})
        cnd_entries: dict[str, Any] = {
            "CNDTrialIndex": run,
            "CNDDataUnit": source_unit,
            "CNDDeviceName": neural.device_name,
            "CNDVersion": _json_value(neural.cnd_version),
            "CNDExternalChannels": list(external_names),
            "CNDExternalDescription": neural.external_description,
            "CNDExternalUnit": (
                _canonical_unit(external_unit, "eeg")
                if external_names and external_unit
                else None
            ),
        }
        if neural.original_trial_positions is not None:
            cnd_entries["CNDOriginalTrialPosition"] = int(
                neural.original_trial_positions[index]
            )
        _update_json(_sidecar(written, "cnd"), cnd_entries)

        if stimulus is not None:
            stimulus_paths.extend(
                _write_stimulus_run(
                    stimulus, index, feature_labels, written, overwrite=overwrite
                )
            )

    return BIDSExport(root, tuple(neural_paths), tuple(stimulus_paths))


def read_cnd_bids(
    bids_root: str | Path,
    *,
    subject: str | int,
    task: str,
    session: str | None = None,
    output_unit: str | None = None,
) -> CNDRecording:
    """Read one subject written by :func:`write_cnd_bids` back into CND.

    ``output_unit`` defaults to the unit of the original CND file (stored as
    ``CNDDataUnit``). Runs are read in run order, so trial order is kept.
    """
    mne_bids = _import_mne_bids()
    root = Path(bids_root).expanduser()
    query = mne_bids.BIDSPath(
        root=root,
        subject=_entity(subject, "subject"),
        session=_entity(session, "session") if session is not None else None,
        task=_entity(task, "task"),
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
    )
    runs = sorted(query.match(), key=lambda path: int(path.run or 0))
    if not runs:
        raise FileNotFoundError(f"No BIDS EEG runs found for {query}")

    neural_raws: list[mne.io.BaseRaw] = []
    external_raws: list[mne.io.BaseRaw] = []
    stim_runs: list[list[dict[str, Any]]] = []
    first_sidecar: dict[str, Any] = {}
    original_positions: list[int] = []
    for bids_path in runs:
        sidecar = _read_json(_sidecar(bids_path, "cnd"))
        if not first_sidecar:
            first_sidecar = sidecar
        with warnings.catch_warnings():
            # Harmless for CND: extChan signals (EOG, mastoids) have no positions.
            warnings.filterwarnings("ignore", message=".*without locations.*")
            warnings.filterwarnings("ignore", message=".*Not setting position.*")
            raw = mne_bids.read_raw_bids(
                bids_path, extra_params={"preload": True}, verbose=False
            )
        raw.set_annotations(None)
        raw.info["bads"] = []
        external = [str(name) for name in sidecar.get("CNDExternalChannels") or []]
        if external:
            external_raws.append(raw.copy().pick(external))
            raw.drop_channels(external)
        neural_raws.append(raw.pick("eeg"))
        if "CNDOriginalTrialPosition" in sidecar:
            original_positions.append(int(sidecar["CNDOriginalTrialPosition"]))
        stim_runs.append(_read_stimulus_run(Path(bids_path.fpath)))

    unit = output_unit or first_sidecar.get("CNDDataUnit") or "V"
    stimulus = _stimulus_from_runs(stim_runs)
    version = first_sidecar.get("CNDVersion")
    return from_mne(
        neural_raws,
        stimulus=stimulus,
        external_raws=external_raws or None,
        external_unit=first_sidecar.get("CNDExternalUnit") if external_raws else None,
        external_description=first_sidecar.get("CNDExternalDescription"),
        output_unit=unit,
        device_name=first_sidecar.get("CNDDeviceName"),
        original_trial_positions=(
            original_positions if len(original_positions) == len(runs) else None
        ),
        cnd_version=version if version is not None else 1.0,
        on_unsupported_metadata="ignore",
    )


# ---------------------------------------------------------------------------
# Stimulus features <-> BIDS continuous recordings
# ---------------------------------------------------------------------------


def _write_stimulus_run(
    stimulus: CNDStimulus,
    trial: int,
    labels: tuple[str, ...],
    neural_path: Any,
    *,
    overwrite: bool,
) -> list[Path]:
    """One ``_recording-<feature>_stim.tsv.gz`` (+ JSON) per feature."""
    written: list[Path] = []
    stim_indices = stimulus.resolved_stimulus_indices
    for feature_index, (name, label) in enumerate(
        zip(stimulus.names, labels, strict=True)
    ):
        values = np.asarray(stimulus.features[feature_index][trial], dtype=np.float64)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if values.ndim != 2:
            raise CNDValidationError(
                f"Stimulus feature {name!r} must be time x dimensions"
            )
        width = values.shape[1]
        columns = (
            [label] if width == 1 else [f"{label}{d:03d}" for d in range(1, width + 1)]
        )
        stim_path = neural_path.copy().update(
            recording=label, suffix="stim", extension=".tsv.gz", check=False
        )
        tsv = Path(stim_path.fpath)
        json_path = tsv.with_name(tsv.name.replace(".tsv.gz", ".json"))
        if not overwrite and (tsv.exists() or json_path.exists()):
            raise FileExistsError(f"{tsv} exists; pass overwrite=True")
        with gzip.open(tsv, "wt", encoding="utf-8") as handle:
            np.savetxt(handle, values, delimiter="\t", fmt="%.17g")
        condition_index = None
        condition_name = None
        if stimulus.condition_indices is not None:
            condition_index = _json_value(stimulus.condition_indices[trial])
            condition_name = _resolve_condition_name(
                stimulus, stimulus.condition_indices[trial]
            )
        sidecar = {
            "SamplingFrequency": float(stimulus.sfreq),
            "StartTime": 0.0,
            "Columns": columns,
            "Description": (
                f"CND stimulus feature '{name}' for this run. Its derivation "
                "(e.g. envelope method) is not recorded in CND."
            ),
            "CNDFeatureName": name,
            "CNDFeatureIndex": feature_index + 1,
            "CNDStimulusIndex": _json_value(stim_indices[trial]),
            "CNDConditionIndex": condition_index,
            "CNDConditionName": condition_name,
            "CNDConditionNames": (
                list(stimulus.condition_names)
                if stimulus.condition_names is not None
                else None
            ),
        }
        json_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        written.extend([tsv, json_path])
    return written


def _read_stimulus_run(neural_file: Path) -> list[dict[str, Any]]:
    prefix = neural_file.name.removesuffix("_eeg.vhdr")
    features: list[dict[str, Any]] = []
    for json_path in sorted(neural_file.parent.glob(f"{prefix}_recording-*_stim.json")):
        sidecar = _read_json(json_path)
        tsv = json_path.with_name(json_path.name.replace(".json", ".tsv.gz"))
        with gzip.open(tsv, "rt", encoding="utf-8") as handle:
            values = np.loadtxt(handle, delimiter="\t", ndmin=2)
        if len(sidecar.get("Columns", [])) == 1:
            values = values[:, 0]
        sidecar["_values"] = values
        features.append(sidecar)
    return sorted(features, key=lambda item: int(item.get("CNDFeatureIndex", 0)))


def _stimulus_from_runs(runs: list[list[dict[str, Any]]]) -> CNDStimulus | None:
    if not any(runs):
        return None
    names = tuple(str(item["CNDFeatureName"]) for item in runs[0])
    for run, items in enumerate(runs, start=1):
        if tuple(str(item["CNDFeatureName"]) for item in items) != names:
            raise CNDValidationError(f"Run {run} has different stimulus features")
    features = tuple(
        tuple(items[f]["_values"] for items in runs) for f in range(len(names))
    )
    first = runs[0][0]
    condition_indices = [items[0].get("CNDConditionIndex") for items in runs]
    names_list = first.get("CNDConditionNames")
    return CNDStimulus(
        names=names,
        features=features,
        sfreq=float(first["SamplingFrequency"]),
        stimulus_indices=tuple(items[0].get("CNDStimulusIndex") for items in runs),
        condition_indices=(
            tuple(condition_indices)
            if all(value is not None for value in condition_indices)
            else None
        ),
        condition_names=tuple(names_list) if names_list else None,
        cnd_version=1.0,
    )


def _sparse_feature_annotations(
    stimulus: CNDStimulus, trial: int, duration: float
) -> tuple[mne.Annotations, dict[str, int]]:
    """Onsets of sparse 1-D features (impulse tracks) as annotations."""
    onsets: list[float] = []
    descriptions: list[str] = []
    for name, feature in zip(stimulus.names, stimulus.features, strict=True):
        values = np.asarray(feature[trial], dtype=np.float64)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or values.size == 0:
            continue
        nonzero = np.flatnonzero(values != 0)
        if nonzero.size == 0 or nonzero.size > _SPARSE_FRACTION * values.size:
            continue
        times = nonzero / stimulus.sfreq
        times = times[times <= duration]
        onsets.extend(times.tolist())
        descriptions.extend([name] * len(times))
    annotations = mne.Annotations(
        onset=onsets, duration=[0.0] * len(onsets), description=descriptions
    )
    event_id = {
        name: code for code, name in enumerate(dict.fromkeys(descriptions), start=1)
    }
    return annotations, event_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_mne_bids() -> Any:
    try:
        import mne_bids
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "BIDS export needs the optional 'bids' dependencies: "
            "uv sync --extra bids  (or pip install 'cnd-mne-converter[bids]')"
        ) from error
    return mne_bids


def _entity(value: str | int, name: str) -> str:
    label = str(value)
    if not label or not label.isalnum():
        raise ValueError(f"BIDS {name} label must be letters/digits only: {label!r}")
    return label


def _feature_labels(names: Sequence[str]) -> tuple[str, ...]:
    labels: list[str] = []
    for index, name in enumerate(names, start=1):
        label = _FEATURE_LABEL.sub("", str(name)) or f"feature{index}"
        if label in labels:
            label = f"{label}{index}"
        labels.append(label)
    return tuple(labels)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _sidecar(bids_path: Any, suffix: str) -> Path:
    path = bids_path.copy().update(suffix=suffix, extension=".json", check=False)
    return Path(path.fpath)


def _ensure_bidsignore(root: Path) -> None:
    """CND-only metadata lives in ``*_cnd.json``, which BIDS tools skip."""
    path = root / ".bidsignore"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if "*_cnd.json" not in lines:
        path.write_text("\n".join([*lines, "*_cnd.json"]) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _update_json(path: Path, entries: dict[str, Any]) -> None:
    data = _read_json(path)
    data.update({key: value for key, value in entries.items() if value is not None})
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def compare_cnd(original: CNDRecording, restored: CNDRecording) -> dict[str, Any]:
    """Compare two CND recordings after a BIDS round trip.

    EEG is stored in BrainVision as 32-bit floats, so neural values are
    compared with a relative tolerance of 1e-6 of each trial's largest value;
    everything else must match exactly.
    """
    checks: dict[str, Any] = {}
    a, b = original.neural, restored.neural
    if a is None or b is None:
        raise CNDValidationError("Both recordings need neural data")
    checks["n_trials"] = a.n_trials == b.n_trials
    checks["sampling_rate"] = a.sfreq == b.sfreq
    checks["channel_names"] = a.channel_names == b.channel_names
    # Legacy CND often has no unit; BIDS requires one, so it is added.
    checks["data_unit"] = a.data_unit is None or a.data_unit == b.data_unit
    if a.data_unit is None:
        checks["unit_added"] = b.data_unit
    checks["trial_lengths"] = [np.shape(t) for t in a.trials] == [
        np.shape(t) for t in b.trials
    ]
    worst = 0.0
    if checks["trial_lengths"]:
        for x, y in zip(a.trials, b.trials, strict=True):
            xa = np.asarray(x, dtype=np.float64)
            ya = np.asarray(y, dtype=np.float64)
            scale = float(np.max(np.abs(xa))) or 1.0
            worst = max(worst, float(np.max(np.abs(xa - ya))) / scale)
    checks["max_relative_neural_error"] = worst
    checks["neural_values"] = checks["trial_lengths"] and worst <= 1e-6
    checks["original_trial_positions"] = (
        a.original_trial_positions == b.original_trial_positions
    )
    if a.external_trials is not None and b.external_trials is not None:
        checks["external_values"] = all(
            np.allclose(np.asarray(x), np.asarray(y), rtol=1e-6, atol=0)
            for x, y in zip(a.external_trials, b.external_trials, strict=False)
        )
    sa, sb = original.stimulus, restored.stimulus
    if sa is not None:
        checks["stimulus_present"] = sb is not None
        if sb is not None:
            checks["stimulus_names"] = sa.names == sb.names
            checks["stimulus_sampling_rate"] = sa.sfreq == sb.sfreq
            checks["stimulus_values"] = checks["stimulus_names"] and all(
                np.array_equal(
                    np.asarray(x, dtype=np.float64).reshape(len(x), -1),
                    np.asarray(y, dtype=np.float64).reshape(len(y), -1),
                )
                for fa, fb in zip(sa.features, sb.features, strict=True)
                for x, y in zip(fa, fb, strict=False)
            )
            checks["stimulus_indices"] = tuple(
                _json_value(v) for v in sa.resolved_stimulus_indices
            ) == tuple(_json_value(v) for v in sb.resolved_stimulus_indices)
            checks["condition_indices"] = sa.condition_indices is None or tuple(
                _json_value(v) for v in sa.condition_indices
            ) == tuple(_json_value(v) for v in (sb.condition_indices or ()))
            checks["condition_names"] = sa.condition_names == sb.condition_names
    checks["passed"] = all(
        value for key, value in checks.items() if isinstance(value, bool)
    )
    return checks
