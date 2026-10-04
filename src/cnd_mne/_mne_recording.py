"""MNE trial views and template-preserving editing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mne
import numpy as np

from ._channels import _canonical_unit, _unit_scale
from ._mne_export import UnsupportedMetadataPolicy, from_mne
from .exceptions import CNDValidationError
from .model import CNDPaths, CNDRecording, CNDStimulus, CNDTrialMetadata


@dataclass(slots=True)
class MNECNDRecording:
    """One MNE ``Raw`` per trial, plus the original CND on ``cnd``."""

    raws: tuple[mne.io.RawArray, ...]
    cnd: CNDRecording
    neural_unit: str

    @property
    def stimulus(self) -> CNDStimulus | None:
        return self.cnd.stimulus

    @property
    def trial_metadata(self) -> tuple[CNDTrialMetadata, ...]:
        """Trial metadata in the same order as ``raws``."""
        return self.cnd.trial_metadata

    @property
    def trial_slices(self) -> tuple[slice, ...]:
        """Sample slices of each trial after :meth:`concatenate`."""
        start = 0
        output: list[slice] = []
        for raw in self.raws:
            stop = start + raw.n_times
            output.append(slice(start, stop))
            start = stop
        return tuple(output)

    def concatenate(self, *, add_trial_annotations: bool = True) -> mne.io.BaseRaw:
        """Glue trials into one ``Raw``. Joins are fake; MNE marks them.

        Copies the per-trial objects. Optional ``CND_TRIAL/N`` annotations
        mark each trial start.
        """
        if not self.raws:
            raise CNDValidationError("Cannot concatenate an empty CND recording")
        combined = mne.concatenate_raws(
            [raw.copy() for raw in self.raws], preload=True, verbose="ERROR"
        )
        if add_trial_annotations:
            sfreq = float(combined.info["sfreq"])
            onsets = [trial_slice.start / sfreq for trial_slice in self.trial_slices]
            annotations = mne.Annotations(
                onset=onsets,
                duration=[0.0] * len(onsets),
                description=[
                    f"CND_TRIAL/{metadata.cnd_index}"
                    for metadata in self.trial_metadata
                ],
            )
            combined.set_annotations(combined.annotations + annotations)
        return combined

    def stimulus_raws(self, feature: str | int) -> tuple[mne.io.RawArray, ...]:
        """One stimulus feature as ``misc`` channels. Same rate and values.

        A spectrogram becomes one channel per bin. No resampling or unit claim.
        """
        stimulus = self.stimulus
        if stimulus is None:
            raise CNDValidationError("CND recording has no stimulus data")
        feature_index = _feature_index(stimulus, feature)
        feature_name = stimulus.names[feature_index]
        output: list[mne.io.RawArray] = []
        for trial in stimulus.features[feature_index]:
            array = np.asarray(trial, dtype=np.float64)
            if array.ndim == 1:
                array = array[:, np.newaxis]
            if array.ndim != 2:
                raise CNDValidationError(
                    f"Stimulus feature {feature_name!r} must be time x dimensions"
                )
            if array.shape[1] == 1:
                names = [feature_name]
            else:
                width = max(2, len(str(array.shape[1])))
                names = [
                    f"{feature_name}[{index:0{width}d}]"
                    for index in range(1, array.shape[1] + 1)
                ]
            info = mne.create_info(
                names, stimulus.sfreq, ch_types=["misc"] * len(names)
            )
            info["description"] = "Imported CND stimulus feature; arbitrary units"
            output.append(mne.io.RawArray(array.T, info, verbose="ERROR"))
        return tuple(output)

    def stimulus_annotations(
        self,
        feature: str | int,
        *,
        threshold: float = 0.0,
        include_values: bool = False,
    ) -> tuple[mne.Annotations, ...]:
        """Turn a sparse 1-D feature (word onsets, etc.) into annotations.

        Samples above ``threshold`` become zero-length marks on the stimulus
        clock. Continuous / multi-D features are rejected.
        """
        stimulus = self.stimulus
        if stimulus is None:
            raise CNDValidationError("CND recording has no stimulus data")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be a finite non-negative number")
        feature_index = _feature_index(stimulus, feature)
        feature_name = stimulus.names[feature_index]
        output: list[mne.Annotations] = []
        for trial in stimulus.features[feature_index]:
            values = np.asarray(trial, dtype=np.float64)
            if values.ndim == 2 and values.shape[1] == 1:
                values = values[:, 0]
            if values.ndim != 1:
                raise CNDValidationError(
                    f"Stimulus feature {feature_name!r} must be one-dimensional "
                    "to create annotations"
                )
            indices = np.flatnonzero(np.abs(values) > threshold)
            if include_values:
                descriptions = [
                    f"CND_STIM/{feature_name}/{values[index]:g}" for index in indices
                ]
            else:
                descriptions = [f"CND_STIM/{feature_name}"] * len(indices)
            output.append(
                mne.Annotations(
                    onset=indices.astype(float) / stimulus.sfreq,
                    duration=np.zeros(len(indices)),
                    description=descriptions,
                )
            )
        return tuple(output)

    def external_raws(
        self,
        *,
        unit: str,
        channel_types: str | Sequence[str] | None = None,
        channel_names: Sequence[str] | None = None,
    ) -> tuple[mne.io.RawArray, ...]:
        """Extra CND channels (EOG, mastoids, …) as their own ``Raw`` objects.

        Pass ``unit``. Optionally pass ``channel_types`` such as ``"eog"``.
        """
        neural = self.cnd.neural
        if neural is None or neural.external_trials is None:
            raise CNDValidationError("CND recording has no external-channel data")
        first = np.asarray(neural.external_trials[0])
        if first.ndim != 2:
            raise CNDValidationError("CND external trials must be time x channels")
        n_channels = int(first.shape[1])
        stored_names = neural.external_fields.get("channelNames")
        stored_types = neural.external_fields.get("channelTypes")
        if channel_names is None:
            channel_names = (
                np.atleast_1d(stored_names).ravel()
                if stored_names is not None
                else tuple(f"EXT{index:03d}" for index in range(1, n_channels + 1))
            )
        names = tuple(str(name) for name in channel_names)
        if isinstance(channel_types, str):
            types = (channel_types,) * n_channels
        elif channel_types is not None:
            types = tuple(channel_types)
        elif stored_types is not None:
            types = tuple(str(value) for value in np.atleast_1d(stored_types).ravel())
        else:
            types = ("misc",) * n_channels
        if len(names) != n_channels or len(types) != n_channels:
            raise CNDValidationError(
                "external channel names/types must match the external channel count"
            )
        if len(set(names)) != len(names):
            raise CNDValidationError("external channel names must be unique")
        scale = _unit_scale(unit, "eeg")
        info = mne.create_info(list(names), neural.sfreq, ch_types=list(types))
        info["description"] = (
            "Imported CND external channels; "
            f"source_unit={_canonical_unit(unit, 'eeg')}; "
            f"source_description={neural.external_description or 'unspecified'}"
        )
        output: list[mne.io.RawArray] = []
        for index, trial in enumerate(neural.external_trials):
            array = np.asarray(trial, dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != n_channels:
                raise CNDValidationError(
                    f"CND external trial {index} has inconsistent channel shape"
                )
            output.append(
                mne.io.RawArray(array.T * scale, info.copy(), verbose="ERROR")
            )
        return tuple(output)

    def to_cnd(
        self,
        *,
        output_unit: str | None = None,
        cnsp_external_cells: bool = False,
        on_unsupported_metadata: UnsupportedMetadataPolicy = "warn",
    ) -> CNDRecording:
        """Export edited MNE values while preserving the CND template.

        Set ``cnsp_external_cells`` to wrap a legacy single external group in
        a MATLAB cell for CNSP scripts that use ``eeg.extChan{1}``.
        """
        return from_mne(
            self.raws,
            template=self.cnd,
            output_unit=output_unit or self.neural_unit,
            cnsp_external_cells=cnsp_external_cells,
            on_unsupported_metadata=on_unsupported_metadata,
        )

    def write_cnd(
        self,
        destination: str | Path,
        *,
        subject: str | int = 1,
        output_unit: str | None = None,
        cnsp_external_cells: bool = False,
        overwrite: bool = False,
        compression: bool = True,
        mat_version: Literal["5", "7.3"] = "5",
        preserve_source_filenames: bool = True,
        on_unsupported_metadata: UnsupportedMetadataPolicy = "warn",
    ) -> CNDPaths:
        """Export edited MNE values and atomically write a CND directory."""
        from .io import write_cnd

        recording = self.to_cnd(
            output_unit=output_unit,
            cnsp_external_cells=cnsp_external_cells,
            on_unsupported_metadata=on_unsupported_metadata,
        )
        neural_filename = None
        stimulus_filename = None
        if preserve_source_filenames:
            if self.cnd.neural is not None and self.cnd.neural.source_path is not None:
                name = self.cnd.neural.source_path.name
                if name.startswith(("dataParticipant_", "pre_dataSub")):
                    neural_filename = name
            if (
                self.cnd.stimulus is not None
                and self.cnd.stimulus.source_path is not None
                and self.cnd.stimulus.source_path.name.startswith("dataStim")
                and self.cnd.stimulus.source_path.name != "dataStim.mat"
                and self.cnd.stimulus.source_path.suffix == ".mat"
            ):
                stimulus_filename = self.cnd.stimulus.source_path.name
        return write_cnd(
            recording,
            destination,
            subject=subject,
            neural_filename=neural_filename,
            stimulus_filename=stimulus_filename,
            overwrite=overwrite,
            compression=compression,
            mat_version=mat_version,
        )


def _feature_index(stimulus: CNDStimulus, feature: str | int) -> int:
    if isinstance(feature, str):
        try:
            return stimulus.names.index(feature)
        except ValueError as error:
            raise KeyError(feature) from error
    feature_index = int(feature)
    if not 0 <= feature_index < stimulus.n_features:
        raise IndexError(feature_index)
    return feature_index
