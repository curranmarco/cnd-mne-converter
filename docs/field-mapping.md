# CND ↔ MNE

CND separates sensor signals from stimulus tracks: speech amplitude, word
onsets, or other events.

| CND field | Python / MNE representation |
| --- | --- |
| `eeg.data` | `rec.cnd.neural.trials`: time × channels; `rec.raws`: channels × time |
| `eeg.fs` | Sampling rate in Hz; copied to MNE `Info["sfreq"]` |
| `dataUnit` | EEG scales to volts; fNIRS concentration scales to molar units |
| `chanlocs` | Channel labels; optional electrode positions (MNE montage) |
| `extChan` | Auxiliary signals, such as eye-movement channels (EOG) |
| `origTrialPosition`, `paddingStartSample` | Preserved metadata; no automatic reordering or onset shift |
| `stim.data`, `names`, `fs` | Feature → trial → array, names, and a separate sampling rate |
| `stimIdxs`, `condIdxs`, `condNames` | Stimulus IDs and experimental conditions |
| fNIRS `datatype` / data grid | HbO → `hbo`, HbR → `hbr`, HbT → `misc`; grid rebuilt on export |
| Extra fields / other recording types | `extra_fields` / `additional_variables`; retained on export |

MATLAB IDs remain one-based. Python trial indices start at zero. Missing
`stimIdxs` resolve to `1..n` in memory without changing the original fields.
Stored trial `i` stays paired with stimulus trial `i`.

## Optional views

```python
envelopes = rec.stimulus_raws("Speech Envelope")
onsets = rec.stimulus_annotations("Word Onsets")
eog = rec.external_raws(unit="uV", channel_types="eog")
continuous = rec.concatenate()  # joins get boundary annotations
```

Stimulus views keep their own clock. Different rates or durations produce
warnings; align them explicitly for analysis. Empty legacy trials are retained.
Strict mode rejects selected specification violations; rate differences remain
warnings. Review experimental synchronisation separately.

Select a recording type with `neural_variable="eeg"`;
`available_neural_variables(path)` lists choices.

Electrode positions require `montage="eeglab"` and a positive
`coordinate_scale_to_meters`. EEGLAB axes are converted to MNE axes. BioSemi
layouts with matching labels use template fiducials; these are not measured
participant anatomy. Two-dimensional layouts supply labels only.

## Export from MNE

```python
from cnd_mne import from_mne, write_cnd

cnd = from_mne(
    eeg_raw,
    stimulus=stimulus,
    external_raws=eog_raw,
    external_unit="V",
    external_description="EOG",
)
write_cnd(cnd, "converted/dataCND", mat_version="7.3")
```

Use `rec.write_cnd(...)` for edits to existing CND. It checks trial lengths and
channel order against the template. MNE annotations, bad-channel lists and
projections have no standard CND mapping; export warns by default. Use
`on_unsupported_metadata="raise"` to reject their loss.

New external exports use MATLAB cells for CNSP's `eeg.extChan{1}` access.
For legacy single-group templates, `cnsp_external_cells=True` opts into that
layout. Scalar stimulus features export as `time × 1` columns. MATLAB v5 is the
default; `mat_version="7.3"` selects HDF5 for large files.
