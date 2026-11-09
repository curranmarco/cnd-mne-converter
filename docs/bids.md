# CND ↔ MNE ↔ BIDS

Install the optional dependencies once: `uv sync --locked --extra dev` (or
`--extra bids` for conversion only).

```bash
# CND -> MNE -> BIDS, then read it back and compare with the CND input
uv run cnd-mne to-bids /path/to/dataCND --subject 1 --bids-root bids \
  --task naturalspeech --neural-unit uV --external-unit uV --line-freq 50 --verify

# BIDS -> MNE -> CND
uv run cnd-mne from-bids bids --subject 1 --task naturalspeech --output dataCND_back
```

```python
from cnd_mne import read_cnd
from cnd_mne.bids import compare_cnd, read_cnd_bids, write_cnd_bids

cnd = read_cnd("dataCND", subject=1)
write_cnd_bids(cnd, "bids", subject=1, task="naturalspeech", neural_unit="uV")
back = read_cnd_bids("bids", subject=1, task="naturalspeech")
print(compare_cnd(cnd, back)["passed"])
```

## Mapping

| CND | BIDS |
| --- | --- |
| Subject file, trial *i* | `sub-<label>/eeg/..._run-<i>_eeg.vhdr` (BrainVision, written by MNE-BIDS) |
| `eeg.data`, `fs`, channel labels, unit | EEG file, `channels.tsv` (units), `eeg.json` |
| `chanlocs` (with `--montage eeglab`) | `electrodes.tsv`, `coordsystem.json` |
| `extChan` (with `--external-unit`) | Extra channels in the same run, typed e.g. `misc` or `eog` |
| `stim.data{f, i}` | `..._run-<i>_recording-<feature>_stim.tsv.gz` + `_stim.json` (own `SamplingFrequency`) |
| Sparse 1-D features (≤10 % non-zero) | Also onsets in `events.tsv` (`trial_type` = feature name) |
| `stimIdxs`, `condIdxs`, `condNames` | Keys in `_stim.json` |
| Unit, device, `origTrialPosition`, extChan names | `..._run-<i>_cnd.json` (listed in `.bidsignore`) |

## Checks

`--verify` compares trial count and lengths, channel names, sampling rates,
EEG and extChan values, stimulus values, stimulus and condition indices and
original trial positions. EEG is stored as 32-bit floats in BrainVision, so
EEG values must agree to 1e-6 of each trial's largest value; stimulus values
are written at full precision and must match exactly. The exported folder
passes the official `bids-validator` (1.15).

## Limits

- EEG only; MEG and fNIRS raise an error.
- Empty CND trials cannot be stored as BIDS runs and raise an error.
- `extChan` signals are left out unless their unit is given.
- BIDS has no field for how a stimulus feature was derived (e.g. envelope
  method); the `_stim.json` description says so.
- `--montage eeglab` positions are written in BIDS `CapTrak` space; they come
  from the CND `chanlocs`, not from a measured digitisation.

## BIDS → CND → BIDS (starting from a real BIDS dataset)

General BIDS goes to CND with the `open-neurodata-to-cnd` pipeline (its
recipe says which events become stimulus features). `to-bids` writes it back,
and `compare-bids` checks the final BIDS against the original:

```bash
# 1. BIDS -> CND (in open-neurodata-to-cnd)
uv run neurodata-to-cnd convert recipes/ds004574-sub001-oddball.json
# 2. CND -> BIDS (in cnd-mne-converter)
uv run cnd-mne to-bids ../open-neurodata-to-cnd/outputs/ds004574/0.1.0/dataCND \
  --subject 1 --bids-subject 001 --task Oddball --bids-root final_bids \
  --line-freq 60 --montage eeglab --coordinate-scale 1 --verify
# 3. Original BIDS vs final BIDS
uv run cnd-mne compare-bids ../open-neurodata-to-cnd/cache/ds004574/1.0.0 final_bids \
  --subject 001 --task Oddball --recipe ../open-neurodata-to-cnd/recipes/ds004574-sub001-oddball.json
```

`compare-bids` passes when EEG (same channels, rate, length, values within
1e-6) and every recipe event rule (same count, onsets within half a sample)
match. It also lists metadata that did not survive, because CND has no field
for it (e.g. `EEGReference`, `EEGGround`).

## In MNE's data viewer

```bash
uv sync --locked --extra dev --extra gui
uv run cnd-mne browse /path/to/dataCND --subject 1           # CND
uv run cnd-mne browse /path/to/bids --subject 001 --task Oddball  # BIDS
```

The MNE viewer gets four toolbar buttons: **Open CND…**, **Open BIDS…**,
**Save as CND…**, **Save as BIDS…**. A recording from CND (or from BIDS made
by `to-bids`) is saved with all trials and stimulus features. Any other
recording is saved as one trial; bad channels and annotations marked in the
viewer go to BIDS (`channels.tsv`, `events.tsv`) but not to CND, which has no
field for them. When no channel is typed EEG (BIDS `type` = `n/a`), saving asks
for confirmation first. In Python: `cnd_mne.gui.add_cnd_bids_toolbar(raw.plot())`.

This is a prototype built on Qt's public window API, not an MNE plugin API.
