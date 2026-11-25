# Speech envelope from BIDS audio

`cnd-mne from-bids` can add the speech envelope (and word onsets) to the CND
when a BIDS dataset includes its audio.

## Finding the audio for each run

1. **BIDS way:** `events.tsv` has a `stim_file` column naming a `.wav` in
   `stimuli/`; the row's `onset` is when it started.
2. **Otherwise:** `--audio` gives a pattern relative to the dataset folder,
   e.g. `"stimuli/audio{run:02d}.wav"` (run 1 → `audio01.wav`), for audio that
   starts with the recording (ds004408's README says so).

## What is added to `dataStim.mat`

- **Speech Envelope**, one track per trial, at `--stim-sfreq` (default 128 Hz,
  as in CNSP), placed at the audio's onset and zero after it ends.
- **Word Onset** (all words), 1 at each word start, when a `.TextGrid` with a
  `words` tier sits next to the audio.
- Event-type features from `events.tsv`, now at the same 128 Hz.

The EEG keeps its own rate; CND stores stimulus features at their own rate.

## Envelope methods (`--envelope-method`)

| Method | What it does | Match with CNSP LalorNatSpeech envelope |
|---|---|---|
| `hilbert` (default) | magnitude of the Hilbert transform, resampled to 128 Hz with an anti-aliasing filter | r = 0.9997 |
| `mtrf` | Python port of mTRF-Toolbox `mTRFenvelope` (Lalor & Foxe 2010): mean power per window, square root, compression `log10(2)` | r = 0.90 |

The CNSP envelope was therefore made with the Hilbert method. CNSP's "Word
Onset Vectors" contain content words only (from separate text files); all of
them are in our Word Onset at the same sample (283/283 and 260/260).

## Check

```bash
uv run --extra bids python scripts/check_envelope.py \
  --bids "$R/open-datasets/ds004408" --subject 001 \
  --cnsp "$R/CNSP-resources/CNSP/datasets/LalorNatSpeech/dataCND" --cnsp-subject 10
```
