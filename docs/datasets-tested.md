# Datasets tested (BIDS → CND)

Each dataset was converted with `cnd-mne from-bids` and with the MNE viewer
(Open BIDS… → Save as CND…), then checked: either written back with `to-bids`
and compared with the original using `compare-bids`, or compared directly with
the values MNE-BIDS reads from the original files.

| Dataset | Format | What it tests | Result |
|---|---|---|---|
| ds004574 (oddball) | EEGLAB `.set` | channels typed `n/a`; events in `value` | passed: EEG error 5e-8, 708/708 events at the same times |
| ds004408 (natural speech) | BrainVision | 2 runs → 2 trials; no `events.tsv` | passed: EEG identical (error 0) |
| ds001787 (meditation) | BDF | sessions; extra GSR/respiration/temperature channels; `trial_type` + `value` | passed: 64 EEG channels (error 3e-8), 87/87 events; extra channels left out |
| ds003555 (clinical, 2 h at 1024 Hz) | EDF | long recording, high rate | passed BIDS → CND: identical (error 2e-16) |
| CNSP LalorNatSpeech sub 10 (CND → BIDS) | CND | 20 trials, extChan, 4 features | passed, and bids-validator accepts the BIDS |

## Problems found and fixed
- `compare-bids` looked for the session in the BIDS written back from CND
  (CND has no sessions): it now falls back to no session.
- Events read through MNE-BIDS are named `trial_type/value` (e.g. `response/2`)
  when both columns exist; `compare-bids` now matches that form.
- Reading a CND with a long recording used about 5× the data size in memory;
  now about 1×. Writing BIDS keeps one copy fewer (MNE-BIDS's own writer still
  needs about 3.5× the data: a 2 h, 23-channel, 1024 Hz recording needs a
  machine with ~8 GB free).
- `from-bids` warns when EEG values are above 1 V (impossible for EEG).

## Problems in the datasets themselves (reported, not fixed)
- ds004408: values are stored 10^7 times too large (the BrainVision header and
  `channels.tsv` say volts/0.1 µV, but the numbers look like microvolts
  multiplied again). The converter keeps the values as stored and warns.
- ds001787: `channels.tsv` names channels Fp1, AF7, … but the BDF file names
  them A1, A2, …; MNE-BIDS keeps the file's names. Its channel types are in a
  top-level `task-meditation_channels.tsv`: without that file, GSR,
  temperature, etc. would be read as EEG (the 1 V warning catches this).
- Bad channels marked in a dataset's `channels.tsv` are not kept in CND (CND
  has no field for them).
