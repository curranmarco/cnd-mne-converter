# CND-MNE

Read CND MATLAB recordings, work on them in MNE-Python, and write them back.
Python 3.10+. Research software; not yet on PyPI.

**CND** stores brain signals and matching stimulus tracks in `.mat` files.
**MNE** is a Python library for analysing brain recordings. This package bridges
both: one MNE `Raw` object per trial, with the CND metadata attached for export.
A trial is one recording segment; trials can have different lengths.

## Start

```bash
git clone https://github.com/finnjclancy/cnd-mne-converter.git
cd cnd-mne-converter
uv sync --locked --extra dev
uv run cnd-mne inspect tests/data/minimal-cnd --subject 1
```

Run Python examples with `uv run python`. The bundled fixture has two trials,
four EEG channels, and a declared microvolt (`uV`) unit:

```python
from cnd_mne import read_cnd_mne

rec = read_cnd_mne("tests/data/minimal-cnd", subject=1)
rec.raws[0].plot()
rec.raws[0].filter(1, 15)
rec.write_cnd("converted/dataCND", subject=1, output_unit="uV")
```

`rec.raws` contains the MNE trials. `rec.cnd` holds stimulus tracks, trial order,
extra fields, and unselected recording types. Keep this wrapper when exporting.
Existing files require `overwrite=True`.

For files without a declared unit, supply the unit confirmed by the data owner:

```python
rec = read_cnd_mne("/path/to/dataCND", subject=1, neural_unit="uV")
```

EEG and the public CND fNIRS layout are supported. Units and electrode scale
must be explicit. Conversion preserves trial lengths, order, and sampling rates.
MEG conversion and TRF result files are unsupported.

## More

- [Usage and field mapping](docs/field-mapping.md): stimuli, auxiliary channels, export.
- [Desktop viewer](docs/gui.md): `uv sync --locked --extra gui`, then `uv run cnd-mne gui`.
- [Notebook](examples/walkthrough.ipynb): bundled fixture, then an optional 120 MB download.
- [Verification](docs/results/README.md): 1,026 public files; 1,017 passed, eight empty, one unreadable.
- [Development](CONTRIBUTING.md): code map and checks.

To verify your own folder:

```bash
uv run cnd-mne verify-dataset /path/to/dataCND --neural-unit uV --output verification.json
```

Add `--serialized-round-trip` to write and reread MATLAB files.
`--strict-spec` checks newly created CND more strictly than legacy files.
