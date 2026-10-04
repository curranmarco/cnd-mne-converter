# Desktop viewer

```bash
uv sync --locked --extra gui
uv run cnd-mne gui
```

Select a CND file or folder. Choose a subject for folders with multiple subject
files. Enter the confirmed neural unit if the file lacks one, then **Open CND**.

| Control | Purpose |
| --- | --- |
| Previous / Next trial | Switch recording segments |
| MNE data viewer | Browse voltage traces over time |
| Sensor layout | Show electrode names and positions |
| Power spectrum | Show signal power by frequency; select a band for a scalp plot |
| Apply EEGLAB coordinates | Use stored electrode positions |
| Scale to metres | Convert coordinate values using a documented multiplier |

A scalp plot interpolates sensor power; it does not locate brain sources.
The amplitude warning flags unusually large values without choosing a unit.

## Import from an existing MNE viewer

```python
import mne
from cnd_mne.gui import add_cnd_import_button

with mne.viz.use_browser_backend("qt"):
    browser = raw.plot(block=False)
importer = add_cnd_import_button(browser)
```

This opt-in Qt toolbar prototype opens imports in a new viewer. It is supplied
by this package, not upstream MNE. Keep the Qt event loop running in scripts:
[standalone example](../examples/mne_import_button.py).

Imported viewers retain `browser.cnd_recording` and `browser.cnd_trial_index`.
Saving still requires the Python export API.

## Recorded checks

Offscreen Qt tests cover import, cancellation, failure, repeated installation,
and variable-length trials. [Browser audit](results/browser-import/audit.json)
and [coordinate audit](results/giovanni-review/gui-audit.json) record real-data checks.

The notebook's Lalor sample uses `uV` as a plotting assumption and `0.095` as a
coordinate multiplier for a 95 mm display sphere. Neither establishes physical
calibration. BioSemi template positions describe the cap, not individual anatomy.

Reproduce after downloading the notebook fixture:

```bash
QT_QPA_PLATFORM=offscreen uv run --extra gui python examples/validate_browser_import.py \
  examples/_scratch/lalor-sample --neural-unit uV
uv run --extra gui python examples/validate_gui.py --help
```

For a manufacturer-coordinate comparison, supply both recordings and the workbook:

```bash
uv run --with xlrd python examples/validate_coordinates.py \
  --lalor /path/to/pre_dataSub1.mat --p3 /path/to/p3/dataCND \
  --coordinates /path/to/Cap_coords_all.xls
```

The workbook source is recorded in [the historical audit](results/giovanni-review/results.json).
New audits and plots go to local `outputs/`.
