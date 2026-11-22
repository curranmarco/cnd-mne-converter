"""Check the MNE viewer's Save as BIDS / Save as CND on original datasets.

Runs the same code as the viewer buttons (no clicking), then compares the
result with the original and prints passed/failed:

  A. original CND  -> Open CND -> Save as BIDS -> read back -> compare with CND
                                 -> Save as CND  ->            compare with CND
  B. original BIDS -> Open BIDS -> Save as CND -> to-bids -> compare-bids

Example (from the cnd-mne-converter folder):

  QT_QPA_PLATFORM=offscreen uv run --extra gui --extra bids \\
    python scripts/check_viewer_save.py --out ~/Desktop/viewer_check \\
    --cnd ../CNSP-resources/CNSP/datasets/LalorNatSpeech/dataCND \\
    --cnd-subject 10 --cnd-unit uV \\
    --bids ../open-neurodata-to-cnd/cache/ds004574/1.0.0 \\
    --bids-subject 001 --bids-task Oddball --event-column value
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mne
import numpy as np
from qtpy import QtWidgets

from cnd_mne import read_cnd
from cnd_mne.bids import compare_bids, compare_cnd, read_cnd_bids, write_cnd_bids
from cnd_mne.gui import CNDImporterApp, add_cnd_bids_toolbar


def check_cnd(folder: Path, subject: int, unit: str, out: Path) -> bool:
    importer = CNDImporterApp()
    importer.path.setText(str(folder))
    importer.subject.setText(str(subject))
    importer.neural_unit.setCurrentText(unit)
    importer._load()  # the "Open CND" button
    browser = importer._open_browser(importer._selected_raw())  # "MNE data viewer"
    toolbar = browser._cnd_bids_toolbar
    original = read_cnd(folder, subject=subject)
    external = "uV" if original.neural.external_trials is not None else None

    toolbar.save_bids(
        out / "A_bids", subject=str(subject), task="cnd", external_unit=external
    )
    to_bids = compare_cnd(
        original, read_cnd_bids(out / "A_bids", subject=str(subject), task="cnd")
    )
    toolbar.save_cnd(out / "A_cnd", subject=subject)
    to_cnd = compare_cnd(original, read_cnd(out / "A_cnd", subject=subject))
    print(
        f"A. CND -> Save as BIDS: passed={to_bids['passed']} "
        f"(max error {to_bids['max_relative_neural_error']:.1e})"
    )
    print(
        f"A. CND -> Save as CND:  passed={to_cnd['passed']} "
        f"(max error {to_cnd['max_relative_neural_error']:.1e})"
    )
    return bool(to_bids["passed"] and to_cnd["passed"])


def check_bids(
    root: Path, subject: str, task: str, column: str | None, out: Path
) -> bool:
    placeholder = mne.io.RawArray(
        np.zeros((1, 2)), mne.create_info(["empty"], 1.0, "misc"), verbose=False
    )
    with mne.viz.use_browser_backend("qt"):
        hidden = placeholder.plot(show=False, block=False)
    match = next(
        path
        for path in sorted(
            (root / f"sub-{subject}" / "eeg").glob(f"*task-{task}*_eeg.*")
        )
        if path.suffix in {".vhdr", ".set", ".edf", ".bdf", ".fif"}
    )
    browser = add_cnd_bids_toolbar(hidden).open_bids(match)  # "Open BIDS…"
    toolbar = browser._cnd_bids_toolbar
    toolbar.save_cnd(
        out / "B_cnd",
        subject=1,  # "Save as CND…"
        treat_misc_as_eeg=toolbar.needs_eeg_confirmation(),
    )
    cnd = read_cnd(out / "B_cnd", subject=1)
    names = cnd.stimulus.names if cnd.stimulus is not None else ()
    print(
        f"B. BIDS -> Save as CND: {cnd.neural.n_channels} channels, "
        f"stimulus features {list(names)}"
    )
    write_cnd_bids(cnd, out / "B_bids", subject=1, task=task)
    report = compare_bids(
        root,
        out / "B_bids",
        subject=subject,
        task=task,
        final_subject="1",
        event_column=column,
    )
    print(
        f"B. BIDS -> CND -> BIDS vs original: passed={report['passed']} "
        f"(EEG max error {report['eeg']['max_relative_error']:.1e}, "
        f"events {[e['n_final'] for e in report['events'].values()]})"
    )
    return bool(report["passed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cnd", type=Path)
    parser.add_argument("--cnd-subject", type=int, default=1)
    parser.add_argument("--cnd-unit", default="", help="if the CND has no unit")
    parser.add_argument("--bids", type=Path)
    parser.add_argument("--bids-subject")
    parser.add_argument("--bids-task")
    parser.add_argument("--event-column")
    args = parser.parse_args()
    out = args.out.expanduser()
    shutil.rmtree(out, ignore_errors=True)
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    results = []
    if args.cnd:
        results.append(
            check_cnd(args.cnd.expanduser(), args.cnd_subject, args.cnd_unit, out)
        )
    if args.bids:
        results.append(
            check_bids(
                args.bids.expanduser(),
                args.bids_subject,
                args.bids_task,
                args.event_column,
                out,
            )
        )
    print("ALL PASSED" if results and all(results) else "SOMETHING FAILED")


if __name__ == "__main__":
    main()
