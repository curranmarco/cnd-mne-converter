"""Command-line interface for inspecting CND files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .inspection import inspect_cnd
from .io import read_cnd
from .verification import verify_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cnd-mne")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("gui", help="open CND files in a small MNE desktop GUI")

    inspect_parser = commands.add_parser("inspect", help="inspect and validate CND")
    inspect_parser.add_argument(
        "path", help="CND directory or subject/stimulus .mat file"
    )
    inspect_parser.add_argument("--stimulus-path")
    inspect_parser.add_argument("--subject")
    inspect_parser.add_argument(
        "--neural-variable",
        help="recording-modality variable to load when a file contains several",
    )
    inspect_parser.add_argument(
        "--no-stimulus", action="store_true", help="do not infer a stimulus file"
    )
    inspect_parser.add_argument(
        "--strict-spec",
        action="store_true",
        help="treat CND 1.0 conformance deviations as errors",
    )

    verify_parser = commands.add_parser(
        "verify-dataset", help="verify every subject and exercise MNE end to end"
    )
    verify_parser.add_argument("path", help="dataset root or dataCND directory")
    verify_parser.add_argument("--dataset-name")
    verify_parser.add_argument(
        "--neural-unit",
        help="explicit source neural unit for MNE checks (for example uV or uM)",
    )
    verify_parser.add_argument(
        "--strict-spec",
        action="store_true",
        help="treat CND 1.0 conformance deviations as errors",
    )
    verify_parser.add_argument(
        "--no-round-trip", action="store_true", help="skip MNE-to-CND round trip"
    )
    verify_parser.add_argument(
        "--serialized-round-trip",
        action="store_true",
        help="also write and reread temporary MATLAB files (slower)",
    )
    verify_parser.add_argument(
        "--mat-version",
        choices=("5", "7.3"),
        default="5",
        help="MATLAB format used by --serialized-round-trip (default: 5)",
    )
    verify_parser.add_argument(
        "--no-mne-smoke",
        action="store_true",
        help="skip the MNE Welch PSD smoke test",
    )
    verify_parser.add_argument(
        "--output", type=Path, help="write the JSON report to this file"
    )

    to_bids = commands.add_parser(
        "to-bids", help="write one CND subject as a BIDS EEG dataset (via MNE)"
    )
    to_bids.add_argument("path", help="CND directory or subject .mat file")
    to_bids.add_argument("--subject", required=True, help="CND subject to read")
    to_bids.add_argument("--bids-root", required=True, type=Path)
    to_bids.add_argument("--task", required=True, help="BIDS task label")
    to_bids.add_argument(
        "--bids-subject", help="BIDS subject label (default: --subject)"
    )
    to_bids.add_argument("--session", help="BIDS session label")
    to_bids.add_argument("--neural-unit", help="unit when the CND file has none")
    to_bids.add_argument("--external-unit", help="include extChan signals in this unit")
    to_bids.add_argument(
        "--external-type", help="MNE type for extChan (e.g. eog, misc)"
    )
    to_bids.add_argument("--montage", choices=("none", "eeglab"), default="none")
    to_bids.add_argument("--coordinate-scale", type=float, help="chanlocs -> metres")
    to_bids.add_argument("--line-freq", type=float, help="mains frequency (50 or 60)")
    to_bids.add_argument("--overwrite", action="store_true")
    to_bids.add_argument(
        "--verify",
        action="store_true",
        help="read the BIDS files back and compare them with the CND input",
    )

    from_bids = commands.add_parser(
        "from-bids", help="read a BIDS subject written by to-bids back into CND"
    )
    from_bids.add_argument("bids_root", type=Path)
    from_bids.add_argument("--subject", required=True, help="BIDS subject label")
    from_bids.add_argument("--task", required=True)
    from_bids.add_argument("--session")
    from_bids.add_argument("--output", required=True, type=Path, help="dataCND folder")
    from_bids.add_argument("--output-unit", help="default: unit of the original CND")
    from_bids.add_argument("--overwrite", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "gui":
        from .gui import launch_gui

        launch_gui()
        return 0
    if args.command == "inspect":
        recording = read_cnd(
            args.path,
            stimulus_path=args.stimulus_path,
            subject=args.subject,
            load_stimulus=not args.no_stimulus,
            neural_variable=args.neural_variable,
        )
        print(
            json.dumps(inspect_cnd(recording, strict_spec=args.strict_spec), indent=2)
        )
        return 0
    if args.command == "verify-dataset":
        report = verify_dataset(
            args.path,
            dataset_name=args.dataset_name,
            neural_unit=args.neural_unit,
            strict_spec=args.strict_spec,
            round_trip=not args.no_round_trip,
            serialized_round_trip=args.serialized_round_trip,
            serialized_mat_version=args.mat_version,
            mne_smoke_test=not args.no_mne_smoke,
        )
        payload = json.dumps(report.to_dict(), indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0 if report.passed else 1
    if args.command == "to-bids":
        from .bids import compare_cnd, read_cnd_bids, write_cnd_bids

        recording = read_cnd(args.path, subject=args.subject)
        bids_subject = args.bids_subject or args.subject
        export = write_cnd_bids(
            recording,
            args.bids_root,
            subject=bids_subject,
            task=args.task,
            session=args.session,
            neural_unit=args.neural_unit,
            external_unit=args.external_unit,
            external_channel_types=args.external_type,
            montage=args.montage,
            coordinate_scale_to_meters=args.coordinate_scale,
            line_freq=args.line_freq,
            overwrite=args.overwrite,
        )
        summary: dict[str, object] = {
            "bids_root": str(export.bids_root),
            "runs": len(export.neural_paths),
            "stimulus_files": len(export.stimulus_paths) // 2,
        }
        if args.verify:
            restored = read_cnd_bids(
                args.bids_root,
                subject=bids_subject,
                task=args.task,
                session=args.session,
                output_unit=recording.neural.data_unit if recording.neural else None,
            )
            summary["round_trip"] = compare_cnd(recording, restored)
        print(json.dumps(summary, indent=2, default=str))
        round_trip = summary.get("round_trip")
        return 1 if isinstance(round_trip, dict) and not round_trip["passed"] else 0
    if args.command == "from-bids":
        from .bids import read_cnd_bids
        from .io import write_cnd

        recording = read_cnd_bids(
            args.bids_root,
            subject=args.subject,
            task=args.task,
            session=args.session,
            output_unit=args.output_unit,
        )
        paths = write_cnd(
            recording, args.output, subject=args.subject, overwrite=args.overwrite
        )
        print(
            json.dumps({"neural": str(paths.neural), "stimulus": str(paths.stimulus)})
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
