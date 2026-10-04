"""CND filename conventions and subject/stimulus discovery."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from .exceptions import CNDReadError


def _resolve_subject_file(directory: Path, subject: str | int | None) -> Path | None:
    if subject is not None:
        subject_candidates = (
            directory / f"dataSub{subject}.mat",
            directory / f"pre_dataSub{subject}.mat",
            directory / f"dataParticipant_{subject}.mat",
        )
        for candidate in subject_candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(subject_candidates[0])
    candidates = sorted(
        {
            *directory.glob("dataSub*.mat"),
            *directory.glob("pre_dataSub*.mat"),
            *directory.glob("dataParticipant_*.mat"),
        }
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise CNDReadError(
            f"{directory} contains {len(candidates)} subject files; pass subject=..."
        )
    return candidates[0]


def _resolve_stimulus_file(
    directory: Path,
    subject: str | int | None,
    *,
    required: bool,
) -> Path | None:
    candidates: Iterable[Path]
    search_directories = [directory]
    sibling_stimulus = directory.parent / "stimCND"
    if sibling_stimulus.is_dir():
        search_directories.append(sibling_stimulus)
    if directory.parent not in search_directories:
        search_directories.append(directory.parent)
    candidate_names = (
        (f"dataStim{subject}.mat", f"dataStim_{subject}.mat", "dataStim.mat")
        if subject is not None
        else ("dataStim.mat",)
    )
    candidates = (
        search_directory / name
        for search_directory in search_directories
        for name in candidate_names
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(directory / "dataStim.mat")
    return None


def _subject_from_filename(path: Path) -> str | None:
    stem = path.stem
    if stem.startswith("dataSub"):
        return stem.removeprefix("dataSub")
    if stem.startswith("pre_dataSub"):
        return stem.removeprefix("pre_dataSub")
    if stem.startswith("dataParticipant_"):
        return stem.removeprefix("dataParticipant_")
    return None


def _canonical_subject_label(subject: str | int) -> str:
    value = str(subject)
    if not value.isdigit() or int(value) < 1:
        raise ValueError("subject must be a positive numeric CND index")
    return str(int(value))


def _cnd_output_filename(
    filename: str | None, *, default: str, pattern: str, kind: str
) -> str:
    value = default if filename is None else filename
    if Path(value).name != value or re.fullmatch(pattern, value) is None:
        raise ValueError(f"{kind}_filename is not a recognised CND .mat filename")
    return value
