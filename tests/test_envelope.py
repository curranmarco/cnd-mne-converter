"""Speech envelopes from audio and their linking into from-bids."""

from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np
import pytest
from scipy.io import wavfile

from cnd_mne.envelope import (
    audio_envelope,
    hilbert_envelope,
    mtrf_envelope,
    textgrid_word_onsets,
)

TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 3.0
tiers? <exists>
size = 1
item []:
\titem [1]:
\t\tclass = "IntervalTier"
\t\tname = "words"
\t\txmin = 0.0
\t\txmax = 3.0
\t\tintervals: size = 4
\t\t\tintervals [1]:
\t\t\t\txmin = 0.0
\t\t\t\txmax = 0.5
\t\t\t\ttext = "sil"
\t\t\tintervals [2]:
\t\t\t\txmin = 0.5
\t\t\t\txmax = 1.0
\t\t\t\ttext = "old"
\t\t\tintervals [3]:
\t\t\t\txmin = 1.0
\t\t\t\txmax = 1.25
\t\t\t\ttext = ""
\t\t\tintervals [4]:
\t\t\t\txmin = 1.25
\t\t\t\txmax = 3.0
\t\t\t\ttext = "man"
"""


def _speech_like(seconds: float, rate: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(seconds * rate)) / rate
    envelope = 1.0 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz "syllable" rhythm
    return envelope * np.sin(2 * np.pi * 1000 * t), envelope


def test_hilbert_envelope_recovers_the_modulation() -> None:
    audio, truth = _speech_like(4.0)
    envelope = hilbert_envelope(audio, 16000, 128)
    assert len(envelope) == 4 * 128
    expected = truth[:: 16000 // 128][: len(envelope)]
    inner = slice(16, -16)  # ignore filter edges
    assert np.corrcoef(envelope[inner], expected[inner])[0, 1] > 0.999
    np.testing.assert_allclose(envelope[inner], expected[inner], rtol=0.02)


def test_mtrf_envelope_matches_matlab_indexing() -> None:
    # Hand-worked from mTRFresample: fsin 10, fsout 5, window 1, no compression.
    x = np.arange(1.0, 11.0)
    y = mtrf_envelope(x, 10, 5, compression=1.0)
    assert len(y) == 5
    assert y[0] == pytest.approx(np.sqrt((1 + 4) / 2))  # samples 1-2
    assert y[1] == pytest.approx(np.sqrt((4 + 9 + 16) / 3))  # samples 2-4
    # default compression log10(2) of a constant-amplitude tone
    tone = 2.0 * np.sin(2 * np.pi * 50 * np.arange(8000) / 8000)
    level = mtrf_envelope(tone, 8000, 100)[5:-5]
    # MATLAB windows include both end samples, so not exactly whole periods
    np.testing.assert_allclose(level, np.sqrt(2.0) ** np.log10(2), rtol=5e-3)


def test_audio_envelope_reads_wav(tmp_path: Path) -> None:
    audio, _ = _speech_like(2.0)
    stereo = np.stack([audio, audio], axis=1)
    wavfile.write(tmp_path / "a.wav", 16000, (stereo * 10000).astype(np.int16))
    assert len(audio_envelope(tmp_path / "a.wav", 100)) == 200
    assert len(audio_envelope(tmp_path / "a.wav", 100, "mtrf")) == 200
    with pytest.raises(ValueError, match="Unknown envelope method"):
        audio_envelope(tmp_path / "a.wav", 100, "nope")  # type: ignore[arg-type]


def test_textgrid_word_onsets(tmp_path: Path) -> None:
    (tmp_path / "a.TextGrid").write_text(TEXTGRID)
    assert textgrid_word_onsets(tmp_path / "a.TextGrid") == [0.5, 1.25]
    with pytest.raises(ValueError, match="No tier"):
        textgrid_word_onsets(tmp_path / "a.TextGrid", tier="phones")


# -- linking audio into from-bids ---------------------------------------------

mne_bids = pytest.importorskip("mne_bids")
pytest.importorskip("pybv")


def _bids_with_audio(root: Path, *, events: bool) -> None:
    for run in (1, 2):
        raw = mne.io.RawArray(
            np.random.default_rng(run).standard_normal((2, 2000)) * 1e-5,
            mne.create_info(["Fz", "Cz"], 200.0, "eeg"),
            verbose=False,
        )
        if events:
            raw.set_annotations(mne.Annotations([2.0], 0.0, ["sound"]))
        mne_bids.write_raw_bids(
            raw,
            mne_bids.BIDSPath(
                subject="01", task="listen", run=run, datatype="eeg", root=root
            ),
            format="BrainVision",
            allow_preload=True,
            event_id={"sound": 1} if events else None,
            verbose=False,
        )
    (root / "stimuli").mkdir()
    for run in (1, 2):
        audio, _ = _speech_like(3.0)
        wavfile.write(root / "stimuli" / f"audio{run:02d}.wav", 16000, audio)
        (root / "stimuli" / f"audio{run:02d}.TextGrid").write_text(TEXTGRID)


def test_from_bids_with_audio_pattern(tmp_path: Path) -> None:
    from cnd_mne.bids import read_any_bids

    _bids_with_audio(tmp_path, events=False)
    stimulus = read_any_bids(
        tmp_path, subject="01", task="listen", audio="stimuli/audio{run:02d}.wav"
    ).stimulus
    assert stimulus is not None
    assert stimulus.names == ("Speech Envelope", "Word Onset")
    assert stimulus.sfreq == 128.0
    envelope = stimulus.features[0][1]
    assert len(envelope) == 1280  # 10 s of EEG at 128 Hz
    assert envelope[: 3 * 128 - 10].min() > 0  # audio from 0 s ...
    assert not envelope[3 * 128 + 5 :].any()  # ... to 3 s, then silence
    assert np.flatnonzero(stimulus.features[1][0]).tolist() == [64, 160]


def test_from_bids_with_stim_file_column(tmp_path: Path) -> None:
    from cnd_mne.bids import read_any_bids

    _bids_with_audio(tmp_path, events=True)
    for events in tmp_path.rglob("*_events.tsv"):
        with events.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        run = events.name.split("run-")[1][:1]
        for row in rows:
            row["stim_file"] = f"audio0{run}.wav"
        with events.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    stimulus = read_any_bids(tmp_path, subject="01", task="listen").stimulus
    assert stimulus is not None
    assert stimulus.names == ("Speech Envelope", "Word Onset", "sound")
    envelope = stimulus.features[0][0]
    assert not envelope[: 2 * 128 - 2].any()  # played at 2 s
    assert envelope[2 * 128 + 10 : 5 * 128 - 10].min() > 0
    assert np.flatnonzero(stimulus.features[1][0]).tolist() == [256 + 64, 256 + 160]
    assert np.flatnonzero(stimulus.features[2][0]).tolist() == [256]


def test_cli_audio_and_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cnd_mne.cli import main
    from cnd_mne.io import read_cnd

    _bids_with_audio(tmp_path / "bids", events=False)
    base = ["from-bids", str(tmp_path / "bids"), "--subject", "01", "--task", "listen"]
    assert (
        main(
            [
                *base,
                "--output",
                str(tmp_path / "cnd"),
                "--audio",
                "stimuli/audio{run:02d}.wav",
                "--envelope-method",
                "mtrf",
            ]
        )
        == 0
    )
    assert read_cnd(tmp_path / "cnd", subject=1).stimulus.names[0] == "Speech Envelope"  # type: ignore[union-attr]
    assert (
        main(
            [
                *base,
                "--output",
                str(tmp_path / "x"),
                "--audio",
                "stimuli/missing{run}.wav",
            ]
        )
        == 1
    )
    assert "Audio file not found" in capsys.readouterr().err
