"""Speech envelopes from audio files, computed the way CNSP datasets are.

Two methods:

* ``"hilbert"`` (default): magnitude of the analytic signal (Hilbert
  transform), resampled to the stimulus rate with an anti-aliasing filter.
  This reproduces the "Speech Envelope Vectors" stored in the CNSP
  LalorNatSpeech CND (r = 0.9997 against the lab's own envelope).
* ``"mtrf"``: a Python port of mTRF-Toolbox ``mTRFenvelope`` (Lalor & Foxe,
  2010): mean power in a window around each output sample, square root, then
  compression to the power ``log10(2)``.

Word onsets come from Praat ``.TextGrid`` files (forced alignment), the way
CNSP builds its "Word Onset Vectors".
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.fft import next_fast_len
from scipy.io import wavfile
from scipy.signal import hilbert, resample_poly

EnvelopeMethod = Literal["hilbert", "mtrf"]


def read_audio(path: str | Path) -> tuple[np.ndarray, float]:
    """Read a ``.wav`` file as mono float samples and its sampling rate."""
    rate, data = wavfile.read(Path(path).expanduser())
    audio = np.asarray(data, dtype=float)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, float(rate)


def hilbert_envelope(
    audio: np.ndarray, sfreq_in: float, sfreq_out: float
) -> np.ndarray:
    """Hilbert-magnitude envelope resampled to ``sfreq_out``."""
    audio = np.asarray(audio, dtype=float)
    padded = next_fast_len(len(audio))  # FFT length with small prime factors (fast)
    envelope = np.abs(hilbert(audio, N=padded))[: len(audio)]
    return _resample(envelope, sfreq_in, sfreq_out)


def mtrf_envelope(
    audio: np.ndarray,
    sfreq_in: float,
    sfreq_out: float,
    window: float = 1,
    compression: float = float(np.log10(2)),
) -> np.ndarray:
    """Port of mTRF-Toolbox ``mTRFenvelope`` (with ``mTRFresample``).

    For output sample ``i`` at time ``t = i / sfreq_out`` it averages the
    squared audio from ``t - n`` to ``t + n`` (``n = window / (2 sfreq_out)``,
    same index rounding as MATLAB), then takes ``sqrt(mean) ** compression``.
    """
    power = np.asarray(audio, dtype=float) ** 2
    n_in = len(power)
    n_out = round(n_in / sfreq_in * sfreq_out)
    times = np.arange(n_out) / sfreq_out
    half = 0.5 * window / sfreq_out
    start = np.maximum(0, np.round(sfreq_in * (times - half)).astype(int))
    stop = np.minimum(n_in, np.round(sfreq_in * (times + half)).astype(int) + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(power)])
    mean_power = (cumulative[stop] - cumulative[start]) / (stop - start)
    return np.sqrt(mean_power) ** compression


def audio_envelope(
    path: str | Path, sfreq_out: float, method: EnvelopeMethod = "hilbert"
) -> np.ndarray:
    """Envelope of a ``.wav`` file at ``sfreq_out`` with the chosen method."""
    audio, sfreq_in = read_audio(path)
    if method == "hilbert":
        return hilbert_envelope(audio, sfreq_in, sfreq_out)
    if method == "mtrf":
        return mtrf_envelope(audio, sfreq_in, sfreq_out)
    raise ValueError(f"Unknown envelope method {method!r}; use 'hilbert' or 'mtrf'")


def textgrid_word_onsets(path: str | Path, tier: str = "words") -> list[float]:
    """Start times (s) of the non-empty intervals in a Praat TextGrid tier."""
    text = Path(path).expanduser().read_text(encoding="utf-8", errors="replace")
    onsets: list[float] = []
    tiers = re.split(r"\n\s*item \[\d+\]:", text)
    for block in tiers[1:]:
        name = re.search(r'name = "([^"]*)"', block)
        if name is None or name.group(1).lower() != tier.lower():
            continue
        for start, label in re.findall(
            r'xmin = ([-\d.eE+]+)\s*\n\s*xmax = [-\d.eE+]+\s*\n\s*text = "([^"]*)"',
            block,
        ):
            if label.strip() and label.strip().lower() not in {"sp", "sil", "<sil>"}:
                onsets.append(float(start))
    if not onsets and tier.lower() not in text.lower():
        raise ValueError(f"No tier named {tier!r} in {path}")
    return onsets


def _resample(signal: np.ndarray, sfreq_in: float, sfreq_out: float) -> np.ndarray:
    ratio = Fraction(sfreq_out / sfreq_in).limit_denominator(10_000)
    resampled = resample_poly(signal, ratio.numerator, ratio.denominator)
    return np.asarray(resampled[: round(len(signal) / sfreq_in * sfreq_out)])
