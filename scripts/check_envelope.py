"""Check our speech envelope against the CNSP lab's own, for the same audio.

ds004408 (OpenNeuro, BIDS with audio) and CNSP's LalorNatSpeech (CND) come
from the same natural-speech experiment: run N used audioNN.wav in both. This
script converts ds004408 to CND with ``from-bids --audio`` (our code), then
compares, run by run, our "Speech Envelope" with CNSP's "Speech Envelope
Vectors", and our "Word Onset" (all words) with CNSP's "Word Onset Vectors"
(content words only, so CNSP's must be a subset of ours).

Example (from the cnd-mne-converter folder):

  uv run --extra bids python scripts/check_envelope.py \\
    --bids "$R/open-datasets/ds004408" --subject 001 \\
    --cnsp "$R/CNSP-resources/CNSP/datasets/LalorNatSpeech/dataCND" --cnsp-subject 10
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from cnd_mne import read_cnd
from cnd_mne.bids import read_any_bids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bids", type=Path, required=True)
    parser.add_argument("--subject", default="001")
    parser.add_argument("--task", default="listening")
    parser.add_argument("--audio", default="stimuli/audio{run:02d}.wav")
    parser.add_argument("--method", default="hilbert", choices=("hilbert", "mtrf"))
    parser.add_argument("--cnsp", type=Path, required=True)
    parser.add_argument("--cnsp-subject", type=int, default=10)
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ds004408's EEG units warning
        ours = read_any_bids(
            args.bids.expanduser(),
            subject=args.subject,
            task=args.task,
            audio=args.audio,
            envelope_method=args.method,
        ).stimulus
    lab = read_cnd(args.cnsp.expanduser(), subject=args.cnsp_subject).stimulus
    assert ours is not None and lab is not None
    print(f"ours: {ours.names} at {ours.sfreq:g} Hz")
    print(f"CNSP: {lab.names[:2]} at {lab.sfreq:g} Hz")

    passed = ours.sfreq == lab.sfreq
    for trial, (envelope, words) in enumerate(zip(*ours.features[:2], strict=True)):
        lab_envelope = np.ravel(lab.features[0][trial])
        sound = np.flatnonzero(envelope)  # our track is zero after the audio ends
        n = min(sound[-1] + 1, len(lab_envelope))
        r = float(np.corrcoef(envelope[:n], lab_envelope[:n])[0, 1])
        our_words = set(np.flatnonzero(words))
        lab_words = set(np.flatnonzero(np.ravel(lab.features[1][trial])))
        found = len(lab_words & our_words)
        ok = r > 0.99 and found == len(lab_words)
        passed &= ok
        print(
            f"run {trial + 1}: envelope r = {r:.4f} | CNSP content words found in "
            f"ours at the same sample: {found}/{len(lab_words)} "
            f"(ours has {len(our_words)} words) -> {'passed' if ok else 'FAILED'}"
        )
    print("ALL PASSED" if passed else "SOMETHING FAILED")


if __name__ == "__main__":
    main()
