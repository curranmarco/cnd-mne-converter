# Verification

Recorded 24–25 August 2026: 1,026 neural files, 17,774 trials.
1,017 passed; eight BabyRhythm files had no samples; SparrKULee1 `dataSub48.mat`
was truncated. A separate zero-byte SparrKULee2 file was skipped.

Checks covered parsing, dimensions, MNE values, stimulus views, a finite power
spectrum, and in-memory template round trips. Numerical error was zero for files
with samples. The test unit `V` meant identity scaling; it did not confirm physical
units. These runs did not test serialized MATLAB round trips across the catalogue.

[Catalogue summary](catalog-summary.json) retains per-dataset counts and outcomes.
Detailed subject reports remain in Git history. To generate fresh reports:

```bash
uv run cnd-mne verify-dataset /path/to/dataCND --neural-unit uV \
  --serialized-round-trip --output outputs/verification/dataset.json
uv run python scripts/summarize_verification.py outputs/verification \
  --output outputs/verification/catalog-summary.json
```

Use confirmed units. Legacy files may warn about missing metadata, different
clocks, or unequal durations; the tolerant reader preserves them.

[Browser import audit](browser-import/audit.json) and
[coordinate/GUI audits](giovanni-review/results.json) retain smaller real-data
checks. [Download checksums](../manifests/README.md).
