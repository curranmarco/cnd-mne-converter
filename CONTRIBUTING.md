# Development

```bash
uv sync --locked --extra dev
uv run pytest --cov=cnd_mne --cov-fail-under=95
uv run ruff check .
uv run ruff format --check .
uv run mypy src/cnd_mne
uv build
```

Start with `model.py`, then the public API in `io.py` and `mne.py`.

| Responsibility | Module |
|---|---|
| MATLAB structures | `_mat_decode.py`, `_mat_encode.py` |
| Files and subject names | `io.py`, `_paths.py` |
| Units and coordinates | `_channels.py` |
| MNE export / trial views | `_mne_export.py`, `_mne_recording.py` |
| Recording contract | `validation.py` |
| Dataset checks / numerical checks / reports | `verification.py`, `_round_trip.py`, `_verification_report.py` |

The CLI and optional Qt window call the same public API.

Preserve samples, trial order, units, clocks, and unknown MATLAB fields. New
mappings need a round-trip test and a malformed-input test. Keep fixtures small
and redistributable. Explain any interpretation change in the PR.
