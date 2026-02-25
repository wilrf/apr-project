# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Pure Python ML research project (no web server, database, or Docker). See `CLAUDE.md` for full architecture and data flow.

### Running tests

Tests must be run in two batches due to PyTorch/pytest interaction issues:

```bash
python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py -v
python3 -m pytest tests/models/test_lstm_model.py -v
```

10 tests in `tests/features/test_pipeline.py` are pre-existing failures (IndexError / KeyError on spread/situational feature tests). These are known issues in the codebase, not environment problems.

### Linting

```bash
black --check src/ tests/
ruff check src/ tests/
```

Ruff emits a deprecation warning about `select` in `pyproject.toml` (should be under `[tool.ruff.lint]`). This is cosmetic and does not affect functionality.

### PATH note

Pip installs scripts to `~/.local/bin`. If `black`, `ruff`, or `pytest` are not found, ensure `PATH` includes `$HOME/.local/bin`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Data dependencies

The full data pipeline requires `data/raw/spreadspoke_scores.csv` (Kaggle download, gitignored) and internet access for `nfl_data_py` API calls. Tests use mocked/synthetic data and do not require these files.
