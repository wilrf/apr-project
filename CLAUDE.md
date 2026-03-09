# CLAUDE.md

Canonical repo instructions and document nomenclature live in [`AGENTS.md`](AGENTS.md).

## Deep Context — Read Before Non-Trivial Changes

- [`docs/architecture-and-analysis.md`](docs/architecture-and-analysis.md) — Full system reference: pipeline, features, models, evaluation, current results. **Read this before modifying any module.**
- [`docs/paper-framing-notes.md`](docs/paper-framing-notes.md) — Paper framing, positioning, key citations.
- [`results/test/report.md`](results/test/report.md) — March 2026 test set results (558 games).

## Non-Obvious Invariants

These cause bugs if violated:

- **Spread threshold**: Only `spread >= 3` games get labeled. Sub-3 games keep `upset = NaN`, excluded via `upset.notna()`.
- **Feature counts are canonical**: LR=46, XGB=70 (46+24 lags), LSTM=14seq×8ts+10matchup. Column lists live as constants in `pipeline.py` — import them, never hardcode.
- **Disagreement threshold**: Uses base upset rate (~0.30), NOT 0.5.
- **LSTM normalization**: Stats from training data only. Both teams share stats; normalization applied separately; padded positions zeroed after.
- **No week-1 games** in labeled data (no prior stats exist for rolling features).
- **Rolling window crosses seasons** by design (3-game window carries over).
- **Pre-March-2026 metrics are stale**: Old ~21% upset rate numbers are historical only.
- Do not edit `.claude/settings.local.json` for repo-wide policy changes.

## Change-Impact Map

When you modify a module, also review its downstream consumers:

```
pipeline.py (columns/lists)     → sequence_builder, target, generate_features,
                                   evaluate_test_set, run_ab_experiment
sequence_builder.py             → lstm_trainer, unified_trainer,
                                   evaluate_test_set, run_ab_experiment
lstm_model.py (architecture)    → lstm_trainer, unified_trainer
lstm_config.py (hyperparams)    → lstm_trainer, unified_trainer,
                                   evaluate_test_set, run_ab_experiment
unified_trainer.py              → evaluate_test_set, run_ab_experiment
cv_splitter.py                  → trainer, lstm_trainer, unified_trainer,
                                   run_ab_experiment
disagreement.py                 → evaluate_test_set, run_ab_experiment
calibration.py                  → evaluate_test_set
merger.py (column output)       → generate_features, verify_data, pipeline.py
betting_loader.py (team abbrs)  → merger, generate_features, verify_data
logistic_model.py (interface)   → unified_trainer, run_ab_experiment
xgboost_model.py (interface)    → unified_trainer, run_ab_experiment,
                                   shap_analysis
metrics.py                      → evaluate_test_set
```

## Coding Conventions

- **Model interface**: All models expose `fit(X, y)` and `predict_proba(X) → np.ndarray` (1D P(upset)).
- **Feature column lists**: Canonical lists are module-level constants in `pipeline.py`. Always import; never duplicate.
- **No-spread variants**: Drop 4 market features (LR/XGB) or 2 (LSTM matchup). Lists also in `pipeline.py`.
- **Tests mirror src/**: `tests/data/`, `tests/features/`, `tests/models/`, `tests/evaluation/`.
- **Validation**: `generate_features.py` runs 8 checks per split. New features must pass all 8.
- **Typing**: Use `from __future__ import annotations` in all modules.

## Data Splits

- Train: 2005–2022 (3,495 labeled), Test: 2023–2025 (558 labeled), upset rate ~30% both.

## Environment Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # includes PyTorch for LSTM
python3 -m src.data.generate_features  # regenerate train.csv + test.csv
```

## Common Commands

```bash
# Tests — LSTM tests are slow (~30s), separated by default
python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py -v
python3 -m pytest tests/models/test_lstm_model.py -v
python3 -m pytest tests/models/test_cv_splitter.py -v

# Lint
black src/ tests/
python3 -m ruff check src/ tests/

# Run pipeline
python3 -m src.data.generate_features
python3 -m src.models.evaluate_test_set
python3 -m src.models.run_ab_experiment --quick
```
