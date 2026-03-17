# CLAUDE.md

Canonical repo instructions and document nomenclature live in [`AGENTS.md`](AGENTS.md).

## Deep Context — Read Before Non-Trivial Changes

- [`docs/architecture-and-analysis.md`](docs/architecture-and-analysis.md) — Full system reference: pipeline, features, models, evaluation, current results. **Read this before modifying any module.**
- [`docs/paper.md`](docs/paper.md) — Research paper draft (abstract through conclusion).
- [`docs/paper-framing-notes.md`](docs/paper-framing-notes.md) — Paper framing, positioning, key citations.
- [`results/test/report.md`](results/test/report.md) — Test set results (558 games, regenerated 2026-03-10).
- [`results/ab_experiment/`](results/ab_experiment/) — A/B spread ablation: CV predictions, significance tests, full results summary.

## Non-Obvious Invariants

These cause bugs if violated:

- **Spread threshold**: Only `spread >= 3` games get labeled. Sub-3 games keep `upset = NaN`, excluded via `upset.notna()`.
- **Feature counts are canonical**: LR=46, XGB=70 (46+24 lags), LSTM=14seq×8ts+10matchup. Column lists live as constants in `pipeline.py` — import them, never hardcode.
- **Disagreement threshold**: Uses base upset rate (~0.30), NOT 0.5.
- **Threshold consistency**: All binary predictions in `DisagreementAnalyzer` (categorization, agreement matrix, CSV exports) must use `self.threshold`. Never use `GamePrediction.lr_pred`/`xgb_pred`/`lstm_pred` properties inside the analyzer — those hardcode 0.5.
- **LSTM history includes all games**: Sub-3 spread games are used in LSTM team history (via `history_df`) but NOT as labeled targets. Callers must pass full DataFrame for history, filtered DataFrame for labels.
- **LSTM normalization**: Stats from training data only. Both teams share stats; normalization applied separately; padded positions zeroed after.
- **LSTM squeeze**: Always use `.squeeze(-1)`, never bare `.squeeze()`. Bare squeeze on `(1,1)` tensors produces 0-d scalars.
- **No week-1 games** in labeled data (no prior stats exist for rolling features).
- **Rolling window crosses seasons** by design (3-game window carries over). LSTM history is keyed by team (not by season).
- **ReportGenerator threshold**: If passing both `threshold=` and `disagreement_analyzer=`, they must agree or `ValueError` is raised.
- **comparison.py is unused**: `ModelComparison` is not imported anywhere in production. Use `DisagreementAnalyzer` instead.
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
report.py                       → disagreement.py (delegates for summary)
comparison.py                   → (UNUSED in production — dead code)
merger.py (column output)       → generate_features, verify_data, pipeline.py
betting_loader.py (team abbrs)  → merger, generate_features, verify_data
logistic_model.py (interface)   → unified_trainer, run_ab_experiment
xgboost_model.py (interface)    → unified_trainer, run_ab_experiment
metrics.py                      → evaluate_test_set
```

## Bug Fix Protocol

Before changing production code for any claimed bug, follow **red-green-trace** (full playbook: `~/.master-prompts/bug-fix.md`):

1. **Red**: Write a test that fails because of the claimed broken behavior. If it passes immediately, STOP — the bug may not exist.
2. **Green**: Apply the smallest fix that flips the test from red to green.
3. **Trace**: Grep all callers of the changed function. Verify at least one test exercises the caller→callee wiring, not just the component in isolation.

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
# Fast tests (221 tests, ~5s) — excludes slow LSTM model + trainer tests
python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py --ignore=tests/models/test_lstm_trainer.py -v

# LSTM tests (~30s each, run separately)
python3 -m pytest tests/models/test_lstm_model.py -v
python3 -m pytest tests/models/test_lstm_trainer.py -v

# Lint
black src/ tests/
python3 -m ruff check src/ tests/

# Run pipeline
python3 -m src.data.generate_features
python3 -m src.models.evaluate_test_set
python3 -m src.models.run_ab_experiment --quick    # LR+XGB only (~1 min)
python3 -m src.models.run_ab_experiment            # All 3 models (~15 min)
```

## Bughunt Status

See [`BUGHUNT.md`](BUGHUNT.md) for the full list of 43 bugs found 2026-03-09. Current status:
- **All HIGH bugs resolved** (H1–H15). H4 was a false positive (algebraically correct formula).
- **CRITICAL bugs resolved** (C1 test gap remains at MEDIUM priority, C2 fully tested).
- MEDIUM/LOW bugs remain open — see BUGHUNT.md for details.
