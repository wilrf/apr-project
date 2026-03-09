# AGENTS.md

This repository uses one canonical document naming scheme. If a topic has multiple docs, the default source of truth is the design spec, not the implementation plan or the review.

## Project

APR research project on NFL upset prediction using LR, XGBoost, and LSTM as diagnostic tools rather than a pure leaderboard exercise. The core contribution is using disagreement among model classes to taxonomize upset mechanisms, not just improving prediction accuracy.

### Current State

The codebase uses a multi-representation architecture where each model gets the same data in a different representation. The canonical feature pipeline in `src/features/pipeline.py` produces 70 features total (46 base + 24 XGB per-game lags). Train/test CSVs are generated through `src.data.generate_features`. The canonical target labels only games with `spread >= 3`; sub-3 spread games remain in the DataFrame with `upset = NaN` for rolling-stat computation and are excluded from training/evaluation via the standard `upset.notna()` filter.

### Key Numbers

- **LR**: 46 base features (42 no-spread)
- **XGB**: 70 features — 46 base + 24 per-game lag stats (66 no-spread)
- **LSTM**: 14 sequence features × 8 timesteps + 10 matchup context (8 matchup no-spread)
- 3,495 labeled train games in the current clean split
- 558 labeled test games in the current clean split
- Upset rate is roughly 30% in both labeled splits
- Full architecture reference: `docs/architecture-and-analysis.md`

## Data Integrity

- March 2026 label cleanup removed 846 train and 193 test fake negatives from the labeled sets.
- `src.data.generate_features` now runs 8 strict validation checks on each split and separately verifies train/test season overlap.
- Current clean labeled sets have no sub-3 games, no NaN/inf feature values, no duplicate labeled `game_id`s, no week-1 labeled games, and no train/test season overlap.
- Metrics produced before the label cleanup, including results based on the old ~21% upset rate, are historical only and must be recomputed before being treated as current.

## Canonical Document Types

- `Design Spec`: `docs/plans/YYYY-MM-DD-topic-slug-design.md`
  - Main doc for a topic.
  - Owns the research question, experiment design, invariants, interpretation rules, and experimental matrix.
- `Implementation Plan`: `docs/plans/YYYY-MM-DD-topic-slug-implementation.md`
  - Execution checklist for one design spec.
  - May add engineering detail, but does not override the design spec.
- `Review Report`: `docs/YYYY-MM-DD-topic-slug-review-report.md`
  - Critique of a design or result set.
  - Never becomes the source of truth by itself; its accepted changes must be reflected back into the design spec.
- `Results`: `docs/YYYY-MM-DD-topic-slug-results.md` or stable results such as `results/test/report.md`
  - Reports measured outcomes from actual runs.
  - Do not treat results docs as design docs.
- `Notes`: `*-notes.md`
  - Working notes, framing notes, scratch analysis.
  - Non-authoritative unless promoted into a design or results doc.
- `Status`: `status-*.md` or `*-status.md`
  - Snapshot of repo/project state.
  - Non-authoritative for experiment design.

## Resolution Rules

- If a user says "main doc", "feature redesign doc", or just names a topic with multiple docs, use the newest matching `Design Spec`.
- If documents disagree, trust order is:
  - newest `Design Spec`
  - matching `Implementation Plan`
  - matching `Review Report`
  - `Results`
  - `Notes` and `Status`
- Reviews recommend changes. Designs ratify them.
- Implementation plans execute designs. They do not redefine the research claim.

## Naming Rules

- Use one stable topic slug across a document family.
- Keep the doc type suffix explicit: `design`, `implementation`, `review-report`, `results`, `notes`, `status`, `pivot`.
- Put dated design and implementation docs under `docs/plans/`.
- Keep review reports and dated result summaries in `docs/`, not `docs/plans/`.
- When a topic has both design and implementation docs, the implementation doc must link to the design spec near the top.
- When a topic has a review report, the review must link to the design spec near the top.

## Completed Topics

- Topic: `feature-redesign`
- Current canonical implementation: [`src/features/pipeline.py`](/Users/wilfowler/Documents/Projects/apr-research/src/features/pipeline.py)
- Historical March 2026 docs remain in `docs/` and `docs/plans/` for reference only

## Repo Structure

```
src/
  data/           # Load, merge, validate raw data → data/features/*.csv
  features/       # 70-feature multi-representation pipeline
  models/         # LR, XGBoost, LSTM, trainers, experiment scripts
  evaluation/     # Disagreement analysis, calibration, metrics, reports
tests/            # Mirrors src/ structure
data/
  features/       # Generated train.csv and test.csv
  raw/            # Source data files
docs/
  plans/          # Design specs and implementation plans
```

## Change-Impact Map

When modifying a module, also review its downstream consumers:

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
- **Feature column lists**: Canonical lists are module-level constants in `pipeline.py`. Always import; never hardcode column names.
- **No-spread variants**: Drop 4 market features (LR/XGB) or 2 (LSTM matchup). Lists also in `pipeline.py`.
- **Tests mirror src/**: `tests/data/`, `tests/features/`, `tests/models/`, `tests/evaluation/`.
- **Validation**: `generate_features.py` runs 8 checks per split. New features must pass all 8.
- **Typing**: Use `from __future__ import annotations` in all modules.

## Common Commands

```bash
python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py -v
python3 -m pytest tests/models/test_lstm_model.py -v
black src/ tests/
python3 -m ruff check src/ tests/
python3 -m src.data.generate_features
python3 -m src.models.evaluate_test_set
python3 -m src.models.run_ab_experiment --quick
```

## Agent-Specific Bridges

- Claude-specific delta lives in [`CLAUDE.md`](/Users/wilfowler/Documents/Projects/apr-research/CLAUDE.md).
- Codex-specific delta lives in [`CODEX.md`](/Users/wilfowler/Documents/Projects/apr-research/CODEX.md).
- Documentation-specific writing rules live in [`docs/AGENTS.md`](/Users/wilfowler/Documents/Projects/apr-research/docs/AGENTS.md).
