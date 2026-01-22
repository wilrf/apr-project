# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project comparing XGBoost vs LSTM models for predicting NFL moneyline upsets. The goal is to understand structural factors that drive upsets and compare how each model reasons about predictions.

**Current Status:** Infrastructure complete, ready for model training and evaluation.

## Commands

```bash
# Run all tests
python3 -m pytest

# Run specific test file
python3 -m pytest tests/models/test_cv_splitter.py -v

# Run tests with coverage
python3 -m pytest --cov=src

# Format code
black src/ tests/
ruff check src/ tests/

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Data Flow
```
nfl_data_py + Kaggle CSV → src/data/ (load & merge) → src/features/pipeline.py → src/models/ → src/evaluation/
```

### Key Modules

**`src/data/`** - Data ingestion
- `nfl_loader.py`: Fetches from nflverse (schedules, play-by-play)
- `betting_loader.py`: Loads Kaggle spreadspoke CSV, normalizes team abbreviations
- `merger.py`: Joins NFL + betting data on season/week/team with audit trail
- `verify_data.py`: Coverage verification script

**`src/features/`** - Feature engineering
- `pipeline.py`: Main pipeline combining all features, handles rolling windows
- `target.py`: Calculates binary upset target (underdog wins, spread ≥ +3)
- `rolling.py`: 5-game rolling averages with early-season padding
- `matchup.py`: Underdog vs favorite differentials

**`src/models/`** - Model implementations
- `xgboost_model.py`: XGBoost wrapper with SHAP support
- `lstm_model.py`: PyTorch siamese LSTM with attention
- `cv_splitter.py`: Time-series cross-validation (6 folds, 2017-2022)
- `trainer.py`: Training pipeline with CV and metrics
- `mlflow_utils.py`: Experiment tracking wrapper

**`src/evaluation/`** - Analysis and reporting
- `metrics.py`: AUC, Brier, calibration, betting ROI
- `comparison.py`: XGBoost vs LSTM disagreement analysis
- `shap_analysis.py`: SHAP explanations for XGBoost
- `report.py`: Automated report generation

## Data Split

- **Training:** 2005-2022 (18 seasons, ~4,608 games)
- **Test:** 2023, 2024, 2025 (3 seasons, ~816 games)
- **CV:** 6-fold time-series (validation years 2017-2022)

See `docs/data-split-change.md` for rationale.

## Key Constraints

1. **Temporal integrity**: Never use future data. Time-series CV only.
2. **Rolling windows**: 5-game averages. Early-season: use available games with masking.
3. **Week 1 exclusion**: Teams need ≥1 prior game for predictions.
4. **Regular season only**: Exclude playoffs.
5. **Underdog threshold**: Spread ≥ +3 to qualify as underdog.

## Key Documents

| Document | Purpose |
|----------|---------|
| `docs/plans/2026-01-16-nfl-upset-xgboost-lstm-design.md` | Authoritative spec |
| `docs/plans/2026-01-16-nfl-upset-implementation-plan.md` | Step-by-step implementation |
| `docs/data-split-change.md` | Data split rationale |
| `findings.md` | Design review findings |

## Data Sources

- `nflverse/nfl_data_py`: Game stats, EPA metrics
- `data/raw/spreadspoke_scores.csv`: Kaggle betting data (must download manually)

## Skills Reference

Use these Claude Code skills when appropriate:
- `superpowers:brainstorming` - Before creative/building work
- `superpowers:systematic-debugging` - Before fixing bugs
- `superpowers:test-driven-development` - Before writing implementation
- `superpowers:verification-before-completion` - Before claiming done
