# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project comparing XGBoost vs LSTM models for predicting NFL moneyline upsets. The goal is to understand structural factors that drive upsets and compare how each model reasons about predictions.

**Current Status:** Infrastructure complete, ready for model training and evaluation.

## Commands

```bash
# Run all tests (in batches due to PyTorch/pytest interaction)
python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py -v
python3 -m pytest tests/models/test_lstm_model.py -v

# Run specific test file
python3 -m pytest tests/models/test_cv_splitter.py -v

# Run tests with coverage
python3 -m pytest --cov=src

# Format code
black src/ tests/
ruff check src/ tests/
```

## Architecture

### Data Flow
```
nfl_data_py + Kaggle CSV → src/data/ → src/features/pipeline.py → src/models/ → src/evaluation/
```

### Key Modules

**`src/data/`** - Data ingestion
- `nfl_loader.py`: Fetches from nflverse (schedules)
- `betting_loader.py`: Loads Kaggle CSV, normalizes team abbreviations
- `merger.py`: Joins NFL + betting data on season/week/team

**`src/features/`** - Feature engineering (61 features)
- `pipeline.py`: Main pipeline with rolling stats, matchup differentials, interactions
- `target.py`: Binary upset target (underdog wins, spread ≥ +3)

**`src/models/`** - Model implementations
- `xgboost_model.py`: XGBoost wrapper with SHAP support
- `lstm_model.py`: PyTorch siamese LSTM with attention
- `cv_splitter.py`: Time-series cross-validation (6 folds)
- `trainer.py`: Training pipeline with CV and metrics

**`src/evaluation/`** - Analysis and reporting
- `metrics.py`: AUC, Brier, calibration (ECE), betting ROI
- `comparison.py`: XGBoost vs LSTM disagreement analysis
- `shap_analysis.py`: SHAP explanations for XGBoost

### Data Directory
```
data/
├── raw/spreadspoke_scores.csv   # Kaggle betting data (gitignored)
└── features/                     # Generated datasets (gitignored)
    ├── train.csv                 # 2005-2022 (4,346 games, 3,497 upset candidates)
    ├── test.csv                  # 2023-2025 (768 games, 559 upset candidates)
    └── columns.csv               # 61 feature names
```

## Data Split

- **Training:** 2005-2022 (18 seasons)
- **Test:** 2023-2025 (3 seasons, out-of-sample)
- **CV:** 6-fold time-series (validation years 2017-2022)

## Key Constraints

1. **Temporal integrity**: Never use future data. All rolling stats use `shift(1)`.
2. **Week 1 exclusion**: Teams need ≥1 prior game for predictions.
3. **Underdog threshold**: Spread ≥ +3 to qualify as underdog.
4. **Regular season only**: Playoffs excluded.

## Key Documents

- `docs/plans/2026-01-16-nfl-upset-xgboost-lstm-design.md` - Authoritative spec
- `docs/plans/2026-01-16-nfl-upset-implementation-plan.md` - Implementation plan
- `docs/data-split-change.md` - Data split rationale
