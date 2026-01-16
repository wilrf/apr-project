# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project comparing XGBoost vs LSTM models for predicting NFL moneyline upsets. The goal is to understand structural factors that drive upsets and compare how each model reasons about predictions.

**Current Status:** Design phase complete. Implementation not yet started.

## Key Design Document

The authoritative project specification is at:
```
docs/plans/2026-01-16-nfl-upset-xgboost-lstm-design.md
```

Reference this document for all implementation decisions regarding:
- Target variable definition (underdog wins outright, spread ≥ +3)
- Data sources and time ranges
- Feature engineering specifications
- Model architectures
- Evaluation metrics and success criteria

## Planned Tech Stack

| Component | Tool |
|-----------|------|
| Data | `nfl_data_py`, `pandas` |
| ML | `xgboost`, `pytorch` |
| Interpretability | `shap`, `captum` |
| Experiment tracking | `mlflow` |
| Hyperparameter tuning | `optuna` |

## Project Structure (Planned)

```
data/raw/           # Downloaded NFL/betting data
data/processed/     # Feature-engineered datasets
data/splits/        # Train/val/test sets
notebooks/          # Jupyter notebooks (01-05)
src/                # Python modules
results/figures/    # Output visualizations
plugins/            # Plugin documentation for LLMs/agents
docs/plans/         # Design documents
```

## Plugins & Skills Reference

Documentation for Claude Code plugins is available in `plugins/`:

| Plugin | File | Purpose |
|--------|------|---------|
| superpowers | `plugins/superpowers.md` | TDD, debugging, brainstorming, verification skills |
| planning-with-files | `plugins/planning-with-files.md` | Manus-style file-based planning |
| supabase | `plugins/supabase.md` | Database/auth MCP integration |
| code-simplifier | `plugins/code-simplifier.md` | Code refinement agent |
| pyright | `plugins/pyright.md` | Python type checking |

**Key skills to use:**
- `superpowers:brainstorming` - Before any creative/building work
- `superpowers:systematic-debugging` - Before fixing any bug
- `superpowers:test-driven-development` - Before writing implementation
- `superpowers:verification-before-completion` - Before claiming done

## Key Implementation Constraints

1. **Temporal integrity**: Never use future data to predict past games. Use time-series cross-validation.
2. **Rolling windows**: 5-game rolling averages for team stats. Pad with zeros + masking for early-season games.
3. **Week 1 exclusion**: Teams must have at least 1 prior game to be included in predictions.
4. **Regular season only**: Exclude playoff games from all datasets.

## Data Sources

- `nflverse/nfl_data_py`: Game stats, EPA metrics
- `Kaggle: spreadspoke/nfl-scores-and-betting-data`: Historical spreads and results
