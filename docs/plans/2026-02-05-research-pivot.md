# Research Pivot: Multi-Model Diagnostic Framework

**Date:** 2026-02-05
**Status:** Implemented

## Executive Summary

This document explains the pivot from "which model is better?" to "what do models reveal about upsets?"

## Original Research Question

> "Can machine learning models predict NFL moneyline upsets better than the market?"

This framing treats models as competitors, optimizing for predictive accuracy.

## New Research Question

> "What do the differing predictive structures of Logistic Regression, XGBoost, and LSTM models reveal about the underlying mechanisms of NFL upsets?"

This framing treats models as **diagnostic tools** that reveal different structural aspects of upset causation.

## Theoretical Basis

### Why Different Models Capture Different Signals

Each model architecture has inherent biases that make it sensitive to specific types of patterns:

#### Logistic Regression: Spread Mispricing
- **Structure:** Linear combination of features → probability
- **What it captures:** When the betting market systematically misprices certain linear feature combinations
- **Interpretation:** If LR alone predicts an upset, the market likely has a linear blind spot (e.g., consistently undervaluing teams with specific rest/travel patterns)

#### XGBoost: Feature Interactions
- **Structure:** Ensemble of decision trees with gradient boosting
- **What it captures:** Non-linear interactions between features that create upset conditions
- **Interpretation:** If XGBoost alone predicts an upset, specific combinations of factors (not individually predictive) create the upset signal

#### LSTM: Temporal Patterns
- **Structure:** Recurrent neural network with memory cells processing game sequences
- **What it captures:** Momentum, fatigue patterns, and sequential dynamics invisible to static models
- **Interpretation:** If LSTM alone predicts an upset, recent game sequences (streaks, injuries, scheduling patterns) provide the signal

### The Diagnostic Framework

Instead of asking "which model wins?", we categorize games by **agreement pattern**:

| Category | Interpretation |
|----------|---------------|
| ALL_CORRECT | Obvious upsets - clear signal all models detect |
| ALL_WRONG | True randomness or factors outside model scope |
| ONLY_LR | Spread mispricing - linear market inefficiency |
| ONLY_XGB | Interaction-driven - non-linear feature combinations |
| ONLY_LSTM | Temporal signal - momentum/fatigue patterns |
| LR_XGB | Static models agree - non-temporal signal |
| LR_LSTM | Linear + temporal agree |
| XGB_LSTM | Non-linear + temporal agree |

## Research Value

### Academic Contribution
- Moves beyond "model horse race" papers
- Provides framework for understanding upset mechanisms
- Offers interpretable insights for sports analytics

### Practical Applications
- Identifies which upset types are predictable
- Suggests where models add value vs. randomness
- Informs feature engineering priorities

## Implementation

### Files Created/Modified

| File | Purpose |
|------|---------|
| `src/models/logistic_model.py` | LR wrapper with coefficient extraction |
| `src/models/unified_trainer.py` | Train all 3 models on identical folds |
| `src/evaluation/disagreement.py` | Categorize predictions by agreement |
| `src/evaluation/comparison.py` | Extended for 3-model analysis |
| `src/evaluation/report.py` | Added disagreement section |

### Best Configurations (from experiments)

| Model | Config | Expected CV AUC |
|-------|--------|-----------------|
| LR | `C=0.1, penalty='l1', solver='saga'` | ~0.656 |
| XGBoost | `max_depth=1, lr=0.01, n_estimators=500` | ~0.648 |
| LSTM | Siamese architecture, 64 hidden | TBD |

## Success Criteria

1. **Category Coverage:** All 8 categories have games assigned
2. **Meaningful Insights:** ONLY_* categories provide interpretable patterns
3. **Academic Framing:** Results support mechanism-based interpretation
4. **Reproducibility:** CV folds are identical across all models

## Future Directions

1. **Deep dives by category:** Profile features that distinguish each category
2. **Temporal analysis:** When in the season do ONLY_LSTM games cluster?
3. **Market efficiency:** Do ONLY_* categories represent exploitable inefficiencies?
4. **Ensemble strategies:** Weight models differently based on category-specific accuracy
