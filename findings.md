# Findings: NFL Upset Prediction Design Review

## Current Review Intake (2026-??)
- Spec doc located at `docs/plans/2026-01-16-nfl-upset-xgboost-lstm-design.md`
- Existing findings below reflect a prior completed review; will reconcile with the current request and confirm if spec content has changed.

## Preliminary Findings (Current Review)
- Success criteria list Brier < 0.24 as better than a 35% baseline; this appears mathematically worse than the naive baseline and should be recalculated.
- Multiple required feature inputs (weather, travel distance, DVOA, reverse line movement) lack data source commitments.
- Multi-source merge strategy (nflverse/PFR/Kaggle) is unspecified, creating high implementation risk.
- Model comparison uses engineered tabular features for XGBoost but raw sequences for LSTM; fairness of comparison needs explicit mitigation or scoping.
  - Confirmed in Success Criteria table (lines ~197-201 of spec) that Brier target is stated as < 0.24 vs 35% baseline.
  - Feature list lines ~54-79 add weather/travel/DVOA/line movement, but the sources table (lines ~23-28) does not identify where those fields come from.
  - Sources table lines 23-28 lists only nflverse/PFR/Kaggle, which do not cover travel distance, weather, DVOA, or public betting splits needed for reverse line movement.

## Document Overview
- **Source:** `/docs/plans/2026-01-16-nfl-upset-xgboost-lstm-design.md`
- **Version:** v4 - logic gaps fixed
- **Status:** Design Complete
- **Review Date:** 2026-01-16

## Strengths Identified

### Data Strategy
- Clear data verification step BEFORE committing to date range - shows risk awareness
- Multiple data sources for cross-validation
- Train/val/test split respects temporal ordering (2005-2023/2024/2025)
- "Regular season only" explicitly stated - avoids playoff variance

### Feature Engineering
- Clear rolling window rules (5 games, well-documented)
- Start-of-season handling specified (use available games)
- Bye week handling documented
- Good variety of feature categories (offense, defense, situational)
- Matchup differentials approach is smart (underdog - favorite)

### Model Architecture
- Siamese architecture is appropriate for team comparison task
- Attention mechanism adds interpretability
- Masking for variable sequence length is correct approach
- Week 1 exclusion makes sense (no prior data)
- Clear architectural diagrams

### Evaluation Framework
- Appropriate metrics (AUC-ROC, Brier Score, Calibration)
- Success criteria with justification
- Interpretability methods specified (SHAP for XGBoost, Captum for LSTM)
- CLV analysis shows sophisticated betting understanding
- Time-series CV respects temporal ordering

---

## Concerns & Gaps

### Critical (Must Fix Before Implementation)

#### 1. BRIER SCORE TARGET IS MATHEMATICALLY WRONG
- **Doc says:** "Brier < 0.24" is better than naive baseline
- **Reality:** Naive baseline for 35% upset rate = 0.35 × 0.65² + 0.65 × 0.35² = **0.2275**
- **Problem:** Brier < 0.24 is actually WORSE than always predicting 35%!
- **Fix:** Target should be Brier < 0.22 or lower

#### 2. POTENTIAL DATA LEAKAGE IN ATS FEATURES
- `ats_streak` (Against The Spread streak) is correlated with target
- Team that consistently beats spreads may be undervalued → upsets
- **MUST** ensure only PRE-GAME ATS record is used, not including current game
- Same concern for `recent_vs_season` - timing boundaries matter

#### 3. LSTM vs XGBOOST COMPARISON IS UNFAIR
- **XGBoost:** Gets ~50-80 hand-engineered features (rolling stats, trends, interactions)
- **LSTM:** Gets raw 5-game sequences (~15 features × 5 timesteps, unprocessed)
- **Problem:** Comparing apples to oranges! XGBoost has advantage of feature engineering
- **Fix Options:**
  1. Give LSTM the same engineered features
  2. Give XGBoost only raw stats
  3. Acknowledge limitation explicitly and frame research question differently

---

### High Priority

#### 4. DATA SOURCE MERGING STRATEGY UNDEFINED
- Three sources (nflverse, PFR, Kaggle) need joining
- **Not specified:**
  - How games will be matched (game ID formats differ)
  - What happens when sources disagree
  - Expected merge success rate
- **Risk:** This is where many data projects fail

#### 5. MISSING DATA SOURCES FOR KEY FEATURES
| Feature | Issue |
|---------|-------|
| Weather data | No source specified for outdoor games |
| Travel distance | No stadium coordinates source |
| Timezone changes | No calculation method |
| DVOA | "(if available)" - paywalled, needs fallback |
| `reverse_line_movement` | Requires public betting % data (typically paid/proprietary) |

#### 6. CLASS IMBALANCE NOT ADDRESSED
- 35% upset rate = imbalanced classes
- **Missing:**
  - Class weights discussion
  - SMOTE/undersampling consideration
  - Impact on calibration at confidence extremes

#### 7. QB/INJURY DATA MISSING
- QB status (starter vs backup) is HUGE upset predictor
- Key player injuries beyond QB
- Coach tenure/quality
- **Note:** Doc acknowledges spreads "incorporate" this, but there may still be signal

---

### Medium Priority

#### 8. LSTM HYPERPARAMETERS ASSUMED
- "2 layers, 64 hidden units, dropout 0.3" - Why these specific values?
- Should come from tuning, not assumption
- Risk: Suboptimal architecture

#### 9. SUCCESS CRITERIA MAY BE TOO AGGRESSIVE
- AUC > 0.55: Beating Vegas by 1-3 AUC points is VERY hard
- ROI > 5% after 10% vig: Most professional bettors achieve 2-3%
- Consider: What if results show AUC = 0.53? Is that "failure"?

#### 10. SINGLE TEST SPLIT RISK
- Final eval uses single 2024 val / 2025 test split
- Results may be specific to those years (what if unusual season?)
- Consider multi-year holdout or rolling origin evaluation

#### 11. TRAINING INFRASTRUCTURE UNSPECIFIED
- Batching strategy for LSTM?
- GPU requirements?
- Training time estimates?
- Early stopping criteria?

---

### Low Priority / Nice-to-Have

#### 12. PROFITABILITY BACKTEST GAPS
- No bet sizing strategy (Kelly criterion?)
- No line availability discussion (can you actually bet at those odds?)
- No bankroll management approach
- No accounting for line movement between prediction and bet

#### 13. ADDITIONAL ENHANCEMENTS
- Ensemble approach combining both models
- More sophisticated temporal CV for final evaluation
- Feature importance stability across years

---

## Technical Decisions in Doc
| Decision | Rationale Given | Assessment |
|----------|-----------------|------------|
| Binary classification (underdog win) | Clear target | ✅ Good - simple, interpretable |
| Spread threshold ≥ +3 | Defines "underdog" | ✅ Good - excludes pick'em games |
| Rolling 5-game window | Captures recent form | ✅ Good - standard approach |
| Siamese LSTM architecture | Shared encoder for teams | ✅ Good - appropriate for comparison |
| Time-series CV | Respects temporal order | ✅ Good - prevents future leakage |
| PyTorch for LSTM | Captum compatibility | ⚠️ More boilerplate than TensorFlow |
| Train 2005-2023 | ~4800 games | ⚠️ EPA data starts 2006, may need adjustment |

---

## Recommendations Summary

### Before Starting Implementation:
1. **Fix Brier score target** (typo/calculation error)
2. **Document data leakage safeguards** for ATS features
3. **Decide on fair comparison approach** for LSTM vs XGBoost

### During Implementation:
4. **Create data merge specification** with fallback strategies
5. **Source or drop features** you can't get data for
6. **Add class imbalance handling** to both models

### For Research Validity:
7. **Lower success criteria expectations** or frame as exploration
8. **Consider multi-year test evaluation** to reduce year-specific risk

---

## Resources Referenced
- nflverse/nfl_data_py: https://github.com/nflverse/nfl_data_py
- Kaggle betting data: https://www.kaggle.com/datasets/spreadspoke/nfl-scores-and-betting-data
- SHAP: https://github.com/slundberg/shap
- Captum: https://captum.ai/

---
*Review completed 2026-01-16*
