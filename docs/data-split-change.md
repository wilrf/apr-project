# Data Split Strategy Change

**Date:** 2026-01-18  
**Status:** Approved for Implementation

---

## Summary

Changed the train/test split to provide **3 complete test seasons** instead of 2 incomplete ones, improving statistical robustness and presentation value for AP Research.

---

## Original Plan

### Data Split:
- **Training:** 2005-2023 (19 seasons, ~4,900 games)
- **Validation:** 2024 season (1 season, ~272 games)
- **Test:** 2025 season (1 season, ~272 games)

### Issues Identified:
1. ❌ **Only 1-2 test seasons** - Limited statistical conclusions
2. ❌ **2025 was incomplete** at project start (playoffs ongoing)
3. ❌ **Weak for AP Research** - Results from single season could be luck
4. ❌ **No year-over-year analysis** - Can't show consistency

### Design Document Reference:
From `docs/plans/2026-01-16-nfl-upset-xgboost-lstm-design.md`:
> - **Training**: 2005-2023 regular seasons (~19 seasons, ~4,800 games)
> - **Validation**: 2024 regular season
> - **Test**: 2025 regular season (current season)

---

## New Plan

### Data Split:
- **Training:** 2005-2022 (18 seasons, ~4,608 games)
- **Test:** 2023, 2024, 2025 (3 seasons, ~816 games)

### Key Changes:
✅ **3 complete test seasons** instead of 1-2  
✅ **All data currently available** (regular season complete)  
✅ **Better statistical power** for conclusions  
✅ **Year-over-year analysis** possible

---

## Rationale

### 1. Data Availability
**Verified:** All regular season games for 2023, 2024, and 2025 are complete.

```
2023: 272 games (100% coverage)
2024: 272 games (100% coverage)  
2025: 272 games (100% coverage - Week 18 finished Jan 2026)
```

**Note:** We only use regular season games. Playoffs excluded by design.

### 2. Statistical Robustness

| Metric | Original Plan | New Plan |
|--------|--------------|----------|
| Test games | ~272-544 | ~816 |
| Test seasons | 1-2 | 3 |
| Statistical power | Weak | Strong |
| Can calculate SD? | No | Yes |
| Year trends? | No | Yes |

**With 3 test years:**
- Can calculate confidence intervals
- Can show consistency vs flukes
- Can identify which situations model excels/struggles
- Stronger claims in research paper

### 3. AP Research Presentation Value

**Before (1-2 test seasons):**
> "Our model achieved 0.58 AUC on the 2024 season."

**After (3 test seasons):**
> "Our model consistently outperformed Vegas across three consecutive seasons (2023-2025), achieving average AUC of 0.58±0.03, demonstrating robust generalization to contemporary NFL dynamics."

### 4. Training Data Trade-off

**Lost Training Data:**
- Reduced from 19 seasons → 18 seasons
- ~256 fewer games (-5.6%)

**Impact Assessment:**
- ✅ 18 years is still substantial (4,608 games)
- ✅ Sufficient for both XGBoost and LSTM
- ✅ Minimal performance impact expected
- ✅ Benefit of 3 test seasons far outweighs cost

### 5. Modern NFL Relevance

Testing on 2023-2025 shows the model works on the **most recent NFL**:
- Current offensive/defensive strategies
- Latest rule changes and enforcement
- Modern QB play styles
- Recent coaching philosophies
- Contemporary betting markets

**This strengthens research claims about real-world applicability.**

---

## Cross-Validation Strategy

### Hyperparameter Tuning (Time-Series CV on Training Data):

```
Fold 1: Train 2005-2016 → Validate 2017
Fold 2: Train 2005-2017 → Validate 2018
Fold 3: Train 2005-2018 → Validate 2019
Fold 4: Train 2005-2019 → Validate 2020
Fold 5: Train 2005-2020 → Validate 2021
Fold 6: Train 2005-2021 → Validate 2022
```

**6-fold CV** provides robust hyperparameter selection.

### Final Model Training:

Train on **all 2005-2022 data** with best hyperparameters.

### Testing:

Evaluate separately on **2023, 2024, 2025** - report individual + aggregate results.

---

## Results Presentation Framework

### For AP Research Paper/Presentation:

**Table 1: Model Performance Across Test Seasons**

| Model | 2023 | 2024 | 2025 | **Mean±SD** | vs Vegas |
|-------|------|------|------|-------------|----------|
| XGBoost AUC | TBD | TBD | TBD | TBD±TBD | +TBD |
| LSTM AUC | TBD | TBD | TBD | TBD±TBD | +TBD |
| XGBoost ROI | TBD | TBD | TBD | TBD±TBD | +TBD |
| LSTM ROI | TBD | TBD | TBD | TBD±TBD | +TBD |

**Table 2: Model Agreement Analysis**

| Comparison | 2023 | 2024 | 2025 | **Avg** |
|------------|------|------|------|---------|
| Agreement Rate | TBD | TBD | TBD | TBD |
| Disagreement Cases | TBD | TBD | TBD | TBD |

**Key Discussion Points:**
1. Consistency across multiple seasons indicates robustness
2. Year-specific variations reveal situational strengths/weaknesses
3. Comparison to Vegas baseline across all test years
4. Temporal trends (improving/degrading over time?)

---

## Implementation Checklist

- [x] Update data loading scripts to use 2005-2022 for training
- [x] Configure test set to include 2023, 2024, 2025
- [x] Update cross-validation to use 6 folds (2017-2022)
- [ ] Modify evaluation scripts to report per-year results
- [ ] Create visualization for year-over-year comparison
- [ ] Update notebooks with new data ranges
- [ ] Document findings for each test season separately

---

## References

- Original design: `docs/plans/2026-01-16-nfl-upset-xgboost-lstm-design.md`
- Implementation plan: `docs/plans/2026-01-16-nfl-upset-implementation-plan.md`
- Data verification: Confirmed 100% regular season coverage 2000-2025

---

## Approval

**Decision:** Proceed with new data split (2005-2022 train, 2023-2025 test)

**Rationale:** Superior statistical robustness and presentation value for AP Research with minimal trade-offs.

**Date Finalized:** 2026-01-18
