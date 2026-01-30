# LSTM Implementation Issues

**Date:** 2026-01-22
**Status:** Pre-training validation complete

## Summary

During validation of the LSTM infrastructure against the design spec (`docs/plans/2026-01-16-nfl-upset-xgboost-lstm-design.md`), the following issues were identified and addressed.

---

## Resolved Issues

### 1. Architecture Was Not True Siamese

**Severity:** High
**Status:** ✅ Fixed

**Problem:**
The original implementation concatenated both teams' features along the feature dimension and passed them through a single LSTM pass. This is fundamentally different from the siamese architecture specified in the design doc.

**Design Spec (lines 154-161):**
```
Team A last 5 games → LSTM Encoder (shared weights) →
                                                      → Concat → Attention → Dense → P(upset)
Team B last 5 games → LSTM Encoder (shared weights) →
                                                      ↑
                                            Matchup Features (spread, context)
```

**Original Implementation:**
```
Concat(Team A features, Team B features) → Single LSTM → Attention → Dense → P(upset)
```

**Fix Applied:**
- Rewrote `SiameseUpsetLSTM` in `src/models/lstm_model.py` to process each team's sequence independently through a shared encoder
- Updated `SiameseLSTMDataset` to hold separate sequence arrays for each team
- Updated `sequence_builder.py` to produce separate `underdog_sequences` and `favorite_sequences`
- Updated `lstm_trainer.py` to handle the new data format

**Files Changed:**
- `src/models/lstm_model.py`
- `src/models/sequence_builder.py`
- `src/models/lstm_trainer.py`
- `tests/models/test_lstm_model.py`
- `tests/models/test_sequence_builder.py`
- `tests/models/test_lstm_trainer.py`

---

## Open Issues

### 2. Missing Sequence Features

**Severity:** Medium
**Status:** ⚠️ Open - Data limitation

**Problem:**
The design spec calls for ~15 features per game timestep in the LSTM sequences. The current implementation only has 4 features available.

**Design Spec (lines 172-178):**
> Input Features Per Game Timestep (~15 features per game):
> - Points scored, points allowed
> - Total yards, yards allowed
> - EPA/play (offense and defense)
> - Turnover margin
> - 3rd down conversion rate
> - Win/loss indicator

**Current Implementation:**
| Feature | Available | Source |
|---------|-----------|--------|
| Points scored | ✅ | Game scores |
| Points allowed | ✅ | Game scores |
| Point differential | ✅ | Derived |
| Win indicator | ✅ | Derived |
| Total yards | ❌ | Requires PBP data |
| Yards allowed | ❌ | Requires PBP data |
| EPA/play (offense) | ❌ | Requires PBP data |
| EPA/play (defense) | ❌ | Requires PBP data |
| Turnover margin | ❌ | Requires PBP data |
| 3rd down conversion % | ❌ | Requires PBP data |

**Impact:**
- LSTM has less signal to learn from per timestep
- Model may underperform compared to what's achievable with full feature set
- Current AUC ~0.577 may improve with additional features

**Potential Remediation:**
1. Integrate play-by-play data from `nfl_loader.load_pbp_data()` into sequence building
2. Aggregate per-game stats (EPA, yards, turnovers) for each team
3. Store aggregated stats alongside game scores in team history
4. Update `SEQUENCE_FEATURES` list and `_build_team_game_history()` function

**Estimated Effort:** Medium (2-4 hours)

**Files to Modify:**
- `src/models/sequence_builder.py` - Add PBP feature extraction
- `src/data/nfl_loader.py` - May need additional aggregation functions

---

### 3. Sequence Feature Normalization Applied Per-Split

**Severity:** Low
**Status:** ⚠️ Open - Minor data leakage risk

**Problem:**
Currently, sequence normalization statistics (mean, std) are computed on the entire dataset passed to `build_siamese_sequences()`. During cross-validation, this means validation data influences the normalization statistics.

**Current Behavior:**
```python
# In build_siamese_sequences():
all_sequences = np.concatenate([underdog_sequences, favorite_sequences], axis=0)
all_masks = np.concatenate([underdog_masks, favorite_masks], axis=0)
_, stats = _normalize_sequences(all_sequences, all_masks)
```

**Correct Behavior:**
Normalization stats should be computed only on training data, then applied to validation/test data.

**Impact:**
- Minor data leakage during CV (validation data contributes to normalization)
- Could slightly inflate CV metrics
- Does not affect final test evaluation if stats are computed on train only

**Potential Remediation:**
1. Add `fit_transform()` and `transform()` methods to sequence builder
2. Store normalization stats from training data
3. Apply stored stats during prediction

**Estimated Effort:** Low (1-2 hours)

---

### 4. No Learning Rate Scheduling

**Severity:** Low
**Status:** ⚠️ Open - Potential improvement

**Problem:**
The LSTM trainer uses a fixed learning rate throughout training. Learning rate scheduling (e.g., reduce on plateau, cosine annealing) often improves convergence and final performance.

**Current Implementation:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
```

**Potential Remediation:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
# In training loop:
scheduler.step(val_loss)
```

**Estimated Effort:** Low (30 min)

---

### 5. No Gradient Clipping

**Severity:** Low
**Status:** ⚠️ Open - Stability improvement

**Problem:**
LSTMs can suffer from exploding gradients, especially with longer sequences or deeper networks. No gradient clipping is currently implemented.

**Potential Remediation:**
```python
# In _train_epoch():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Estimated Effort:** Low (15 min)

---

### 6. Feature Pipeline Test Failures

**Severity:** Low
**Status:** ⚠️ Open - Test infrastructure issue

**Problem:**
10 tests in `tests/features/test_pipeline.py` fail due to test fixture issues, not actual code bugs. The fixtures create data that gets entirely filtered out when `exclude_week_1=True`.

**Evidence:**
- Integration tests pass
- Real data processing works correctly
- Train/test CSVs generated successfully

**Potential Remediation:**
- Update test fixtures to include games from week 2+
- Or adjust test assertions to handle empty DataFrames

**Estimated Effort:** Low (1 hour)

---

## Architecture Verification

### Current State (Post-Fix)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Siamese LSTM Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Underdog History        Favorite History                        │
│  (n, 5, 4)              (n, 5, 4)                                │
│       │                      │                                   │
│       ▼                      ▼                                   │
│  ┌─────────────────────────────────────┐                        │
│  │      Shared LSTM Encoder            │                        │
│  │   (hidden=64, layers=2, dropout=0.3)│                        │
│  └─────────────────────────────────────┘                        │
│       │                      │                                   │
│       ▼                      ▼                                   │
│  ┌─────────┐            ┌─────────┐                             │
│  │Attention│            │Attention│                             │
│  └─────────┘            └─────────┘                             │
│       │                      │                                   │
│       ▼                      ▼                                   │
│  Encoding A              Encoding B                              │
│  (n, 64)                 (n, 64)                                 │
│       │                      │                                   │
│       └──────────┬───────────┘                                   │
│                  │                                               │
│                  ▼                                               │
│            ┌───────────┐                                         │
│            │  Concat   │◄──── Matchup Features (n, 10)          │
│            └───────────┘                                         │
│                  │                                               │
│                  ▼                                               │
│         ┌───────────────┐                                        │
│         │ Dense: 138→64 │                                        │
│         │ Dense: 64→32  │                                        │
│         │ Dense: 32→1   │                                        │
│         │ Sigmoid       │                                        │
│         └───────────────┘                                        │
│                  │                                               │
│                  ▼                                               │
│            P(upset)                                              │
│            [0.0 - 1.0]                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Sequence Features (4 per team)
1. `points_scored` - Points scored in that game
2. `points_allowed` - Points allowed in that game
3. `point_diff` - Point differential (scored - allowed)
4. `win` - Binary win indicator (1.0 or 0.0)

### Matchup Features (10)
1. `spread_magnitude` - Absolute spread value
2. `home_indicator` - Is underdog the home team?
3. `divisional_game` - Divisional matchup flag
4. `rest_advantage` - Rest days differential
5. `week_number` - Week / 18 (normalized)
6. `primetime_game` - Primetime indicator
7. `is_dome` - Indoor game flag
8. `cold_weather` - Temperature < 40°F
9. `windy_game` - Wind > 15 mph
10. `over_under_normalized` - (O/U - 45) / 10

---

## Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| `test_lstm_model.py` | 8 | ✅ All passing |
| `test_sequence_builder.py` | 14 | ✅ All passing |
| `test_lstm_trainer.py` | 11 | ✅ All passing |
| **Total LSTM Tests** | **33** | **✅ All passing** |

---

## Validation Metrics

Quick 3-fold CV on 2017-2022 data (5 epochs, reduced model):

| Metric | Mean | Std |
|--------|------|-----|
| AUC-ROC | 0.577 | ±0.049 |
| Brier Score | 0.208 | ±0.007 |
| Log Loss | 0.606 | ±0.017 |

Note: These are preliminary results with minimal training. Full training with hyperparameter tuning expected to improve.

---

## Recommendations

### Before Training
1. **Optional:** Add missing sequence features from PBP data (Issue #2)
2. **Optional:** Fix normalization to prevent minor leakage (Issue #3)

### During Training
1. Consider adding learning rate scheduling (Issue #4)
2. Consider adding gradient clipping (Issue #5)

### Post-Training
1. Fix feature pipeline test fixtures (Issue #6)
