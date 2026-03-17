# Verified Numbers — Exhaustive Extraction from Code Outputs

> Generated 2026-03-15 from raw CSVs, saved results, and source code.
> Numbers pulled from data artifacts only — NOT from paper.md.

---

## 1. DATASET COUNTS

### Total Games (post week-1 exclusion)

| Split | All rows | Labeled (spread ≥ 3) | Unlabeled (spread < 3) |
|-------|----------|----------------------|------------------------|
| Train (2005–2022) | 4,351 | 3,495 | 856 |
| Test (2023–2025) | 768 | 558 | 210 |
| **Combined** | **5,119** | **4,053** | **1,066** |

Week-1 games excluded entirely (no prior rolling stats). Only weeks 2–18.

### Train Per-Season (labeled only, 2005–2022)

| Season | Total | Labeled | Unlabeled | Upsets | Upset Rate |
|--------|-------|---------|-----------|--------|------------|
| 2005 | 239 | 196 | 43 | 45 | 22.9592% |
| 2006 | 240 | 195 | 45 | 77 | 39.4872% |
| 2007 | 240 | 201 | 39 | 58 | 28.8557% |
| 2008 | 240 | 201 | 39 | 60 | 29.8507% |
| 2009 | 240 | 209 | 31 | 62 | 29.6651% |
| 2010 | 240 | 197 | 43 | 65 | 32.9949% |
| 2011 | 240 | 200 | 40 | 57 | 28.5000% |
| 2012 | 240 | 197 | 43 | 63 | 31.9797% |
| 2013 | 240 | 179 | 61 | 47 | 26.2570% |
| 2014 | 240 | 189 | 51 | 56 | 29.6296% |
| 2015 | 240 | 185 | 55 | 64 | 34.5946% |
| 2016 | 240 | 184 | 56 | 62 | 33.6957% |
| 2017 | 241 | 187 | 54 | 48 | 25.6684% |
| 2018 | 240 | 192 | 48 | 57 | 29.6875% |
| 2019 | 240 | 194 | 46 | 63 | 32.4742% |
| 2020 | 240 | 193 | 47 | 54 | 27.9793% |
| 2021 | 256 | 209 | 47 | 68 | 32.5359% |
| 2022 | 255 | 187 | 68 | 55 | 29.4118% |
| **TOTAL** | **4,351** | **3,495** | **856** | **1,061** | **30.3577%** |

(2021–2022 have 256/255 rows due to NFL 17-game expansion.)

### Test Per-Season (labeled only, 2023–2025)

| Season | Total | Labeled | Unlabeled | Upsets | Upset Rate |
|--------|-------|---------|-----------|--------|------------|
| 2023 | 256 | 185 | 71 | 55 | 29.7297% |
| 2024 | 256 | 192 | 64 | 45 | 23.4375% |
| 2025 | 256 | 181 | 75 | 59 | 32.5967% |
| **TOTAL** | **768** | **558** | **210** | **159** | **28.4946%** |

### Spread Buckets — CV (1,162 validation games)

| Bucket | Games | Upsets | Upset Rate |
|--------|-------|--------|------------|
| Small (3–6.5) | 700 | ~263 | 37.6% |
| Medium (7–13.5) | 402 | ~76 | 18.9% |
| Large (14+) | 60 | 6 | 10.0% |
| **Total** | **1,162** | **~345** | **29.69%** |

### Spread Buckets — Test (558 games)

| Bucket | Games | Upsets | Upset Rate |
|--------|-------|--------|------------|
| Small (3–6.5) | 360 | ~114 | 31.7% |
| Medium (7–13.5) | 175 | ~45 | 25.7% |
| Large (14+) | 23 | 0 | 0.0% |
| **Total** | **558** | **159** | **28.49%** |

### CV Fold Breakdown (expanding window)

| Fold | Train Seasons | Val Season | Val Games | Upsets | Upset Rate |
|------|---------------|------------|-----------|--------|------------|
| 1 | 2005–2016 | 2017 | 187 | 48 | 25.6684% |
| 2 | 2005–2017 | 2018 | 192 | 57 | 29.6875% |
| 3 | 2005–2018 | 2019 | 194 | 63 | 32.4742% |
| 4 | 2005–2019 | 2020 | 193 | 54 | 27.9793% |
| 5 | 2005–2020 | 2021 | 209 | 68 | 32.5359% |
| 6 | 2005–2021 | 2022 | 187 | 55 | 29.4118% |
| **Total** | | | **1,162** | **345** | **29.69%** |

---

## 2. FEATURE COUNTS

| Model | With Spread | Without Spread |
|-------|-------------|----------------|
| LR | 46 | 42 (drop 4 market) |
| XGB | 70 (46 + 24 lag) | 66 (drop 4 market) |
| LSTM seq | 14 × 8 timesteps × 2 teams | 14 × 8 × 2 (unchanged) |
| LSTM matchup | 10 | 8 (drop spread_magnitude, total_line) |

### LR / Base 46 Features (FEATURE_COLUMNS)

**Rolling Efficiency (10):** underdog_pass_epa_roll3, underdog_rush_epa_roll3, underdog_success_rate_roll3, underdog_cpoe_roll3, underdog_turnover_margin_roll3, favorite_pass_epa_roll3, favorite_rush_epa_roll3, favorite_success_rate_roll3, favorite_cpoe_roll3, favorite_turnover_margin_roll3

**Differentials (5):** pass_epa_diff, rush_epa_diff, success_rate_diff, cpoe_diff, turnover_margin_diff

**Volatility & Trend (12):** underdog_total_epa_std_roll3, favorite_total_epa_std_roll3, total_epa_std_diff, underdog_success_rate_std_roll3, favorite_success_rate_std_roll3, success_rate_std_diff, underdog_total_epa_trend, favorite_total_epa_trend, total_epa_trend_diff, underdog_success_rate_trend, favorite_success_rate_trend, success_rate_trend_diff

**Schedule Context (5):** underdog_rest_days, favorite_rest_days, rest_days_diff, short_week_game, divisional_game

**Market (4):** home_implied_points, away_implied_points, spread_magnitude, total_line

**Elo (3):** underdog_elo, favorite_elo, elo_diff

**Environment (5):** temperature, wind_speed, is_dome, temperature_missing, wind_speed_missing

**Game Context (2):** underdog_is_home, week_number

**Total: 10 + 5 + 12 + 5 + 4 + 3 + 5 + 2 = 46**

### XGB Additional 24 Lag Features

3 lags × 2 roles × 4 stats = 24:
```
{underdog,favorite}_last{1,2,3}_{total_epa,success_rate,turnover_margin,margin}
```

### LSTM 14 Sequence Features (per timestep per team)

total_epa, pass_epa, rush_epa, success_rate, cpoe, turnover_margin, points_scored, points_allowed, point_diff, opponent_elo, win, was_home, days_since_last_game, short_week

### LSTM 10 Matchup Context Features

spread_magnitude, total_line, underdog_elo, favorite_elo, underdog_is_home, underdog_rest_days, favorite_rest_days, week_number, divisional_game, is_dome

---

## 3. CROSS-VALIDATION RESULTS (6-fold, 1,162 games)

### Model Performance

| Model | AUC-ROC | Brier Score | Log Loss |
|-------|---------|-------------|----------|
| LR | 0.6497 | 0.1974 | 0.5807 |
| XGB | 0.6377 | 0.1991 | 0.5855 |
| LSTM | 0.6407 | 0.1985 | 0.5832 |

Ranking: LR > LSTM > XGB

### Bootstrap 95% CIs on Pairwise AUC Differences (10,000 resamples)

| Comparison | Mean | 95% CI | Significant? |
|------------|------|--------|--------------|
| LR − XGB | +0.012 | [−0.007, +0.031] | No |
| LR − LSTM | +0.009 | [−0.014, +0.033] | No |
| XGB − LSTM | −0.003 | [−0.030, +0.024] | No |

All three models statistically indistinguishable.

### CV Probability Correlation Matrix

|      | LR    | XGB   | LSTM  |
|------|-------|-------|-------|
| LR   | 1.000 | 0.874 | 0.784 |
| XGB  | 0.874 | 1.000 | 0.699 |
| LSTM | 0.784 | 0.699 | 1.000 |

### CV Raw Probability Ranges

| Model | Min | Max |
|-------|-----|-----|
| LR | 0.030 | 0.522 |
| XGB | 0.057 | 0.618 |
| LSTM | 0.000 | 0.724 |

---

## 4. TEST SET RESULTS (558 games, Platt-calibrated)

### Model Performance

| Model | AUC-ROC | Brier Score | Log Loss |
|-------|---------|-------------|----------|
| LR | 0.5622 | 0.2026 | 0.5942 |
| XGB | 0.5755 | 0.2013 | 0.5915 |
| LSTM | 0.5202 | 0.2072 | 0.6051 |

**Baseline Brier Score: 0.2038**

Ranking: XGB > LR > LSTM

### Generalization Gaps

| Model | CV AUC | Test AUC | Gap |
|-------|--------|----------|-----|
| LR | 0.6497 | 0.5622 | −0.0875 |
| XGB | 0.6377 | 0.5755 | −0.0622 |
| LSTM | 0.6407 | 0.5202 | −0.1205 |

### Per-Season AUC — ⚠️ DISCREPANCY ACROSS FILES

**From `report.md` (test set evaluation script):**

| Season | Games | Upset Rate | LR AUC | XGB AUC | LSTM AUC |
|--------|-------|------------|--------|---------|----------|
| 2023 | 185 | 29.7% | 0.512 | 0.521 | 0.469 |
| 2024 | 192 | 23.4% | 0.552 | 0.554 | 0.549 |
| 2025 | 181 | 32.6% | 0.617 | 0.639 | 0.556 |

**From `full_results_2026-03-09T1430.md` (A/B experiment script):**

| Season | Games | Upset Rate | LR AUC | XGB AUC | LSTM AUC |
|--------|-------|------------|--------|---------|----------|
| 2023 | 185 | 29.7% | 0.512 | 0.521 | 0.489 |
| 2024 | 192 | 23.4% | 0.552 | 0.554 | 0.586 |
| 2025 | 181 | 32.6% | 0.617 | 0.639 | 0.493 |

**Cause:** LSTM is stochastic (random init). These come from different training runs. LR and XGB are deterministic — their numbers match across files.

### Test Probability Correlation Matrix — ⚠️ DISCREPANCY ACROSS FILES

| Source | LR-XGB | LR-LSTM | XGB-LSTM |
|--------|--------|---------|----------|
| report.md | 0.878 | 0.429 | 0.408 |
| full_results | 0.878 | 0.311 | 0.273 |
| significance analysis | 0.878 | 0.426 | 0.381 |

**Cause:** Same stochastic LSTM issue. LR-XGB is consistent (0.878) across all files.

### Post-Calibration Probability Ranges (Test)

| Model | Min | Max |
|-------|-----|-----|
| LR | 0.194 | 0.435 |
| XGB | 0.216 | 0.498 |
| LSTM | 0.220 | 0.528 |

---

## 5. SPREAD ABLATION (CV)

### AUC With vs Without Spread

| Model | With Spread | Without Spread | Delta |
|-------|-------------|----------------|-------|
| LR | 0.6497 | 0.5707 | −0.079 |
| XGB | 0.6377 | 0.5662 | −0.072 |
| LSTM | 0.6407 | 0.5739 | −0.067 |

Without spread ranking: **LSTM > LR > XGB** (inverts from with-spread)

### Bootstrap 95% CIs on AUC Delta (no_spread − with_spread)

| Model | Mean | 95% CI | Significant? |
|-------|------|--------|--------------|
| LR | −0.079 | [−0.108, −0.050] | Yes |
| XGB | −0.072 | [−0.101, −0.042] | Yes |
| LSTM | −0.067 | [−0.098, −0.036] | Yes |

### Delta-of-Deltas (LSTM vs LR/XGB degradation difference)

| Comparison | Mean | 95% CI | Significant? |
|------------|------|--------|--------------|
| LSTM_delta − LR_delta | +0.012 | [−0.019, +0.044] | No |
| LSTM_delta − XGB_delta | +0.005 | [−0.030, +0.040] | No |

LSTM degrades less in point estimate but NOT statistically significant.

### No-Spread Correlation Matrix vs With-Spread

| Pair | With Spread | Without Spread | Change |
|------|-------------|----------------|--------|
| LR-XGB | 0.874 | 0.742 | −0.132 |
| LR-LSTM | 0.784 | 0.559 | −0.225 |
| XGB-LSTM | 0.699 | 0.419 | −0.280 |

### Agreement Rates

| Metric | With Spread | Without Spread |
|--------|-------------|----------------|
| All-three agreement | 74.7% | 55.3% |
| LR-XGB agreement | 87.0% | 77.6% |
| LR-LSTM agreement | 83.0% | 69.2% |
| XGB-LSTM agreement | 79.4% | 63.9% |
| LSTM exclusive % | 5.6% | 11.0% |

---

## 6. DISAGREEMENT TABLE (CV, with spread) — ⚠️ CORRECTED

### Recomputed from `predictions_with_spread.csv` at threshold = 0.2969

| Category | N | Pct | Upset Rate | Avg LR | Avg XGB | Avg LSTM |
|----------|---|-----|------------|--------|---------|----------|
| all_correct | 528 | 45.4% | 37.1% | 0.259 | 0.265 | 0.258 |
| all_wrong | 333 | 28.7% | 20.4% | 0.350 | 0.351 | 0.353 |
| only_lr | 31 | 2.7% | 22.6% | 0.292 | 0.320 | 0.320 |
| only_xgb | 48 | 4.1% | 22.9% | 0.317 | 0.286 | 0.344 |
| only_lstm | 72 | 6.2% | 20.8% | 0.326 | 0.345 | 0.236 |
| lr_xgb | 78 | 6.7% | 26.9% | 0.285 | 0.275 | 0.307 |
| lr_lstm | 45 | 3.9% | 37.8% | 0.290 | 0.308 | 0.281 |
| xgb_lstm | 27 | 2.3% | 37.0% | 0.311 | 0.292 | 0.257 |

### Numbers From `full_results` Report (for comparison)

| Category | N (report) | N (CSV recomputed) |
|----------|-----------|-------------------|
| all_correct | 518 | 528 |
| all_wrong | 340 | 333 |
| only_lr | 28 | 31 |
| only_xgb | 42 | 48 |
| only_lstm | **65** | **72** |
| lr_xgb | 79 | 78 |
| lr_lstm | 54 | 45 |
| xgb_lstm | 36 | 27 |

The report numbers sum to 1,162 and the recomputed numbers sum to 1,162. The difference is in how the threshold is applied — the report was generated by `DisagreementAnalyzer` (which uses `self.threshold = base_rate`), but these counts shifted between runs or code versions.

---

## 7. SPREAD-STRATIFIED DISAGREEMENT — RECOMPUTED FROM CSV

### Small Spread (3–6.5): 700 games, 37.6% upset rate

| Category | N | Pct | Upset Rate |
|----------|---|-----|------------|
| all_correct | 215 | 30.7% | 91.2% |
| all_wrong | 271 | 38.7% | 3.0% |
| only_lr | 27 | 3.9% | 11.1% |
| only_xgb | 39 | 5.6% | 23.1% |
| **only_lstm** | **60** | **8.6%** | **8.3%** |
| lr_xgb | 38 | 5.4% | 52.6% |
| lr_lstm | 29 | 4.1% | 44.8% |
| xgb_lstm | 21 | 3.0% | 42.9% |

LSTM exclusives: **60 total — 5 upsets caught, 55 non-upsets rejected**
all_wrong: 271 total — 8 missed upsets, 263 false alarms

### Medium Spread (7–13.5): 402 games, 18.9% upset rate

| Category | N | Pct | Upset Rate |
|----------|---|-----|------------|
| all_correct | 259 | 64.4% | 0.0% |
| all_wrong | 56 | 13.9% | 96.4% |
| only_lr | 4 | 1.0% | 100.0% |
| only_xgb | 9 | 2.2% | 22.2% |
| **only_lstm** | **12** | **3.0%** | **83.3%** |
| lr_xgb | 40 | 10.0% | 2.5% |
| lr_lstm | 16 | 4.0% | 25.0% |
| xgb_lstm | 6 | 1.5% | 16.7% |

LSTM exclusives: **12 total — 10 upsets caught, 2 non-upsets rejected**
all_wrong: 56 total — 54 missed upsets, 2 false alarms

### Large Spread (14+): 60 games, 10.0% upset rate

| Category | N | Pct |
|----------|---|-----|
| all_correct | 54 | 90.0% |
| all_wrong | 6 | 10.0% |
| All other categories | 0 | 0% |

LSTM exclusives: **0**
all_wrong: 6 missed upsets, 0 false alarms

### LSTM Role Inversion by Spread

| Bucket | LSTM Exclusives | Upsets Caught | Non-Upsets Rejected | Primary Role |
|--------|----------------|---------------|---------------------|-------------|
| Small (3–6.5) | 60 | 5 (8.3%) | 55 (91.7%) | False-alarm filter |
| Medium (7–13.5) | 12 | 10 (83.3%) | 2 (16.7%) | Upset detector |
| Large (14+) | 0 | 0 | 0 | — |
| **Total** | **72** | **15** | **57** | — |

### ⚠️ LSTM EXCLUSIVE DISCREPANCY: 65 vs 72

**The `full_results` report flat table says 65; the stratified table in the same report sums to 60+12+0=72. The CSV recomputation gives 72.**

Root cause: The report's flat table (65) is **not reproducible** from the saved `predictions_with_spread.csv` at any threshold. The stratified table (72) **is** reproducible. The flat table appears to be from a different LSTM training run or an earlier code version (before bughunt fixes changed LSTM predictions). The CSV is the authoritative data source.

**Corrected numbers: 72 LSTM exclusives total — 15 upsets, 57 non-upsets.**

(The report's claim of "12 upsets, 53 non-upsets" from the flat table is also inconsistent with the stratified breakdown of 5+10+0=15 upsets.)

---

## 8. TOP-K ANALYSIS (Test Set)

Base rate: 28.5%

**From `full_results` report:**

| K | LR | XGB | LSTM | Ensemble |
|---|-----|-----|------|----------|
| 10 | 5/10 (50%, 1.8×) | 6/10 (60%, 2.1×) | 3/10 (30%, 1.1×) | 6/10 (60%, 2.1×) |
| 20 | 8/20 (40%, 1.4×) | 9/20 (45%, 1.6×) | 7/20 (35%, 1.2×) | 10/20 (50%, 1.8×) |
| 30 | 12/30 (40%, 1.4×) | 12/30 (40%, 1.4×) | 8/30 (27%, 0.9×) | 13/30 (43%, 1.5×) |
| 50 | 19/50 (38%, 1.3×) | 22/50 (44%, 1.5×) | 16/50 (32%, 1.1×) | 18/50 (36%, 1.3×) |
| 75 | 24/75 (32%, 1.1×) | 28/75 (37%, 1.3×) | 24/75 (32%, 1.1×) | 25/75 (33%, 1.2×) |
| 100 | 31/100 (31%, 1.1×) | 33/100 (33%, 1.2×) | 33/100 (33%, 1.2×) | 32/100 (32%, 1.1×) |

**From `report.md` (different LSTM run):**

| K | LR | XGB | LSTM | Ensemble |
|---|-----|-----|------|----------|
| 10 | 5/10 (50%, 1.8×) | 6/10 (60%, 2.1×) | 4/10 (40%, 1.4×) | 5/10 (50%, 1.8×) |
| 20 | 8/20 (40%, 1.4×) | 9/20 (45%, 1.6×) | 7/20 (35%, 1.2×) | 9/20 (45%, 1.6×) |
| 50 | 19/50 (38%, 1.3×) | 22/50 (44%, 1.5×) | 14/50 (28%, 1.0×) | 17/50 (34%, 1.2×) |

---

## 9. LR COEFFICIENTS (Standardized, With Spread)

Source: `results/ab_experiment/lr_coefs_with_spread.json`

| Feature | Coefficient |
|---------|-------------|
| **spread_magnitude** | **−0.5388** |
| temperature | +0.0772 |
| favorite_turnover_margin_roll3 | −0.0694 |
| underdog_rush_epa_roll3 | +0.0647 |
| short_week_game | −0.0638 |
| underdog_total_epa_std_roll3 | +0.0597 |
| success_rate_diff | +0.0539 |
| favorite_rest_days | −0.0498 |
| underdog_success_rate_std_roll3 | −0.0294 |
| success_rate_std_diff | −0.0264 |
| favorite_cpoe_roll3 | +0.0249 |
| divisional_game | −0.0239 |
| wind_speed | +0.0228 |
| favorite_total_epa_std_roll3 | +0.0205 |
| underdog_cpoe_roll3 | +0.0143 |

22 of 46 features zeroed by L1 (including all Elo features, underdog_is_home, week_number).

**Confirmed: spread_magnitude = −0.5388** (paper rounds to −0.539).

### LR Coefficients (Without Spread)

Top features shift when spread removed:

| Feature | Coefficient |
|---------|-------------|
| elo_diff | +0.141 |
| underdog_is_home | +0.124 |
| success_rate_diff | +0.120 |
| underdog_rush_epa_roll3 | +0.118 |

16 of 42 features zeroed without spread (vs 22/46 with spread).

---

## 10. ENSEMBLE RESULTS

| Strategy | CV AUC | CV No-Spread AUC | Test AUC |
|----------|--------|-------------------|----------|
| Simple avg (3 models) | 0.6553 | 0.5809 | 0.5685 |
| LR+XGB only | 0.6488 | 0.5712 | 0.5707 |
| Soft veto (2× LSTM wt) | 0.6552 | 0.5794 | 0.5639 |
| Hard veto (LSTM override) | 0.6490 | 0.5636 | 0.5353 |
| XGB alone (reference) | 0.6377 | 0.5662 | 0.5755 |

**Paper claims: ensemble AUC 0.655 CV, 0.649 LR+XGB.**
- 0.655 CV → matches 0.6553 (rounded to 3 sig figs)
- 0.649 LR+XGB → matches 0.6488 (rounded to 3 sig figs)

Both confirmed.

---

## 11. HYPERPARAMETERS

### Logistic Regression

| Parameter | Value |
|-----------|-------|
| C | 0.1 |
| penalty | l1 |
| solver | saga |
| max_iter | 1000 |
| random_state | 42 |
| Preprocessing | StandardScaler (fit on train only) |

### XGBoost

| Parameter | Model Default | Actual (UnifiedTrainer) |
|-----------|---------------|------------------------|
| max_depth | 6 | **2** |
| learning_rate | 0.1 | **0.03** |
| n_estimators | 100 | **300** |
| min_child_weight | 1 | 1 |
| objective | binary:logistic | binary:logistic |
| eval_metric | logloss | logloss |
| random_state | 42 | 42 |
| subsample | 1.0 (lib default) | 1.0 |
| colsample_bytree | 1.0 (lib default) | 1.0 |
| scale_pos_weight | 1.0 (lib default) | 1.0 |

### LSTM (Siamese Architecture)

| Parameter | Model Default | Config (TUNED, actual) |
|-----------|---------------|----------------------|
| hidden_size | 64 | 64 |
| num_layers | 2 | **3** |
| dropout | 0.3 | **0.25** |
| sequence_features | 14 | 14 |
| matchup_features | 10 (8 no-spread) | 10 (8 no-spread) |
| sequence_length | 8 | 8 |

| Training Parameter | Value |
|-------------------|-------|
| optimizer | Adam |
| learning_rate | 0.001 |
| weight_decay | 0 (PyTorch default) |
| batch_size | 64 |
| max_epochs | 25 |
| patience (early stopping) | 6 |
| loss | BCELoss |

**Architecture:** Siamese LSTM encoder (shared weights) → attention over timesteps → concat [underdog_64 + favorite_64 + matchup_10] = 138 → Linear(138,64) → ReLU → Dropout(0.25) → Linear(64,32) → ReLU → Dropout(0.25) → Linear(32,1) → Sigmoid

---

## 12. BINOMIAL TEST — ⚠️ INVALIDATED BY CORRECTED COUNTS

### As reported (using flat table N=65):
- 53/65 non-upset rejections (81.5%) vs expected 70.3%
- `binomtest(53, 65, 0.7031, alternative='greater')` → **p = 0.029**
- 95% CI: [0.000, 0.282]

### Corrected (using CSV-recomputed N=72):
- 57/72 non-upset rejections (79.2%) vs expected 70.3%
- `binomtest(57, 72, 0.7031, alternative='greater')` → **p = 0.061**
- **Not significant at α = 0.05**

The p = 0.029 in the paper is based on inconsistent counts. With corrected counts from the authoritative CSV, the effect is directionally the same but does not reach significance.

---

## SPREAD-STRATIFIED AUC (CV, per model)

| Bucket | N | Base Rate | LR AUC | XGB AUC | LSTM AUC |
|--------|---|-----------|--------|---------|----------|
| Small (3–6.5) | 700 | 37.6% | 0.570 | 0.553 | 0.562 |
| Medium (7–13.5) | 402 | 18.9% | 0.551 | 0.536 | 0.519 |
| Large (14+) | 60 | 10.0% | 0.438 | 0.176 | 0.657 |

---

## CROSS-FILE DISCREPANCY SUMMARY

| Item | report.md | full_results | significance | CSV recomputed |
|------|-----------|-------------|-------------|----------------|
| LSTM test AUC (overall) | 0.5202 | 0.5202 | — | — |
| LSTM 2023 AUC | 0.469 | 0.489 | — | — |
| LSTM 2024 AUC | 0.549 | 0.586 | — | — |
| LSTM 2025 AUC | 0.556 | 0.493 | — | — |
| Test LR-LSTM corr | 0.429 | 0.311 | 0.426 | — |
| Test XGB-LSTM corr | 0.408 | 0.273 | 0.381 | — |
| LSTM exclusives (flat) | — | 65 | — | **72** |
| Top-K LSTM K=10 | 4/10 | 3/10 | — | — |

**Root cause for all discrepancies:** LSTM is stochastic (random weight initialization). `report.md` and `full_results` were generated from different training runs. The saved CSV predictions in `results/ab_experiment/` correspond to the `full_results` run. The test set predictions in `results/test/predictions.csv` correspond to the `report.md` run.

The 65-vs-72 LSTM exclusive discrepancy is a separate issue: the flat table in `full_results` is inconsistent with both the stratified table in the same file and the CSV. The CSV (72) is authoritative.
