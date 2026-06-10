# Architecture & Analysis

A comprehensive reference for how the NFL upset prediction system works end-to-end: data pipeline, feature design, model choices, evaluation infrastructure, and current results.

*Last updated: 2026-03-09*

---

## 1. Research Question

**How do the agreement and disagreement patterns of three structurally distinct ML architectures reveal the mechanisms and predictability boundaries of NFL upsets?**

This is not a model competition. We deliberately choose architectures that process data differently, then use *where they agree and disagree* as a diagnostic tool. The core contribution is a taxonomy of upset types, not a leaderboard.

---

## 2. Data Pipeline

### Sources

| Source | Module | What it provides |
|--------|--------|-----------------|
| nfl_data_py schedules | `src/data/nfl_loader.py` | Game schedule: teams, scores, dates, rest days, division flags |
| Kaggle spreadspoke | `src/data/betting_loader.py` | Betting lines: spread, over/under, favorite ID |
| nfl_data_py play-by-play | `src/data/epa_loader.py` | Advanced stats: pass/rush EPA, success rate, CPOE, turnover margin |

### Merge Flow

```
NFL Schedules ──┐
                 ├── merge_nfl_betting_data() ──┐
Betting Data ───┘     (join on season/week/     │
                       home/away teams)          ├── merge_epa_data() ── merged DataFrame
                                                 │
PBP → EPA ──────────────────────────────────────┘
     load_game_advanced_stats()                    (join on game_id)
```

**Key details:**
- NFL schedules are the spine (left join). Betting data joins on `(season, week, home_team, away_team)`.
- Team abbreviations are normalized for relocations: STL→LA, SD→LAC, OAK→LV, etc. (`betting_loader.TEAM_ABBR_MAP`).
- EPA extraction reads only the 16 columns needed from multi-GB PBP parquet files.
- The merge produces one row per game with scores, betting lines, and 10 advanced stats (4 EPA + success rate + CPOE + turnover margin per side).

### Elo Ratings

`src/data/elo.py` computes simple pre-game Elo ratings:
- K-factor = 20, home advantage = 50, base rating = 1500
- Ratings carry across seasons (no reset)
- Computed inline during feature engineering, joined on `game_id`

### Feature Generation Script

`src/data/generate_features.py` runs the full pipeline:

1. Load schedules for 2005-2025
2. Load and merge betting data
3. Load and merge EPA from PBP
4. Run `FeatureEngineeringPipeline.transform()`
5. Split by season: train (2005-2022), test (2023-2025)
6. Run 8 strict validation checks on each split
7. Save `data/features/train.csv` and `data/features/test.csv`

**Validation checks:**
1. All labeled games have spread >= 3
2. No impossible upsets (upset=1 without underdog or winner)
3. Upset rate in plausible range (15-40%)
4. All 70 feature columns present and finite
5. No NaN/inf in features for labeled games
6. No week-1 games (no prior stats available)
7. No duplicate game_ids
8. Season range check

---

## 3. Target Definition

**An upset occurs when the underdog wins a game with spread >= 3.**

```python
# Games with spread < 3 get upset = NaN (excluded from training/evaluation)
# Games with spread >= 3 get upset = 0 (favorite won) or 1 (underdog won)
```

**Why spread >= 3?** Games with smaller spreads are near toss-ups by definition. Including them as `upset=0` would flood the dataset with ~20% fake negatives where upsets are structurally impossible. This was a critical bug fix in March 2026 that changed:
- Train: 4,341 → 3,495 labeled games
- Test: 751 → 558 labeled games
- Upset rate: ~21% → ~30%

Sub-3 games remain in the DataFrame for rolling stat computation but are excluded via `upset.notna()` during training and evaluation.

**Upset tiers** (for analysis, not modeling):
- Tier 1: spread 3-6.5 (moderate underdogs)
- Tier 2: spread 7-13.5 (significant underdogs)
- Tier 3: spread 14+ (massive underdogs)

---

## 4. Feature Engineering: Multi-Representation Design

The central design principle: **each model gets the same underlying data in a different representation**, optimized for that model's structural strengths. This means disagreement between models reflects genuine architectural differences in how they process information, not differences in input data.

### 4.1 Shared Base: Team Rolling Stats

For each game, the pipeline (`src/features/pipeline.py`):

1. Converts game rows to one row per team per game (`_aggregate_team_games`)
2. Computes per-team rolling statistics over a 3-game window (`_calculate_team_rollups`):
   - Rolling means: pass EPA, rush EPA, success rate, CPOE, turnover margin, total EPA
   - Rolling std: total EPA volatility, success rate volatility
   - Trend: current value minus rolling mean (is the team trending up or down?)
   - Per-game lags: individual stats from last 1, 2, 3 games
3. Maps team stats to game rows using underdog/favorite roles

**Why rolling-3?** Short windows capture recent form while remaining robust with NFL's 17-game seasons. Longer windows dilute the recency signal that matters for upset detection.

**Why cross-season rolling?** NFL team composition changes between seasons, but the window is short enough (3 games) that the late-season carry-over is minimal, and it avoids cold-start issues at the beginning of each season.

### 4.2 LR Representation: "The Summary" (46 features)

Logistic regression gets pre-computed summary statistics — everything distilled to a single number per aspect.

| Group | Count | Features |
|-------|-------|----------|
| Rolling efficiency | 10 | pass/rush EPA, success rate, CPOE, turnover margin (x2 roles) |
| Differentials | 5 | underdog minus favorite for each efficiency metric |
| Volatility & trend | 12 | Std and trend for EPA and success rate (x2 roles + diffs) |
| Schedule context | 5 | Rest days (x2 roles + diff), short week flag, divisional game |
| Market | 4 | Home/away implied points, spread magnitude, total line |
| Elo | 3 | Underdog Elo, favorite Elo, Elo difference |
| Environment | 5 | Temperature, wind speed, dome flag, 2 missingness indicators |
| Game context | 2 | Underdog is home, week number |

**Design rationale:** LR can only learn linear combinations. These 46 features are already engineered to capture the relationships LR needs (differentials, trend slopes, etc.). LR's role is baseline — it captures what a simple linear model can extract from the same information.

### 4.3 XGBoost Representation: "The Details" (70 features)

XGBoost gets everything LR gets **plus** 24 per-game lag features:

```
{role}_last{lag}_{stat}
  role: underdog, favorite
  lag:  1, 2, 3 (most recent, two games ago, three games ago)
  stat: total_epa, success_rate, turnover_margin, margin
```

**Design rationale:** XGBoost can discover non-linear interactions and splits that rolling averages destroy. For example: "the underdog's 3-game rolling EPA is good, BUT their most recent game was terrible" — the average hides this, but XGB with per-game lags can learn it. The 24 extra features give XGB access to the game-by-game texture that LR's summary statistics erase.

**XGBoost hyperparameters:** `max_depth=2`, `learning_rate=0.03`, `n_estimators=300`. The shallow depth is deliberate — with 70 features, deeper trees risk overfitting to noise in a 3,495-game dataset.

### 4.4 LSTM Representation: "The Movie" (14 seq x 8 timesteps + 10 matchup)

The LSTM processes raw game-by-game sequences — no pre-computed rolling stats.

**Sequence features** (14 per timestep, `src/models/sequence_builder.py`):

| Feature | What it captures |
|---------|-----------------|
| total_epa | Overall offensive efficiency |
| pass_epa | Passing game quality |
| rush_epa | Running game quality |
| success_rate | Play-by-play efficiency |
| cpoe | QB performance vs. expectation |
| turnover_margin | Ball security |
| points_scored | Scoring output |
| points_allowed | Defensive performance |
| point_diff | Net outcome (margin) |
| opponent_elo | Quality of competition |
| win | Binary outcome |
| was_home | Venue context |
| days_since_last_game | Schedule/rest |
| short_week | Schedule flag |

**Matchup context** (10 features — things the sequence *can't* know):

| Feature | Rationale |
|---------|-----------|
| spread_magnitude | Market's assessment of this specific matchup |
| total_line | Expected scoring environment |
| underdog_elo, favorite_elo | Long-range team strength |
| underdog_is_home | Venue for this game |
| underdog_rest_days, favorite_rest_days | Schedule for this game |
| week_number | Season timing |
| divisional_game | Rivalry factor |
| is_dome | Weather/venue |

**Design rationale:** The LSTM's strength is temporal pattern recognition. By giving it raw game-by-game data instead of pre-computed averages, it can learn patterns that summaries destroy: win streaks, momentum shifts, performance trajectories, and the ordering/shape of recent results. The matchup context is deliberately minimal (10 features instead of the full 46) to force the LSTM to rely on its sequence encoder rather than shortcutting through pre-computed rolling stats.

**Architecture:** Siamese LSTM with attention (`src/models/lstm_model.py`):
```
Underdog's last 8 games → Shared LSTM Encoder → Attention → Encoding A
Favorite's last 8 games → Shared LSTM Encoder → Attention → Encoding B
Concat(Encoding A, Encoding B, Matchup) → Dense(64→32→1) → P(upset)
```

The **siamese** design means both teams are processed by the same encoder — the model learns general team performance patterns, not team-specific ones. The **attention** mechanism lets the model weight recent games differently (e.g., weighting the last game more heavily when detecting momentum).

**LSTM hyperparameters** (`src/models/lstm_config.py`):
- hidden_size=64, num_layers=3, dropout=0.25
- learning_rate=0.001, batch_size=64, epochs=25, patience=6

**Sequence normalization:** Statistics are computed from training data only and applied to validation/test data, preventing data leakage. Both teams' sequences contribute to shared statistics, then normalization is applied separately to each team's sequences. Padded positions (teams with fewer than 8 prior games) are zeroed out after normalization.

### 4.5 No-Spread Variants

Each representation has a no-spread variant for the A/B spread ablation experiment:
- LR: 42 features (drops 4 market features)
- XGB: 66 features (drops 4 market features)
- LSTM matchup: 8 features (drops spread_magnitude, total_line)
- LSTM sequence: unchanged (14 features — sequence data has no market info)

---

## 5. Model Architectures & Rationale

### Why these three models?

The three models are chosen to be **structurally complementary**, not competitive:

| Model | Architecture | What it captures | Blind spots |
|-------|-------------|-----------------|-------------|
| **LR** | Linear, L1-regularized | Systematic linear mispricing in the spread | Can't see interactions, sequences, or non-linearities |
| **XGBoost** | Gradient-boosted trees | Non-linear feature interactions and game-by-game patterns | Can't see temporal ordering or momentum |
| **LSTM** | Siamese recurrent network | Temporal dynamics, momentum, trajectory shape | Limited to sequence data; small matchup context |

**The insight:** When all three models agree on an upset, it's a strong signal. When only one model sees it, the mechanism is specific to that model's structural strength. When no model sees it, the upset was driven by information outside our data (injuries, motivation, weather surprises) or genuine randomness.

### Logistic Regression

`src/models/logistic_model.py` — `UpsetLogisticRegression`

- L1 regularization (LASSO) with C=0.1 — strong regularization that zeros out irrelevant features
- StandardScaler applied before fitting
- Provides interpretable coefficients on standardized scale
- Role: baseline and "spread mispricing detector"

### XGBoost

`src/models/xgboost_model.py` — `UpsetXGBoost`

- `max_depth=2` — shallow trees to prevent overfitting
- `learning_rate=0.03` with 300 estimators — slow learning
- Built-in feature importance via gain
- Role: interaction detector (can find "low spread + bad recent game + road team" patterns)

### Siamese LSTM

`src/models/lstm_model.py` — `SiameseUpsetLSTM`

- 3-layer LSTM with hidden_size=64
- Attention mechanism over timesteps
- Weight-shared encoder (true siamese — same weights for both teams)
- BCELoss with Adam optimizer
- Role: temporal pattern detector (momentum, fatigue, hot/cold streaks)

---

## 6. Training & Evaluation Infrastructure

### Cross-Validation

`src/models/cv_splitter.py` — `TimeSeriesCVSplitter`

Expanding-window time series CV with 6 folds:
```
Fold 1: Train 2005-2016, Val 2017
Fold 2: Train 2005-2017, Val 2018
Fold 3: Train 2005-2018, Val 2019
Fold 4: Train 2005-2019, Val 2020
Fold 5: Train 2005-2020, Val 2021
Fold 6: Train 2005-2021, Val 2022
```

**Why expanding window?** Standard k-fold would leak future data into training. Expanding window respects temporal ordering and mirrors real-world deployment (you always train on all available history).

### Unified Trainer

`src/models/unified_trainer.py` — `UnifiedTrainer`

Trains all three models on identical CV folds:
- LR gets base 46 features
- XGBoost gets expanded 70 features (base + per-game lags)
- LSTM gets separate sequences built from the same fold data
- LSTM normalization stats are computed from training fold only

This ensures disagreement analysis reflects genuine model differences, not data differences.

**Data structures:**
- `GamePrediction` — per-game prediction record with all three model probabilities
- `FoldResult` — all predictions and metrics from one CV fold
- `UnifiedCVResults` — aggregated results across all folds with mean/std metrics

### Test Set Evaluation

`src/models/evaluate_test_set.py`

1. Trains final models on full 2005-2022 training set
2. Generates predictions on held-out 2023-2025 test set
3. Applies post-hoc Platt calibration (fit on 2021-2022 held-out predictions)
4. Runs disagreement analysis, top-K analysis, probability bucket analysis
5. Saves predictions CSV and markdown report

### Post-Hoc Calibration

`src/evaluation/calibration.py`

- **Platt scaling:** Fits a logistic sigmoid to map raw probabilities → calibrated probabilities
- **Isotonic regression:** Non-parametric, monotonic calibrator (available but Platt is default)
- Calibration set: Train models on 2005-2020, predict on 2021-2022, fit calibrators on those predictions
- Applied to test predictions from models trained on full 2005-2022

**Why calibrate?** Different model architectures produce probability distributions with different shapes. Calibration makes probabilities comparable across models, which is essential for disagreement analysis.

### A/B Spread Ablation

`src/models/run_ab_experiment.py`

Runs the same CV pipeline twice:
- **Experiment A:** Full features (LR=46, XGB=70, LSTM=10 matchup)
- **Experiment B:** No-spread features (LR=42, XGB=66, LSTM=8 matchup)

This isolates the spread's contribution. If a model collapses without spread, it was primarily echoing the market. If it retains predictive power, it captures genuine team-quality signal independent of the market's assessment.

**Quick mode** (`--quick`): LR + XGB only (~1 minute). Full mode includes LSTM (~15 minutes).

---

## 7. Disagreement Analysis Framework

`src/evaluation/disagreement.py` — `DisagreementAnalyzer`

### Categories

Each game is categorized by which models correctly predicted the outcome:

| Category | Models correct | Interpretation |
|----------|---------------|----------------|
| `all_correct` | LR + XGB + LSTM | Obvious upset — clear signal all architectures detect |
| `all_wrong` | None | Genuine randomness or hidden information |
| `only_lr` | LR only | Linear spread mispricing |
| `only_xgb` | XGB only | Non-linear interaction pattern |
| `only_lstm` | LSTM only | Temporal/momentum signal |
| `lr_xgb` | LR + XGB | Static models agree — non-temporal signal |
| `lr_lstm` | LR + LSTM | Linear + temporal agree |
| `xgb_lstm` | XGB + LSTM | Non-linear + temporal agree |

### Threshold

The analyzer uses the **base upset rate** as its threshold (not 0.5). A model "predicts upset" when it assigns probability higher than the average upset rate. This is the principled choice for minority-class problems — with a ~30% base rate, using 0.5 would mean no model ever predicts an upset.

### Outputs

- `get_category_stats()` — count, percentage, upset rate, average probabilities per category
- `get_agreement_matrix()` — pairwise agreement rates (LR-XGB, LR-LSTM, XGB-LSTM, all three)
- `get_correlation_matrix()` — Pearson correlations of probability outputs
- `get_exclusive_insights()` — detailed analysis of ONLY_LR, ONLY_XGB, ONLY_LSTM categories

---

## 8. Current Results

*Generated 2026-03-09 on the multi-representation architecture.*

### Data

- **Training:** 2005-2022, 3,495 upset candidates
- **Testing:** 2023-2025, 558 upset candidates
- **CV:** 6-fold expanding-window (val seasons 2017-2022), 1,162 predictions
- **Upset rate:** ~30% in both train and test

### 8.1 CV Performance (6-fold, 1,162 games)

| Model | AUC-ROC | Brier Score | Log Loss |
|-------|---------|-------------|----------|
| LR | 0.6497 | 0.1974 | 0.5807 |
| XGB | 0.6377 | 0.1991 | 0.5855 |
| LSTM | 0.6407 | 0.1985 | 0.5832 |

**CV ranking:** LR > LSTM > XGB. All three models are competitive — LSTM is not the weak link.

CV probability correlations:

|      | LR    | XGB   | LSTM  |
|------|-------|-------|-------|
| LR   | 1.000 | 0.874 | 0.784 |
| XGB  | 0.874 | 1.000 | 0.699 |
| LSTM | 0.784 | 0.699 | 1.000 |

### 8.2 Test Set Performance (558 games, calibrated)

| Model | AUC-ROC | Brier Score | Log Loss |
|-------|---------|-------------|----------|
| LR | 0.5622 | 0.2026 | 0.5942 |
| XGB | 0.5755 | 0.2013 | 0.5915 |
| LSTM | 0.5202 | 0.2072 | 0.6051 |
| *Baseline Brier* | — | *0.2038* | — |

**Test ranking:** XGB > LR > LSTM. XGBoost is the strongest on held-out data, consistent with Grinsztajn et al. (2022).

Test probability correlations:

|      | LR    | XGB   | LSTM  |
|------|-------|-------|-------|
| LR   | 1.000 | 0.878 | 0.311 |
| XGB  | 0.878 | 1.000 | 0.273 |
| LSTM | 0.311 | 0.273 | 1.000 |

### 8.3 CV-to-Test Gap

| Model | CV AUC | Test AUC | Gap |
|-------|--------|----------|-----|
| LR | 0.650 | 0.562 | -0.088 |
| XGB | 0.638 | 0.576 | -0.062 |
| LSTM | 0.641 | 0.520 | -0.121 |

LSTM shows the largest generalization gap (0.12 AUC), suggesting temporal patterns learned in 2005-2022 don't fully transfer to 2023-2025. XGB generalizes best. The LSTM-LR/XGB correlation also drops dramatically from CV (0.78/0.70) to test (0.31/0.27), meaning LSTM diverges more in truly out-of-sample data.

### 8.4 A/B Spread Ablation (CV)

| Model | With Spread | Without Spread | Delta |
|-------|-------------|----------------|-------|
| LR | 0.6497 | 0.5707 | -0.079 |
| XGB | 0.6377 | 0.5662 | -0.072 |
| LSTM | 0.6407 | 0.5739 | -0.067 |

**Without spread, LSTM wins:** ranking flips to LSTM > LR > XGB. LSTM degrades the least (-0.067 vs -0.079 for LR), supporting the thesis that it captures temporal signal independent of the market line.

Correlation and disagreement effects of removing spread:
- LR-XGB correlation drops from 0.874 → 0.742 (models diversify)
- All-three agreement rate drops from 74.7% → 55.3%
- LSTM exclusive predictions double from 5.6% → 11.0% of games

### 8.5 Disagreement Analysis

**CV disagreement (with spread, threshold = base rate 0.297):**

| Category | N | % | Upset Rate | Interpretation |
|----------|---|---|------------|----------------|
| all_correct | 528 | 45.4% | 36.7% | Clear signal all architectures detect |
| all_wrong | 340 | 29.3% | 20.9% | Outside model capabilities |
| only_lr | 28 | 2.4% | 28.6% | Linear mispricing signal |
| only_xgb | 48 | 4.1% | 25.0% | Interaction pattern |
| only_lstm | 65 | 5.6% | 18.5% | Temporal signal (mostly non-upset rejection) |
| lr_xgb | 78 | 6.7% | 29.5% | Static models agree |
| lr_lstm | 48 | 4.1% | 33.3% | Linear + temporal agree |
| xgb_lstm | 27 | 2.3% | 33.3% | Non-linear + temporal agree |

**LSTM exclusive analysis:** Of 65 LSTM-only-correct predictions in CV, 53 are non-upsets correctly rejected (LSTM says "no" while LR/XGB say "yes") and 12 are upsets correctly caught. The LSTM's primary exclusive value is **moderating false alarms** from the static models, not detecting upsets the others miss.

**Note on test set disagreement:** The test set uses calibrated (Platt-scaled) probabilities, which compress the probability range to [0.19, 0.50]. Threshold-based categorization on calibrated probabilities has reduced discriminative power compared to raw probabilities or rank-based analysis. The top-K analysis (Section 8.6) is more informative for test set disagreement.

### 8.6 Top-K Analysis (Test Set)

| K | LR | XGB | LSTM | Ensemble |
|---|-----|-----|------|----------|
| 10 | 5/10 (50%, 1.8x) | 6/10 (60%, 2.1x) | 3/10 (30%, 1.1x) | 6/10 (60%, 2.1x) |
| 20 | 8/20 (40%, 1.4x) | 9/20 (45%, 1.6x) | 7/20 (35%, 1.2x) | 10/20 (50%, 1.8x) |
| 50 | 19/50 (38%, 1.3x) | 22/50 (44%, 1.5x) | 16/50 (32%, 1.1x) | 18/50 (36%, 1.3x) |

XGB's top 10 hit at 60% (2.1x lift). The ensemble's top 20 hit at 50% (1.8x lift). Signal exists but fades at higher K values.

### 8.7 Per-Season Performance (Test Set)

| Season | Games | Upset Rate | Best AUC (model) | LSTM AUC |
|--------|-------|-----------|-----------------|----------|
| 2023 | 185 | 29.7% | 0.521 (XGB) | 0.489 |
| 2024 | 192 | 23.4% | 0.586 (LSTM) | 0.586 |
| 2025 | 181 | 32.6% | 0.639 (XGB) | 0.493 |

LSTM is the best model in 2024 but worst in 2023 and 2025, suggesting it is more season-dependent than the static models.

---

## 9. The Upset Taxonomy

The disagreement framework reveals four structural types of upsets:

### Type 1: Market Errors
- **Who catches it:** LR/XGB (static models)
- **What it is:** The spread was wrong based on available statistics
- **Reality:** Extremely rare (<1%). The market is efficient on static features.

### Type 2: Temporal/Momentum Upsets
- **Who catches it:** LSTM only
- **What it is:** Temporal patterns invisible to point-in-time snapshots
- **Mechanism:** Hot team effect, form shifts, schedule-driven fatigue
- **Statistical status:** LSTM is competitive in CV (AUC 0.641) and wins without spread (0.574), but its exclusive catches are primarily non-upset rejections (53/65 in CV). Significance tests pending.

### Type 3: Hidden Information
- **Who catches it:** Nobody
- **What it is:** Large-spread upsets where information isn't in historical data
- **Mechanism:** QB injuries, teams resting starters, tanking, weather surprises
- **Key insight:** Not a modeling failure — it's a data availability problem

### Type 4: Stochastic
- **Who catches it:** Nobody
- **What it is:** Close games that could go either way
- **Mechanism:** Game variance, bounces, referee calls, execution on the day
- **Key insight:** The irreducible floor of NFL prediction

---

## 10. Open Questions & Next Steps

### Completed
1. ~~**Run CV with multi-representation architecture**~~ — Done. Results in Section 8.1. LSTM is competitive (AUC 0.641).
2. ~~**Run A/B spread ablation**~~ — Done. Results in Section 8.4. LSTM wins without spread (0.574).
3. ~~**Investigate LSTM exclusive predictions**~~ — Done. LSTM exclusives are primarily non-upset rejections (Section 8.5).

### Research
4. **XGB max_depth tuning** — currently 2, may need 3+ to exploit the 70-feature space effectively.
5. **Significance testing** — rerun permutation, binomial, and Mann-Whitney tests on LSTM-only catches under the current architecture.
6. **External data for Type 3** — could injury reports or motivation signals reduce the "hidden information" category?
7. **Investigate LSTM CV-to-test gap** — LSTM drops 0.12 AUC from CV to test (largest of all models). Is this temporal non-stationarity, or overfitting to 2005-2022 temporal patterns?
8. **Calibration and disagreement** — current test set disagreement uses calibrated probabilities with compressed range. Consider raw-probability or rank-based disagreement analysis for the paper.

---

## Appendix: File Map

### Data Pipeline
| File | Role |
|------|------|
| `src/data/nfl_loader.py` | Load NFL schedules via nfl_data_py |
| `src/data/betting_loader.py` | Load Kaggle spreadspoke betting data |
| `src/data/epa_loader.py` | Extract EPA and advanced stats from PBP data |
| `src/data/merger.py` | Merge schedule + betting + EPA |
| `src/data/elo.py` | Pre-game Elo rating computation |
| `src/data/generate_features.py` | End-to-end pipeline with validation |

### Feature Engineering
| File | Role |
|------|------|
| `src/features/pipeline.py` | 70-feature multi-representation pipeline |
| `src/features/target.py` | Upset target computation helpers |

### Models
| File | Role |
|------|------|
| `src/models/logistic_model.py` | L1-regularized logistic regression |
| `src/models/xgboost_model.py` | XGBoost classifier |
| `src/models/lstm_model.py` | Siamese LSTM with attention |
| `src/models/lstm_config.py` | Tuned LSTM hyperparameters |
| `src/models/sequence_builder.py` | LSTM sequence construction and normalization |
| `src/models/cv_splitter.py` | Time-series cross-validation |
| `src/models/unified_trainer.py` | Multi-model training on identical folds |
| `src/models/evaluate_test_set.py` | Held-out test set evaluation |
| `src/models/run_ab_experiment.py` | A/B spread ablation experiment |

### Evaluation
| File | Role |
|------|------|
| `src/evaluation/disagreement.py` | Model disagreement categorization |
| `src/evaluation/calibration.py` | Post-hoc Platt/isotonic calibration |
| `src/evaluation/metrics.py` | Calibration error, Brier score, baselines |
| `src/evaluation/report.py` | Report generation |

### Data Splits
| Split | Seasons | Labeled Games | Upset Rate |
|-------|---------|--------------|------------|
| Train | 2005-2022 | 3,495 | ~30% |
| Test | 2023-2025 | 558 | 28.5% |
