# NFL Upset Prediction: XGBoost vs LSTM Comparison

**Date:** 2026-01-16
**Status:** Design Complete (v4 - logic gaps fixed)

## Overview

Research project comparing XGBoost and LSTM models for predicting NFL moneyline upsets, with focus on understanding the structural factors that drive upset predictions and comparing how each model reasons about the problem.

## Problem Definition

### Target Variable
- **Binary classification**: Did the underdog win outright? (1 = yes, 0 = no)
- **Underdog definition**: Team with betting spread ≥ +3

### Research Questions
1. Which structural factors most predict NFL upsets?
2. Do XGBoost and LSTM rely on different signals? (Difference in reasoning between the two models)
3. Are there profitable betting opportunities at high-confidence thresholds?

## Data Strategy

### Sources
| Source | What it provides |
|--------|------------------|
| `nflverse/nfl_data_py` | Play-by-play, game stats, EPA, team metrics, schedules |
| `nflverse team/stadium metadata` | Stadium coordinates, roof type, timezone for travel/weather flags |
| `Pro Football Reference` | Historical game results, team standings |
| `Kaggle: spreadspoke/nfl-scores-and-betting-data` | Historical spreads, over/unders, results (1966-present) |
| `Weather (Meteostat or NOAA)` | Game-time weather by stadium coordinates when nflverse is missing |
| `FTN Data / Football Outsiders (DVOA)` | DVOA efficiency metrics (licensed) |
| `Odds history + public splits (SBR/OddsPortal + Action Network/Sports Insights)` | Opening/closing lines and public bet % for line movement |

**Data Verification Step (BEFORE committing to date range):**
1. Download Kaggle dataset, check spread coverage for 2005-2010
2. If >10% of games missing spreads in early years, adjust start date to 2010
3. Verify nflverse EPA data availability (started ~2006)
4. Verify weather, line movement, and DVOA coverage for target years; trim features if gaps are large
5. Document any gaps in `data/README.md`

### Feature Source Notes
- **Travel/timezone**: Compute from nflverse team/stadium metadata (lat/long, timezone) + schedule; no external join needed.
- **Weather**: Use nflverse pbp weather fields when available; backfill missing values via Meteostat/NOAA by stadium coords + kickoff time; treat indoor/closed roof as neutral.
- **DVOA**: Requires FTN/Football Outsiders data (licensed). If unavailable, drop DVOA and use EPA-based efficiency features instead.
- **Line movement**: Requires opening and closing lines from odds history. If only one line is available, drop line movement features.
- **Reverse line movement**: Requires public bet % splits. If unavailable, drop `reverse_line_movement`.

### Dataset Merge Strategy
1. Use nflverse schedules as the spine with canonical `game_id`, `season`, `week`, `game_date`, `home_team`, `away_team`.
2. Normalize team abbreviations to nflverse `team_abbr` mapping (handles relocations like STL->LAR, SD->LAC, OAK->LV).
3. Join Kaggle betting lines to the spine on `season`, `week`, `home_team`, `away_team`; fall back to `game_date` if week is missing. If multiple rows per game, keep the closing line or latest timestamp and retain both opening/closing when available.
4. Join PFR standings/team context on `season` + `team_abbr` only; do not overwrite game outcomes or scores.
5. Log unmatched/duplicate rows to `data/merge_audit.csv` and exclude unresolved games from modeling; track the drop rate.

### Time Range
- **Training**: 2005-2022 regular seasons (~18 seasons, ~4,608 games)
- **Test**: 2023, 2024, 2025 regular seasons (3 seasons, ~816 games)

**Note**: Regular season games only - excludes playoffs. See `docs/data-split-change.md` for rationale on 3-season test set.

## Feature Engineering

### Team Performance Features (Rolling Averages)

**What is a rolling average?** Instead of using a team's full season stats, we use their stats from their last 5 games. This captures "how they're playing right now" rather than how they played in Week 1. As the season progresses, the window "rolls" forward.

**Rolling Window Rules:**
- **Window size**: Last 5 games
- **Start of season**: Use all available games (if team has only played 3 games, use those 3)
- **Bye weeks**: Skip the bye, still use last 5 actual games played

| Category | Features |
|----------|----------|
| Offense | Points/game, yards/game, EPA/play, 3rd down %, red zone % |
| Defense | Points allowed, yards allowed, defensive EPA, sacks, turnovers forced |
| Efficiency | DVOA (if available), point differential, turnover margin |

### Situational/Contextual Features
| Category | Features |
|----------|----------|
| Rest/Travel | Days since last game, travel distance, timezone changes |
| Game Context | Home/away, divisional game, primetime game, weather (outdoor) |
| Season Context | Week number, playoff implications, win streak/losing streak |

### Spread-Related Features
| Feature | Why it matters |
|---------|----------------|
| Spread magnitude | +3 vs +10 underdogs behave differently |
| Over/under | High-scoring games = more variance = more upsets? |

**Line Movement Features** (potential market inefficiency signals):
| Feature | Calculation |
|---------|-------------|
| `line_movement` | Closing spread − Opening spread |
| `line_direction` | Did line move toward or away from underdog? |
| `movement_magnitude` | Absolute value of line movement |
| `reverse_line_movement` | Public betting one way, line moving opposite (sharp indicator) |

### Derived Features

#### Matchup Differentials (underdog - favorite)
| Feature | Calculation |
|---------|-------------|
| `offense_defense_mismatch` | Underdog pass offense EPA − Favorite pass defense EPA |
| `rush_attack_advantage` | Underdog rush yards/game − Favorite rush yards allowed/game |
| `turnover_edge` | Underdog turnover margin − Favorite turnover margin |
| `red_zone_differential` | Underdog RZ TD% − Favorite RZ defense TD% allowed |
| `pressure_matchup` | Underdog sacks allowed − Favorite sacks generated |

#### Momentum/Trend Indicators
| Feature | Calculation |
|---------|-------------|
| `point_diff_trend` | Slope of point differential over last 5 games |
| `consistency_score` | Std deviation of points scored (low = consistent) |
| `ats_streak` | Consecutive games covering/not covering spread |
| `win_streak_diff` | Underdog win streak − Favorite win streak |
| `recent_vs_season` | Last 3 game avg − Season avg (hot/cold indicator) |

#### Situational Splits
| Feature | Calculation |
|---------|-------------|
| `home_away_split` | Team's home record − away record differential |
| `vs_winning_teams` | Record against teams with winning records |
| `divisional_factor` | Historical performance in divisional games |
| `rest_advantage` | Days rest difference (underdog − favorite) |
| `travel_fatigue` | Miles traveled difference |

#### Interaction Features
| Feature | Why |
|---------|-----|
| `spread × momentum` | Big underdogs on hot streaks |
| `rest × travel` | Well-rested + home vs tired + road |
| `mismatch × spread` | Exploitable matchups in big underdog spots |

## Model Architectures

### XGBoost
- **Input**: Flat feature vector per game (~50-80 features)
- **Output**: Probability 0-1 (logistic objective)
- **Strengths**: Handles tabular data well, fast training, highly interpretable
- **Hyperparameters**: max_depth, learning_rate, n_estimators, min_child_weight

```
Game Features → XGBoost → P(upset)
[80 features]              [0.0 - 1.0]
```

### LSTM
- **Framework**: PyTorch (for Captum interpretability compatibility)
- **Input**: Sequence of last 5 games for each team
- **Output**: Probability 0-1 (sigmoid activation)
- **Strengths**: Learns temporal patterns automatically, captures momentum

**Architecture Details:**
```
Team A last 5 games → LSTM Encoder (shared weights) →
                                                      → Concat → Attention → Dense → P(upset)
Team B last 5 games → LSTM Encoder (shared weights) →
                                                      ↑
                                            Matchup Features (spread, context)
```

**Matchup Features Added Post-Encoding** (these describe THIS specific game, not team history):
- Spread magnitude (+3, +7, etc.)
- Home/away indicator
- Divisional game flag
- Rest advantage (days difference)
- Week number

**Siamese Architecture**: Both teams share the same LSTM encoder weights. This "team performance encoder" learns general patterns applicable to any team.

**Input Features Per Game Timestep** (~15 features per game):
- Points scored, points allowed
- Total yards, yards allowed
- EPA/play (offense and defense)
- Turnover margin
- 3rd down conversion rate
- Win/loss indicator

**Attention Mechanism**: Adds interpretability - shows which past games the model weighted most heavily for each prediction.

**Hyperparameters:**
- LSTM: 2 layers, 64 hidden units, dropout 0.3
- Sequence length: 5 games (fixed)
- Dense layers: 64 → 32 → 1

**Handling Variable Sequence Length (Start of Season):**
- **Problem**: LSTMs require fixed-length input, but early-season games have <5 prior games
- **Solution**: Pad sequences with zeros + apply masking layer
- **Example**: Week 2 team has 1 prior game → [Game1, 0, 0, 0, 0] with mask [1, 0, 0, 0, 0]
- **Minimum requirement**: Team must have at least 1 prior game to be included (Week 1 excluded from predictions)

### Key Architectural Difference
- **XGBoost**: You engineer the temporal features (rolling averages, trends)
- **LSTM**: Model learns what temporal patterns matter from raw sequences

## Evaluation Framework

### Training Metric
**Log Loss** (also called binary cross-entropy): This is what the models optimize during training. It penalizes confident wrong predictions more heavily than uncertain ones. Lower is better.

### Prediction Accuracy Metrics
| Metric | What it measures |
|--------|------------------|
| AUC-ROC | Overall discrimination ability |
| Brier Score | Probability calibration (lower = better) |
| Precision @ threshold | Of predicted upsets, how many were correct? |
| Recall @ threshold | Of actual upsets, how many did we catch? |
| Calibration curve | Are 70% predictions actually right 70% of the time? |

### Success Criteria

**What would make this research "successful"?**

| Metric | Target | Why this target |
|--------|--------|-----------------|
| AUC-ROC | > 0.55 | Random guessing = 0.50. Vegas implied lines are ~0.52-0.54. Beating this shows we found signal. |
| Calibration | Brier < baseline | Baseline = constant prediction at empirical upset rate r (Brier = r*(1-r)) |
| Profitability | ROI > 5% | After accounting for the ~10% vig, need meaningful edge |
| Model difference | Agreement < 85% | If models agree 95%+ they learned the same thing - less interesting |

**Calibration baseline note:** Compute r from the training set upset rate. Example: r=0.35 => baseline Brier=0.2275, so target <0.2275.

**What if models perform similarly to random?** That's still a valid research finding - it would suggest NFL upset markets are highly efficient.

### Interpretability Analysis
| Model | Method |
|-------|--------|
| **XGBoost** | Feature importance (gain), SHAP values per prediction, partial dependence plots |
| **LSTM** | Attention weights (if added), gradient-based saliency, hidden state analysis |

### Research Analysis
1. **Which features drive XGBoost's upset predictions?**
   - SHAP summary plot → global feature importance
   - SHAP waterfall → why did it predict THIS game as upset?

2. **What temporal patterns does LSTM learn?**
   - Does it weight recent games more heavily?
   - Does it detect momentum shifts humans miss?

3. **Where do they disagree?**
   - Find games where XGBoost says "upset" but LSTM says "no" (and vice versa)
   - Analyze *why* → reveals different structural theories

### Model Comparison Methodology

**Quantifying "Reasoning Difference":**

| Analysis | Method |
|----------|--------|
| Feature importance correlation | Spearman correlation between XGBoost SHAP and LSTM aggregated importance (see below) |
| Prediction agreement rate | % of games where both models agree (same side of 0.5 threshold) |
| Disagreement analysis | Deep dive into games where models disagree by >0.3 probability |
| Confidence calibration | Compare calibration curves - do they over/underconfident in same situations? |

**Making LSTM/XGBoost Importance Comparable:**
- **Problem**: XGBoost SHAP gives per-feature importance, LSTM attention gives per-timestep importance
- **Solution**: Aggregate LSTM attention by feature category
  1. For each prediction, get attention weights per timestep [w1, w2, w3, w4, w5]
  2. Multiply each timestep's features by its attention weight
  3. Sum across timesteps to get "attention-weighted feature importance"
  4. Average across all predictions for global comparison
- **Result**: Both models now have per-feature importance scores that can be correlated

**Disagreement Case Study Protocol:**
1. Identify top 20 games with largest prediction disagreement
2. For each: extract XGBoost SHAP explanation + LSTM attention weights
3. Categorize disagreement type: momentum-driven? matchup-driven? situational?
4. Document patterns in when each model is correct

### Profitability Backtest

**Standard Metrics:**
| Metric | Calculation |
|--------|-------------|
| ROI | (Profit / Total wagered) × 100 |
| Win rate @ threshold | % correct when P(upset) > 0.5, 0.6, 0.7 |
| Expected value | Avg return per bet at each confidence level |

**Closing Line Value (CLV) Analysis:**
| Metric | Why it matters |
|--------|----------------|
| CLV | Did model's implied probability beat the closing line? |
| Sharp vs model | Compare model predictions to closing line movement direction |
| Edge sustainability | Is model finding real inefficiencies or noise? |

**Note**: Positive CLV is the gold standard for betting models - indicates finding value the market eventually corrects to.

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Data ingestion | `nfl_data_py`, `pandas` | nflverse Python wrapper |
| Feature engineering | `pandas`, `numpy`, `scikit-learn` | Rolling calculations, preprocessing |
| XGBoost | `xgboost` | Industry standard, SHAP support |
| LSTM | `pytorch` | Flexibility, Captum compatibility |
| Interpretability | `shap`, `captum` | SHAP for XGBoost, Captum for LSTM |
| Visualization | `matplotlib`, `seaborn`, `plotly` | Charts and plots |
| Experiment tracking | `mlflow` | Track hyperparameters, compare runs |
| Hyperparameter tuning | `optuna` | Bayesian optimization |

## Project Structure

```
nfl-upset-prediction/
├── data/
│   ├── raw/              # Downloaded NFL data
│   ├── processed/        # Feature-engineered datasets
│   └── splits/           # Train/val/test sets
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_xgboost_model.ipynb
│   ├── 04_lstm_model.ipynb
│   └── 05_comparison_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models/
│   └── evaluation.py
├── results/
│   └── figures/
└── README.md
```

## Hyperparameter Tuning Strategy

**What are hyperparameters?** Settings you choose before training (like "how deep should the tree be?" or "how many training rounds?"). We need to find good settings without "cheating" by looking at our test data.

**Time-Series Cross-Validation:**

Unlike regular cross-validation where you randomly split data, we must respect time order (can't use 2020 games to predict 2019).

```
Fold 1: Train on 2005-2016 → Validate on 2017
Fold 2: Train on 2005-2017 → Validate on 2018
Fold 3: Train on 2005-2018 → Validate on 2019
Fold 4: Train on 2005-2019 → Validate on 2020
Fold 5: Train on 2005-2020 → Validate on 2021
Fold 6: Train on 2005-2021 → Validate on 2022
```

**Final model**: Train on all 2005-2022 data with best hyperparameters, then evaluate separately on 2023, 2024, and 2025 (test).

**Why this matters**: This prevents "peeking" at future data and gives realistic performance estimates.

## Implementation Order

1. Data collection & cleaning (verify betting line data quality for all years)
2. Exploratory data analysis (upset rates, feature distributions, correlations)
3. Feature engineering pipeline
4. Baseline models (logistic regression, "always predict no upset")
5. Train XGBoost model + hyperparameter tuning
6. Train LSTM model + hyperparameter tuning
7. Interpretability analysis (SHAP, attention weights)
8. Model comparison (disagreement analysis, reasoning differences)
9. Profitability backtest + CLV analysis

## Expected Outputs

1. **Research findings**: Which structural factors most predict NFL upsets
2. **Model comparison**: How XGBoost vs LSTM reason differently
3. **Betting insights**: Profitable thresholds (if any exist)
4. **Visualizations**: SHAP plots, calibration curves, disagreement analysis

## Limitations & Assumptions

### Market Efficiency Assumption
This study uses Vegas opening/closing spreads as a baseline, acknowledging that spreads already incorporate:
- QB availability and quality
- Injury reports
- Weather conditions
- Travel and rest factors
- Coaching tendencies
- Public and sharp money

**Implication**: Our features are testing whether additional signal exists *beyond* what the efficient market has already priced in. We are hunting for market inefficiencies, not replicating Vegas's information.

### Data Limitations
- Historical betting line data quality may vary for earlier seasons (2005-2010)
- Line movement data may be incomplete for some games
- Play-by-play data quality improves over time

### Methodological Limitations
- LSTM vs XGBoost comparison uses different data representations (sequences vs flat features), which affects fairness of comparison
- Class imbalance (~35% upset rate) may affect model calibration at extreme confidence levels
- Regular season only - playoff dynamics may differ significantly

### What This Study Does NOT Address
- Real-time prediction (we use pre-game features only)
- Player-level analysis (beyond what's priced into spreads)
- In-game betting / live odds
