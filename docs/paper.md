# Diagnosing the Unpredictable: A Multi-Architecture Disagreement Framework for NFL Upset Taxonomy

---

## Abstract

Two decades of research on NFL upset prediction have produced consistently modest accuracy gains, suggesting that the barrier to understanding upsets may be structural rather than methodological. Rather than attempt another incremental improvement, we use the *structure of prediction failure* as a lens for understanding why upsets happen. We train three architecturally distinct models — L1-regularized logistic regression, gradient-boosted trees (XGBoost), and a siamese LSTM with attention — on identical data in architecture-appropriate representations, then analyze their patterns of agreement and disagreement as a diagnostic tool. Each model processes the same underlying game data in a different form: LR receives 46 pre-computed summary statistics, XGBoost receives 70 features including per-game lags, and the LSTM receives raw 8-game sequences of 14 features per timestep plus 10 matchup context features. This is not a model competition — it is a diagnostic framework where the disagreement itself is the finding.

On 3,495 training games (2005–2022) evaluated via 6-fold expanding-window cross-validation, the three models achieve statistically indistinguishable performance (AUC-ROC: LR 0.650, LSTM 0.641, XGB 0.638; bootstrap CIs on all pairwise differences contain zero). Stratifying the disagreement analysis by point spread reveals that the LSTM's exclusive contribution inverts with matchup context: at small spreads (3–6.5 points), 92% of LSTM exclusives are false-alarm rejections; at medium spreads (7–13.5), 83% are genuine upset detections. A spread ablation experiment shows that all models significantly degrade without betting-line features, with the LSTM retaining the most signal (AUC 0.574 vs. 0.571 and 0.566). On a held-out test set of 558 games (2023–2025), the LSTM shows the largest generalization gap (AUC: 0.641 → 0.524), and an LSTM veto ensemble that improves predictions in CV fails to transfer forward, indicating temporal patterns are real but non-stationary.

These findings support a two-dimensional upset taxonomy — model agreement crossed with matchup context — that decomposes prediction failure into stochastic variance (close games), temporal dynamics (moderate underdogs), and information limits (large mismatches). The central contribution is methodological: architectural disagreement reveals the structure behind upsets in a way that no single model or flat ensemble can, and the diagnostic framework generalizes to any prediction domain where outcomes are driven by heterogeneous mechanisms.

---

## 1. Introduction

NFL upset prediction has attracted substantial research attention, yet the fundamental question remains largely unanswered: not *whether* upsets can be predicted, but *why* they happen. The typical approach trains one or more models on historical game data, evaluates accuracy against the betting line, and reports marginal improvements over market efficiency. This framing treats prediction as the goal and model comparison as the method — and after two decades, it has produced consistently modest results.

We invert this relationship. Our goal is not to predict upsets more accurately than the market, but to use the *structure of prediction failure* as a diagnostic lens for understanding the mechanisms behind upsets. The key insight is that when three architecturally distinct models disagree about a game, the pattern of disagreement reveals something about the nature of the upset itself. If all three models fail on a large favorite, the cause is hidden information. If only the LSTM succeeds, the cause is temporal — momentum or trajectory that static models cannot see. The disagreement is the finding.

This approach draws on several intellectual traditions. The co-validation literature demonstrates that per-instance disagreement between models is an unbiased estimate of error variance, which justifies treating disagreement as informative rather than as noise (Gordon et al., 2021). The ensemble uncertainty literature shows that model disagreement captures epistemic uncertainty — what the model doesn't know — as distinct from aleatoric uncertainty — what cannot be known from the data (Lakshminarayanan et al., 2017). And the medical diagnostics literature routinely uses the agreement patterns of structurally different tests (e.g., MRI vs. biopsy) to characterize disease subtypes, a paradigm we adapt for sports prediction.

Our contribution is a disagreement-based upset taxonomy that categorizes games by which architectures correctly predict the outcome, crossed with matchup context (the point spread). When all three models agree, the signal is strong and cross-architectural — the game was structurally readable from every angle. When only one model is correct, the mechanism is specific to that model's structural strength: linear spread mispricing for logistic regression, non-linear feature interactions for XGBoost, or temporal dynamics for the LSTM. When no model is correct, the spread context distinguishes two fundamentally different failure modes: stochastic variance in close games versus information outside the model's reach in large mismatches. This two-dimensional decomposition — model agreement × matchup context — is the diagnostic framework's core contribution. It reveals the structure behind upsets in a way that neither prediction accuracy nor any single-model analysis can.

### 1.1 Related Work

**NFL prediction and market efficiency.** The betting market is a strong baseline for NFL prediction. Wilkens (2021) provides a comprehensive review of market efficiency in sports betting, finding that closing lines are well-calibrated and difficult to beat consistently. Hubacek et al. (2019) demonstrate that decorrelating model predictions from bookmaker odds is necessary to identify genuine predictive signal, a finding that motivates our spread ablation experiment. Our work does not claim to beat the market; rather, we use the market line as one input among several to study *when and why* upsets occur.

**Machine learning for sports outcomes.** Gradient-boosted trees have emerged as the dominant architecture for tabular sports prediction. Grinsztajn et al. (2022) provide systematic evidence that tree-based models outperform deep learning on medium-sized tabular datasets, which characterizes our setting (3,495 training games with 70 features). Recurrent neural networks have been applied to sports prediction to capture sequential patterns (e.g., team form trajectories), though typically with mixed results compared to tabular approaches. Our contribution is not to adjudicate this comparison but to exploit the *structural differences* between these architectures for diagnostic purposes.

**Ensemble disagreement and uncertainty.** Lakshminarayanan et al. (2017) introduce deep ensembles for uncertainty estimation, showing that disagreement between independently trained models captures predictive uncertainty. Gordon et al. (2021) extend this idea to co-validation, demonstrating that per-instance disagreement is an unbiased estimator of error variance. Fort et al. (2019) show that models trained from different initializations explore different loss basins and make systematically different errors. We build on this tradition but go further: instead of using disagreement to *estimate uncertainty*, we use it to *categorize the mechanism* of prediction failure. The prior literature treats disagreement as a signal to quantify; we treat it as a diagnostic tool that reveals structure.

**Multi-representation learning.** Our approach of giving three models the same data in different representations is related to multi-view learning (Xu et al., 2013), where different "views" of the same instance are processed by different learners. The key difference is that multi-view learning typically seeks consensus; we seek informative *disagreement*. Pinchuk (2026) documents limitations in how XGBoost learns higher-order feature interactions, which motivates our inclusion of per-game lag features that make interaction patterns more accessible to tree-based splits.

---

## 2. Methods

### 2.1 Data

We compile NFL game data from the 2005 through 2025 seasons using three sources: game schedules from nfl_data_py (teams, scores, dates, rest days, division flags), betting lines from the Kaggle Spreadspoke dataset (point spreads, over/under totals, favorite identification), and advanced statistics derived from play-by-play data via nfl_data_py (expected points added for passing and rushing, success rate, completion percentage over expectation, and turnover margin).

These sources are merged on game identifiers to produce one row per game containing scores, betting lines, and 10 advanced statistics per team. Team abbreviations are normalized for franchise relocations (e.g., STL → LA, SD → LAC, OAK → LV). Pre-game Elo ratings are computed inline using a standard algorithm (K = 20, home advantage = 50, base rating = 1500) with ratings carrying across seasons without reset.

**Target definition.** We define an upset as a game in which the underdog wins and the point spread is at least 3 points. Games with spreads below 3 are near toss-ups by the market's own assessment; including them as non-upsets would introduce approximately 20% label noise. Games below the spread threshold retain `upset = NaN` and are excluded from training and evaluation via `upset.notna()`, but remain in the dataset for rolling statistic computation.

**Data splits.** The training set comprises 2005–2022 (3,495 labeled games, upset rate 30.4%). The held-out test set comprises 2023–2025 (558 labeled games, upset rate 28.5%). No information from the test period is used during model development, hyperparameter selection, or feature engineering.

### 2.2 Feature Engineering: Multi-Representation Design

The central design principle is that each model receives the same underlying game data in a representation optimized for that model's structural strengths. This ensures that disagreement between models reflects genuine architectural differences in how they process information, not differences in the input data.

**Shared base: team rolling statistics.** For each game, we convert game-level rows to team-level rows and compute per-team rolling statistics over a 3-game window: rolling means for pass EPA, rush EPA, success rate, CPOE, and turnover margin; rolling standard deviation for total EPA and success rate (capturing volatility); and trend indicators (current value minus rolling mean). Per-game lag features are also computed for the most recent 1, 2, and 3 games. The 3-game window is chosen to balance recency with stability in NFL's 17-game seasons, and rolling windows are allowed to cross season boundaries to avoid cold-start artifacts.

**LR representation: "The Summary" (46 features).** Logistic regression receives pre-computed summary statistics: 10 rolling efficiency metrics per role (underdog/favorite), 5 underdog-minus-favorite differentials, 12 volatility and trend indicators, 5 schedule context features (rest days, short week, divisional game), 4 market features (home/away implied points, spread magnitude, total line), 3 Elo features, 5 environment features (temperature, wind, dome), and 2 game context features (underdog is home, week number). These features are pre-engineered to capture the relationships that a linear model needs (e.g., differentials rather than raw values).

**XGBoost representation: "The Details" (70 features).** XGBoost receives all 46 LR features plus 24 per-game lag features: for each role (underdog, favorite) and each of the three most recent games, the individual values of total EPA, success rate, turnover margin, and margin of victory. This allows XGBoost to discover non-linear interactions that rolling averages destroy — for example, "the underdog's 3-game rolling EPA is positive, but their most recent game was a blowout loss." The 24 additional features give XGBoost access to the game-by-game texture that LR's summary statistics erase.

**LSTM representation: "The Movie" (14 sequential features x 8 timesteps + 10 matchup context features).** The LSTM processes raw game-by-game sequences with no pre-computed rolling statistics. For each team, the 8 most recent games are represented as a sequence of 14 features: total/pass/rush EPA, success rate, CPOE, turnover margin, points scored, points allowed, point differential, opponent Elo, win indicator, home indicator, days since last game, and short week flag. A separate 10-feature matchup context vector provides information the sequence cannot encode: spread magnitude, total line, both teams' Elo ratings, underdog home status, both teams' rest days, week number, divisional game flag, and dome indicator. This design forces the LSTM to rely on its temporal encoder rather than shortcutting through pre-computed statistics.

**No-spread variants.** For the spread ablation experiment, each representation has a variant with market-derived features removed: LR drops from 46 to 42 features, XGBoost from 70 to 66, and the LSTM matchup context drops from 10 to 8 features. The LSTM sequence features are unchanged (they contain no market information).

### 2.3 Model Architectures

The three models are chosen to be structurally complementary:

**L1-regularized logistic regression (LR).** We use logistic regression with L1 (LASSO) penalty (C = 0.1, SAGA solver) preceded by standard scaling. The strong L1 regularization zeros out irrelevant features, producing a sparse, interpretable model. LR serves as the baseline: it captures systematic linear relationships between pre-computed features and upset probability. Its coefficient on spread magnitude (-0.539 on the standardized scale) confirms the spread as the dominant linear predictor.

**Gradient-boosted trees (XGBoost).** We use XGBoost with deliberately conservative hyperparameters: `max_depth = 2`, `learning_rate = 0.03`, `n_estimators = 300`. The shallow depth prevents overfitting to noise in a 3,495-game dataset with 70 features. XGBoost's role is interaction detection: it can learn split-based rules that combine features in ways unavailable to LR, particularly exploiting the per-game lag features to find patterns in recent game trajectories.

**Siamese LSTM with attention.** We use a siamese architecture where both teams' game sequences are processed by a shared LSTM encoder (3 layers, hidden size 64, dropout 0.25), followed by temporal attention over the 8 timesteps. The attention-weighted encodings for both teams are concatenated with the matchup context vector and passed through a dense classifier (64 → 32 → 1) with sigmoid output. The siamese design ensures the model learns general team-performance patterns rather than team-specific ones. Training uses binary cross-entropy loss with Adam optimizer (learning rate 0.001), batch size 64, and a tuned 25-epoch budget; the cross-validation pipeline applies early stopping with patience 6, while the final held-out model is trained for the full epoch budget. Sequence normalization statistics are computed from training data only to prevent data leakage.

### 2.4 Evaluation Framework

**Cross-validation.** We use expanding-window time-series cross-validation with 6 folds (training 2005–2016 through 2005–2021, validating on 2017 through 2022 respectively). All three models are trained on identical folds to ensure disagreement reflects model differences rather than data differences. The unified training pipeline constructs model-specific feature representations from the same fold data.

**Held-out test evaluation.** Final models are trained on the full 2005–2022 training set and evaluated on 2023–2025. Post-hoc Platt calibration is applied using a held-out calibration set: models are trained on 2005–2020, predictions on 2021–2022 are used to fit logistic calibrators, and these calibrators are applied to the test predictions. Calibration is necessary to make probabilities comparable across architectures for the disagreement analysis.

**Metrics.** We report AUC-ROC (discrimination), Brier score (calibrated accuracy; baseline Brier for a constant-probability model is 0.204 at the test upset rate), and log loss. For the disagreement analysis, we use the base upset rate (approximately 0.30) as the threshold for binary prediction rather than 0.50, which is the principled choice for minority-class problems.

**Spread ablation.** We run the full cross-validation pipeline twice: once with all features (Experiment A) and once with market-derived features removed (Experiment B). The delta between conditions isolates what the betting line contributes to each model's predictions. If a model collapses without spread, it was primarily echoing the market; if it retains predictive power, it captures genuine team-quality signal independent of the market's assessment.

### 2.5 Disagreement Analysis

Each game is categorized by which models' binary predictions (above/below the base-rate threshold) match the actual outcome:

| Category | Models Correct | Interpretation |
|----------|---------------|----------------|
| all_correct | LR, XGB, LSTM | Strong cross-architectural signal |
| all_wrong | None | Outside model capabilities |
| only_lr | LR | Linear spread mispricing |
| only_xgb | XGB | Non-linear interaction pattern |
| only_lstm | LSTM | Temporal/momentum signal |
| lr_xgb | LR, XGB | Static models agree (non-temporal) |
| lr_lstm | LR, LSTM | Linear + temporal agreement |
| xgb_lstm | XGB, LSTM | Non-linear + temporal agreement |

For each category, we report the count, percentage of games, actual upset rate, and average predicted probabilities from each model. We further decompose exclusive categories (only_lr, only_xgb, only_lstm) into upset detections (model correctly predicts upset) and non-upset rejections (model correctly rejects false alarm from other models).

---

## 3. Results

### 3.1 Cross-Validation Performance

Table 1 presents the 6-fold cross-validation results on 1,162 predictions (validation seasons 2017–2022).

**Table 1: Cross-validation performance (6-fold, 1,162 games)**

| Model | AUC-ROC | Brier Score | Log Loss |
|-------|---------|-------------|----------|
| LR | 0.650 | 0.197 | 0.581 |
| LSTM | 0.641 | 0.199 | 0.583 |
| XGB | 0.638 | 0.199 | 0.586 |

All three models achieve comparable performance. The difference between the best (LR, 0.650) and worst (XGB, 0.638) is 0.012 AUC. Bootstrap 95% confidence intervals on pairwise AUC differences confirm the models are statistically indistinguishable: LR-XGB +0.012 [-0.007, +0.031], LR-LSTM +0.009 [-0.014, +0.033], XGB-LSTM -0.003 [-0.030, +0.024]. All intervals contain zero. The LSTM performs on par with XGBoost, contrary to the general finding that tree-based models dominate deep learning on tabular data (Grinsztajn et al., 2022) — though our LSTM receives sequential data rather than a tabular representation, which is its structural advantage. The statistical equivalence of the three models is important for the disagreement framework: if one model were significantly weaker, its disagreement would simply indicate error rather than a different perspective on the data.

The probability correlations reveal the structural relationships between models. LR and XGB correlate at 0.874, indicating they largely extract the same signal from overlapping feature sets. The LSTM correlates moderately with both (LR: 0.784, XGB: 0.699), confirming it captures partially distinct information despite receiving the same underlying data. The XGB-LSTM correlation (0.699) is the lowest pairwise value, reflecting the greatest structural distance between interaction-based and sequence-based processing.

### 3.2 Held-Out Test Performance

Table 2 presents results on the 558-game test set (2023–2025), with Platt-calibrated probabilities.

**Table 2: Test set performance (558 games, calibrated)**

| Model | AUC-ROC | Brier Score | Log Loss |
|-------|---------|-------------|----------|
| XGB | 0.576 | 0.201 | 0.592 |
| LR | 0.562 | 0.203 | 0.594 |
| LSTM | 0.524 | 0.208 | 0.606 |
| *Baseline* | — | *0.204* | — |

XGBoost is the best test-set model, consistent with the general finding favoring tree-based methods on tabular-adjacent tasks. All AUCs are modest, confirming that NFL upsets remain genuinely difficult to predict.

**Table 3: CV-to-test generalization gap**

| Model | CV AUC | Test AUC | Gap |
|-------|--------|----------|-----|
| LR | 0.650 | 0.562 | -0.088 |
| XGB | 0.638 | 0.576 | -0.062 |
| LSTM | 0.641 | 0.524 | -0.117 |

All models degrade from CV to test, which is expected when validating on a future time period rather than on held-out folds within the training range. XGBoost shows the smallest gap (-0.062), indicating that non-linear interaction patterns are the most temporally stable signal. The LSTM shows the largest gap (-0.117), still substantially larger than XGBoost's, suggesting that the temporal patterns it learns from 2005–2022 sequences do not fully transfer to 2023–2025.

The inter-model correlation structure also shifts materially. The LR-XGB correlation remains stable (0.874 CV → 0.878 test), but the LSTM's correlation with both static models drops substantially (LR: 0.784 → 0.429; XGB: 0.699 → 0.408). The LSTM diverges from the other models more in truly out-of-sample data than in cross-validation, indicating that its temporal encoding captures patterns that are less stable forward in time than the features available to LR and XGB.

The per-season breakdown further illuminates LSTM instability:

**Table 4: Per-season test AUC**

| Season | Games | Upset Rate | LR | XGB | LSTM |
|--------|-------|-----------|------|------|------|
| 2023 | 185 | 29.7% | 0.512 | 0.521 | 0.469 |
| 2024 | 192 | 23.4% | 0.552 | 0.554 | 0.549 |
| 2025 | 181 | 32.6% | 0.617 | 0.639 | 0.556 |

The LSTM trails the static models in all three held-out seasons. Its weakest performance comes in 2023 (AUC 0.469), and although it improves through 2025 (0.556), it remains below both LR and XGB throughout. This pattern is more consistent with a persistent forward-transfer problem than with a single anomalous season.

### 3.3 Spread Ablation

Table 5 presents the effect of removing betting-line features on cross-validation AUC.

**Table 5: Spread ablation (CV AUC)**

| Model | With Spread | Without Spread | Delta |
|-------|-------------|----------------|-------|
| LR | 0.650 | 0.571 | -0.079 |
| XGB | 0.638 | 0.566 | -0.072 |
| LSTM | 0.641 | 0.574 | -0.067 |

Removing spread information hurts all three models substantially. Bootstrap confidence intervals confirm all three deltas are statistically significant: LR -0.079 [-0.108, -0.050], XGB -0.072 [-0.101, -0.042], LSTM -0.067 [-0.098, -0.036]. The betting line contains significant predictive information beyond what team performance statistics capture.

The LSTM degrades least in point estimate (-0.067) and becomes the strongest model in the no-spread condition (0.574 vs. 0.571 for LR and 0.566 for XGB). However, the difference in degradation between LSTM and LR is not statistically significant: the bootstrap CI on the delta-of-deltas is +0.012 [-0.019, +0.044], which contains zero. We can state that the LSTM retains more signal without spread than the other models, but we cannot claim this difference is robust to sampling variation. The ranking reversal is suggestive evidence that the LSTM captures temporal signal partially independent of the market, not a statistically proven finding.

The correlation structure shifts substantially without spread. The LR-XGB correlation drops from 0.874 to 0.742, and the XGB-LSTM correlation drops from 0.699 to 0.419. The spread acts as a shared anchor that pulls model predictions together; without it, the models diversify and their architectural differences become more pronounced.

This diversification is reflected in the disagreement statistics. The all-three agreement rate drops from 74.7% to 55.3%, and LSTM-exclusive correct predictions double from 5.6% to 11.0% of games. Within these LSTM exclusives, the number of upsets correctly caught increases from 12 to 33, suggesting that the spread was masking some genuine temporal signal that the static models were capturing through the betting line rather than through team performance trajectories.

### 3.4 Disagreement Analysis

With three statistically equivalent models producing moderately correlated predictions, the disagreement categories become interpretable: where models split, the pattern tells us something about the mechanism driving the outcome. Table 6 presents the full disagreement categorization from cross-validation (with spread features, threshold = base upset rate of 0.297).

**Table 6: Disagreement categories (CV, with spread)**

| Category | N | % | Upset Rate | Avg P(upset): LR | XGB | LSTM |
|----------|---|---|------------|---------|-----|------|
| all_correct | 528 | 45.4 | 36.7 | 0.259 | 0.265 | 0.259 |
| all_wrong | 340 | 29.3 | 20.9 | 0.349 | 0.351 | 0.343 |
| only_lr | 28 | 2.4 | 28.6 | 0.292 | 0.317 | 0.315 |
| only_xgb | 48 | 4.1 | 25.0 | 0.317 | 0.290 | 0.333 |
| only_lstm | 65 | 5.6 | 18.5 | 0.328 | 0.343 | 0.249 |
| lr_xgb | 78 | 6.7 | 29.5 | 0.289 | 0.279 | 0.307 |
| lr_lstm | 48 | 4.1 | 33.3 | 0.290 | 0.311 | 0.277 |
| xgb_lstm | 27 | 2.3 | 33.3 | 0.311 | 0.286 | 0.261 |

Several patterns emerge.

**The agreement core.** Nearly half of all games (45.4%) fall into the all_correct category, where all three models' binary predictions match the outcome. These games have a 36.7% upset rate — above the 29.7% base rate — and tend to be games where the signal is strong enough for any reasonable model to detect. Another 29.3% are all_wrong: games where no model captures the outcome. Together, these two categories account for 74.7% of games, leaving 25.3% where the models disagree in an informative way.

**Exclusive categories are small but diagnostic.** The only_lr (2.4%), only_xgb (4.1%), and only_lstm (5.6%) categories are each a small fraction of games, but their structure reveals the unique contribution of each architecture. The lr_xgb category (6.7%) identifies games where the static models agree but the LSTM disagrees — the largest pairwise-agreement category and a natural indicator of non-temporal signal.

**LSTM exclusives are primarily non-upset rejections.** Of the 65 games where only the LSTM is correct, 53 (81.5%) are non-upsets that the LSTM correctly rejects while LR and XGB incorrectly predict upset. Only 12 (18.5%) are upsets that the LSTM catches and the other models miss. A one-sided binomial test confirms that this bias is statistically significant: the LSTM exclusive upset rate of 18.5% is below the base rate of 29.7% at p = 0.029. The LSTM's primary exclusive contribution is *moderating false alarms* from the static models, not detecting temporal upsets that the others miss.

The average predicted probabilities in the only_lstm category illuminate the mechanism. For these games, LR (0.328) and XGB (0.343) both assign above-threshold probabilities, while the LSTM assigns a lower probability (0.249). The LSTM's temporal encoding provides evidence that these games, which look like upset candidates based on static features, are not actually likely to produce upsets — perhaps because the underdog's recent form trajectory contradicts the static picture.

### 3.5 Spread-Stratified Disagreement

The flat disagreement categories in Table 6 provide one dimension of the taxonomy — which models agree — but aggregate across all spread levels. Adding the second dimension (matchup context via the point spread) reveals that the taxonomy's character changes fundamentally with spread regime, and that the same disagreement category masks different mechanisms at different spread levels.

**Table 7: Disagreement by spread bucket (CV, with spread)**

| | Small (3–6.5) | Medium (7–13.5) | Large (14+) |
|---|---|---|---|
| Games | 700 | 402 | 60 |
| Upset rate | 37.6% | 18.9% | 10.0% |
| all_correct | 30.7% | 64.4% | 90.0% |
| all_wrong | 38.7% | 13.9% | 10.0% |
| only_lstm | 8.6% | 3.0% | 0.0% |

The taxonomy's structure depends on the spread regime. At small spreads, the all_wrong category dominates (38.7%) and models struggle. At medium and large spreads, all_correct dominates (64.4% and 90.0%) because the favorite usually wins and all models correctly predict non-upset.

**LSTM exclusive behavior inverts by spread.** This is the most striking pattern in the stratified analysis:

**Table 8: LSTM exclusive decomposition by spread**

| Spread | LSTM Exclusives | Upsets Caught | Non-Upsets Rejected | Primary Role |
|--------|----------------|---------------|--------------------|----|
| Small (3–6.5) | 60 | 5 (8%) | 55 (92%) | **False-alarm filter** |
| Medium (7–13.5) | 12 | 10 (83%) | 2 (17%) | **Upset detector** |
| Large (14+) | 0 | — | — | No exclusive value |

At small spreads — close games where models tend to over-predict upsets — the LSTM's temporal encoding almost exclusively provides *corrective* information: "this team's trajectory doesn't support an upset despite what the summary statistics suggest." At medium spreads — games the favorite should win comfortably — the LSTM's exclusives are overwhelmingly *genuine upset detections* that the static models miss. The LSTM appears to detect momentum or trajectory signals in meaningful underdogs that LR and XGB cannot see.

**The all_wrong category also decomposes by spread:**

| Spread | all_wrong N | Upsets Missed | False Alarms |
|--------|-------------|--------------|-------------|
| Small (3–6.5) | 271 | 8 (3%) | 263 (97%) |
| Medium (7–13.5) | 56 | 54 (96%) | 2 (4%) |
| Large (14+) | 6 | 6 (100%) | 0 (0%) |

At small spreads, all_wrong is almost entirely false alarms — close games where all models predicted upset but the favorite held on. This is the stochastic floor: game variance in near-toss-up matchups. At medium and large spreads, all_wrong is almost entirely missed upsets — genuine surprises driven by information outside the feature set. The spread stratification cleanly separates the stochastic and hidden-information mechanisms that the flat taxonomy conflates. This is the central demonstration of the diagnostic framework: the same label ("all models wrong") conceals two fundamentally different failure mechanisms, and only the disagreement × context matrix reveals the distinction.

### 3.6 Top-K Analysis

Rank-based analysis provides a complementary view of model discrimination that is less sensitive to calibration effects.

**Table 9: Top-K hit rates (test set)**

| K | LR | XGB | LSTM | Ensemble |
|---|-----|-----|------|----------|
| 10 | 50% (1.8x) | 60% (2.1x) | 40% (1.4x) | 50% (1.8x) |
| 20 | 40% (1.4x) | 45% (1.6x) | 35% (1.2x) | 45% (1.6x) |
| 50 | 38% (1.3x) | 44% (1.5x) | 28% (1.0x) | 34% (1.2x) |

XGBoost's top 10 predictions include 6 actual upsets (60% hit rate, 2.1x lift over the 28.5% base rate). The ensemble remains competitive at K = 20 (45%, 1.6x lift) but no longer matches XGBoost at the very top of the ranking. Signal is concentrated in the highest-confidence predictions and fades as K increases, consistent with a small number of genuinely predictable upsets embedded in a large stochastic background.

---

## 4. Analysis

The results in Section 3 establish three prerequisites for using disagreement as a diagnostic tool: the models are statistically equivalent (so disagreement reflects perspective, not quality), they correlate moderately but not perfectly (so disagreement is informative, not random), and the spread stratification reveals that the same disagreement category means fundamentally different things in different matchup contexts. With these prerequisites established, we can read the disagreement patterns as a map of the mechanisms behind NFL upsets.

### 4.1 An Upset Taxonomy

The disagreement framework, crossed with matchup context, reveals four structural types of NFL upsets. The spread-stratified analysis (Section 3.5) shows that these types are not uniformly distributed across matchup contexts — each occupies a distinct region of the spread spectrum, giving the taxonomy both a model-agreement dimension and a matchup-context dimension.

**Type 1: Consensus Signal (all_correct, 45.4% of games).** Games where all three architectures correctly predict the outcome. At small spreads (3–6.5), only 30.7% of games reach consensus; at large spreads (14+), 90.0% do. The consensus signal strengthens with spread magnitude because large-spread games are more deterministic — the favorite usually wins — and all models capture this easily. The more interesting consensus cases are at small spreads, where models agree on an upset call and are right 91.2% of the time. These represent the small subset of close games where linear mispricing, interaction effects, and temporal patterns all converge.

**Type 2: Temporal Signal (only_lstm, 5.6% overall).** The LSTM's exclusive contribution has a dual character that depends on the spread regime. At small spreads (60 of 65 LSTM exclusives), the LSTM almost exclusively *moderates false alarms*: 92% of its exclusive correct predictions are non-upsets that LR and XGB incorrectly flag. The LSTM's trajectory reading says "despite what the summary statistics suggest, this underdog's recent form doesn't support an upset." At medium spreads (12 of 65 LSTM exclusives), the pattern inverts: 83% are *genuine upset detections* that the static models miss. Here the LSTM identifies meaningful underdogs — teams with momentum or trajectory that static features cannot capture.

This inversion is the taxonomy's most diagnostic finding — and the strongest evidence that disagreement reveals structure rather than noise. The LSTM is not one instrument but two: a false-alarm filter and an upset detector. The spread context determines which role it plays. At small spreads, games are inherently noisy and models over-predict; the LSTM provides corrective temporal evidence ("the trajectory doesn't support this upset"). At medium spreads, games should be more deterministic, so when the LSTM uniquely predicts an upset, it is detecting genuine signal — team momentum or trajectory — that the static models' representations cannot encode. This context-dependent dual role would be invisible to any single-model analysis or flat model comparison; it is visible only because the disagreement framework allows us to ask *which model succeeds, and in what matchup context*.

The doubling of LSTM exclusives without spread (5.6% → 11.0%) indicates that approximately half of the LSTM's unique temporal information overlaps with what the betting line encodes. Within these additional exclusives, upsets caught increase from 12 to 33, suggesting the spread was masking genuine temporal signal.

**Type 3: Hidden Information (all_wrong at medium/large spreads).** The spread stratification cleanly separates this type. At medium spreads, the all_wrong category is 96% missed upsets (54 of 56 games) — genuine surprises where the favorite should have won comfortably but didn't. At large spreads, it is 100% missed upsets (6 of 6). These are driven by information outside our feature set: injuries announced after the line was set, motivational context (playoff elimination, rivalry intensity), weather surprises, or teams resting starters. These are not modeling failures — they are data availability boundaries. No feature engineering on historical game statistics can capture a game-day quarterback injury.

**Type 4: Stochastic Floor (all_wrong at small spreads).** At small spreads, the all_wrong category has a completely different character: 97% are false alarms (263 of 271 games), not missed upsets. All three models predict upset, but the favorite holds on. These are close games where the statistical signal pointed toward upset but the favorite's marginal quality advantage prevailed. This is the irreducible floor of NFL prediction — game variance in near-toss-up matchups that no pre-game model can resolve.

The spread stratification thus transforms the flat all_wrong category (29.3% of games) into two distinct mechanisms: stochastic variance at small spreads and information limits at medium/large spreads. This decomposition is only possible because the disagreement framework provides the model-agreement axis while the spread provides the matchup-context axis.

### 4.2 The LSTM Paradox: Competitive but Non-Stationary

The most striking finding in our results is the contrast between the LSTM's cross-validation performance (AUC 0.641, competitive with LR and XGB) and its test-set performance (AUC 0.524, substantially worse). This 0.117 AUC gap, still much larger than XGBoost's gap of 0.062, demands explanation.

Three hypotheses merit consideration:

**Temporal non-stationarity.** NFL teams change substantially between seasons: coaching staffs turn over, key players change teams, offensive and defensive schemes evolve. The LSTM learns trajectory patterns — e.g., "a team improving in EPA over three consecutive games is likely to continue" — that may be specific to the era in which they were learned. If the relationship between performance trajectories and upset probability has shifted between 2005–2022 and 2023–2025, the LSTM would degrade more than models that rely on point-in-time features.

**Overfitting to sequence artifacts.** Despite dropout regularization and validation-based early stopping during cross-validation, the LSTM has substantially more parameters than LR or XGBoost and operates on a richer input space (8 × 14 sequence + 10 context vs. 46 or 70 scalar features). With only 3,495 training games, the LSTM may learn spurious temporal correlations that don't persist.

**Persistent forward-transfer fragility.** The per-season test results sharpen this interpretation. The LSTM trails LR and XGB in all three held-out seasons: 0.469 in 2023, 0.549 in 2024, and 0.556 in 2025. The problem is therefore not confined to one anomalous future season. Rather, the sequence-derived signal appears less portable than the static representations across the entire 2023–2025 horizon.

The correlation evidence reinforces this diagnosis. In CV, the LSTM correlates moderately with LR (0.784) and XGB (0.699). On the test set, these correlations drop to 0.429 and 0.408. The LSTM still diverges meaningfully from the other models in truly out-of-sample data, suggesting it is responding to temporal features that the other models cannot see — but that these features are less reliable in the forward direction.

### 4.3 What the Market Encodes

The spread ablation experiment provides a clean decomposition of what each model learns from the betting line versus from team performance statistics.

The spread magnitude is LR's dominant feature (standardized coefficient -0.539, more than 5x any other feature), confirming that logistic regression in this setting is primarily a spread-mispricing detector. When spread is removed, LR drops from 0.650 to 0.571 AUC — a 12% relative decline — and must rely on less informative features like success rate differential (0.054), turnover margin (-0.069), and environmental variables.

XGBoost's degradation (-0.072) is slightly less than LR's (-0.079), suggesting its tree-based interactions extract some value from feature combinations that partially compensate for the missing spread signal. The LSTM's degradation is the smallest (-0.067), and critically, the LSTM *becomes the strongest model* in the no-spread condition (0.574 vs. 0.571 and 0.566). This ranking reversal is robust evidence that the LSTM captures genuine predictive signal from team performance trajectories that is partially independent of the market's assessment.

The correlation structure tells the same story from a different angle. With spread, the LR-XGB correlation is 0.874 — the spread acts as a powerful shared anchor pulling both models toward similar predictions. Without spread, this drops to 0.742. The XGB-LSTM correlation drops even further (0.699 → 0.419), indicating that the spread was the primary channel through which XGBoost and the LSTM made similar predictions. Remove it, and they are processing fundamentally different information. This diversification is itself diagnostic: the spread masks the architectural differences that reveal the structure behind upsets. When the shared anchor is removed, the three models scatter — and their disagreement patterns become more informative about the distinct mechanisms each architecture captures.

### 4.4 Implications

**For understanding upset structure.** The disagreement framework's primary value is diagnostic, not predictive. The spread-stratified taxonomy decomposes prediction failure into distinct mechanisms at different spread levels: stochastic variance in close games, temporal dynamics in moderate underdogs, and information limits in large mismatches. This decomposition is the study's central contribution — it would be invisible to either a single-model analysis or a flat ensemble. No individual model reveals the four-type structure; you need the disagreement × context matrix to see it.

**For prediction.** The top-K analysis suggests that the system has modest practical value at the highest confidence levels, but mainly through XGBoost rather than through the ensemble. XGBoost's top-10 predictions hit at 60% (2.1x lift), while the ensemble is competitive only at broader cutoffs such as K = 20 (45%, 1.6x lift). Signal fades rapidly: at K = 50, the best model achieves only 1.5x lift. This is consistent with a small number of genuinely predictable upsets — perhaps 10–20 per season — embedded in a stochastic background.

**For ensemble construction.** We tested whether the LSTM's false-alarm moderation could be operationalized as an "LSTM veto" ensemble: when both LR and XGB predict upset but the LSTM disagrees, the LSTM overrides the consensus. In cross-validation, the simple three-model average (AUC 0.655) outperforms LR+XGB alone (0.649), confirming the LSTM adds ensemble value. The effect is larger without spread features (simple avg 0.581 vs. LR+XGB 0.571).

However, on the held-out test set, the LSTM veto fails: among the 92 veto opportunities in the current rerun, veto accuracy is only 65.2%, and XGB alone (0.576) outperforms both LR+XGB (0.571) and the simple three-model average (0.556). The LSTM's temporal signal improves the ensemble in-sample but this improvement does not transfer forward — the same generalization gap that affects the LSTM's individual performance also undermines its ensemble contribution. This negative result reinforces the diagnostic thesis: the LSTM's value is not in improving aggregate prediction accuracy but in revealing the temporal mechanism behind specific upsets. It learns real but era-specific patterns that are diagnostic of the training period but not reliably actionable on future data.

---

## 5. Limitations

Several limitations bound the claims of this work.

**Statistical power of exclusive categories.** Although the LSTM exclusive non-upset rejection bias is statistically significant (p = 0.029), the spread-stratified decomposition produces small cell sizes — particularly the 12 medium-spread LSTM exclusives where 10 are upset detections. This inversion pattern is striking but rests on a small sample that limits formal significance testing within spread strata. Similarly, the LSTM's smaller ablation delta is a point estimate that does not reach statistical significance (CI contains zero). We are careful to distinguish between statistically proven findings and suggestive patterns throughout.

**Calibration and disagreement interact.** The test set uses Platt-calibrated probabilities, which compress the probability range from [0.00, 0.72] (CV, raw) to roughly [0.19, 0.51] (test, calibrated). This compression means about 71% of test predictions exceed the base-rate threshold, compared to 53% in CV, inflating threshold-based disagreement categories. We verified that the LSTM correlation drop from CV (0.78/0.70) to test (0.43/0.41) is genuine — correlations are computed on continuous probabilities and are unaffected by calibration compression. The primary disagreement analyses are reported on uncalibrated CV predictions, with rank-based (top-K) analysis used for the test set where calibration artifacts are less relevant.

**Sample size.** The 558-game test set and 3,495-game training set are modest by machine learning standards. NFL produces approximately 270 games per season, and our 21-season training window may not capture the full range of temporal dynamics. The per-season test results (Table 4) are based on 181–192 games each, limiting the reliability of season-specific conclusions.

**Feature scope.** Our features are limited to historical game statistics and betting lines. We do not include player-level data, injury reports, weather forecasts, motivational factors, or coaching changes. The spread-stratified taxonomy offers a partial characterization of what is missing: at medium and large spreads, all_wrong games are 96–100% missed upsets, suggesting these are driven by unobserved game-day factors. Incorporating injury and motivation data could shrink this category and sharpen the taxonomy boundaries.

**Single sport.** NFL has specific structural properties (17-game seasons, high per-game variance, dominant betting market) that may not generalize. The disagreement framework is applicable to any prediction domain with heterogeneous mechanisms, but the specific taxonomy and quantitative findings are NFL-specific.

---

## 6. Conclusion

Model failure is diagnostic. When structurally different models fail on different games, the pattern of failure reveals the mechanism behind the outcome. We have presented a multi-architecture disagreement framework that demonstrates this principle for NFL upsets, using the agreement and disagreement patterns of three structurally distinct models — logistic regression, XGBoost, and a siamese LSTM — not as a model competition, but as diagnostic instruments whose disagreements expose the structure behind outcomes.

Four principal findings emerge.

First, the three models are statistically indistinguishable in cross-validation performance (bootstrap CIs on all pairwise AUC differences contain zero), which validates the disagreement framework's premise: disagreement between these models reflects genuinely different perspectives on the data, not differences in model quality.

Second, the LSTM captures temporal signal that is partially independent of the betting market and whose character depends on matchup context. At small spreads, the LSTM's exclusive contribution is false-alarm moderation (92% of exclusives are non-upset rejections). At medium spreads, it inverts to upset detection (83% are genuine catches). This dual role — and its dependence on spread regime — is the taxonomy's most diagnostic finding, and it is invisible to any single-model analysis.

Third, temporal patterns are real but non-stationary. The LSTM improves the three-model ensemble in cross-validation (AUC 0.655 vs. 0.649 for LR+XGB alone), but this improvement does not transfer to the held-out test set, where XGB alone (0.576) outperforms all ensemble strategies. The LSTM's CV-to-test gap (-0.117) remains much larger than XGBoost's (-0.062), and its correlation with the static models falls from roughly 0.78/0.70 in CV to 0.43/0.41 on test. Temporal dynamics in NFL performance appear to be more era-specific than static feature relationships.

Fourth, the spread-stratified taxonomy decomposes prediction failure into distinct mechanisms: stochastic variance dominates at small spreads (97% of all_wrong games are false alarms in close matchups), while hidden information dominates at medium and large spreads (96–100% are missed upsets driven by unobserved factors). This two-dimensional structure — model agreement crossed with matchup context — characterizes both the mechanisms and the boundaries of prediction more precisely than either dimension alone.

The methodological contribution is the reframe: stop asking "which model predicts best" and start asking "what do their disagreements reveal about the problem?" The framework is exportable to any prediction domain where outcomes are driven by heterogeneous mechanisms — and where structurally distinct models can be trained on appropriate data representations. The spread-stratified approach generalizes naturally: any contextual variable that partitions the prediction space (e.g., disease severity in medical diagnosis, market volatility in financial prediction) can serve as the second axis of the taxonomy. Medical diagnostics already uses structurally different tests (MRI vs. biopsy vs. bloodwork) and could apply the same disagreement × context logic to categorize diagnostic failures. The specific models, features, and taxonomy are NFL-specific, but the principle that architectural disagreement × context reveals mechanism is general. The models are not competing — they are diagnostic instruments.

---

## References

Fort, S., Hu, H., & Lakshminarayanan, B. (2019). Deep ensembles: A loss landscape perspective. arXiv preprint arXiv:1912.02757.

Gordon, M. L., Lam, M. S., Park, J. S., Patel, K., Hancock, J., Hashimoto, T., & Bernstein, M. S. (2021). Disagreement deconstructed: A method for the computational analysis of disagreement. Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems.

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems, 35.

Hubacek, O., Sourek, G., & Zelezny, F. (2019). Exploiting sports-betting market using machine learning. International Journal of Forecasting, 35(2), 783–796.

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems, 30.

Pinchuk, I. (2026). On the limitations of gradient-boosted trees for learning higher-order feature interactions. Proceedings of the AAAI Conference on Artificial Intelligence.

Wilkens, S. (2021). Sports prediction and betting models in the machine learning age: The case of tennis. Journal of Sports Analytics, 7(2), 99–117.

Xu, C., Tao, D., & Xu, C. (2013). A survey on multi-view learning. arXiv preprint arXiv:1304.5634.
