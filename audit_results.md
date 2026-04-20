# AP Research Numerical Audit

- Doc Type: Results
- Topic: Numerical Audit
- Topic Slug: numerical-audit
- Date: 2026-04-20
- Status: Complete

## Scope and Source Inventory

Primary numeric sources examined:

- `data/features/train.csv`
- `data/features/test.csv`
- `results/ab_experiment/predictions_with_spread.csv`
- `results/ab_experiment/predictions_without_spread.csv`
- `results/test/predictions.csv`
- `results/ab_experiment/lr_coefs_with_spread.json`
- Current markdown summaries in `docs/` and `results/` for consistency checks
- Presentation sources: `docs/AP_Research_POD_Revised.pptx`, `docs/AP_Research_POD_Corrected.pptx`, and `tools/presentation/rewrite_presentation.py`

No `.ipynb` notebooks are present in the repo outside ignored virtualenv/cache directories. `AP_Research_Slides_Content.md` is not present, so slide reconciliation uses the latest PowerPoint artifact (`docs/AP_Research_POD_Revised.pptx`, 20 slides) and its generator (`tools/presentation/rewrite_presentation.py`).

All current model metrics below are recomputed from saved prediction CSVs, not copied from prose summaries. Bootstrap intervals use 10,000 paired nonparametric resamples.

## 1. Dataset Composition

Computed at `tools/numerical_audit_compute.py:245` from `data/features/train.csv` and `data/features/test.csv`.

| Split | Total rows | Labeled games | Upsets | Upset base rate | Sub-3 unlabeled | Other unlabeled |
| --- | --- | --- | --- | --- | --- | --- |
| Train (2005-2022) | 4351 | 3495 | 1061 | 30.358% | 849 | 7 |
| Test (2023-2025) | 768 | 558 | 159 | 28.495% | 209 | 1 |

The exact modeling counts are the labeled games: 3,495 training games and 558 test games. Total generated rows are larger because sub-3-spread games remain for rolling-stat continuity.

| Season | Total rows | Labeled games | Upsets | Upset rate |
| --- | --- | --- | --- | --- |
| 2023 | 256 | 185 | 55 | 29.730% |
| 2024 | 256 | 192 | 45 | 23.438% |
| 2025 | 256 | 181 | 59 | 32.597% |

## 2. Model Performance in Cross-Validation

Metrics computed at `tools/numerical_audit_compute.py:121`; bootstrap CIs at `tools/numerical_audit_compute.py:277` from `results/ab_experiment/predictions_with_spread.csv`.

| Model | AUC | AUC 95% CI | Brier | Log loss |
| --- | --- | --- | --- | --- |
| LR | 0.6497 | [0.6158, 0.6828] | 0.1974 | 0.5807 |
| XGB | 0.6377 | [0.6020, 0.6716] | 0.1991 | 0.5855 |
| LSTM | 0.6372 | [0.6026, 0.6707] | 0.1997 | 0.5858 |

Pairwise AUC differences (left minus right):

| Comparison | Point estimate | Bootstrap mean | 95% CI |
| --- | --- | --- | --- |
| LR-XGB | 0.0120 | 0.0121 | [-0.0067, 0.0308] |
| LR-LSTM | 0.0125 | 0.0124 | [-0.0123, 0.0372] |
| XGB-LSTM | 0.0005 | 0.0003 | [-0.0278, 0.0282] |

Pairwise probability correlations:

| Pair | Correlation |
| --- | --- |
| LR-XGB | 0.874225 |
| LR-LSTM | 0.764312 |
| XGB-LSTM | 0.673688 |

Inconsistency flagged: `results/ab_experiment/full_results_2026-03-09T1430.md` reports LSTM CV AUC `0.6407` and LSTM correlations `.784/.699`; the current raw CSV gives LSTM AUC `0.6372` and correlations `0.7643/0.6737`.

## 3. Disagreement Analysis (CV Set)

Computed at `tools/numerical_audit_compute.py:153` using the documented single global CV base-rate threshold `0.296902`. Total CV games: 1162; upsets: 345; base rate: 29.690%.

| Category | N | % games | Upsets | Non-upsets | Upset rate |
| --- | --- | --- | --- | --- | --- |
| all_correct | 528 | 45.44% | 196 | 332 | 37.12% |
| all_wrong | 333 | 28.66% | 68 | 265 | 20.42% |
| only_lstm | 72 | 6.20% | 15 | 57 | 20.83% |
| only_lr | 31 | 2.67% | 7 | 24 | 22.58% |
| only_xgb | 48 | 4.13% | 11 | 37 | 22.92% |
| lr_xgb | 78 | 6.71% | 21 | 57 | 26.92% |
| lr_lstm | 45 | 3.87% | 17 | 28 | 37.78% |
| xgb_lstm | 27 | 2.32% | 10 | 17 | 37.04% |

Category count check: 1162 / 1162.
LSTM-exclusive skew test: 15/72 upsets (20.83%) versus CV base rate 29.69% gives one-sided binomial p = `0.061332`. The old p = `0.029` is tied to the stale 65-game count.

Important threshold inconsistency: the saved CV `lr_pred`/`xgb_pred`/`lstm_pred` columns were written with fold-specific thresholds, not the single global CV threshold used above. Those stored columns give `only_lstm = 62` rather than 72.

## 4. LSTM Spread Bucket Analysis

Computed at `tools/numerical_audit_compute.py:390`. Buckets: small 3-6.5, medium 7-13.5, large 14+.

### Analysis A: LSTM-Exclusive Correct Predictions

| Bucket | Count | Upsets caught | Non-upsets rejected | Upset share |
| --- | --- | --- | --- | --- |
| small | 60 | 5 | 55 | 8.3% |
| medium | 12 | 10 | 2 | 83.3% |
| large | 0 | 0 | 0 | NA |

Analysis A currently totals 72, not 65. The bucket split is 60 + 12 + 0 = 72, so those bucket numbers match current LSTM-exclusive correct predictions under the documented global threshold.

### Analysis B: LSTM Disagrees With LR+XGB Consensus

| Bucket | LSTM vs consensus N | LSTM right | LSTM wrong | Hit rate |
| --- | --- | --- | --- | --- |
| small | 98 | 60 | 38 | 61.2% |
| medium | 52 | 12 | 40 | 23.1% |
| large | 0 | 0 | 0 | NA |

Analysis B totals 150 (98 small + 52 medium + 0 large). Therefore the 72 number is not an LR+XGB-consensus disagreement count; it is the current LSTM-exclusive-correct count. The 65 number appears only in stale markdown summaries.

## 5. Spread Ablation Results

Computed at `tools/numerical_audit_compute.py:121` and `tools/numerical_audit_compute.py:277` from `predictions_with_spread.csv` and `predictions_without_spread.csv`. Delta is without-spread AUC minus with-spread AUC.

| Model | AUC with spread | AUC without spread | Delta | Delta 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| LR | 0.6497 | 0.5707 | -0.0790 | [-0.1076, -0.0503] | <0.0002 |
| XGB | 0.6377 | 0.5662 | -0.0715 | [-0.1015, -0.0414] | <0.0002 |
| LSTM | 0.6372 | 0.5682 | -0.0690 | [-0.1032, -0.0358] | <0.0002 |

No-spread pairwise probability correlations:

| Pair | Correlation |
| --- | --- |
| LR-XGB | 0.741594 |
| LR-LSTM | 0.515562 |
| XGB-LSTM | 0.372294 |

LSTM-exclusive current global-threshold count: with spread 72/1162 = 6.20%; without spread 125/1162 = 10.76%. The slide claim `5.6% -> 11.0%` is stale for the current raw CSV; current is `6.20% -> 10.76%`.
LSTM-exclusive upsets: with spread 15; without spread 44. The current raw CSV gives `15 -> 44`, not `12 -> 33`.

LR coefficient check (standardized by `StandardScaler` in `src/models/logistic_model.py`):

| Feature | Coefficient | Absolute ratio vs spread |
| --- | --- | --- |
| spread_magnitude | -0.538763 | 1.000 |
| temperature | 0.077182 | 6.980 |
The spread coefficient is `-0.538763`. The next-largest coefficient by absolute value is `temperature` at `0.077182`, so the spread-to-next ratio is `6.980`.

## 6. Out-of-Sample Test Set (2023-2025)

Computed at `tools/numerical_audit_compute.py:121` and `tools/numerical_audit_compute.py:352` from `results/test/predictions.csv` (saved calibrated probabilities).

| Model | Test AUC | Test Brier | Test log loss | CV AUC | CV-test gap |
| --- | --- | --- | --- | --- | --- |
| LR | 0.5622 | 0.2026 | 0.5942 | 0.6497 | 0.0875 |
| XGB | 0.5755 | 0.2013 | 0.5915 | 0.6377 | 0.0622 |
| LSTM | 0.5263 | 0.2089 | 0.6084 | 0.6372 | 0.1109 |

Per-season test AUC:

| Season | N | Upsets | Upset rate | LR AUC | XGB AUC | LSTM AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 | 185 | 55 | 29.73% | 0.5119 | 0.5210 | 0.4435 |
| 2024 | 192 | 45 | 23.44% | 0.5522 | 0.5536 | 0.5344 |
| 2025 | 181 | 59 | 32.60% | 0.6167 | 0.6395 | 0.5921 |

Test pairwise probability correlations:

| Pair | Correlation |
| --- | --- |
| LR-XGB | 0.877627 |
| LR-LSTM | 0.373438 |
| XGB-LSTM | 0.309158 |

Test pairwise AUC differences (left minus right):

| Comparison | Point estimate | 95% CI | p |
| --- | --- | --- | --- |
| XGB-LSTM | 0.0492 | [-0.0145, 0.1117] | 0.135786 |
| LR-LSTM | 0.0359 | [-0.0233, 0.0953] | 0.240376 |
| LR-XGB | -0.0134 | [-0.0405, 0.0133] | 0.306169 |
All current test-set pairwise AUC-difference bootstrap CIs include zero. XGB is best by point estimate, but not significantly better than LSTM or LR on this 558-game test set.

## 7. Top-K Hit Rates on Test Set

Computed at `tools/numerical_audit_compute.py:428`. Ensemble is a simple unweighted average: `(lr_prob + xgb_prob + lstm_prob) / 3`.

Base rate: 159/558 = 28.49%.

| K | LR | XGB | LSTM | Ensemble |
| --- | --- | --- | --- | --- |
| 10 | 5/10 (50%, 1.75x) | 6/10 (60%, 2.11x) | 4/10 (40%, 1.40x) | 6/10 (60%, 2.11x) |
| 20 | 8/20 (40%, 1.40x) | 9/20 (45%, 1.58x) | 7/20 (35%, 1.23x) | 8/20 (40%, 1.40x) |
| 50 | 19/50 (38%, 1.33x) | 22/50 (44%, 1.54x) | 14/50 (28%, 0.98x) | 18/50 (36%, 1.26x) |

## 8. Taxonomy Coverage

Computed at `tools/numerical_audit_compute.py:451` from CV categories and spread buckets.

| Taxonomy type | Definition used | N | % of all CV games | % of all_wrong failures |
| --- | --- | --- | --- | --- |
| Consensus | all_correct | 528 | 45.44% | NA |
| Temporal/LSTM-exclusive | only_lstm | 72 | 6.20% | NA |
| Hidden information | all_wrong at medium/large spreads | 62 | 5.34% | 18.62% |
| Irreducible variance | all_wrong at small spreads | 271 | 23.32% | 81.38% |

Named taxonomy coverage totals 80.29% of CV games. Mixed categories total 19.71%.

Mixed-category upset-rate tests use exact two-sided binomial tests against the CV base rate.

| Mixed category | N | % games | Upsets | Upset rate | p vs base |
| --- | --- | --- | --- | --- | --- |
| lr_xgb | 78 | 6.71% | 21 | 26.92% | 0.710128 |
| only_xgb | 48 | 4.13% | 11 | 22.92% | 0.346333 |
| lr_lstm | 45 | 3.87% | 17 | 37.78% | 0.253863 |
| only_lr | 31 | 2.67% | 7 | 22.58% | 0.438673 |
| xgb_lstm | 27 | 2.32% | 10 | 37.04% | 0.404010 |
No mixed category is statistically distinguishable from the CV base rate at alpha = 0.05. A principled collapse would be to keep them as an explicit `mixed/ambiguous architecture signal` bucket. Folding `lr_xgb` into a static/non-temporal type is defensible, but the other mixed categories combine temporal and static models and should not be forced into the four named types without a new design rule.

## 9. Calibration Details

Probability summaries computed at `tools/numerical_audit_compute.py:508`.

Platt calibration is applied to all three final test-model outputs in `src/models/evaluate_test_set.py`: LR, XGB, and LSTM calibrators are fit on held-out 2021-2022 predictions and applied to 2023-2025 raw test probabilities. The saved `results/test/predictions.csv` contains calibrated probabilities.

Primary CV analyses use uncalibrated CV predictions from `results/ab_experiment/predictions_with_spread.csv`. Spread ablation uses uncalibrated CV predictions from both A/B CSVs. Test metrics, test correlations, and test Top-K in this audit use the saved calibrated test probabilities.

Current saved probability distributions:

| Dataset | Model | Min | Max | Mean | Std |
| --- | --- | --- | --- | --- | --- |
| CV raw | LR | 0.029753 | 0.521571 | 0.296780 | 0.104826 |
| CV raw | XGB | 0.057000 | 0.617653 | 0.299968 | 0.103666 |
| CV raw | LSTM | 0.000358 | 0.724441 | 0.293323 | 0.123062 |
| Test Platt-calibrated | LR | 0.194216 | 0.434911 | 0.321627 | 0.052376 |
| Test Platt-calibrated | XGB | 0.216109 | 0.497659 | 0.317802 | 0.050075 |
| Test Platt-calibrated | LSTM | 0.217904 | 0.547503 | 0.330310 | 0.074778 |

Exact pre-calibration test probability summary stats for the current saved `results/test/predictions.csv` are not recoverable from repo artifacts: raw test probabilities and fitted Platt calibrators were not saved. Existing docs record only old pre-calibration ranges from a rerun, not mean/std. Rerunning `src.models.evaluate_test_set` would create a new stochastic LSTM run, so it would not verify the current saved test file.

## 10. Feature Details

Computed at `tools/numerical_audit_compute.py:522` from feature-pipeline and sequence-builder constants.

| Item | Count/value |
| --- | --- |
| LR features | 46 |
| LR no-spread features | 42 |
| XGB features | 70 |
| XGB no-spread features | 66 |
| LSTM sequence representation | 14 features x 8 timesteps |
| LSTM matchup context | 10 |
| LSTM matchup no-spread context | 8 |
| Rolling window | 3-game |
| XGB per-game lags | 1, 2, 3 |
There is no 5-game rolling window in the canonical pipeline. The flat pipeline uses 3-game rolling means/std/trends and XGB gets last-1/last-2/last-3 per-game lag stats; the LSTM uses 8 prior games per team.

## Cross-File Inconsistency Summary

| Item | Current CSV recomputation | Conflicting file/claim | Status |
| --- | --- | --- | --- |
| LSTM CV AUC | 0.6372 | `results/ab_experiment/full_results_2026-03-09T1430.md`: 0.6407 | Inconsistent |
| CV LSTM correlations | LR-LSTM 0.764, XGB-LSTM 0.674 | `full_results`: .784/.699 | Inconsistent |
| LSTM-exclusive count | 72 (15 upsets, 57 non-upsets) | `full_results` flat table: 65 (12/53) | Inconsistent |
| No-spread LSTM AUC | 0.5682 | `full_results`: 0.5739 | Inconsistent |
| Test LSTM AUC | 0.5263 | `docs/2026-03-15-paper-rewrite-audit-results.md`: 0.5240 | Inconsistent with current test CSV |
| Test LSTM correlations | LR-LSTM 0.373, XGB-LSTM 0.309 | Slides/docs: .429/.408 or .311/.273 | Inconsistent across runs |
