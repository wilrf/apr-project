# Claim–Evidence Reconciliation Audit

- Doc Type: Audit / Reconciliation
- Date: 2026-06-10
- Status: Complete (corrections applied)
- Author: Task 12 (autonomous claim–evidence reconciliation)

## Purpose

Audit every quantitative and interpretive claim in the in-repo docs against the
saved `results/` prediction artifacts, then apply **minimal, evidence-grounded**
corrections so the docs match the artifacts. No models were re-run; all metrics
were recomputed from the existing CSVs using `src/evaluation/metrics.py`
(`safe_roc_auc_score`, `safe_log_loss`) and `sklearn.metrics.brier_score_loss`
(as the pipeline does).

**Scope of docs audited:** `README.md`, `docs/paper.md`,
`docs/architecture-and-analysis.md`, `docs/model-cards/*.md`.

## Authoritative source of truth

The **primary prediction CSVs are canonical** because they are the raw saved model
outputs:

- `results/ab_experiment/predictions_with_spread.csv` (CV, 1,162 rows)
- `results/ab_experiment/predictions_without_spread.csv` (CV no-spread, 1,162 rows)
- `results/test/predictions.csv` (held-out test, 558 rows)
- `results/ab_experiment/lr_coefs_with_spread.json`

`results/audit_computed.json` is a faithful recomputation of these CSVs and was
independently reproduced during this audit (every headline figure matched to 6
decimal places). Two prior audits reached the same conclusions:
`docs/development/audit_results.md` (2026-04-20) and
`docs/2026-03-15-paper-rewrite-audit-results.md`.

### FLAGGED: stale derived markdown artifacts (not corrected — out of doc scope)

Two **derived** markdown artifacts in `results/` were generated from an *earlier,
different LSTM training run* and do **not** reproduce from the current prediction
CSVs. They are out of scope for editing (the task forbids touching `results/`
artifact numbers), but they are flagged here because several doc claims were
(incorrectly) sourced from them:

- `results/ab_experiment/significance_and_analysis_2026-03-10.md` — reports LSTM
  exclusive `12/65`, non-upset rejection `53/65`, and binomial `p = 0.029`.
  **The current CSV gives `15/72`, `57/72`, and `p = 0.061` (not significant).**
- `results/ab_experiment/full_results_2026-03-09T1430.md` — reports LSTM CV AUC
  `0.6407`, no-spread LSTM AUC `0.5739`, and the `65 (12/53)` exclusive count.
  **The current CSV gives `0.6372`, `0.5682`, and `72 (15/57)`.**

Recommendation for a future task with `results/` write scope: regenerate these two
derived markdown summaries from the current CSVs, or annotate them as superseded.

## Resolution of the two named open items

### 1. XGBoost `max_depth` (model card said 6)

**Determination: the model is trained with `max_depth = 2` in every production
path. The model-card value of `6` was WRONG** (it cited the wrapper's constructor
default, not the value actually used).

Evidence:

- `src/models/xgboost_model.py:22` — constructor *default* `max_depth: int = 6`
  (a library-style default, never used directly in training).
- `src/models/unified_trainer.py:205–210` — `self.xgb_params` default dict sets
  `"max_depth": 2`. `UnifiedTrainer()` is instantiated with no `xgb_params` by
  `src/models/evaluate_test_set.py:796` and `src/models/run_ab_experiment.py:326`,
  so this `max_depth=2` default is the value used for the test-set and full-AB
  artifacts.
- `src/models/run_ab_experiment.py:194–195` — quick-AB path hardcodes
  `max_depth=2`.

Every saved prediction artifact (`predictions_with_spread.csv`, `predictions.csv`)
was therefore produced with `max_depth=2`. The paper (`docs/paper.md:69`) and the
architecture doc (`docs/architecture-and-analysis.md:149, 241`) already state
`max_depth=2` correctly. Only the XGBoost model card was wrong.

### 2. LSTM headline CV / Test AUC

**Determination: the artifact-backed values are CV AUC `0.6372` and Test AUC
`0.5263`** (recomputed directly from the current CSVs; matches
`results/audit_computed.json`).

- `MEMORY.md` (0.6407 / 0.5202) and the stale derived artifacts are NOT supported
  by the current CSVs.
- `docs/verified-numbers.md` (0.5202 / 0.5240 / Brier 0.2072) is stale and was NOT
  edited (it is a development extraction snapshot, out of audit scope, and is
  internally labeled as a point-in-time extraction).
- README and the model cards already used `0.6372` / `0.5263` / Brier `0.2089` —
  **confirmed correct, no change needed.**
- The paper and architecture doc used the stale `0.641` / `0.524` (and `0.5202`) —
  **corrected to `0.637` / `0.526` (`0.6372` / `0.5263`).**

## Recomputed canonical values (from current CSVs)

| Metric | LR | XGB | LSTM |
|---|---|---|---|
| CV AUC | 0.6497 | 0.6377 | **0.6372** |
| CV Brier | 0.1974 | 0.1991 | **0.1997** |
| Test AUC | 0.5622 | 0.5755 | **0.5263** |
| Test Brier | 0.2026 | 0.2013 | **0.2089** |
| No-spread CV AUC | 0.5707 | 0.5662 | **0.5682** |
| Ablation delta | −0.0790 | −0.0715 | **−0.0690** |

No-spread ranking (from CSV): **LR (0.571) > LSTM (0.568) > XGB (0.566)** — the
LSTM is the no-spread *runner-up*, it does **not** win the ablation.

CV correlations: LR-XGB 0.874, LR-LSTM **0.764**, XGB-LSTM **0.674**.
Test correlations: LR-XGB 0.878, LR-LSTM **0.373**, XGB-LSTM **0.309**.

CV disagreement at base-rate threshold (0.2969): all_correct 528, all_wrong **333**,
only_lr **31**, only_xgb 48, only_lstm **72** (15 upsets / 57 non-upsets),
lr_xgb 78, lr_lstm **45**, xgb_lstm 27. Sum = 1162.

LSTM-exclusive one-sided binomial (15/72 vs 0.2969) **p = 0.061 (not significant)**.
Spread buckets of only_lstm: small 60 (5 caught / 55 rejected), medium 12 (10 / 2),
large 0 → totals 72 and 15. (The per-bucket numbers were already correct; only the
flat total `65` and split `12/53` were stale.)

LSTM-exclusive without spread: 125 games (44 upsets / 81 non-upsets); 10.76% of CV.
Per-season test LSTM AUC: 2023 **0.443**, 2024 **0.534**, 2025 **0.592**
(`results/test/report.md` confirms 0.443 / 0.534 / 0.592).

Ensemble (confirmed): CV avg3 0.6553 (~0.655), CV LR+XGB 0.6488 (~0.649),
no-spread avg3 0.5809 (~0.581), no-spread LR+XGB 0.5712 (~0.571).
XGB Top-10 test = 6/10 = 60%, lift 2.106 (~2.1x). LR coef `spread_magnitude` =
−0.5388 (~−0.539), 6.98x the next feature.

## Findings table

Verdicts: **Confirmed** = matches artifact; **Imprecise** = wording/rounding
overstates or misreads; **Wrong** = number/claim contradicts artifact;
**Flagged** = believed off but not cleanly groundable, left as-is.

| Claim | Location | Artifact | Recomputed | Verdict | Action |
|---|---|---|---|---|---|
| Train 3,495 labeled, 30.36%; Test 558, 28.49% | README §Data; all cards; paper §2.1; arch §8 | audit_computed.json / train.csv / test.csv | 3495/0.3036, 558/0.2849 | Confirmed | none |
| Feature counts LR 46 / XGB 70 / LSTM 14×8+10 (no-spread 42/66/8) | README; cards; paper §2.2; arch §4 | pipeline.py constants | matches | Confirmed | none |
| CV AUC LR 0.6497 / XGB 0.6377 | README; cards; paper T1; arch §8.1 | predictions_with_spread.csv | 0.6497 / 0.6377 | Confirmed | none |
| **CV LSTM AUC 0.641** | paper Abstract, T1, T3, §3.1, §4.2; arch §8.1, §8.3, §8.4, Type 2 | predictions_with_spread.csv | **0.6372** | **Wrong** | → 0.637 / 0.6372 |
| CV LSTM AUC 0.6372 | README; lstm card | predictions_with_spread.csv | 0.6372 | Confirmed | none |
| Test AUC XGB 0.5755 / LR 0.5622 | README; cards; paper T2; arch §8.2 | predictions.csv | 0.5755 / 0.5622 | Confirmed | none |
| **Test LSTM AUC 0.524 / 0.5202** | paper Abstract, T2, T3, §4.2; arch §8.2 (0.5202), §8.3 (0.520) | predictions.csv | **0.5263** | **Wrong** | → 0.526 / 0.5263 |
| Test LSTM AUC 0.5263 | README; lstm card | predictions.csv | 0.5263 | Confirmed | none |
| **Test LSTM Brier 0.2072 / 0.2051** | arch §8.2 (0.2072) | predictions.csv | **0.2089** | **Wrong** | → 0.2089 |
| Test LSTM Brier 0.2089 | lstm card; README implied | predictions.csv | 0.2089 | Confirmed | none |
| **CV-test gap LSTM −0.117 / −0.121** | paper T3 (−0.117), §4.2 (0.117); arch §8.3 (−0.121) | computed | **−0.1109** | Imprecise/Wrong | → −0.111 (paper), −0.111 (arch) |
| CV-test gap LR −0.088 / XGB −0.062 | paper T3; arch §8.3 | computed | −0.0875 / −0.0622 | Confirmed | none |
| **CV corr LR-LSTM 0.784 / XGB-LSTM 0.699** | paper §3.1, §3.3, §4.2, §4.3, §5; arch §8.1 | predictions_with_spread.csv | **0.764 / 0.674** | **Wrong** | → 0.764 / 0.674 |
| **Test corr LR-LSTM 0.429 / XGB-LSTM 0.408** | paper §3.2, §4.2, §5; lstm card uses 0.373/0.309 | predictions.csv | **0.373 / 0.309** | **Wrong** (paper) | → 0.373 / 0.309 |
| **Test corr LR-LSTM 0.311 / XGB-LSTM 0.273** | arch §8.2, §8.3 | predictions.csv | **0.373 / 0.309** | **Wrong** (arch) | → 0.373 / 0.309 |
| Test corr LR-LSTM 0.373 / XGB-LSTM 0.309 | README; lstm card | predictions.csv | 0.373 / 0.309 | Confirmed | none |
| No-spread CV AUC LR 0.571 / XGB 0.566 | README; cards; paper T5; arch §8.4 | predictions_without_spread.csv | 0.5707 / 0.5662 | Confirmed | none |
| **No-spread CV LSTM AUC 0.574 / 0.5739** | paper Abstract, T5, §3.3, §4.3; arch §8.4, Type 2 | predictions_without_spread.csv | **0.5682** | **Wrong** | → 0.568 |
| **"LSTM wins / becomes strongest without spread"** | paper Abstract, §3.3, §4.3; arch §8.4 ("LSTM wins"), Type 2 | predictions_without_spread.csv | rank = **LR 0.571 > LSTM 0.568 > XGB 0.566** | **Wrong** | reword: LSTM degrades least but does NOT win; LR remains top |
| Ablation deltas LR −0.079 / XGB −0.072 / LSTM −0.067 | README; cards; paper T5; arch §8.4 | both CSVs | −0.0790/−0.0715/−0.0690 | Confirmed | none |
| **only_lstm = 65 (12 upsets / 53 non-upsets)** | paper §3.4 T6, §3.4 prose, §4.1 Type2, §5; arch §8.5, Type 2 | predictions_with_spread.csv @ base-rate thr | **72 (15 / 57)** | **Wrong** | → 72 (15 / 57) |
| only_lstm 5.6% of games | paper T6, §3.3, §4.1; arch §8.4, §8.5 | CSV | 6.20% | Imprecise | → 6.2% |
| LSTM-exclusive % without spread 11.0% | README; paper §3.3, §4.1; arch §8.4 | CSV | 10.76% | Confirmed (rounds 11.0%) | none |
| **Binomial p = 0.029, "statistically significant"** | paper §3.4, §5 | CSV @ base-rate thr | **p = 0.061, NOT significant** | **Wrong** | → p = 0.061, soften to "not significant" |
| **only_lstm upset rate 18.5%** | paper §3.4 (×2), §4.1; arch §8.5 (18.5%) | CSV | **20.8%** | **Wrong** | → 20.8% |
| **Disagreement Table 6 counts** all_wrong 340 / only_lr 28 / lr_lstm 48 | paper §3.4 T6; arch §8.5 (340/28/48) | CSV @ base-rate thr | **333 / 31 / 45** | **Wrong** | → 333 / 31 / 45 |
| all_wrong upset rate 20.9% | paper T6 (20.9), §3.4; arch §8.5 (20.9) | CSV | **20.4%** | Imprecise/Wrong | → 20.4% |
| all_correct upset rate 36.7% | paper T6, §3.4; arch §8.5 | CSV | 37.1% | Imprecise | → 37.1% |
| only_xgb 48 (4.1%), lr_xgb 78 (6.7%), xgb_lstm 27 (2.3%) | paper T6; arch §8.5 | CSV | 48 / 78 / 27 | Confirmed | none |
| **LSTM ablation upsets caught "12 → 33"** | paper §3.3, §4.1 | both CSVs @ base-rate thr | **15 → 44** | **Wrong** | → 15 → 44 |
| Spread-bucket only_lstm: small 60 (5/55), medium 12 (10/2), large 0 | paper T8; arch (via §8.5) | CSV | exact match | Confirmed | none |
| all_wrong by bucket: small 271 (8/263), medium 56 (54/2), large 6 (6/0) | paper §3.5 | CSV | match | Confirmed | none |
| Bucket games 700 / 402 / 60; upset rates 37.6/18.9/10.0 | paper T7; verified-numbers | CSV | match | Confirmed | none |
| **Per-season test LSTM 0.469 / 0.549 / 0.556** | paper T4, §3.2, §4.2 | report.md / predictions.csv | **0.443 / 0.534 / 0.592** | **Wrong** | → 0.443 / 0.534 / 0.592 |
| **Per-season test LSTM 0.489 / 0.586 / 0.493** | arch §8.7 | report.md / predictions.csv | **0.443 / 0.534 / 0.592** | **Wrong** | → 0.443 / 0.534 / 0.592; reword "best in 2024" claim |
| Per-season LR/XGB 0.512/0.521, 0.552/0.554, 0.617/0.639 | paper T4; arch §8.7 | predictions.csv | match | Confirmed | none |
| XGB Top-10 = 6/10 = 60%, 2.1x lift | README; xgb card; paper §3.6, §4.4; arch §8.6 | predictions.csv | 6/10, 2.106x | Confirmed | none |
| Top-K test table (K=10/20/50) | paper T9; arch §8.6; report.md | predictions.csv | matches report.md (LSTM K10=4/10) | Confirmed (paper); arch §8.6 LSTM K10=3/10 | arch §8.6 LSTM 3/10→4/10, ensemble K20 to match report |
| LR spread coef −0.539, >5x next | paper §3.3?, §4.3 | lr_coefs_with_spread.json | −0.5388, 6.98x | Confirmed | none |
| Ensemble CV avg3 0.655 / LR+XGB 0.649 | paper §4.4, §6 | CSV | 0.6553 / 0.6488 | Confirmed | none |
| Ensemble no-spread avg3 0.581 / LR+XGB 0.571 | paper §4.4 | CSV | 0.5809 / 0.5712 | Confirmed | none |
| Test ensemble avg3 "0.556" | paper §4.4 | CSV | 0.5548 (~0.555) | Imprecise (±0.001) | left as-is (within rounding; qualitative claim holds) |
| **LSTM veto test "92 opportunities, 65.2% accuracy"** | paper §4.4 | not a single canonical artifact (veto semantics vary) | my recompute: 94 / 68.1%; significance artifact: 68/63.2% | **Flagged** | **left as-is** (cannot ground to one artifact; qualitative claim "veto fails on test" holds under every definition) |
| Baseline Brier 0.204 / 0.2038 | paper §2.4, T2; arch §8.2 | calculate_baseline_brier(0.2849) | 0.2038 | Confirmed | none |
| XGB max_depth = 6 | xgboost.md:9 | unified_trainer/run_ab_experiment | **2** in all training paths | **Wrong** | → max_depth = 2 |
| XGB max_depth = 2 | paper §2.3; arch §4.3, §5 | trainers | 2 | Confirmed | none |
| LSTM hidden 64, layers 3, dropout 0.25 | paper §2.3; arch §4.4 | lstm_config.py | matches | Confirmed | none |

## Summary of edits applied

All edits propagate the **same** artifact-backed value to every doc that repeats
it. See git diff for exact text. Files changed: `docs/paper.md`,
`docs/architecture-and-analysis.md`, `docs/model-cards/xgboost.md`. README and the
LR/LSTM model cards required **no changes** (already artifact-correct).

Nothing in `results/` was modified. No models were re-run.
