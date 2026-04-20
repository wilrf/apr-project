# Slide Reconciliation

- Doc Type: Results
- Topic: Numerical Audit
- Topic Slug: numerical-audit
- Date: 2026-04-20
- Status: Complete

`AP_Research_Slides_Content.md` was not found in the repo. I reconciled numerical project-result claims against the latest available deck artifact, `docs/AP_Research_POD_Revised.pptx`, and its generator, `tools/presentation/rewrite_presentation.py`. Citation years, journal volume/page numbers, and general definition examples are not treated as project-result claims.

| Slide | Claim | Status | Correct / note | Source |
| --- | --- | --- | --- | --- |
| 3 | Upset = 3+ point underdog wins | CONFIRMED | Pipeline labels only spread >= 3. | src/features/pipeline.py |
| 3 | Upsets happen in roughly 29% of games | CONFIRMED | Train 30.4%, test 28.5%, CV 29.7%. | tools/numerical_audit_compute.py:245 |
| 6 | 18 NFL regular seasons, 2005-2022 for training | CONFIRMED | 3495 labeled training games across 2005-2022. | tools/numerical_audit_compute.py:245 |
| 6 | Hold out 2023-2025 as blind test | CONFIRMED | 558 labeled test games across 2023-2025. | tools/numerical_audit_compute.py:245 |
| 6 | LR 46 stats; XGB 70 features; LSTM raw 8-game sequences | CONFIRMED | LR 46, XGB 70, LSTM 14 x 8 plus 10 matchup context. | tools/numerical_audit_compute.py:522 |
| 6 | 6-fold expanding-window CV | CONFIRMED | Current CV predictions cover 1162 games from validation seasons 2017-2022. | tools/numerical_audit_compute.py:121 |
| 7 | Training: 3,495 games | CONFIRMED | 3,495 labeled train games. | tools/numerical_audit_compute.py:245 |
| 7 | Test: 558 games | CONFIRMED | 558 labeled test games. | tools/numerical_audit_compute.py:245 |
| 7 | Base rate: ~30% upsets | CONFIRMED | Train 30.4%, test 28.5%. | tools/numerical_audit_compute.py:245 |
| 8 | CV validation years 2017-2022, N = 1,162 | CONFIRMED | Current CV prediction CSV has 1162 rows. | tools/numerical_audit_compute.py:121 |
| 8 | Pairwise AUC-difference CIs contain zero | CONFIRMED | All current CV pairwise AUC-difference bootstrap CIs include zero. | tools/numerical_audit_compute.py:277 |
| 8 | LR-XGB correlation .874 | CONFIRMED | Current 0.874. | tools/numerical_audit_compute.py:121 |
| 8 | LSTM correlations .784 with LR and .699 with XGB | INCORRECT | Current LR-LSTM 0.764 and XGB-LSTM 0.674. | tools/numerical_audit_compute.py:121 |
| 8 | XGB-LSTM .699 is the lowest correlation | INCORRECT | XGB-LSTM is still lowest, but current value is 0.674. | tools/numerical_audit_compute.py:121 |
| 9 | CV set 1,162 games, threshold ~0.30 | CONFIRMED | N=1162, threshold=0.297. | tools/numerical_audit_compute.py:153 |
| 9 | 74.7% all-agree categories; 25.3% split | INCORRECT | Current global-threshold all-agree is 861/1162 = 74.1%; split is 25.9%. | tools/numerical_audit_compute.py:153 |
| 9 | 65 LSTM-exclusive games | INCORRECT | Current global-threshold only_lstm is 72. Stored fold-threshold columns give 62. | tools/numerical_audit_compute.py:153 |
| 9 | 53 non-upset rejections (81.5%) and 12 upsets | INCORRECT | Current only_lstm is 57 non-upsets (79.2%) and 15 upsets. | tools/numerical_audit_compute.py:153 |
| 9 | Binomial p = 0.029 | INCORRECT | Current one-sided p for only_lstm upset rate below base rate is 0.061332. | tools/numerical_audit_compute.py:153 |
| 10 | Close games 3-6.5; LSTM right 92% as false-alarm filter | CONFIRMED | Current small-spread only_lstm: 55/60 non-upset rejections = 91.7%. | tools/numerical_audit_compute.py:390 |
| 10 | Medium spreads 7-13.5; catches 83% as real upsets | CONFIRMED | Current medium-spread only_lstm: 10/12 upsets = 83.3%. | tools/numerical_audit_compute.py:390 |
| 10 | LSTM watches last 8 games | CONFIRMED | SEQUENCE_LENGTH = 8. | tools/numerical_audit_compute.py:522 |
| 11 | All spread-ablation drops significant | CONFIRMED | All current bootstrap delta CIs are negative; p < 0.0002 by two-sided sign bootstrap. | tools/numerical_audit_compute.py:277 |
| 11 | Without spread: LSTM .574 > LR .571 > XGB .566 | INCORRECT | Current no-spread AUCs: LR 0.571, LSTM 0.568, XGB 0.566. Ranking is LR > LSTM > XGB. | tools/numerical_audit_compute.py:121 |
| 11 | LSTM degrades least (-.067) | INCORRECT | Current LSTM delta is -0.069; it is still least negative by point estimate. | tools/numerical_audit_compute.py:121 |
| 11 | LSTM smaller delta vs LR is not significant | CONFIRMED | Current delta difference still has uncertainty crossing zero; individual delta CIs are negative. | tools/numerical_audit_compute.py:277 |
| 11 | LSTM-exclusive predictions double 5.6% -> 11.0% | INCORRECT | Current global-threshold values are 6.2% -> 10.8%. | tools/numerical_audit_compute.py:153 |
| 11 | Upsets caught jump 12 -> 33 | INCORRECT | Current global-threshold only_lstm upsets are 15 -> 44. | tools/numerical_audit_compute.py:153 |
| 11 | LR-XGB correlation .874 -> .742 | CONFIRMED | Current 0.874 -> 0.742. | tools/numerical_audit_compute.py:121 |
| 12 | Test base rate = 28.5% | CONFIRMED | Current test base rate 28.5%. | tools/numerical_audit_compute.py:121 |
| 12 | LSTM largest generalization gap (-.117) | INCORRECT | Current LSTM CV-test gap is 0.111; it remains largest. | tools/numerical_audit_compute.py:121 |
| 12 | XGB generalizes best (-.062) | CONFIRMED | Current XGB gap 0.062. | tools/numerical_audit_compute.py:121 |
| 12 | LSTM trails in all three test seasons | CONFIRMED | Current LSTM AUC is below LR and XGB in 2023, 2024, and 2025. | tools/numerical_audit_compute.py:121 |
| 12 | LSTM correlations collapse .784 -> .429 and .699 -> .408 | INCORRECT | Current collapse is 0.764 -> 0.373 and 0.674 -> 0.309. | tools/numerical_audit_compute.py:121 |
| 12 | XGB top-10 hit rate 60%, 2.1x lift | CONFIRMED | Current XGB top 10 is 6/10 = 60.0%, lift 2.11x. | tools/numerical_audit_compute.py:428 |
| 13 | All three agree / structurally readable: 45% of games | CONFIRMED | Current all_correct is 45.4%. | tools/numerical_audit_compute.py:153 |
| 13 | Only LSTM sees it: 6% of games | CONFIRMED | Current only_lstm is 6.2%. | tools/numerical_audit_compute.py:153 |
| 13 | Temporal signal statistically confirmed | AMBIGUOUS | The current overall only_lstm skew p-value is 0.0613, not <0.05; bucket inversion remains descriptive with small cells. | tools/numerical_audit_compute.py:153 |
| 14 | LSTM AUC .641 tied with LR .650 and XGB .638 | INCORRECT | Current AUCs: LR 0.650, XGB 0.638, LSTM 0.637. Statistical tie confirmed. | tools/numerical_audit_compute.py:121 |
| 14 | LSTM test AUC .524 | INCORRECT | Current saved test CSV gives LSTM AUC 0.526. | tools/numerical_audit_compute.py:121 |
| 14 | LSTM gap -.117 | INCORRECT | Current LSTM gap is 0.111. | tools/numerical_audit_compute.py:121 |
| 14 | LSTM-static correlation .784 -> .429 | INCORRECT | Current LR-LSTM is 0.764 -> 0.373. | tools/numerical_audit_compute.py:121 |
| 15 | LR lost .079; XGB lost .072 | CONFIRMED | Current LR -0.079, XGB -0.072. | tools/numerical_audit_compute.py:121 |
| 15 | LSTM lost .067 | INCORRECT | Current LSTM delta is -0.069. | tools/numerical_audit_compute.py:121 |
| 15 | All spread drops p < .001 | CONFIRMED | Current bootstrap p < 0.0002 for all three model deltas. | tools/numerical_audit_compute.py:277 |
| 15 | Spread coefficient 5x larger than any team stat | CONFIRMED | Spread coefficient -0.539 is 6.98x the next-largest coefficient overall. | tools/numerical_audit_compute.py:542 |
| 15 | No-spread ranking LSTM .574 > LR .571 > XGB .566 | INCORRECT | Current no-spread ranking is LR 0.571 > LSTM 0.568 > XGB 0.566. | tools/numerical_audit_compute.py:121 |
| 15 | XGB-LSTM correlation .699 -> .419 | INCORRECT | Current XGB-LSTM is 0.674 -> 0.372. | tools/numerical_audit_compute.py:121 |
| 15 | LSTM-exclusive catches doubled 12 -> 33 upsets | INCORRECT | Current only_lstm upsets are 15 -> 44. | tools/numerical_audit_compute.py:153 |
| 16 | LR-XGB agree 87% of predictions | CONFIRMED | Current LR-XGB agreement 87.0%. | tools/numerical_audit_compute.py:153 |
| 16 | LSTM unique signal doubles without spread | AMBIGUOUS | Current exact percentages are 6.2% -> 10.8% (1.74x), not exactly 2.0x. | tools/numerical_audit_compute.py:153 |
| 17 | LSTM inversion rests on 12 medium-spread exclusives | CONFIRMED | Current medium-spread only_lstm count is 12. | tools/numerical_audit_compute.py:390 |
| 17 | Platt calibration compresses test probabilities to [0.19, 0.51] | INCORRECT | Current saved calibrated test ranges are LR [0.194, 0.435], XGB [0.216, 0.498], LSTM [0.218, 0.548]. Overall max is 0.548, not 0.51. | tools/numerical_audit_compute.py:508 |
| 17 | Primary analyses use uncalibrated CV predictions | CONFIRMED | CV A/B prediction CSVs are raw; calibration is only applied in held-out test evaluation. | src/models/evaluate_test_set.py |
| 17 | 3,495 training games, 558 test games | CONFIRMED | Counts are labeled games. | tools/numerical_audit_compute.py:245 |
| 17 | Per-season test results rest on 181-192 games | CONFIRMED | 2023:185, 2024:192, 2025:181 labeled games. | tools/numerical_audit_compute.py:245 |
| 18 | Consensus signal 45% | CONFIRMED | Current all_correct 45.4%. | tools/numerical_audit_compute.py:451 |
| 18 | Temporal signal 6%; close 92%, medium 83% | CONFIRMED | Current only_lstm 6.2%; small non-upset rejections 91.7%; medium upset share 83.3%. | tools/numerical_audit_compute.py:390 |
| 18 | Hidden information ~24% of failures | INCORRECT | Current all_wrong medium/large is 18.6% of failures, or 5.3% of all CV games. | tools/numerical_audit_compute.py:451 |
| 18 | Irreducible variance ~76% of failures | INCORRECT | Current all_wrong small is 81.4% of failures, or 23.3% of all CV games. | tools/numerical_audit_compute.py:451 |
