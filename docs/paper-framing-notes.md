# Paper Framing Notes

Running bullet list of framing decisions, key sentences, and positioning claims for the write-up. Separate from the technical design spec.

---

## Core Positioning
- Existing work uses ML to *predict* upsets; we use ML disagreement to *categorize* them
- This is a paper about the *limits* of prediction, not about prediction success
- The diagnostic framework is exportable to any prediction domain with heterogeneous mechanisms

## Key Framing Sentences
- "We don't pick the best model. We use where three architecturally distinct models agree and disagree as a diagnostic tool."
- "The co-validation literature (Gordon et al. 2021) shows per-instance disagreement is an unbiased estimate of the variance of error, which justifies treating disagreement as informative rather than noise."
- "Each model gets the same data in a different representation: LR sees 46 pre-computed summary statistics, XGBoost sees 70 features including per-game lags, and the LSTM sees 8 raw game-by-game sequences of 14 features plus 10 matchup context features."

## Novelty Claim
- No published work uses model disagreement patterns to taxonomize the nature of prediction failures in sports analytics
- Closest precedents: ensemble uncertainty estimation (Lakshminarayanan et al. 2017), co-validation for error estimation, medical diagnostic modality comparison
- The 4-type upset taxonomy (Market Error, Interaction, Temporal, Unpredictable) is a genuine contribution

## Key Empirical Findings (March 2026)

### CV Performance (6-fold, 1,162 games)
- All three models competitive: LR 0.650 > LSTM 0.641 > XGB 0.638
- LSTM is NOT the weak link — it performs on par with XGB

### Spread Ablation
- Removing spread hurts all models: LR -0.079, XGB -0.072, LSTM -0.067
- **LSTM degrades least and wins without spread** (0.574 > 0.571 > 0.566)
- This is the strongest evidence for distinct temporal signal independent of the market
- LSTM exclusive predictions double from 5.6% → 11.0% without spread
- All-model agreement drops from 74.7% → 55.3%

### LSTM Exclusive Contribution
- In CV, LSTM's exclusive value is primarily **moderating false alarms** (53/65 exclusives are non-upset rejections)
- Only 12/65 are upsets caught that both LR and XGB miss
- This reframes the LSTM's role: not "temporal upset detector" but "temporal false-alarm filter"

### CV-to-Test Gap
- LSTM: 0.641 → 0.520 (largest gap, -0.121)
- LR: 0.650 → 0.562 (-0.088), XGB: 0.638 → 0.576 (-0.062)
- Temporal patterns generalize less well forward than static features
- LSTM-LR/XGB correlation drops from ~0.75 (CV) to ~0.30 (test) — LSTM diverges more in truly out-of-sample data

## Honest Limitations to Foreground
- LSTM temporal signal exists in CV but generalizes poorly forward (0.12 AUC gap vs 0.06 for XGB)
- LSTM exclusive catches are mostly non-upset rejections, not upset detections
- Calibrated probabilities compress disagreement analysis — threshold-based categories lose discriminative power
- ~29% of CV predictions are "all wrong" — the irreducible floor remains large
- Significance tests still pending under the current architecture
- This builds credibility — reviewers trust papers that acknowledge limits

## Key Citations for Framing
- Grinsztajn et al. (2022) — trees still beat deep learning on tabular data
- Hubacek et al. (2019) — decorrelating from bookmaker odds
- Wilkens (2021) — market efficiency in sports betting
- Gordon et al. (2021) — disagreement as diagnostic (CHI)
- Lakshminarayanan et al. (2017) — ensemble disagreement for uncertainty
- Pinchuk (2026) — XGBoost interaction learning limitations
