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

## Honest Limitations to Foreground
- LSTM exclusive catches may not survive significance testing (permutation p=0.17 in prior run)
- ~42% of upsets are genuinely unpredictable from historical data
- Weak temporal signal exists but may not be actionable
- This builds credibility — reviewers trust papers that acknowledge limits

## Key Citations for Framing
- Grinsztajn et al. (2022) — trees still beat deep learning on tabular data
- Hubacek et al. (2019) — decorrelating from bookmaker odds
- Wilkens (2021) — market efficiency in sports betting
- Gordon et al. (2021) — disagreement as diagnostic (CHI)
- Lakshminarayanan et al. (2017) — ensemble disagreement for uncertainty
- Pinchuk (2026) — XGBoost interaction learning limitations
