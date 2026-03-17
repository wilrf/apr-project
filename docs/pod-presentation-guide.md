# AP Research POD Presentation Guide

## Corrected Slides + New Conclusion Section

This document provides slide-by-slide content for the POD presentation,
aligned with the current paper (docs/paper.md) and optimized for the
2025 AP Research POD rubric (24 points, 7 rows).

**Rubric rows to hit during presentation (Rows 1-4):**
- Row 1 (Research Design, 3 pts): State research question AND method AND argument/conclusion
- Row 2 (Establish Argument, 6 pts): Logically organized argument with consequences/implications
- Row 3 (Reflect, 3 pts): Explain how research steps led to personal conclusions
- Row 4 (Engage Audience, 6 pts): Dynamic delivery, targeted for audience

**Rubric rows scored during oral defense (Rows 5-7):**
- Row 5 (2 pts): Explain *why* inquiry choices were appropriate
- Row 6 (2 pts): Specific details showing depth of new understanding
- Row 7 (2 pts): How the project is significant for your own learning

---

## Slide 1: Title (NO CHANGES NEEDED)

**The Anatomy of NFL Upsets**
Comparing Logistic Regression, XGBoost, and LSTM

Wil Fowler | AP Research | 2025-2026

---

## Slide 2: Key Terms (MINOR EDITS)

**Key Terms You Need to Know**

*Here are the key terms that come up throughout this presentation.*

- **AUC (Area Under the Curve):** A score from 0 to 1 measuring how well a model
  ranks outcomes. 0.50 = random guessing, 1.0 = perfect. Think of it like a grade.
- **p-value:** The probability a result happened by pure luck. Below .05 = likely real.
  Above .05 = cannot rule out chance.
- **Bootstrap Confidence Interval:** A range computed by resampling the data thousands
  of times. If the interval contains zero, the difference is not statistically significant.
- **Cross-Validation:** Training on older data, testing on newer data the model has never
  seen. Prevents memorizing answers.
- **LSTM:** A model that processes games in sequence (week 1, week 2, week 3...)
  instead of looking at each game alone. Detects momentum and fatigue.
- **Point Spread:** How many points oddsmakers think the favorite will win by. A spread
  of 7 = the market expects a touchdown margin.
- **Spread Ablation:** Removing the point spread from the models to see how much they
  depend on it versus the raw stats.

> **CHANGE:** Replaced "Effect Size (Cohen's d)" with "Bootstrap Confidence Interval"
> since the paper uses bootstrap CIs throughout, not Cohen's d.

---

## Slide 3: Research Question (NO CHANGES NEEDED)

**Research Question**

*To what extent can disagreement patterns among logistic regression, XGBoost,
and LSTM models, combined with spread ablation analysis, reveal distinct
categories of NFL upsets?*

- NFL upsets happen when a 3+ point underdog wins. That's about 29% of games.
- Most models try to predict wins and losses. This study looks at *why* certain upsets
  happen by analyzing which models get them right and which don't.
- Spread ablation removes betting line data to test whether models actually learn
  something on their own or just echo the market.

---

## Slide 4: Literature Review (MINOR UPDATES)

**Literature Review**

**Sports Prediction**
- Logistic regression and statistical models remain competitive with complex ML for
  NFL game prediction (Boulier & Stekler, 2003)
- Tree-based models (XGBoost) outperform deep learning on typical tabular data but
  not necessarily on sequential data (Grinsztajn et al., 2022)
- Deep learning including LSTMs shows promise for capturing sequential patterns in
  sports outcomes (Huang & Li, 2021)

**Market Efficiency**
- NFL betting lines outperform both expert judgment and statistical models in predicting
  game outcomes (Song et al., 2007; Wilkens, 2021)
- Decorrelating model predictions from bookmaker odds is necessary to identify genuine
  signal (Hubacek et al., 2019)

**Model Disagreement**
- Ensemble disagreement reveals areas of classification uncertainty (Krogh & Vedelsby, 1995)
- Per-instance disagreement between models is an unbiased estimate of error variance
  (Gordon et al., 2021)
- No prior work uses model disagreement to *categorize* upsets into distinct types

> **CHANGES:** Added Grinsztajn (2022), Wilkens (2021), Hubacek (2019), Gordon (2021)
> from the paper. Updated framing of tree-based vs deep learning point.

---

## Slide 5: Gap in the Literature (NO CHANGES NEEDED)

**Gap in the Literature**

*No one has used model disagreement patterns to classify NFL upsets into distinct
types. Existing research treats all upsets the same.*

- Prior studies focus on binary win/loss prediction, not understanding why specific
  upsets occur
- Ensemble disagreement research tries to improve accuracy, not use disagreement as
  a diagnostic tool
- No one has tested whether prediction models learn independent signals or just
  reflect what the betting market already knows

---

## Slide 6: Method Overview (MAJOR REWRITE)

**Method Overview**

*Quantitative, quasi-experimental with observational data and ablation analysis*

1. Collected 18 NFL regular seasons (2005-2022) of game data. Held out
   2023-2025 for testing.

2. Each model receives the **same data in a different representation**:
   - LR gets 46 summary statistics ("The Summary")
   - XGBoost gets 70 features including per-game lags ("The Details")
   - LSTM gets raw 8-game sequences of 14 features + 10 matchup context ("The Movie")

3. Trained all three with 6-fold expanding-window time-series cross-validation.

4. Classified each game by which models' binary predictions (above/below the
   ~30% base upset rate) matched the actual outcome.

5. Retrained all models without market-derived features to isolate what the
   betting line contributes.

> **CHANGES:**
> - "61 features" → multi-representation design (46/70/sequential)
> - "predicted probability > .50" → base upset rate threshold (~0.30)
> - "16 spread-derived features" → "market-derived features" (4 for LR/XGB, 2 for LSTM)
> - Added the multi-representation framing which is central to the paper

---

## Slide 7: Data & Features (MAJOR REWRITE)

**Data & Multi-Representation Design**

**Dataset**
- Training: 3,495 games (2005-2022)
- Test: 558 games (2023-2025)
- Upset-eligible: spread of 3+ points
- Base rate: ~30% upsets

**Three Representations of the Same Data**

| Model | Representation | Features | What It Sees |
|-------|---------------|----------|-------------|
| LR | "The Summary" | 46 | Rolling averages, differentials, market, Elo |
| XGBoost | "The Details" | 70 | LR's 46 + 24 per-game lag stats (last 3 games) |
| LSTM | "The Movie" | 14×8 seq + 10 ctx | Raw game-by-game sequences + matchup context |

**Models**
- Logistic Regression: Linear baseline. **L1 (LASSO)** regularization (C = 0.1).
- XGBoost: Gradient-boosted trees. Conservative: max_depth=2, lr=0.03.
- LSTM: **Siamese** architecture with attention. **8-game** sequences. Shared encoder
  for both teams.

> **CHANGES:**
> - 3,490 → 3,495
> - Flat "61 features" → multi-representation table
> - L2 → L1 (LASSO)
> - 5-game → 8-game sequences
> - "Grid search tuned" → specific conservative hyperparameters
> - Added siamese architecture and attention

---

## Slide 8: CV Performance (MAJOR REWRITE)

**Results: Cross-Validation Performance**

| Model | AUC | Brier Score | Log Loss |
|-------|-----|-------------|----------|
| Logistic Regression | .650 | .197 | .581 |
| LSTM | .641 | .199 | .583 |
| XGBoost | .638 | .199 | .586 |

*6-fold expanding-window CV, validation years 2017-2022, N = 1,162 games*

- All three models are **statistically indistinguishable**. Bootstrap 95% CIs on all
  pairwise AUC differences contain zero.
- This validates the disagreement framework: disagreement reflects *different
  perspectives*, not differences in model quality.
- LR-XGB predictions correlate at .874. LSTM correlates moderately with both
  (.784 and .699), confirming it captures partially distinct information.

> **CHANGES (ALL NUMBERS):**
> - LR: .656 → .650, XGB: .648 → .638, LSTM: .545 → **.641**
> - LSTM is now competitive, not significantly worse
> - "Both significantly outperformed LSTM (p < .001)" →
>   "All three statistically indistinguishable"
> - Correlations: .950 → .874 (LR-XGB), .537/.515 → .784/.699 (LSTM)
> - Narrative completely inverts: old version had weak LSTM, new has competitive LSTM

---

## Slide 9: Disagreement Categories (MAJOR REWRITE)

**Results: Disagreement Categories**

| Category | N | % | Upset Rate | Meaning |
|----------|---|---|-----------|---------|
| All correct | 528 | 45.4% | 36.7% | Strong cross-architectural signal |
| All wrong | 340 | 29.3% | 20.9% | Outside model capabilities |
| Only LSTM | 65 | 5.6% | 18.5% | Temporal signal (unique to LSTM) |
| LR + XGB only | 78 | 6.7% | 29.5% | Static models agree, LSTM disagrees |
| Only XGB | 48 | 4.1% | 25.0% | Non-linear interaction pattern |
| LR + LSTM | 48 | 4.1% | 33.3% | Linear + temporal agreement |
| Only LR | 28 | 2.4% | 28.6% | Linear spread mispricing |
| XGB + LSTM | 27 | 2.3% | 33.3% | Non-linear + temporal |

*CV set: 1,162 games, threshold = base upset rate (~0.30)*

- 74.7% of games fall into all-agree categories (all correct + all wrong).
- **Key finding:** Of 65 LSTM-exclusive games, 53 (81.5%) are **non-upset rejections**
  — the LSTM moderates false alarms from the other models. Only 12 are upsets caught.
- This is statistically significant (binomial p = 0.029).

> **CHANGES (COMPLETE REWRITE):**
> - Old used binary correct/wrong with 0%/100% upset rates
> - New uses 8-category system with nuanced upset rates
> - Old: "LSTM caught 97% of upsets" → New: "LSTM primarily rejects false alarms (81.5%)"
> - Narrative completely inverts

---

## Slide 10: Spread-Stratified Disagreement (NEW SLIDE)

**Results: The LSTM's Role Depends on Spread**

*This is the study's most diagnostic finding.*

| Spread Bucket | LSTM Exclusives | Upsets Caught | Non-Upsets Rejected | Primary Role |
|--------------|----------------|---------------|--------------------|----|
| Small (3-6.5 pts) | 60 | 5 (8%) | 55 (92%) | **False-alarm filter** |
| Medium (7-13.5 pts) | 12 | 10 (83%) | 2 (17%) | **Upset detector** |
| Large (14+ pts) | 0 | — | — | No exclusive value |

- At **small spreads** (close games): The LSTM says "despite what the stats suggest,
  this underdog's recent trajectory doesn't support an upset." It *moderates* false alarms.
- At **medium spreads** (clear favorites): The LSTM detects genuine upsets the static
  models miss — teams with momentum the summary stats can't capture.
- **The LSTM is not one thing.** Which role it plays depends on the matchup context.

> **NEW SLIDE — this is the paper's central finding and doesn't exist in the old deck.**

---

## Slide 11: Spread Ablation (CORRECTED NUMBERS)

**Results: Spread Ablation**

| Model | AUC With Spread | AUC Without | Delta | p |
|-------|----------------|-------------|-------|---|
| LR | .650 | .571 | -.079 | < .001 |
| XGB | .638 | .566 | -.072 | < .001 |
| LSTM | .641 | .574 | -.067 | < .001 |

- All three models depend heavily on spread data. Bootstrap CIs confirm all deltas
  are statistically significant.
- The LSTM degrades least (-.067) and **becomes the strongest model** without
  spread (.574 vs .571 and .566).
- But the LSTM's smaller delta is NOT statistically significant vs LR's delta
  (CI contains zero). Suggestive, not proven.
- Without spread, LSTM-exclusive correct predictions **double** (5.6% → 11.0%) and
  upsets caught increase from 12 to 33. The spread was masking temporal signal.
- LR-XGB correlation drops from .874 to .742. Spread was the shared anchor.

> **CHANGES:**
> - All AUC numbers updated to match paper
> - Added nuance: LSTM's smaller delta not statistically significant
> - Added doubling of LSTM exclusives finding
> - Changed "no model significantly outperformed" → "LSTM becomes strongest"

---

## Slide 12: Test Set (MAJOR REWRITE)

**Results: Out-of-Sample Test (2023-2025)**

| Model | CV AUC | Test AUC | Gap |
|-------|--------|----------|-----|
| XGB | .638 | .576 | -.062 |
| LR | .650 | .562 | -.088 |
| LSTM | .641 | .524 | **-.117** |

- The LSTM shows the **largest** generalization gap (-.117), not the smallest.
  Temporal patterns learned from 2005-2022 do not transfer well to 2023-2025.
- XGBoost generalizes best (-.062). Non-linear interaction patterns are
  the most temporally stable signal.
- LSTM correlation with static models drops dramatically: LR .784 → .429,
  XGB .699 → .408. It diverges more in truly out-of-sample data.
- Per-season: LSTM trails in all three test seasons (2023: .469, 2024: .549,
  2025: .556). Not one bad year — persistent forward-transfer fragility.

**Top-K Hit Rates** *(base rate = 28.5%)*

| K | LR | XGB | LSTM | Ensemble |
|---|-----|-----|------|----------|
| 10 | 50% (1.8x) | **60% (2.1x)** | 40% (1.4x) | 50% (1.8x) |
| 20 | 40% (1.4x) | **45% (1.6x)** | 35% (1.2x) | 45% (1.6x) |
| 50 | 38% (1.3x) | 44% (1.5x) | 28% (1.0x) | 34% (1.2x) |

- XGBoost's top 10 hit at 60% (2.1x lift). Signal concentrates in highest-confidence
  predictions and fades quickly.

> **CHANGES (CRITICAL — NARRATIVE INVERTS):**
> - Old: "LSTM generalized best (-.005)" → New: "LSTM has largest gap (-.117)"
> - Old: "LR and XGB agreed 100%" → Removed (not in paper)
> - All numbers corrected
> - Top-K numbers corrected
> - Added per-season breakdown and correlation drop

---

## Slide 13: The Four Types (MAJOR REWRITE)

**Analysis: A Two-Dimensional Upset Taxonomy**

*The disagreement framework crossed with spread context reveals four structural
types of NFL upsets.*

- **Type 1 — Consensus Signal (45.4%):** All models agree and are correct. At small
  spreads only 31% reach consensus; at large spreads 90% do. The easy cases.

- **Type 2 — Temporal Signal (5.6%):** Only the LSTM is correct. **Dual character:**
  at small spreads it's a false-alarm filter (92%); at medium spreads it's an upset
  detector (83%). Which role depends on matchup context.

- **Type 3 — Hidden Information (all_wrong at medium/large spreads):** 96-100%
  are missed upsets. Driven by injuries, motivation, game-day factors outside the
  data. Not a modeling failure — a data availability boundary.

- **Type 4 — Stochastic Floor (all_wrong at small spreads):** 97% are false alarms
  in close games. All models predicted upset, but the favorite's marginal advantage
  held. Irreducible game variance.

> **CHANGES:**
> - Old flat taxonomy → New 2D taxonomy (agreement × spread context)
> - Old "Market Errors (<1%)" → Absorbed into consensus or removed
> - Old "Momentum Upsets (19%)" → "Temporal Signal (5.6%)" with dual character
> - Old percentages (19%, 24%, 57%) → New percentages from paper
> - Added spread stratification as the key diagnostic innovation

---

## Slide 14: Temporal Signal Deep Dive (REPLACES "Types 1 and 2")

**The LSTM Paradox: Competitive but Non-Stationary**

*The most striking finding: the LSTM matches LR and XGB in cross-validation
but shows the largest gap to the test set.*

**In cross-validation (2005-2022):**
- LSTM AUC .641 — competitive with LR (.650) and XGB (.638)
- Adds ensemble value: 3-model average AUC .655 vs LR+XGB .649
- LSTM exclusives double without spread (5.6% → 11.0%)

**On the test set (2023-2025):**
- LSTM drops to .524 — largest gap of any model (-.117)
- LSTM veto ensemble fails to transfer forward
- XGB alone (.576) outperforms all ensemble strategies

**Why?** Three hypotheses:
1. **Temporal non-stationarity:** NFL changes between eras. Trajectory patterns
   from 2005-2022 may not hold in 2023-2025.
2. **Overfitting to sequence artifacts:** More parameters, richer input, only 3,495 games.
3. **Persistent forward-transfer fragility:** LSTM trails in all 3 test seasons, not just one.

> **REPLACES old "Types 1 and 2" slide entirely. Old data about "97% LSTM catches"
> and Cohen's d = 0.13 no longer applies.**

---

## Slide 15: What the Spread Tells Us (CORRECTED)

**What the Spread Ablation Tells Us**

*When I removed the point spread, models lost significant predictive power.
But what they lost — and what they kept — was different.*

- LR lost .079 AUC, XGB lost .072, LSTM lost .067. All highly significant (p < .001).
- The spread contains more predictive info than all other team stats combined.
  LR's coefficient on spread magnitude is -0.539 (standardized) — 5x any other feature.
- **But the LSTM kept the most.** Without spread, LSTM (.574) > LR (.571) > XGB (.566).
  The ranking reverses. The LSTM captures temporal signal partially independent
  of the market.
- Model agreement collapsed: LR-XGB correlation .874 → .742, XGB-LSTM .699 → .419.
  The spread was the glue holding predictions together.
- LSTM exclusives doubled without spread, and upsets caught went from 12 to 33.
  The spread was masking genuine temporal signal.
- **Strong evidence for market efficiency**, but also evidence that sequential
  processing captures something the market line encodes differently.

> **CHANGES:** All numbers corrected to match paper. Added nuance about LSTM
> becoming strongest without spread. Changed "equally mediocre" framing.

---

## Slide 16: The Bottom Line (CORRECTED)

**The Bottom Line**

*"Can machine learning predict NFL upsets?" depends on which type and which era.*

- **Consensus Signal (45%):** Yes, when all architectures converge. But these are
  the "easy" games where the signal is strong.
- **Temporal Signal (6%):** The LSTM finds real patterns — false-alarm moderation
  at small spreads, upset detection at medium spreads. But these patterns are
  **non-stationary** and don't transfer forward reliably.
- **Hidden Information:** No. Not without injury reports, motivation data, and other
  game-day factors outside historical statistics.
- **Stochastic Floor:** No. Close games are determined by in-game variance that
  no pre-game model can resolve.
- The spread-stratified taxonomy decomposes prediction failure into distinct
  mechanisms at different spread levels. This decomposition is **invisible** to any
  single-model analysis.
- **The real contribution is the method:** using architectural disagreement × context
  as a diagnostic framework. Exportable to any domain with heterogeneous mechanisms.

> **CHANGES:** Updated types to match paper taxonomy. Added non-stationarity
> finding. Changed percentages. Removed "81% unpredictable" framing.

---

## Slide 17: Limitations (NEW SLIDE — REQUIRED FOR ROW 2)

**Limitations**

- **Small cell sizes in spread strata:** The LSTM inversion (false-alarm filter →
  upset detector) rests on 12 medium-spread exclusives. Striking pattern,
  but limited formal significance within strata.
- **Calibration artifacts:** Platt calibration compresses test probabilities to
  [0.19, 0.51], inflating threshold-based disagreement. Primary analyses use
  uncalibrated CV predictions; rank-based analysis used for test set.
- **Sample size:** 3,495 training games, 558 test games. NFL produces ~270
  games/season. Per-season results based on 181-192 games each.
- **Feature scope:** No player-level data, injury reports, weather forecasts,
  or motivational factors. The "hidden information" category is partly a data
  availability limitation.
- **Single sport:** NFL has specific structural properties (17-game seasons,
  high variance, dominant betting market). The disagreement framework
  generalizes; the specific taxonomy is NFL-specific.

---

## Slide 18: Conclusion & Implications (NEW — CRITICAL FOR ROWS 2-3)

**Conclusion: Answering the Research Question**

*To what extent can disagreement patterns reveal distinct categories of NFL upsets?*

**Four principal findings:**

1. **The three models are statistically equivalent** (bootstrap CIs contain zero),
   validating disagreement as reflecting different perspectives, not quality differences.

2. **The LSTM captures temporal signal whose character depends on context.**
   At small spreads: false-alarm moderation (92%). At medium spreads: upset
   detection (83%). This dual role is the taxonomy's most diagnostic finding.

3. **Temporal patterns are real but non-stationary.** The LSTM improves the ensemble
   in CV but this does not transfer forward. The largest generalization gap (-.117)
   indicates era-specific dynamics.

4. **Spread stratification decomposes prediction failure** into stochastic variance
   (small spreads, 97% false alarms) and hidden information (medium/large, 96-100%
   missed upsets). Neither dimension alone reveals this structure.

**Implication for the field:** Architectural disagreement × context is a viable
diagnostic framework for any prediction domain with heterogeneous mechanisms —
medical diagnosis, financial prediction, climate modeling.

**Implication for NFL prediction:** ~81% of upsets are driven by stochastic variance
or hidden information. Better algorithms won't help; better data sources might.

---

## Slide 19: Future Research (NEW — SUPPORTS ROW 2)

**Future Research**

- **Incorporate game-day data:** Injury reports, weather, and motivation factors
  could shrink the "hidden information" category and sharpen taxonomy boundaries.
- **Player-level features:** Individual performance trajectories rather than
  team-level aggregates may give the LSTM more stable temporal signal.
- **Retraining windows:** Online or rolling-window LSTM training might address
  temporal non-stationarity by updating sequence patterns as the game evolves.
- **Other sports / domains:** Apply the disagreement × context framework to NBA
  (more games, less variance), soccer (different market structure), or non-sports
  domains like medical diagnostics where heterogeneous mechanisms drive outcomes.
- **Significance testing within strata:** Larger datasets (more seasons or multiple
  sports) could provide the statistical power to formally test the LSTM inversion
  pattern within spread buckets.

---

## Slide 20: References (UPDATE)

**References**

Boulier, B. L., & Stekler, H. O. (2003). Predicting the outcomes of NFL games. *International Journal of Forecasting*, 19(2), 257-270.

Dietterich, T. G. (2000). Ensemble methods in machine learning. *MCS 2000*, LNCS 1857.

Glickman, M. E., & Stern, H. S. (1998). A state-space model for NFL scores. *JASA*, 93(441), 25-35.

Gordon, M. L., et al. (2021). Disagreement deconstructed. *CHI '21*.

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 35*.

Hubacek, O., Sourek, G., & Zelezny, F. (2019). Exploiting sports-betting market using ML. *IJF*, 35(2), 783-796.

Krogh, A., & Vedelsby, J. (1995). Neural network ensembles, cross validation, and active learning. *NIPS 7*.

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 30*.

Wilkens, S. (2021). Sports prediction and betting models in the ML age. *J Sports Analytics*, 7(2).

---

## ORAL DEFENSE PREP NOTES

These are not slides — they're preparation for the 3 defense questions
(Rows 5, 6, 7 of the rubric).

### Row 5: Research/Inquiry Process Question
*"Why did you make the choices you did during your research process?"*

**Strong answer elements:**
- Why three architecturally *distinct* models instead of three similar ones:
  "I needed the disagreement to be structurally informative, not just noise from
  different random seeds. LR captures linear relationships, XGB captures
  interactions, LSTM captures temporal dynamics."
- Why multi-representation design: "If all models saw the same 61 features, I
  couldn't tell if disagreement came from architecture or data. By giving each
  model its optimal representation, disagreement reflects genuine structural
  differences in how they process information."
- Why base-rate threshold (~0.30) instead of 0.50: "With a 30% upset rate,
  using 0.50 would mean almost no model ever predicts 'upset.' The base rate
  is the principled choice for minority-class problems."
- Why expanding-window CV instead of random splits: "NFL data is temporal.
  Random splits would let the model see future data during training — data leakage."

### Row 6: Depth of Understanding Question
*"What new understanding did your research generate?"*

**Strong answer elements:**
- The LSTM inversion by spread is genuinely new: "No one has shown that a
  temporal model's role *changes qualitatively* depending on matchup context.
  At small spreads it filters noise; at medium spreads it detects signal."
- The 2D taxonomy: "Previous work categorizes upsets as a single list. Crossing
  model agreement with spread context reveals that 'all models wrong' means
  completely different things at different spread levels — stochastic variance
  vs hidden information."
- Non-stationarity as a finding, not a failure: "The LSTM's poor test performance
  isn't just 'the model didn't work.' It tells us something about the domain:
  temporal patterns in NFL performance are era-specific. That's a finding about
  the sport, not just about the model."

### Row 7: Reflection Question
*"How has this project been significant for your own understanding?"*

**Strong answer elements:**
- "I started this project trying to build a better prediction model. The most
  important thing I learned is that prediction failure is *more interesting*
  than prediction success. The disagreement patterns taught me more about why
  upsets happen than any accuracy metric."
- "Working with the LSTM taught me that more complex ≠ better. The LSTM
  matched the simpler models in CV but failed to generalize. This changed how
  I think about model complexity — it's not about having the most powerful
  architecture, it's about matching the architecture to the signal's stability."
- "The spread ablation was humbling. The betting market had already captured
  most of what my models were learning. The real contribution wasn't beating
  the market — it was using the market as a diagnostic tool."
- "I learned to distinguish between statistical significance and practical
  significance. Some patterns are real (p < .05) but too small to act on.
  That distinction matters in any field."

---

## SLIDE-BY-SLIDE CHANGE SUMMARY

| Slide | Status | Key Changes |
|-------|--------|-------------|
| 1 (Title) | Keep | No changes |
| 2 (Key Terms) | Edit | Replace Cohen's d → Bootstrap CI |
| 3 (Research Q) | Keep | No changes |
| 4 (Lit Review) | Edit | Add Grinsztajn, Wilkens, Gordon citations |
| 5 (Gap) | Keep | No changes |
| 6 (Method) | **Rewrite** | Multi-rep design, base-rate threshold |
| 7 (Data) | **Rewrite** | 3-representation table, L1, 8-game, siamese |
| 8 (CV Results) | **Rewrite** | All numbers, LSTM competitive not weak |
| 9 (Disagreement) | **Rewrite** | 8 categories, LSTM = false-alarm filter |
| 10 (Spread-Strat) | **NEW** | LSTM inversion by spread bucket |
| 11 (Ablation) | Edit | Corrected numbers, LSTM strongest without spread |
| 12 (Test Set) | **Rewrite** | LSTM worst gap not best, all numbers |
| 13 (Four Types) | **Rewrite** | 2D taxonomy, new type descriptions |
| 14 (LSTM Paradox) | **NEW** (replaces old 13) | Non-stationarity analysis |
| 15 (Spread) | Edit | Corrected numbers and framing |
| 16 (Bottom Line) | Edit | Updated taxonomy, added non-stationarity |
| 17 (Limitations) | **NEW** | Required for Row 2 top score |
| 18 (Conclusion) | **NEW** | Answers research question directly |
| 19 (Future Research) | **NEW** | Supports Row 2 implications |
| 20 (References) | Edit | Updated to match paper citations |
