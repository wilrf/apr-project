# NFL Upset Taxonomy: Paper Notes

**Working Title:** "The Anatomy of Upsets: Using Multi-Model Disagreement to Reveal the Structure of NFL Prediction"

**Date:** 2026-02-05

---

## The Story

Everyone assumes upsets are random. This paper shows they're not — they're structurally different from each other, and you can prove it by using three model architectures as diagnostic lenses.

---

## The Arc

### Opening
NFL upsets are treated as chaos — unpredictable, random. But what if different upsets have different causes, and the reason no single model predicts them well is that we're treating them as one phenomenon when they're actually multiple distinct types?

### The Method
We don't pick the "best" model. We deliberately choose three architectures that each see data differently:
- **Logistic Regression**: Linear relationships between features
- **XGBoost**: Non-linear feature interactions
- **LSTM**: Temporal sequences and momentum

We use *where they agree and disagree* as our analytical tool.

### The Surprise Finding
**97% of caught upsets were caught by LSTM alone.**

The "market error" category (where LR/XGB catch upsets) barely exists (0.6%). The market is already efficient on static features. LR and XGBoost see the same thing (r=0.95), and that thing is already priced into the spread.

The only predictable signal is temporal — momentum, form shifts, fatigue patterns.

---

## The Four Types of Upsets

### Type 1: Market Errors (0.6% of upsets, N=2)
- **Spread:** 3.0 pts avg
- **Who catches it:** LR/XGB (static models)
- **What it is:** The line was objectively wrong based on available statistics
- **Reality:** Almost never happens. Market is efficient on static features.
- **Examples:** BUF over DEN, NO over NE

### Type 2: Momentum Upsets (18.6% of upsets, N=64)
- **Spread:** 3.6 pts avg
- **Who catches it:** LSTM only (LR/XGB miss)
- **Model probabilities:** LSTM 0.64, LR 0.40, XGB 0.38
- **What it is:** Temporal patterns invisible to point-in-time snapshots
- **Mechanism:** "Hot team" effect, form shifts, schedule-driven fatigue
- **Examples:** JAX over BAL, NYJ over JAX, PIT over KC, WAS over LV
- **Key insight:** These represent 97% of all caught upsets

### Type 3: Hidden Information (23.5% of upsets, N=81)
- **Spread:** 9.4 pts avg
- **Who catches it:** Nobody (all models ~0.18 probability)
- **What it is:** Big upsets where information isn't in historical data
- **Mechanism:** QB injuries, teams resting starters, tanking, weather surprises
- **Examples:** MIA over ATL (13.5-pt dog), CAR over NE (9-pt dog)
- **Key insight:** Not a modeling failure — it's an information availability problem

### Type 4: Stochastic (57.3% of upsets, N=197)
- **Spread:** 4.2 pts avg
- **Who catches it:** Nobody (models gave ~0.35 probability)
- **What it is:** Close games that could go either way
- **Mechanism:** Game variance, bounces, referee calls, execution on the day
- **Includes:** "Near misses" (models gave 40-50% prob) and true randomness
- **Key insight:** This is the irreducible floor of NFL prediction

---

## Key Statistics

### Model Performance (6-fold CV, 2017-2022 validation)
| Model | AUC-ROC | Brier Score | Log Loss |
|-------|---------|-------------|----------|
| LR | 0.656 ± 0.020 | 0.197 ± 0.008 | 0.579 ± 0.018 |
| XGB | 0.648 ± 0.021 | 0.198 ± 0.008 | 0.583 ± 0.018 |
| LSTM | 0.545 ± 0.027 | 0.242 ± 0.014 | 0.752 ± 0.066 |

### Model Correlations
|      | LR    | XGB   | LSTM  |
|------|-------|-------|-------|
| LR   | 1.000 | 0.950 | 0.537 |
| XGB  | 0.950 | 1.000 | 0.515 |
| LSTM | 0.537 | 0.515 | 1.000 |

### Catch Rate by Spread Size
| Spread Size | Total Upsets | Caught | Catch Rate |
|-------------|--------------|--------|------------|
| Small (3-4.5) | 185 | 57 | 30.8% |
| Medium (5-7) | 64 | 7 | 10.9% |
| Large (7.5+) | 64 | 1 | 1.6% |

### The Punchline Numbers
- Total upsets analyzed: 344
- Caught by at least one model: 66 (19.2%)
- Caught by LSTM only: 64 (18.6%)
- LSTM's share of caught upsets: **97%**

---

## The Paper's Argument

### What This Is NOT
- A model comparison ("LSTM beats XGBoost")
- A betting system ("we can predict upsets profitably")
- A claim that upsets are predictable

### What This IS
- A diagnostic framework using model disagreement
- A taxonomy of upset types based on predictive structure
- Evidence that different mechanisms produce different upsets
- Honest characterization of what's predictable and what isn't

### The Central Insight
The question "Can ML predict upsets?" has FOUR different answers:

| Type | Can ML Predict? | Why/Why Not |
|------|-----------------|-------------|
| Market Errors | Yes, trivially | But they're <1% of upsets |
| Momentum | Yes, with LSTM | Static models can't see it |
| Hidden Info | No | Need external data (injuries, motivation) |
| Stochastic | No | Genuine variance, accept the floor |

---

## Why Panels Will Like This

1. **NOT A HORSE RACE:** We're not claiming "LSTM beats XGBoost." We're saying they see different things, and that's the point.

2. **METHODOLOGICAL CONTRIBUTION:** Multi-model disagreement as a diagnostic tool — exportable to any prediction domain with heterogeneous mechanisms.

3. **SUBSTANTIVE FINDING:** The upset taxonomy itself — first empirical characterization of upset types by predictive structure.

4. **HONEST ABOUT LIMITS:** Type 3 and 4 are explicitly "unpredictable" — not overselling builds credibility.

5. **ACTIONABLE IMPLICATIONS:**
   - Bettors: Focus on Type 2 (small spread + LSTM signal)
   - Researchers: Type 3 needs external data integration
   - Market efficiency: Static features are priced in; temporal dynamics are not

---

## Sample Games by Type

### Type 2: Momentum Upsets (LSTM caught, others missed)
| Game | Underdog | Favorite | Spread | LR | XGB | LSTM |
|------|----------|----------|--------|-----|-----|------|
| 2017_03_OAK_WAS | WAS | LV | 3.0 | 0.38 | 0.37 | 0.66 |
| 2017_03_BAL_JAX | JAX | BAL | 4.0 | 0.47 | 0.36 | 0.58 |
| 2017_04_JAX_NYJ | NYJ | JAX | 3.0 | 0.41 | 0.39 | 0.70 |
| 2017_06_PIT_KC | PIT | KC | 3.5 | 0.36 | 0.33 | 0.61 |
| 2017_12_JAX_ARI | ARI | JAX | 5.0 | 0.36 | 0.36 | 0.65 |

### Type 3: Hidden Information (All models missed, large spread)
| Game | Underdog | Favorite | Spread | LR | XGB | LSTM |
|------|----------|----------|--------|-----|-----|------|
| 2019_17_MIA_NE | MIA | NE | 17.0 | 0.07 | 0.12 | 0.00 |
| 2020_15_NYJ_LA | NYJ | LA | 17.0 | 0.07 | 0.14 | 0.01 |
| 2021_18_IND_JAX | JAX | IND | 14.0 | 0.10 | 0.15 | 0.08 |
| 2018_03_BUF_MIN | BUF | MIN | 16.5 | 0.09 | 0.14 | 0.03 |

---

## Files Generated

| File | Description |
|------|-------------|
| `results/unified_cv_predictions.csv` | All predictions with LR/XGB/LSTM probabilities |
| `results/disagreement_detailed.csv` | Per-game category assignments |
| `results/diagnostic_summary.md` | Technical summary |
| `docs/paper-notes.md` | This file |

---

## Next Steps

1. **Validate on test set (2023-2025)** — Does taxonomy hold out-of-sample?
2. **Deep dive on Type 2 games** — What specific temporal patterns does LSTM detect?
3. **Add external data for Type 3** — Can injury reports reduce "hidden information" category?
4. **Write abstract and introduction** — Frame the contribution clearly
5. **Create figures** — Taxonomy diagram, disagreement matrix visualization

---

## Draft Abstract

> NFL upsets are typically treated as unpredictable events driven by randomness. We challenge this assumption using a novel diagnostic framework: rather than competing three machine learning architectures (Logistic Regression, XGBoost, LSTM) to find the "best" model, we use their patterns of agreement and disagreement to reveal the underlying structure of upset mechanisms. Our analysis of 3,490 NFL games (2005-2022) reveals a four-type taxonomy: Market Errors (<1%), where static models detect mispricing; Momentum Upsets (19%), where only the LSTM detects temporal patterns invisible to point-in-time features; Hidden Information Upsets (24%), requiring external data like injury reports; and Stochastic events (57%), representing irreducible game variance. Strikingly, 97% of predictable upsets fall into the Momentum category, suggesting that the only exploitable edge lies in temporal dynamics, not static feature analysis. This finding has implications for both sports analytics methodology and market efficiency theory: the betting market appears efficient with respect to static information but may underweight recent form and momentum patterns.
