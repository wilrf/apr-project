# Model Card — Logistic Regression ("The Summary")

**Representation:** A single static statistical snapshot of each matchup.

## Features
- 46 base features (42 in the no-spread variant): rolling averages, differentials,
  market line, and Elo. Canonical list: `src/features/pipeline.py`.

## Training data
- 3,495 labeled games, 2005–2022. Labels apply only to games with `spread >= 3`;
  sub-3 games are retained for rolling-feature continuity and excluded via `upset.notna()`.
- Base upset rate ≈ 30% (train 30.36%, test 28.49%). Decision threshold is the base
  rate, **not** 0.5.

## Evaluation (frozen artifacts)
| Split | AUC | Brier |
|-------|-----|-------|
| 6-fold expanding-window CV | 0.6497 | — |
| Held-out test (2023–2025) | 0.5622 | 0.2026 |

Spread ablation (CV AUC): 0.6497 with spread → 0.5707 without spread (−0.0790).

## Intended use
- Diagnostic baseline: the most interpretable model in the set; LR coefficients
  give directional influence directly (see the dashboard Feature Weights view).

## Limitations
- Linear snapshot only; cannot represent interactions or temporal dynamics.
- Like all three models, loses substantial signal when the market spread is removed.
