# Model Card — XGBoost ("The Details")

**Representation:** The static snapshot plus short lag structure — the same
matchup view as the baseline, enriched with per-game stats from recent games.

## Features
- 70 features (66 in the no-spread variant): 46 base features + 24 lag features
  (per-game stats for the last 3 games). Canonical list: `src/features/pipeline.py`.
- Tree depth: `max_depth = 6` (`src/models/xgboost_model.py`).

## Training data
- 3,495 labeled games, 2005–2022. Labels apply only to games with `spread >= 3`;
  sub-3 games are retained for rolling-feature continuity and excluded via `upset.notna()`.
- Base upset rate ≈ 30% (train 30.36%, test 28.49%). Decision threshold is the base
  rate, **not** 0.5.

## Evaluation (frozen artifacts)
| Split | AUC | Brier |
|-------|-----|-------|
| 6-fold expanding-window CV | 0.6377 | — |
| Held-out test (2023–2025) | 0.5755 | 0.2013 |

- Best held-out generalizer of the three models.
- XGBoost's top 10 held-out predictions contain 6 real upsets — a 60% hit rate,
  roughly 2.1x lift over the 28.49% test base rate.
- Spread ablation (CV AUC): 0.6377 with spread → 0.5662 without spread (−0.0715).

## Intended use
- Short-lag detail model: tests whether nonlinear interactions over recent
  per-game stats add signal beyond the static snapshot. The best test-set
  generalizer in this study.

## Limitations
- Still a snapshot-plus-lags view; it does not model the full temporal sequence.
- Like all three models, loses substantial signal when the market spread is removed.
