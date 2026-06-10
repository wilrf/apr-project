# Model Card — Siamese LSTM ("The Movie")

**Representation:** Each team encoded as a recent game-by-game sequence, with a
small matchup-context vector — raw temporal data rather than pre-computed averages.

## Features
- 14 sequence features × 8 timesteps + 10 matchup features (8 matchup features in
  the no-spread variant, which drops `spread_magnitude` and `total_line`).
  Canonical lists: `src/features/pipeline.py`.

## Training data
- 3,495 labeled games, 2005–2022. Labels apply only to games with `spread >= 3`;
  sub-3 games are retained for rolling-feature continuity and excluded via `upset.notna()`.
  Sub-3 games are still used in each team's LSTM sequence history, but never as labeled targets.
- Base upset rate ≈ 30% (train 30.36%, test 28.49%). Decision threshold is the base
  rate, **not** 0.5.

## Evaluation (frozen artifacts)
| Split | AUC | Brier |
|-------|-----|-------|
| 6-fold expanding-window CV | 0.6372 | — |
| Held-out test (2023–2025) | 0.5263 | 0.2089 |

- Competitive in-sample but the largest CV→test drop of the three models.
- Most behaviorally distinct model on the held-out set: test-set probability
  correlations are 0.373 (LR–LSTM) and 0.309 (XGB–LSTM), versus 0.878 for LR–XGB.
- Spread ablation (CV AUC): 0.6372 with spread → 0.5682 without spread (−0.0690).

## Intended use
- Diagnostic instrument, not a leaderboard winner: the temporal representation is
  most useful as a behaviorally distinct view (a temporal false-alarm filter) whose
  disagreement with the snapshot models is the signal of interest, rather than for
  raw held-out accuracy.

## Limitations
- Largest in-sample-to-held-out generalization gap; weakest test-set AUC of the three.
- Limited matchup context by design (10 features) to force reliance on the sequence encoder.
- Like all three models, loses substantial signal when the market spread is removed.
