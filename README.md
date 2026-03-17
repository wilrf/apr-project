# NFL Upset Taxonomy via Multi-Architecture Disagreement

This project studies NFL upset prediction as a diagnostic problem, not a leaderboard problem.
Three model classes see the same games through different representations:

- Logistic regression reads a static feature snapshot.
- XGBoost reads the same snapshot plus short lag structure.
- A true siamese LSTM reads each team as a recent sequence.

The question is not only which model scores highest. The question is what their agreement and
disagreement reveal about the kinds of signal that are actually present in upset prediction.

Disagreement is treated here as architectural evidence about signal type, not as direct causal
proof. If one model class succeeds where the others fail, that suggests a game may depend more on
static summaries, nonlinear interactions, or recent temporal dynamics. It does not by itself prove
that a single mechanism caused the upset.

## What This Repo Contributes

- A controlled multi-representation setup: same labeled games, different model classes, different input views.
- A focus on disagreement analysis rather than only aggregate accuracy.
- An honest held-out evaluation on 2023-2025 games, where the models do not all generalize equally well.
- A spread-ablation experiment that tests how much signal remains once market features are removed.

The strongest claim is not "deep learning beats simpler models." It does not. The stronger claim is
that model-class disagreement is informative about representation-sensitive failure modes.

## Current Snapshot

The numbers below are taken from the current saved prediction artifacts in:

- `results/ab_experiment/predictions_with_spread.csv`
- `results/ab_experiment/predictions_without_spread.csv`
- `results/test/predictions.csv`

### Data

- Train: 3,495 labeled games from 2005-2022, upset rate 30.36%
- Test: 558 labeled games from 2023-2025, upset rate 28.49%
- Labels apply only to games with `spread >= 3`
- Sub-3 spread games remain in the generated datasets for rolling-feature continuity and are excluded from modeling via `upset.notna()`

### Model Representations

| Model | Representation |
|-------|----------------|
| LR | 46 base features |
| XGB | 70 features = 46 base + 24 lag features |
| LSTM | 14 sequence features x 8 timesteps + 10 matchup features |

## Results At A Glance

| Model | CV AUC | Test AUC | Test Brier |
|-------|--------|----------|------------|
| LR | 0.6497 | 0.5622 | 0.2026 |
| XGB | 0.6377 | 0.5755 | 0.2013 |
| LSTM | 0.6372 | 0.5263 | 0.2089 |

What matters in those numbers:

- In 6-fold expanding-window CV, all three models are close. Pairwise bootstrap CIs on CV AUC differences include zero.
- On the held-out 2023-2025 test set, XGBoost generalizes best.
- The LSTM is competitive in-sample, but its held-out performance drops the most.
- The value of the LSTM in this repo is diagnostic more than leaderboard-driven.

### Spread Ablation

Removing market features degrades all three models and compresses them into a near-tie:

| Model | CV AUC With Spread | CV AUC Without Spread | Delta |
|-------|--------------------|-----------------------|-------|
| LR | 0.6497 | 0.5707 | -0.0790 |
| XGB | 0.6377 | 0.5662 | -0.0715 |
| LSTM | 0.6372 | 0.5682 | -0.0690 |

The careful reading is:

- All three models lose substantial signal without the betting line.
- The no-spread condition is much tighter than the with-spread condition.
- The current artifacts do not support a strong claim that the LSTM clearly wins this ablation.
- They do support the narrower claim that temporal information retains some signal even after market features are removed.

### Held-Out Test Behavior

- Test-set probability correlations are `0.878` for LR-XGB, `0.373` for LR-LSTM, and `0.309` for XGB-LSTM.
- That makes the LSTM the most behaviorally distinct model on truly out-of-sample data.
- XGBoost's top 10 held-out predictions contain 6 real upsets, a 60% hit rate and roughly 2.1x lift over the 28.49% base rate.

## Interpretation

The project supports a measured version of the disagreement thesis:

- If multiple model classes succeed on the same games, the signal is representation-robust.
- If they split, the game may depend on a representation-specific cue.
- If they all fail, the event may be weakly signaled in the current feature set or driven by factors the repo does not observe.

That is a useful research framing, but it is not the same as claiming direct causal identification.

## Repo Structure

```text
src/data/           Data loading, merging, validation, and feature generation
src/features/       Canonical feature pipeline and target definition
src/models/         LR, XGBoost, LSTM, trainers, and experiment runners
src/evaluation/     Metrics, calibration, disagreement analysis, and reports
data/features/      Generated train/test feature CSVs
results/            Saved prediction artifacts and reports
docs/               Paper and presentation materials
```

## Reproduce

```bash
python3 -m src.data.generate_features
python3 -m src.models.run_ab_experiment --quick
python3 -m src.models.evaluate_test_set
python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py -v
```
