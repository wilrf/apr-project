# NFL Upset Taxonomy via Multi-Architecture Disagreement

AP Research project investigating **why** NFL upsets happen using the structure of prediction failure across three architecturally distinct models.

## Research Question

> How do agreement and disagreement patterns of three structurally distinct ML architectures reveal the mechanisms and predictability boundaries of NFL upsets?

## Approach

Three models see the same data in different representations:

| Model | Representation | Features |
|-------|---------------|----------|
| **Logistic Regression** (L1) | Summary statistics | 46 pre-computed features |
| **XGBoost** | Detailed tabular | 70 features (46 base + 24 per-game lags) |
| **Siamese LSTM** | Temporal sequences | 8 timesteps x 14 features + 10 matchup |

This is not a model competition. The disagreement patterns themselves are the finding — they reveal whether an upset was driven by linear mispricing, non-linear interactions, temporal dynamics, or hidden information.

## Key Results (CV: 3,495 games, 2005-2022)

- Models achieve statistically indistinguishable AUC-ROC (LR 0.650, LSTM 0.641, XGB 0.638)
- LSTM exclusives invert by spread: **92% false-alarm rejections** at small spreads, **83% upset detections** at medium spreads
- All models degrade significantly without betting-line features; LSTM retains the most signal

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.data.generate_features   # build train.csv + test.csv
```

## Usage

```bash
python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py --ignore=tests/models/test_lstm_trainer.py -v
python3 -m src.models.evaluate_test_set
python3 -m src.models.run_ab_experiment --quick   # LR+XGB only (~1 min)
python3 -m src.models.run_ab_experiment           # all 3 models (~15 min)
```

## Project Structure

```
src/data/           Data loading (NFL schedules, betting lines, play-by-play EPA)
src/features/       Feature engineering pipeline and target definition
src/models/         Model implementations, training, and evaluation scripts
src/evaluation/     Metrics, calibration, disagreement analysis, reporting
tests/              Mirror of src/ structure
docs/               Research paper, framing notes, presentation materials
results/            Generated reports and experiment outputs
```
