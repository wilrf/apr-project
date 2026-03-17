# NFL Upset Taxonomy via Multi-Architecture Disagreement

AP Research project investigating **why** NFL upsets happen using the structure of prediction failure across three architecturally distinct models.

## Research Question

> How do agreement and disagreement patterns of structurally distinct ML architectures reveal the mechanisms and predictability boundaries of NFL upsets?

## Approach

Three models process the same game data in architecture-appropriate representations: logistic regression sees summary statistics, XGBoost sees detailed tabular features with per-game lags, and a siamese LSTM sees raw temporal sequences.

This is not a model competition. The disagreement patterns themselves are the finding — they reveal whether an upset was driven by linear mispricing, non-linear interactions, temporal dynamics, or hidden information. The result is a two-dimensional upset taxonomy: model agreement crossed with matchup context (point spread).

## Project Structure

```
src/data/           Data loading (NFL schedules, betting lines, play-by-play EPA)
src/features/       Feature engineering pipeline and target definition
src/models/         Model implementations, training, and evaluation scripts
src/evaluation/     Metrics, calibration, disagreement analysis, reporting
docs/               Research paper, framing notes, presentation materials
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.data.generate_features
```
