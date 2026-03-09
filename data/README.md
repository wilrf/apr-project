# Data Directory

## Structure

```
data/
├── raw/                    # Source data (external, gitignored)
│   └── spreadspoke_scores.csv    # Kaggle betting data (1966-2025)
│
└── features/               # Generated datasets (gitignored)
    ├── train.csv           # Training set (2005-2022)
    └── test.csv            # Test set (2023-2025)
```

## Datasets

| File | Description |
|------|-------------|
| `raw/spreadspoke_scores.csv` | Kaggle NFL betting data |
| `features/train.csv` | Training games with 70 features (46 base + 24 XGB per-game lags) |
| `features/test.csv` | Test games with 70 features |

## Data Split

- **Train:** 2005-2022 (18 seasons)
- **Test:** 2023-2025 (3 seasons, out-of-sample)

## Upset Candidates

Only games with `spread >= 3` receive an upset label. Sub-3 games stay in the CSV with `upset = NaN` for rolling stat computation.
- Train: 3,495 labeled games
- Test: 558 labeled games
- Upset rate: ~30% in both splits

## Regenerating Data

```bash
# Run from project root
python3 -m src.data.generate_features
```

## Sources

- NFL schedules: `nfl_data_py` library (nflverse)
- Betting lines: [Kaggle spreadspoke dataset](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data)
