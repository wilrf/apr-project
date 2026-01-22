# Data Directory

## Structure

```
data/
├── raw/                    # Source data (external, gitignored)
│   └── spreadspoke_scores.csv    # Kaggle betting data (1966-2025)
│
└── features/               # Generated datasets (gitignored)
    ├── train.csv           # Training set (2005-2022)
    ├── test.csv            # Test set (2023-2025)
    └── columns.csv         # Feature column definitions
```

## Datasets

| File | Rows | Description |
|------|------|-------------|
| `raw/spreadspoke_scores.csv` | 14,358 | Kaggle NFL betting data |
| `features/train.csv` | 4,346 | Training games with 61 features |
| `features/test.csv` | 768 | Test games with 61 features |
| `features/columns.csv` | 61 | Feature column names |

## Data Split

- **Train:** 2005-2022 (18 seasons)
- **Test:** 2023-2025 (3 seasons, out-of-sample)

## Upset Candidates

Filter to `spread_magnitude >= 3` for upset prediction:
- Train: ~3,497 games
- Test: ~559 games

## Regenerating Data

```bash
# Run from project root
python3 scripts/export_data.py
```

## Sources

- NFL schedules: `nfl_data_py` library (nflverse)
- Betting lines: [Kaggle spreadspoke dataset](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data)
