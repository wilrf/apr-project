# Model Cards

Each model in this project is documented as a model card: its representation,
features, training data, intended use, evaluation, and limitations. All numbers
are anchored to `results/` and match the figures reported in the top-level README.

- [Logistic Regression](logistic-regression.md) — the static snapshot baseline
- [XGBoost](xgboost.md) — snapshot + short lag structure
- [Siamese LSTM](lstm.md) — each team as a recent sequence

> These models are diagnostic instruments for studying upset *mechanisms* via
> disagreement, not a production betting system. See the repository README.
