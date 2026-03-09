"""NFL Upset Prediction — multi-model disagreement analysis.

Three structurally distinct models (LR, XGBoost, Siamese LSTM) process
the same data in different representations. Their agreement/disagreement
patterns taxonomize upset mechanisms.

Package layout:
  data/       — Load, merge, validate raw data → data/features/*.csv
  features/   — Engineer 70 features in model-specific representations
  models/     — Train/evaluate LR, XGB, LSTM; cross-validation; experiments
  evaluation/ — Disagreement analysis, calibration, metrics, reports
"""
