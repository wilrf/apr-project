"""Model evaluation, disagreement analysis, and reporting.

Core modules:
  disagreement.py — Categorizes games by which models predicted correctly.
                    Uses base upset rate (~0.30) as threshold, NOT 0.5.
                    8 categories: all_correct, all_wrong, only_lr, only_xgb,
                    only_lstm, lr_xgb, lr_lstm, xgb_lstm.
  calibration.py  — Post-hoc Platt/isotonic calibration. Fit on held-out
                    (2021-2022) predictions, applied to test predictions.
  metrics.py      — Calibration error, Brier score, baseline comparisons.
  report.py       — Markdown report generation.
  comparison.py   — Model comparison utilities.
  shap_analysis.py — SHAP feature importance for XGBoost.

If you change disagreement categories → update evaluate_test_set, run_ab_experiment.
If you change calibration interface → update evaluate_test_set.
"""
