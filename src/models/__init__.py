"""Model training, evaluation, and experiments.

Model interface contract — all models expose:
  fit(X, y)              — train on feature matrix X and labels y
  predict_proba(X)       — return 1D np.ndarray of P(upset)

Models and their feature representations:
  logistic_model.py  — UpsetLogisticRegression  (46 LR features)
  xgboost_model.py   — UpsetXGBoost             (70 XGB features)
  lstm_model.py      — SiameseUpsetLSTM         (14seq×8ts + 10 matchup)

Training infrastructure:
  cv_splitter.py      — Expanding-window time-series CV (6 folds, 2017-2022 val)
  unified_trainer.py  — Trains all 3 models on identical folds with model-specific
                        feature routing. Produces GamePrediction/FoldResult/UnifiedCVResults.
  lstm_trainer.py     — LSTM-specific training loop with DataLoader + early stopping
  sequence_builder.py — Builds team game-history sequences for LSTM; normalizes from
                        training stats only to prevent leakage

Experiment scripts:
  evaluate_test_set.py  — Held-out 2023-2025 evaluation with calibration
  run_ab_experiment.py  — Spread ablation (with vs without market features)

If you change the model interface → update unified_trainer and experiment scripts.
If you change sequence_builder → update lstm_trainer, unified_trainer, evaluate/run_ab.
"""
