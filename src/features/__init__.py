"""Feature engineering — multi-representation pipeline.

Core module: pipeline.py (FeatureEngineeringPipeline)
  Input:  merged DataFrame (one row per game, from data/merger.py)
  Output: DataFrame with 70 feature columns + metadata + upset label

Representations (same data, different views):
  LR  = 46 base features (rolling avgs, diffs, market, Elo, environment)
  XGB = 70 features (base 46 + 24 per-game lag stats)
  LSTM = 14 sequence features × 8 timesteps + 10 matchup context

Canonical column lists are module-level constants in pipeline.py:
  FEATURE_COLUMNS (46), XGB_FEATURE_COLUMNS (70), LSTM_SEQUENCE_FEATURES (14),
  LSTM_MATCHUP_FEATURES (10), and their no-spread variants.

If you change column lists → update sequence_builder, target, generate_features,
  evaluate_test_set, run_ab_experiment (they import these constants).
"""
