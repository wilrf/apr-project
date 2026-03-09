"""Data loading, merging, and validation.

Pipeline: nfl_loader → betting_loader → epa_loader → merger → generate_features
Output:   data/features/train.csv (2005-2022), data/features/test.csv (2023-2025)

Key contracts:
- merger.py produces one row per game with scores, betting lines, and 10 EPA stats.
- generate_features.py runs 8 validation checks on each split before saving.
- betting_loader.TEAM_ABBR_MAP normalizes relocations (STL→LA, SD→LAC, OAK→LV).
- elo.py computes pre-game Elo inline during feature engineering (K=20, home=50).

If you change merger column output → update pipeline.py expected columns.
If you change team abbreviation mapping → check merger, generate_features, verify_data.
"""
