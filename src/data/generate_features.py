"""End-to-end canonical feature generation script with strict validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.betting_loader import load_betting_data
from src.data.epa_loader import load_game_advanced_stats
from src.data.merger import merge_epa_data, merge_nfl_betting_data
from src.data.nfl_loader import load_schedules
from src.features.pipeline import (
    FEATURE_COLUMNS,
    XGB_FEATURE_COLUMNS,
    FeatureEngineeringPipeline,
)

# Data paths
RAW_DIR = Path("data/raw")
FEATURES_DIR = Path("data/features")

# Season ranges
TRAIN_SEASONS = list(range(2005, 2023))  # 2005-2022
TEST_SEASONS = list(range(2023, 2026))  # 2023-2025
ALL_SEASONS = TRAIN_SEASONS + TEST_SEASONS


def _atomic_write_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Write a CSV via a temporary file, then atomically replace the target."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(output_path)


def validate_dataset(df: pd.DataFrame, label: str) -> None:
    """
    Strict validation of a feature dataset. Raises on any violation.

    Checks:
    1. No season overlap between train/test (caller responsibility)
    2. Target integrity: every labeled game has spread >= 3
    3. No sub-3 spread games in the labeled set
    4. Upset rate is in a plausible range (15-40%)
    5. All 46 canonical features present and finite for labeled games
    6. No future data leakage: week 1 excluded, rolling stats NaN-free
    7. No duplicate game_ids in labeled set
    """
    labeled = df[df["upset"].notna()].copy()
    n_labeled = len(labeled)
    n_total = len(df)

    print(f"\n   --- Validating {label} ({n_labeled} labeled / {n_total} total) ---")

    # Check 1: Target integrity — every labeled game must have spread >= 3
    spread = labeled["spread_magnitude"]
    sub3_labeled = (spread < 3).sum()
    if sub3_labeled > 0:
        raise ValueError(
            f"{label}: {sub3_labeled} labeled games have spread < 3. "
            "Sub-3 games must have upset=NaN."
        )
    print("   [PASS] All labeled games have spread >= 3")

    # Check 2: No impossible upsets (upset=1 requires an underdog and winner)
    upsets = labeled[labeled["upset"] == 1]
    bad_upsets = upsets[upsets["underdog"].isna() | upsets["winner"].isna()]
    if len(bad_upsets) > 0:
        raise ValueError(
            f"{label}: {len(bad_upsets)} upset=1 games lack underdog or winner."
        )
    print("   [PASS] All upset=1 games have valid underdog and winner")

    # Check 3: Upset rate in plausible range
    upset_rate = labeled["upset"].mean()
    if not 0.15 <= upset_rate <= 0.40:
        raise ValueError(
            f"{label}: upset rate {upset_rate:.1%} outside plausible range [15%, 40%]."
        )
    print(f"   [PASS] Upset rate = {upset_rate:.1%} (plausible)")

    # Check 4: All canonical features present (base 46 + 24 XGB per-game)
    all_features = XGB_FEATURE_COLUMNS  # superset of FEATURE_COLUMNS
    missing = [f for f in all_features if f not in labeled.columns]
    if missing:
        raise ValueError(f"{label}: missing feature columns: {missing}")
    print(
        f"   [PASS] All {len(all_features)} feature columns present "
        f"({len(FEATURE_COLUMNS)} base + "
        f"{len(all_features) - len(FEATURE_COLUMNS)} XGB)"
    )

    # Check 5: No NaN/inf in features for labeled games
    feature_data = labeled[all_features]
    nan_counts = feature_data.isna().sum()
    bad_features = nan_counts[nan_counts > 0]
    if len(bad_features) > 0:
        raise ValueError(
            f"{label}: NaN values in features for labeled games: "
            f"{dict(bad_features)}"
        )
    inf_count = np.isinf(feature_data.values).sum()
    if inf_count > 0:
        raise ValueError(f"{label}: {inf_count} infinite values in features.")
    print("   [PASS] No NaN/inf in features for labeled games")

    # Check 6: No week 1 games (exclude_week_1=True)
    week1 = labeled[labeled["week"] == 1]
    if len(week1) > 0:
        raise ValueError(
            f"{label}: {len(week1)} week-1 games in labeled set. "
            "Week 1 should be excluded (no prior rolling stats)."
        )
    print("   [PASS] No week-1 games in labeled set")

    # Check 7: No duplicate game_ids
    if "game_id" in labeled.columns:
        dupes = labeled["game_id"].duplicated().sum()
        if dupes > 0:
            raise ValueError(f"{label}: {dupes} duplicate game_ids.")
        print("   [PASS] No duplicate game_ids")

    # Check 8: Season range
    seasons = sorted(labeled["season"].unique())
    print(f"   [PASS] Seasons: {seasons[0]}-{seasons[-1]} ({len(seasons)} seasons)")


def validate_no_overlap(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Verify no season overlap between train and test."""
    train_seasons = set(train["season"].unique())
    test_seasons = set(test["season"].unique())
    overlap = train_seasons & test_seasons
    if overlap:
        raise ValueError(f"Season overlap between train and test: {overlap}")
    print(
        f"   [PASS] No season overlap (train ends {max(train_seasons)}, "
        f"test starts {min(test_seasons)})"
    )


def main():
    print("=" * 60)
    print("FEATURE GENERATION PIPELINE")
    print("=" * 60)

    # Step 1: Load NFL schedules
    print("\n1. Loading NFL schedules...")
    schedules = load_schedules(ALL_SEASONS, regular_season_only=True)
    print(f"   Loaded {len(schedules)} games ({ALL_SEASONS[0]}-{ALL_SEASONS[-1]})")

    # Step 2: Load betting data
    print("\n2. Loading betting data...")
    betting = load_betting_data(
        RAW_DIR / "spreadspoke_scores.csv",
        min_season=ALL_SEASONS[0],
        max_season=ALL_SEASONS[-1],
    )
    print(f"   Loaded {len(betting)} betting rows")

    # Step 3: Merge schedule + betting
    print("\n3. Merging schedule and betting data...")
    merged, audit = merge_nfl_betting_data(schedules, betting)
    print(f"   Merged: {len(merged)} games (merge rate: {audit['merge_rate']:.1%})")

    # Step 4: Load advanced PBP stats
    print("\n4. Loading advanced stats from play-by-play...")
    epa_data = load_game_advanced_stats(ALL_SEASONS)
    print(f"   Loaded advanced stats for {len(epa_data)} games")

    # Step 5: Merge EPA
    print("\n5. Merging EPA data...")
    merged = merge_epa_data(merged, epa_data)
    epa_coverage = merged["home_off_pass_epa"].notna().mean()
    print(f"   EPA coverage: {epa_coverage:.1%}")

    # Step 6: Run feature pipeline
    print("\n6. Running feature engineering pipeline...")
    pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
    featured = pipeline.transform(merged)
    print(f"   Generated {len(featured)} rows")
    print(f"   Feature columns: {len(pipeline.get_feature_columns())}")

    # Step 7: Split train/test
    print("\n7. Splitting train/test...")
    train = featured[featured["season"].isin(TRAIN_SEASONS)].copy()
    test = featured[featured["season"].isin(TEST_SEASONS)].copy()

    train_labeled = train[train["upset"].notna()]
    test_labeled = test[test["upset"].notna()]
    print(f"   Train: {len(train_labeled)} labeled / {len(train)} total")
    print(f"   Test:  {len(test_labeled)} labeled / {len(test)} total")

    # Step 8: Strict validation
    print("\n8. Running strict validation...")
    validate_no_overlap(train, test)
    validate_dataset(train, "Train")
    validate_dataset(test, "Test")

    # Step 9: Save
    print("\n9. Saving CSVs...")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(train, FEATURES_DIR / "train.csv")
    _atomic_write_csv(test, FEATURES_DIR / "test.csv")
    print(f"   Saved {FEATURES_DIR / 'train.csv'} ({len(train)} rows)")
    print(f"   Saved {FEATURES_DIR / 'test.csv'} ({len(test)} rows)")

    print("\n" + "=" * 60)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
