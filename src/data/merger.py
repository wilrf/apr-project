"""Data merging utilities for NFL and betting datasets."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import TypedDict, Tuple

from src.data.betting_loader import TEAM_ABBR_MAP


def normalize_team_abbr(abbr: str) -> str:
    """Normalize team abbreviation to current franchise name."""
    if pd.isna(abbr):
        return abbr
    return TEAM_ABBR_MAP.get(abbr, abbr)


class MergeAudit(TypedDict):
    """Audit information from data merge."""
    unmatched_nfl: pd.DataFrame
    unmatched_betting: pd.DataFrame
    duplicate_matches: pd.DataFrame
    merge_rate: float


def merge_nfl_betting_data(
    nfl_df: pd.DataFrame,
    betting_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, MergeAudit]:
    """
    Merge NFL schedule data with betting data.

    Uses nflverse schedules as the spine. Joins on season, week, home_team, away_team.

    Args:
        nfl_df: NFL schedule DataFrame with game_id as canonical identifier
        betting_df: Betting DataFrame from Kaggle spreadspoke

    Returns:
        Tuple of (merged DataFrame, audit dictionary)
    """
    # Make copies to avoid modifying originals
    nfl_df = nfl_df.copy()
    betting_df = betting_df.copy()

    # Normalize NFL team abbreviations to match betting data (handles relocations)
    # STL -> LA, SD -> LAC, OAK -> LV, etc.
    for col in ["home_team", "away_team"]:
        if col in nfl_df.columns:
            nfl_df[col] = nfl_df[col].apply(normalize_team_abbr)

    # Standardize betting columns for merge
    betting_df = betting_df.rename(columns={
        "schedule_season": "season",
        "schedule_week": "week",
        "team_home": "home_team",
        "team_away": "away_team",
    })

    # Track original counts for audit
    nfl_count = len(nfl_df)

    # Merge on canonical keys
    merged = nfl_df.merge(
        betting_df,
        on=["season", "week", "home_team", "away_team"],
        how="left",
        indicator=True,
    )

    # Identify unmatched rows
    unmatched_nfl = merged[merged["_merge"] == "left_only"].copy()
    matched = merged[merged["_merge"] == "both"].copy()

    # Check for duplicates (multiple betting rows per game)
    duplicate_matches = matched[matched.duplicated(subset=["game_id"], keep=False)]

    # Keep first match if duplicates exist
    matched = matched.drop_duplicates(subset=["game_id"], keep="first")

    # Clean up merge indicator
    matched = matched.drop(columns=["_merge"])

    # Calculate merge rate
    merge_rate = len(matched) / nfl_count if nfl_count > 0 else 0.0

    # Prepare audit columns
    unmatched_cols = ["game_id", "season", "week", "home_team", "away_team"]
    unmatched_cols = [c for c in unmatched_cols if c in unmatched_nfl.columns]

    audit: MergeAudit = {
        "unmatched_nfl": unmatched_nfl[unmatched_cols] if unmatched_cols else unmatched_nfl,
        "unmatched_betting": pd.DataFrame(),  # Would need separate tracking
        "duplicate_matches": duplicate_matches,
        "merge_rate": merge_rate,
    }

    return matched, audit


def save_merge_audit(audit: MergeAudit, filepath: Path) -> None:
    """Save merge audit to CSV for review."""
    audit["unmatched_nfl"].to_csv(filepath, index=False)
