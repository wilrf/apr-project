"""Game-level advanced stats extracted from nfl_data_py play-by-play."""

from __future__ import annotations

from typing import List

import nfl_data_py as nfl
import numpy as np
import pandas as pd

# Columns needed from PBP data — keeps memory manageable (full PBP is multi-GB)
_PBP_COLUMNS = [
    "game_id",
    "play_id",
    "home_team",
    "away_team",
    "posteam",
    "play_type",
    "epa",
    "success",
    "cpoe",
    "interception",
    "fumble_lost",
    "total_home_pass_epa",
    "total_away_pass_epa",
    "total_home_rush_epa",
    "total_away_rush_epa",
]


def _build_cumulative_epa_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """Return one row per game with the final cumulative EPA totals."""
    sort_cols = [column for column in ["game_id", "play_id"] if column in pbp.columns]
    return (
        pbp.sort_values(sort_cols)
        .groupby("game_id", sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _build_rate_stats_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate success rate, CPOE, and turnover margin to game level."""
    required = {
        "posteam",
        "play_type",
        "success",
        "cpoe",
        "interception",
        "fumble_lost",
    }
    if not required.issubset(pbp.columns):
        empty = pbp[["game_id"]].drop_duplicates().copy()
        for column in [
            "home_success_rate",
            "away_success_rate",
            "home_cpoe",
            "away_cpoe",
            "home_turnover_margin",
            "away_turnover_margin",
        ]:
            empty[column] = np.nan
        return empty

    offensive_plays = pbp[pbp["posteam"].notna()].copy()
    offense_mask = offensive_plays["play_type"].isin(["pass", "run"])
    offense_stats = offensive_plays[offense_mask].copy()

    grouped = (
        offense_stats.groupby(["game_id", "posteam"], dropna=False)
        .agg(
            success_rate=("success", "mean"),
            cpoe=("cpoe", "mean"),
            giveaways=("interception", "sum"),
            fumbles_lost=("fumble_lost", "sum"),
        )
        .reset_index()
    )
    grouped["turnovers"] = grouped["giveaways"].fillna(0) + grouped[
        "fumbles_lost"
    ].fillna(0)

    teams = offensive_plays[["game_id", "home_team", "away_team"]].drop_duplicates()
    merged = grouped.merge(teams, on="game_id", how="left")

    home = (
        merged[merged["posteam"] == merged["home_team"]][
            ["game_id", "success_rate", "cpoe", "turnovers"]
        ]
        .rename(
            columns={
                "success_rate": "home_success_rate",
                "cpoe": "home_cpoe",
                "turnovers": "home_turnovers",
            }
        )
        .reset_index(drop=True)
    )
    away = (
        merged[merged["posteam"] == merged["away_team"]][
            ["game_id", "success_rate", "cpoe", "turnovers"]
        ]
        .rename(
            columns={
                "success_rate": "away_success_rate",
                "cpoe": "away_cpoe",
                "turnovers": "away_turnovers",
            }
        )
        .reset_index(drop=True)
    )

    result = teams[["game_id"]].drop_duplicates().merge(home, on="game_id", how="left")
    result = result.merge(away, on="game_id", how="left")

    home_turnovers = result["home_turnovers"].fillna(0)
    away_turnovers = result["away_turnovers"].fillna(0)
    result["home_turnover_margin"] = away_turnovers - home_turnovers
    result["away_turnover_margin"] = home_turnovers - away_turnovers

    return result.drop(columns=["home_turnovers", "away_turnovers"])


def load_game_advanced_stats(seasons: List[int]) -> pd.DataFrame:
    """
    Load game-level EPA, success rate, CPOE, and turnover stats from PBP data.

    Args:
        seasons: List of NFL seasons to load (e.g., [2005, 2006, ...])

    Returns:
        DataFrame with one row per game and home/away advanced stats.
    """
    pbp = nfl.import_pbp_data(seasons, columns=_PBP_COLUMNS)

    game_epa = _build_cumulative_epa_frame(pbp).rename(
        columns={
            "total_home_pass_epa": "home_off_pass_epa",
            "total_home_rush_epa": "home_off_rush_epa",
            "total_away_pass_epa": "away_off_pass_epa",
            "total_away_rush_epa": "away_off_rush_epa",
        }
    )[
        [
            "game_id",
            "home_off_pass_epa",
            "home_off_rush_epa",
            "away_off_pass_epa",
            "away_off_rush_epa",
        ]
    ]

    rate_stats = _build_rate_stats_frame(pbp)
    return game_epa.merge(rate_stats, on="game_id", how="left")


def load_game_epa(seasons: List[int]) -> pd.DataFrame:
    """
    Load game-level EPA totals from play-by-play data.

    Args:
        seasons: List of NFL seasons to load (e.g., [2005, 2006, ...])

    Returns:
        DataFrame with one row per game and columns:
        game_id, home_off_pass_epa, home_off_rush_epa,
        away_off_pass_epa, away_off_rush_epa
    """
    advanced = load_game_advanced_stats(seasons)
    return advanced[
        [
            "game_id",
            "home_off_pass_epa",
            "home_off_rush_epa",
            "away_off_pass_epa",
            "away_off_rush_epa",
        ]
    ]
