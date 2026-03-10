"""Canonical feature engineering pipeline for NFL upset prediction."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.elo import compute_pre_game_elo

ROLLING_WINDOW = 3

_ROLLING_EFFICIENCY_GROUP = [
    "underdog_pass_epa_roll3",
    "underdog_rush_epa_roll3",
    "underdog_success_rate_roll3",
    "underdog_cpoe_roll3",
    "underdog_turnover_margin_roll3",
    "favorite_pass_epa_roll3",
    "favorite_rush_epa_roll3",
    "favorite_success_rate_roll3",
    "favorite_cpoe_roll3",
    "favorite_turnover_margin_roll3",
]

_DIFFERENTIAL_GROUP = [
    "pass_epa_diff",
    "rush_epa_diff",
    "success_rate_diff",
    "cpoe_diff",
    "turnover_margin_diff",
]

_VOLATILITY_TREND_GROUP = [
    "underdog_total_epa_std_roll3",
    "favorite_total_epa_std_roll3",
    "total_epa_std_diff",
    "underdog_success_rate_std_roll3",
    "favorite_success_rate_std_roll3",
    "success_rate_std_diff",
    "underdog_total_epa_trend",
    "favorite_total_epa_trend",
    "total_epa_trend_diff",
    "underdog_success_rate_trend",
    "favorite_success_rate_trend",
    "success_rate_trend_diff",
]

_SCHEDULE_CONTEXT_GROUP = [
    "underdog_rest_days",
    "favorite_rest_days",
    "rest_days_diff",
    "short_week_game",
    "divisional_game",
]

_MARKET_GROUP = [
    "home_implied_points",
    "away_implied_points",
    "spread_magnitude",
    "total_line",
]

_ELO_GROUP = [
    "underdog_elo",
    "favorite_elo",
    "elo_diff",
]

_ENVIRONMENT_GROUP = [
    "temperature",
    "wind_speed",
    "is_dome",
    "temperature_missing",
    "wind_speed_missing",
]

_GAME_CONTEXT_GROUP = [
    "underdog_is_home",
    "week_number",
]

# Per-game lag features for XGBoost: individual game stats for last 3 games.
# Lets XGB discover game-by-game patterns that rolling averages hide
# (e.g., "terrible last game but good 3-game average").
_XGB_GAME_STATS = ["total_epa", "success_rate", "turnover_margin", "margin"]
_XGB_PER_GAME_GROUP: List[str] = [
    f"{role}_last{lag}_{stat}"
    for lag in [1, 2, 3]
    for role in ["underdog", "favorite"]
    for stat in _XGB_GAME_STATS
]  # 3 lags × 2 roles × 4 stats = 24 features

FEATURE_COLUMNS: List[str] = (
    _ROLLING_EFFICIENCY_GROUP
    + _DIFFERENTIAL_GROUP
    + _VOLATILITY_TREND_GROUP
    + _SCHEDULE_CONTEXT_GROUP
    + _MARKET_GROUP
    + _ELO_GROUP
    + _ENVIRONMENT_GROUP
    + _GAME_CONTEXT_GROUP
)

FEATURE_COLUMNS_NO_SPREAD: List[str] = [
    feature for feature in FEATURE_COLUMNS if feature not in set(_MARKET_GROUP)
]

# XGBoost gets the base 46 features PLUS per-game lag features (70 total).
# The lag features give XGB individual game stats that rolling averages destroy,
# letting it learn non-linear patterns like "bad last game, good 3-game average".
XGB_FEATURE_COLUMNS: List[str] = FEATURE_COLUMNS + _XGB_PER_GAME_GROUP  # 70

XGB_FEATURE_COLUMNS_NO_SPREAD: List[str] = [
    f for f in XGB_FEATURE_COLUMNS if f not in set(_MARKET_GROUP)
]  # 66

# LSTM matchup context: minimal, only things the sequence CAN'T know.
# The LSTM sequence encoder learns team form, trends, and volatility from
# raw game-by-game data. Matchup context provides only external context
# (opponent strength, market view, venue, weather, schedule).
LSTM_MATCHUP_FEATURES: List[str] = [
    "spread_magnitude",  # market's view
    "total_line",  # expected scoring environment
    "underdog_elo",  # underdog long-range strength
    "favorite_elo",  # favorite long-range strength
    "underdog_is_home",  # venue for this game
    "underdog_rest_days",  # schedule for this game
    "favorite_rest_days",  # schedule for this game
    "week_number",  # season context
    "divisional_game",  # rivalry factor
    "is_dome",  # weather/venue
]  # 10 total

_LSTM_MARKET_FEATURES = {"spread_magnitude", "total_line"}
LSTM_MATCHUP_FEATURES_NO_SPREAD: List[str] = [
    f for f in LSTM_MATCHUP_FEATURES if f not in _LSTM_MARKET_FEATURES
]  # 8 total


def _identify_underdog(row: pd.Series) -> str | None:
    """Identify the underdog team whenever the betting row names a favorite."""
    if pd.isna(row.get("team_favorite_id")) or pd.isna(row.get("spread_favorite")):
        return None

    favorite = row["team_favorite_id"]
    if favorite == row["home_team"]:
        return row["away_team"]
    if favorite == row["away_team"]:
        return row["home_team"]
    return None


def calculate_upset_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the canonical upset target.

    Only games with spread >= 3 receive a target label (0 or 1).
    Sub-3 spread games get upset=NaN and are excluded from training/evaluation.
    Games remain in the DataFrame for feature computation (rolling stats need them).
    """
    result = df.copy()
    spread_abs = pd.to_numeric(result["spread_favorite"], errors="coerce").abs()

    result["favorite"] = result["team_favorite_id"].where(
        result["team_favorite_id"].notna() & spread_abs.notna()
    )
    result["underdog"] = result.apply(_identify_underdog, axis=1)
    result["winner"] = None

    home_wins = result["home_score"] > result["away_score"]
    away_wins = result["away_score"] > result["home_score"]
    result.loc[home_wins, "winner"] = result.loc[home_wins, "home_team"]
    result.loc[away_wins, "winner"] = result.loc[away_wins, "away_team"]

    # Default to NaN — only eligible games get a label
    result["upset"] = np.nan

    eligible = spread_abs >= 3
    has_winner = result["winner"].notna()
    has_favorite = result["favorite"].notna()

    # Eligible games with a clear winner get upset = 0 or 1
    labelable = eligible & has_winner & has_favorite
    result.loc[labelable, "upset"] = (
        result.loc[labelable, "winner"] == result.loc[labelable, "underdog"]
    ).astype(float)

    result["upset_tier"] = pd.Categorical(
        np.select(
            [
                spread_abs.between(3, 6.5, inclusive="both"),
                spread_abs.between(7, 13.5, inclusive="both"),
                spread_abs >= 14,
            ],
            ["tier_1", "tier_2", "tier_3"],
            default="unlabeled",
        )
    )

    return result


def _sum_if_any(*values: float) -> float:
    """Return the sum of finite values, preserving NaN when all are missing."""
    valid = [value for value in values if pd.notna(value)]
    if not valid:
        return np.nan
    return float(sum(valid))


def _aggregate_team_games(df: pd.DataFrame) -> pd.DataFrame:
    """Convert game rows to one row per team per game."""
    records = []

    for _, row in df.iterrows():
        game_date = pd.to_datetime(row.get("gameday"), errors="coerce")
        for is_home, team, opponent, scored, allowed, prefix in [
            (
                1.0,
                row["home_team"],
                row["away_team"],
                row.get("home_score", np.nan),
                row.get("away_score", np.nan),
                "home",
            ),
            (
                0.0,
                row["away_team"],
                row["home_team"],
                row.get("away_score", np.nan),
                row.get("home_score", np.nan),
                "away",
            ),
        ]:
            pass_epa = row.get(f"{prefix}_off_pass_epa", np.nan)
            rush_epa = row.get(f"{prefix}_off_rush_epa", np.nan)
            records.append(
                {
                    "game_id": row["game_id"],
                    "season": row["season"],
                    "week": row["week"],
                    "team": team,
                    "opponent": opponent,
                    "is_home": is_home,
                    "game_date": game_date,
                    "points_scored": float(scored) if pd.notna(scored) else np.nan,
                    "points_allowed": float(allowed) if pd.notna(allowed) else np.nan,
                    "pass_epa": float(pass_epa) if pd.notna(pass_epa) else np.nan,
                    "rush_epa": float(rush_epa) if pd.notna(rush_epa) else np.nan,
                    "total_epa": _sum_if_any(pass_epa, rush_epa),
                    "success_rate": row.get(f"{prefix}_success_rate", np.nan),
                    "cpoe": row.get(f"{prefix}_cpoe", np.nan),
                    "turnover_margin": row.get(f"{prefix}_turnover_margin", np.nan),
                    "days_since_last_game": row.get(f"{prefix}_rest", np.nan),
                }
            )

    return pd.DataFrame(records)


def _sort_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically with safe fallbacks for tests."""
    sort_cols = [
        column
        for column in ["season", "week", "game_date", "game_id"]
        if column in team_games
    ]
    return team_games.sort_values(sort_cols).copy()


def _calculate_team_rollups(
    team_games: pd.DataFrame,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Compute rolling means, volatility, and trend for one team."""
    result = _sort_team_games(team_games)
    idx0 = result.index[0]

    for col in [
        "pass_epa",
        "rush_epa",
        "success_rate",
        "cpoe",
        "turnover_margin",
        "total_epa",
    ]:
        shifted = result[col].shift(1)
        result[f"{col}_roll{window}"] = shifted.rolling(
            window=window,
            min_periods=1,
        ).mean()
        result.loc[idx0, f"{col}_roll{window}"] = np.nan

    for col in ["total_epa", "success_rate"]:
        shifted = result[col].shift(1)
        result[f"{col}_std_roll{window}"] = shifted.rolling(
            window=window,
            min_periods=2,
        ).std()
        result.loc[idx0, f"{col}_std_roll{window}"] = np.nan

        result[f"{col}_trend"] = (
            shifted
            - shifted.rolling(
                window=window,
                min_periods=1,
            ).mean()
        )
        result.loc[idx0, f"{col}_trend"] = np.nan

    # Per-game lag features for XGBoost: individual stats from last 1, 2, 3 games.
    # shift(1) = most recent game before this one, shift(2) = two games ago, etc.
    margin = result["points_scored"] - result["points_allowed"]
    for lag in [1, 2, 3]:
        for col in ["total_epa", "success_rate", "turnover_margin"]:
            result[f"{col}_last{lag}"] = result[col].shift(lag)
        result[f"margin_last{lag}"] = margin.shift(lag)

    return result


def _compute_team_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.Series]:
    """Return (team, game_id) -> rolling-stat row across all seasons."""
    team_games = _aggregate_team_games(df)
    lookup: Dict[Tuple[str, str], pd.Series] = {}

    for team, group in team_games.groupby("team"):
        rolled = _calculate_team_rollups(group)
        for _, row in rolled.iterrows():
            lookup[(team, row["game_id"])] = row

    return lookup


def _get_total_line(df: pd.DataFrame) -> pd.Series:
    """Return the available over/under column as a float series."""
    for candidate in ["over_under_line", "total_line", "total"]:
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _get_numeric_series(
    df: pd.DataFrame,
    primary: str,
    fallback: str | None = None,
    default: float = np.nan,
) -> pd.Series:
    """Return a numeric series with safe defaults."""
    if primary in df.columns:
        return pd.to_numeric(df[primary], errors="coerce")
    if fallback is not None and fallback in df.columns:
        return pd.to_numeric(df[fallback], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _get_string_series(
    df: pd.DataFrame,
    column: str,
    default: str,
) -> pd.Series:
    """Return a string series with safe defaults."""
    if column in df.columns:
        return df[column].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype=str)


def get_feature_columns() -> List[str]:
    """Return the canonical flat feature list."""
    return FEATURE_COLUMNS.copy()


def get_no_spread_feature_columns() -> List[str]:
    """Return the feature list without the direct market features."""
    return FEATURE_COLUMNS_NO_SPREAD.copy()


def get_xgb_feature_columns() -> List[str]:
    """Return the expanded XGB feature list (base 46 + 24 per-game lag)."""
    return XGB_FEATURE_COLUMNS.copy()


def get_xgb_no_spread_feature_columns() -> List[str]:
    """Return the XGB feature list without market features."""
    return XGB_FEATURE_COLUMNS_NO_SPREAD.copy()


class FeatureEngineeringPipeline:
    """Multi-representation feature pipeline: 46 base + 24 XGB per-game lag features."""

    def __init__(
        self,
        exclude_week_1: bool = True,
        rolling_window: int = ROLLING_WINDOW,
    ):
        self.exclude_week_1 = exclude_week_1
        self.rolling_window = rolling_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the canonical feature engineering pipeline."""
        result = calculate_upset_target(df)

        elo = compute_pre_game_elo(result)
        result = result.merge(elo, on="game_id", how="left")

        team_lookup = _compute_team_lookup(result)

        spread_abs = pd.to_numeric(result["spread_favorite"], errors="coerce").abs()
        favorite_is_home = result["favorite"] == result["home_team"]
        total_line = _get_total_line(result).fillna(45.0)

        result["spread_magnitude"] = spread_abs.fillna(0.0)
        result["total_line"] = total_line
        result["home_implied_points"] = np.where(
            favorite_is_home,
            (total_line + spread_abs.fillna(0.0)) / 2.0,
            (total_line - spread_abs.fillna(0.0)) / 2.0,
        )
        result["away_implied_points"] = total_line - result["home_implied_points"]

        result["underdog_is_home"] = (result["underdog"] == result["home_team"]).astype(
            float
        )
        result["week_number"] = pd.to_numeric(result["week"], errors="coerce").astype(
            float
        )
        home_rest = _get_numeric_series(result, "home_rest")
        away_rest = _get_numeric_series(result, "away_rest")
        result["divisional_game"] = (
            pd.to_numeric(
                (
                    result["div_game"]
                    if "div_game" in result.columns
                    else pd.Series(0.0, index=result.index)
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .astype(float)
        )

        result["underdog_rest_days"] = np.where(
            result["underdog_is_home"] == 1.0,
            home_rest,
            away_rest,
        )
        result["favorite_rest_days"] = np.where(
            result["underdog_is_home"] == 1.0,
            away_rest,
            home_rest,
        )
        result["rest_days_diff"] = (
            result["underdog_rest_days"] - result["favorite_rest_days"]
        )
        result["short_week_game"] = (
            (result["underdog_rest_days"].fillna(7) <= 5)
            | (result["favorite_rest_days"].fillna(7) <= 5)
        ).astype(float)

        raw_temp = _get_numeric_series(result, "temp", fallback="weather_temperature")
        raw_wind = _get_numeric_series(result, "wind", fallback="weather_wind_mph")
        roof = _get_string_series(result, "roof", "outdoors")
        result["temperature_missing"] = raw_temp.isna().astype(float)
        result["wind_speed_missing"] = raw_wind.isna().astype(float)
        result["temperature"] = raw_temp.fillna(70.0)
        result["wind_speed"] = raw_wind.fillna(0.0)
        result["is_dome"] = roof.isin(["dome", "closed"]).astype(float)

        rolling_stats = [
            "pass_epa_roll3",
            "rush_epa_roll3",
            "success_rate_roll3",
            "cpoe_roll3",
            "turnover_margin_roll3",
            "total_epa_std_roll3",
            "success_rate_std_roll3",
            "total_epa_trend",
            "success_rate_trend",
        ]
        # Per-game lag stats for XGBoost (internal column names in team rollups)
        for prefix in ["underdog", "favorite"]:
            for stat in rolling_stats:
                result[f"{prefix}_{stat}"] = np.nan
            # Initialize XGB lag columns
            for lag in [1, 2, 3]:
                for stat in _XGB_GAME_STATS:
                    result[f"{prefix}_last{lag}_{stat}"] = np.nan

        for idx, row in result.iterrows():
            for prefix in ["underdog", "favorite"]:
                team = row.get(prefix)
                key = (team, row["game_id"])
                if team is None or key not in team_lookup:
                    continue

                lookup_row = team_lookup[key]
                for stat in rolling_stats:
                    result.at[idx, f"{prefix}_{stat}"] = lookup_row.get(stat, np.nan)
                # Extract per-game lag stats for XGBoost
                for lag in [1, 2, 3]:
                    for stat in _XGB_GAME_STATS:
                        internal_key = f"{stat}_last{lag}"
                        external_key = f"{prefix}_last{lag}_{stat}"
                        result.at[idx, external_key] = lookup_row.get(
                            internal_key, np.nan
                        )

        result["pass_epa_diff"] = (
            result["underdog_pass_epa_roll3"] - result["favorite_pass_epa_roll3"]
        )
        result["rush_epa_diff"] = (
            result["underdog_rush_epa_roll3"] - result["favorite_rush_epa_roll3"]
        )
        result["success_rate_diff"] = (
            result["underdog_success_rate_roll3"]
            - result["favorite_success_rate_roll3"]
        )
        result["cpoe_diff"] = (
            result["underdog_cpoe_roll3"] - result["favorite_cpoe_roll3"]
        )
        result["turnover_margin_diff"] = (
            result["underdog_turnover_margin_roll3"]
            - result["favorite_turnover_margin_roll3"]
        )

        result["total_epa_std_diff"] = (
            result["underdog_total_epa_std_roll3"]
            - result["favorite_total_epa_std_roll3"]
        )
        result["success_rate_std_diff"] = (
            result["underdog_success_rate_std_roll3"]
            - result["favorite_success_rate_std_roll3"]
        )
        result["total_epa_trend_diff"] = (
            result["underdog_total_epa_trend"] - result["favorite_total_epa_trend"]
        )
        result["success_rate_trend_diff"] = (
            result["underdog_success_rate_trend"]
            - result["favorite_success_rate_trend"]
        )

        result["underdog_elo"] = np.where(
            result["underdog_is_home"] == 1.0,
            result["home_elo_pre"],
            result["away_elo_pre"],
        )
        result["favorite_elo"] = np.where(
            result["underdog_is_home"] == 1.0,
            result["away_elo_pre"],
            result["home_elo_pre"],
        )
        result["elo_diff"] = result["underdog_elo"] - result["favorite_elo"]

        if self.exclude_week_1:
            result = result[result["week"] > 1].copy()

        # Fill NaN with 0.0 for all feature columns (base + XGB extras)
        all_feature_cols = set(FEATURE_COLUMNS + _XGB_PER_GAME_GROUP)
        for feature in all_feature_cols:
            result[feature] = pd.to_numeric(result[feature], errors="coerce").fillna(
                0.0
            )

        return result

    def get_feature_columns(self) -> List[str]:
        """Return the canonical flat feature list."""
        return FEATURE_COLUMNS.copy()

    def get_no_spread_feature_columns(self) -> List[str]:
        """Return the canonical no-spread flat feature list."""
        return FEATURE_COLUMNS_NO_SPREAD.copy()

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return feature groups for analysis and reporting."""
        return {
            "rolling_efficiency": list(_ROLLING_EFFICIENCY_GROUP),
            "differentials": list(_DIFFERENTIAL_GROUP),
            "volatility_trend": list(_VOLATILITY_TREND_GROUP),
            "schedule_context": list(_SCHEDULE_CONTEXT_GROUP),
            "market": list(_MARKET_GROUP),
            "elo": list(_ELO_GROUP),
            "environment": list(_ENVIRONMENT_GROUP),
            "game_context": list(_GAME_CONTEXT_GROUP),
        }

    def get_target_column(self) -> str:
        """Return the target column name."""
        return "upset"
