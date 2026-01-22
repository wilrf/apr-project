"""Feature engineering pipeline combining all feature calculations.

Generates ~55 features for NFL upset prediction from schedule + betting data.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple

from src.features.target import calculate_upset_target

# Constants
ROLLING_WINDOW = 5
RECENT_WINDOW = 3  # For recent form comparison


# =============================================================================
# ROLLING STATS CALCULATION
# =============================================================================

def _aggregate_team_game_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Convert game-level data to team-game records (home + away rows per game)."""
    records = []
    for _, g in df.iterrows():
        home, away = g.get("home_score", np.nan), g.get("away_score", np.nan)
        valid = pd.notna(home) and pd.notna(away)
        base = {"game_id": g.get("game_id", f"{g['season']}_{g['week']}_{g['home_team']}_{g['away_team']}"),
                "season": g["season"], "week": g["week"]}
        for is_home, team, opp, scored, allowed in [
            (1, g["home_team"], g["away_team"], home, away),
            (0, g["away_team"], g["home_team"], away, home)
        ]:
            diff = scored - allowed if valid else np.nan
            win = (1 if scored > allowed else 0) if valid else np.nan
            records.append({**base, "team": team, "opponent": opp, "is_home": is_home,
                          "points_scored": scored, "points_allowed": allowed,
                          "point_diff": diff, "win": win, "loss": 1 - win if valid else 0})
    return pd.DataFrame(records)


def _calculate_rolling_stats_for_team(team_games: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Calculate rolling stats for a team's games using shift(1) to prevent data leakage."""
    r = team_games.sort_values("week").copy()
    idx0 = r.index[0]

    # Rolling means and std for core stats
    for col in ["points_scored", "points_allowed", "point_diff", "win"]:
        if col in r.columns:
            shifted = r[col].shift(1)
            r[f"{col}_roll{window}"] = shifted.rolling(window=window, min_periods=1).mean()
            r.loc[idx0, f"{col}_roll{window}"] = np.nan
            if col == "points_scored":
                r[f"{col}_std{window}"] = shifted.rolling(window=window, min_periods=2).std()
                r.loc[idx0, f"{col}_std{window}"] = np.nan

    # Recent form (3-game window)
    for col in ["points_scored", "point_diff"]:
        if col in r.columns:
            r[f"{col}_recent{RECENT_WINDOW}"] = r[col].shift(1).rolling(window=RECENT_WINDOW, min_periods=1).mean()
            r.loc[idx0, f"{col}_recent{RECENT_WINDOW}"] = np.nan

    # Streaks, trend, season stats
    r["win_streak"] = _calculate_win_streak(r["win"].shift(1))
    r["loss_streak"] = _calculate_win_streak(r["loss"].shift(1))
    r["point_diff_trend"] = _calculate_trend(r["point_diff"].shift(1), window)
    r["season_win_pct"] = r["win"].shift(1).expanding(min_periods=1).mean()
    r.loc[idx0, "season_win_pct"] = np.nan

    # Home/away splits
    for loc, mask in [("home", r["is_home"]), ("away", 1 - r["is_home"])]:
        wins = (r["win"] * mask).shift(1).expanding().sum()
        games = mask.shift(1).expanding().sum()
        r[f"{loc}_win_pct"] = wins / games.replace(0, np.nan)

    return r


def _calculate_win_streak(wins: pd.Series) -> pd.Series:
    """Calculate current win streak (consecutive wins)."""
    streak = pd.Series(0, index=wins.index)
    current_streak = 0

    for i, win in enumerate(wins):
        if pd.isna(win):
            streak.iloc[i] = 0
            current_streak = 0
        elif win == 1:
            current_streak += 1
            streak.iloc[i] = current_streak
        else:
            current_streak = 0
            streak.iloc[i] = 0

    return streak


def _calculate_trend(values: pd.Series, window: int) -> pd.Series:
    """Calculate linear trend (slope) over rolling window."""
    def slope(arr):
        if len(arr) < 2 or np.isnan(arr).all():
            return np.nan
        x = np.arange(len(arr))
        valid = ~np.isnan(arr)
        if valid.sum() < 2:
            return np.nan
        coeffs = np.polyfit(x[valid], arr[valid], 1)
        return coeffs[0]

    return values.rolling(window=window, min_periods=2).apply(slope, raw=True)


def _compute_all_rolling_stats(df: pd.DataFrame) -> Dict[Tuple[str, int, int], pd.Series]:
    """
    Compute rolling stats for all teams across all seasons.

    Returns dict mapping (team, season, week) to stats.
    """
    team_stats = _aggregate_team_game_stats(df)

    rolling_lookup = {}

    for (team, season), group in team_stats.groupby(["team", "season"]):
        rolling = _calculate_rolling_stats_for_team(group)
        for _, row in rolling.iterrows():
            key = (team, season, row["week"])
            rolling_lookup[key] = row

    return rolling_lookup


# =============================================================================
# SITUATIONAL FEATURES
# =============================================================================

def _add_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add situational/contextual features."""
    r = df.copy()

    def _col(name, default=0):
        """Get column or default Series."""
        return r[name] if name in r.columns else pd.Series(default, index=r.index)

    r["week_number"] = r["week"] / 18.0
    r["divisional_game"] = pd.to_numeric(_col("div_game", 0), errors="coerce").fillna(0).astype(float)

    # Rest days (vectorized)
    underdog_is_home = r["underdog"] == r["home_team"]
    r["underdog_rest"] = np.where(underdog_is_home, _col("home_rest", np.nan), _col("away_rest", np.nan))
    r["favorite_rest"] = np.where(underdog_is_home, _col("away_rest", np.nan), _col("home_rest", np.nan))
    r["rest_advantage"] = (pd.Series(r["underdog_rest"]) - pd.Series(r["favorite_rest"])).fillna(0).values

    # Rest-based flags
    for team in ["underdog", "favorite"]:
        r[f"{team}_off_bye"] = (r[f"{team}_rest"] >= 10).astype(float)
        r[f"{team}_short_week"] = (r[f"{team}_rest"] <= 5).astype(float)

    r["home_indicator"] = underdog_is_home.astype(float)
    r["neutral_site"] = pd.to_numeric(_col("stadium_neutral", 0), errors="coerce").fillna(0).astype(float)

    # Weather
    r["temperature"] = pd.to_numeric(_col("temp", _col("weather_temperature", 70)), errors="coerce").fillna(70)
    r["wind_speed"] = pd.to_numeric(_col("wind", _col("weather_wind_mph", 0)), errors="coerce").fillna(0)
    r["is_dome"] = _col("roof", "outdoors").isin(["dome", "closed"]).astype(float)
    outdoors = r["is_dome"] == 0
    r["cold_weather"] = ((r["temperature"] < 40) & outdoors).astype(float)
    r["windy_game"] = ((r["wind_speed"] > 15) & outdoors).astype(float)

    # Primetime detection (vectorized)
    weekday = _col("weekday", "Sunday").fillna("").astype(str).str.lower()
    gametime = _col("gametime", "1:00 PM").fillna("").astype(str).str.lower()
    r["primetime_game"] = (weekday.isin(["monday", "thursday"]) |
                           ((weekday == "sunday") & (gametime.str.contains("8:|20:")))).astype(float)

    return r


# =============================================================================
# SPREAD-RELATED FEATURES
# =============================================================================

def _add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add spread and betting-related features."""
    r = df.copy()
    spread_col = r["spread_favorite"] if "spread_favorite" in r.columns else pd.Series(0, index=r.index)
    r["spread_magnitude"] = pd.to_numeric(spread_col, errors="coerce").abs().fillna(0)

    # Over/under (try multiple column names)
    ou = next((pd.to_numeric(r[c], errors="coerce") for c in ["over_under_line", "total_line", "total"] if c in r.columns), None)
    r["over_under"] = ou.fillna(45.0) if ou is not None else 45.0
    r["over_under_normalized"] = (r["over_under"] - 45) / 10
    r["high_total"], r["low_total"] = (r["over_under"] > 48).astype(float), (r["over_under"] < 42).astype(float)

    # Spread categories
    sm = r["spread_magnitude"]
    r["spread_small"] = ((sm >= 3) & (sm < 7)).astype(float)
    r["spread_medium"] = ((sm >= 7) & (sm < 10)).astype(float)
    r["spread_large"] = (sm >= 10).astype(float)

    # Implied scores
    r["favorite_implied_score"] = (r["over_under"] + sm) / 2
    r["underdog_implied_score"] = (r["over_under"] - sm) / 2

    return r


# =============================================================================
# MATCHUP DIFFERENTIAL FEATURES
# =============================================================================

def _add_matchup_features(df: pd.DataFrame, rolling_lookup: Dict[Tuple[str, int, int], pd.Series]) -> pd.DataFrame:
    """Add matchup differential features using rolling stats."""
    r = df.copy()
    rolling_cols = ["points_scored_roll5", "points_allowed_roll5", "point_diff_roll5", "win_roll5",
                    "points_scored_std5", "points_scored_recent3", "point_diff_recent3", "win_streak",
                    "loss_streak", "point_diff_trend", "season_win_pct", "home_win_pct", "away_win_pct"]

    # Initialize and populate rolling stats
    for prefix in ["underdog", "favorite"]:
        for col in rolling_cols:
            r[f"{prefix}_{col}"] = np.nan
        for idx, row in r.iterrows():
            key = (row.get(prefix), row["season"], row["week"])
            if key[0] and key in rolling_lookup:
                for col in rolling_cols:
                    if col in rolling_lookup[key].index:
                        r.at[idx, f"{prefix}_{col}"] = rolling_lookup[key][col]

    # Differentials (all fillna(0))
    diffs = [("offense_defense_mismatch", "underdog_points_scored_roll5", "favorite_points_allowed_roll5"),
             ("defense_offense_mismatch", "favorite_points_scored_roll5", "underdog_points_allowed_roll5"),
             ("point_diff_differential", "underdog_point_diff_roll5", "favorite_point_diff_roll5"),
             ("win_pct_differential", "underdog_win_roll5", "favorite_win_roll5"),
             ("season_win_pct_diff", "underdog_season_win_pct", "favorite_season_win_pct"),
             ("consistency_diff", "favorite_points_scored_std5", "underdog_points_scored_std5"),
             ("momentum_diff", "underdog_point_diff_trend", "favorite_point_diff_trend"),
             ("recent_form_diff", "underdog_point_diff_recent3", "favorite_point_diff_recent3"),
             ("win_streak_diff", "underdog_win_streak", "favorite_win_streak")]
    for name, col_a, col_b in diffs:
        r[name] = (r[col_a] - r[col_b]).fillna(0)

    # Situational split differential
    is_home = r["home_indicator"] == 1
    r["underdog_relevant_split"] = np.where(is_home, r["underdog_home_win_pct"], r["underdog_away_win_pct"])
    r["favorite_relevant_split"] = np.where(is_home, r["favorite_away_win_pct"], r["favorite_home_win_pct"])
    r["situational_split_diff"] = (r["underdog_relevant_split"] - r["favorite_relevant_split"]).fillna(0)

    return r


# =============================================================================
# INTERACTION FEATURES
# =============================================================================

def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features combining multiple signals."""
    r = df.copy()
    sm = r["spread_magnitude"]

    # Multiplication interactions
    r["spread_x_momentum"] = sm * r["momentum_diff"]
    r["spread_x_win_streak"] = sm * r["underdog_win_streak"].fillna(0)
    r["rest_x_home"] = r["rest_advantage"] * r["home_indicator"]
    r["spread_x_point_diff"] = sm * (r["point_diff_differential"] / 10)
    r["momentum_x_home"] = r["momentum_diff"] * r["home_indicator"]
    r["divisional_x_spread"] = r["divisional_game"] * sm
    r["weather_x_spread"] = (r["cold_weather"] + r["windy_game"]) * sm

    # Indicator features
    r["hot_underdog"] = ((r["underdog_win_streak"].fillna(0) >= 2) & (sm > 6)).astype(float)
    r["cold_favorite"] = (r["favorite_loss_streak"].fillna(0) >= 2).astype(float)
    r["bounce_back"] = ((r["underdog_loss_streak"].fillna(0) >= 1) &
                        (r["underdog_season_win_pct"].fillna(0) > 0.5)).astype(float)

    return r


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================

class FeatureEngineeringPipeline:
    """
    Pipeline for generating ~55 features for NFL upset prediction.

    Features include:
    - Rolling team performance stats
    - Situational/contextual features
    - Spread and betting features
    - Matchup differentials
    - Momentum/trend indicators
    - Interaction features
    """

    def __init__(
        self,
        exclude_week_1: bool = True,
        rolling_window: int = ROLLING_WINDOW,
    ):
        """
        Initialize pipeline.

        Args:
            exclude_week_1: Whether to exclude Week 1 games (no prior data)
            rolling_window: Window size for rolling stats
        """
        self.exclude_week_1 = exclude_week_1
        self.rolling_window = rolling_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline.

        Args:
            df: Raw merged game data

        Returns:
            DataFrame with all engineered features (~55 features)
        """
        result = df.copy()

        # Step 1: Calculate target variable (identifies underdog/favorite)
        result = calculate_upset_target(result)

        # Step 2: Compute rolling stats for all teams
        rolling_lookup = _compute_all_rolling_stats(result)

        # Step 3: Add spread features
        result = _add_spread_features(result)

        # Step 4: Add situational features
        result = _add_situational_features(result)

        # Step 5: Add matchup differential features
        result = _add_matchup_features(result, rolling_lookup)

        # Step 6: Add interaction features
        result = _add_interaction_features(result)

        # Step 7: Exclude Week 1 if configured
        if self.exclude_week_1:
            result = result[result["week"] > 1].copy()

        # Step 8: Fill any remaining NaNs in feature columns
        for col in self.get_feature_columns():
            if col in result.columns:
                result[col] = result[col].fillna(0)

        return result

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get features organized by category for analysis."""
        return {
            "spread": ["spread_magnitude", "over_under", "over_under_normalized", "high_total", "low_total",
                       "spread_small", "spread_medium", "spread_large", "favorite_implied_score", "underdog_implied_score"],
            "situational": ["week_number", "divisional_game", "home_indicator", "neutral_site", "primetime_game",
                           "rest_advantage", "underdog_rest", "favorite_rest", "underdog_off_bye", "favorite_off_bye",
                           "underdog_short_week", "favorite_short_week", "is_dome", "cold_weather", "windy_game"],
            "underdog_performance": ["underdog_points_scored_roll5", "underdog_points_allowed_roll5",
                                     "underdog_point_diff_roll5", "underdog_win_roll5", "underdog_points_scored_std5",
                                     "underdog_win_streak", "underdog_loss_streak", "underdog_season_win_pct"],
            "favorite_performance": ["favorite_points_scored_roll5", "favorite_points_allowed_roll5",
                                     "favorite_point_diff_roll5", "favorite_win_roll5", "favorite_points_scored_std5",
                                     "favorite_win_streak", "favorite_loss_streak", "favorite_season_win_pct"],
            "matchup": ["offense_defense_mismatch", "defense_offense_mismatch", "point_diff_differential",
                       "win_pct_differential", "season_win_pct_diff", "consistency_diff", "momentum_diff",
                       "recent_form_diff", "win_streak_diff", "situational_split_diff"],
            "interaction": ["spread_x_momentum", "spread_x_win_streak", "rest_x_home", "spread_x_point_diff",
                           "momentum_x_home", "divisional_x_spread", "weather_x_spread", "hot_underdog",
                           "cold_favorite", "bounce_back"],
        }

    def get_feature_columns(self) -> List[str]:
        """Get list of all feature column names (~55 features)."""
        return [col for group in self.get_feature_groups().values() for col in group]

    def get_target_column(self) -> str:
        """Get the target column name."""
        return "upset"
