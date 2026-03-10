"""Sequence builder for Siamese LSTM model.

Converts DataFrame to separate team game history sequences for the
siamese LSTM architecture.
Each team's sequence is processed independently through the shared encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.pipeline import (
    LSTM_MATCHUP_FEATURES,
    LSTM_MATCHUP_FEATURES_NO_SPREAD,
)


@dataclass
class NormalizationStats:
    """Statistics for normalizing LSTM inputs (computed from training data only)."""

    sequence_stats: Dict[str, Tuple[float, float]]  # feature -> (mean, std)
    matchup_stats: Dict[str, Tuple[float, float]]  # feature -> (mean, std)


# Expanded sequence: 14 features per game per team.
# The LSTM learns rolling patterns, trends, and momentum from raw game-by-game data.
# No pre-computed rolling averages — the sequence IS the raw temporal data.
SEQUENCE_FEATURES = [
    "total_epa",  # offensive efficiency
    "pass_epa",
    "rush_epa",
    "success_rate",
    "cpoe",
    "turnover_margin",
    "points_scored",  # score context
    "points_allowed",
    "point_diff",  # net outcome (margin)
    "opponent_elo",  # quality of competition
    "win",  # binary outcome
    "was_home",  # venue
    "days_since_last_game",  # schedule
    "short_week",  # schedule flag
]  # 14 features

SEQUENCE_FEATURES_NO_SPREAD = SEQUENCE_FEATURES.copy()

# Minimal matchup context: only things the sequence CAN'T know.
MATCHUP_FEATURES = LSTM_MATCHUP_FEATURES.copy()  # 10
MATCHUP_FEATURES_NO_SPREAD = LSTM_MATCHUP_FEATURES_NO_SPREAD.copy()  # 8

# 8-game lookback: more temporal context for trend detection
SEQUENCE_LENGTH = 8


@dataclass
class SiameseLSTMData:
    """Container for Siamese LSTM-ready data with separate team sequences."""

    # Underdog sequences: shape (n_samples, seq_len, n_sequence_features)
    underdog_sequences: np.ndarray

    # Favorite sequences: shape (n_samples, seq_len, n_sequence_features)
    favorite_sequences: np.ndarray

    # Matchup features: shape (n_samples, n_matchup_features)
    matchup_features: np.ndarray

    # Targets: shape (n_samples,)
    targets: np.ndarray

    # Masks for underdog sequences: shape (n_samples, seq_len)
    underdog_masks: np.ndarray

    # Masks for favorite sequences: shape (n_samples, seq_len)
    favorite_masks: np.ndarray

    # Game identifiers for tracking
    game_ids: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.targets)

    @property
    def sequence_features_per_team(self) -> int:
        return self.underdog_sequences.shape[2]

    @property
    def n_matchup_features(self) -> int:
        return self.matchup_features.shape[1]


def _safe_float(row: pd.Series, key: str) -> float:
    """Extract a float from a row, returning NaN for missing values."""
    val = row.get(key, np.nan)
    return float(val) if pd.notna(val) else np.nan


def _build_team_game_history(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build game history for each team across all seasons.

    History crosses season boundaries so the LSTM can see late-season games
    when predicting early the following season.

    Args:
        df: DataFrame with game data including scores

    Returns:
        Dictionary mapping team to DataFrame of team's games in chronological order
    """
    history: Dict[str, List[dict]] = {}

    for _, row in df.iterrows():
        season = row["season"]
        week = row["week"]
        home_team = row["home_team"]
        away_team = row["away_team"]
        # Get scores
        home_score = row.get("home_score", row.get("score_home", np.nan))
        away_score = row.get("away_score", row.get("score_away", np.nan))

        if pd.isna(home_score) or pd.isna(away_score):
            continue

        home_rest = row.get("home_rest", 7.0)
        away_rest = row.get("away_rest", 7.0)
        home_rest = 7.0 if pd.isna(home_rest) else float(home_rest)
        away_rest = 7.0 if pd.isna(away_rest) else float(away_rest)

        # EPA and stat fields (graceful when missing)
        epa = {
            col: _safe_float(row, col)
            for col in [
                "home_off_pass_epa",
                "home_off_rush_epa",
                "away_off_pass_epa",
                "away_off_rush_epa",
                "home_success_rate",
                "away_success_rate",
                "home_cpoe",
                "away_cpoe",
                "home_turnover_margin",
                "away_turnover_margin",
            ]
        }

        def _sum_if_any(a: float, b: float) -> float:
            valid = [v for v in (a, b) if pd.notna(v)]
            return float(sum(valid)) if valid else np.nan

        # Build records for both teams with perspective swap
        perspectives = [
            (
                home_team,
                away_team,
                home_score,
                away_score,
                1.0,
                home_rest,
                "home",
                "away",
            ),
            (
                away_team,
                home_team,
                away_score,
                home_score,
                0.0,
                away_rest,
                "away",
                "home",
            ),
        ]
        for (
            team,
            opp,
            scored,
            allowed,
            is_home,
            rest,
            prefix,
            opp_prefix,
        ) in perspectives:
            pass_epa = epa[f"{prefix}_off_pass_epa"]
            rush_epa = epa[f"{prefix}_off_rush_epa"]

            # Opponent Elo (from pre-game ratings merged by pipeline)
            opp_elo_val = row.get(f"{opp_prefix}_elo_pre", 1500.0)
            opponent_elo = float(opp_elo_val) if pd.notna(opp_elo_val) else 1500.0

            if team not in history:
                history[team] = []
            history[team].append(
                {
                    "season": season,
                    "week": week,
                    "points_scored": float(scored),
                    "points_allowed": float(allowed),
                    "point_diff": float(scored - allowed),
                    "win": (
                        1.0 if scored > allowed else (0.5 if scored == allowed else 0.0)
                    ),
                    "was_home": is_home,
                    "rest_days": rest,
                    "days_since_last_game": rest,
                    "short_week": 1.0 if rest <= 5.0 else 0.0,
                    "opponent_elo": opponent_elo,
                    "off_pass_epa": pass_epa,
                    "off_rush_epa": rush_epa,
                    "def_pass_epa": epa[f"{opp_prefix}_off_pass_epa"],
                    "def_rush_epa": epa[f"{opp_prefix}_off_rush_epa"],
                    "pass_epa": pass_epa,
                    "rush_epa": rush_epa,
                    "total_epa": _sum_if_any(pass_epa, rush_epa),
                    "success_rate": epa[f"{prefix}_success_rate"],
                    "cpoe": epa[f"{prefix}_cpoe"],
                    "turnover_margin": epa[f"{prefix}_turnover_margin"],
                }
            )

    # Convert to DataFrames sorted chronologically
    result = {}
    for team, games in history.items():
        team_df = (
            pd.DataFrame(games).sort_values(["season", "week"]).reset_index(drop=True)
        )
        result[team] = team_df

    return result


def _get_team_sequence(
    team: str,
    season: int,
    week: int,
    team_history: Dict[str, pd.DataFrame],
    seq_length: int = SEQUENCE_LENGTH,
    sequence_features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the last N games for a team prior to a given week.

    History crosses season boundaries: early-season games draw from the
    prior season's tail.

    Args:
        team: Team abbreviation
        season: Season year
        week: Week number (games BEFORE this week)
        team_history: Dictionary mapping team → chronological game DataFrame
        seq_length: Number of games to include

    Returns:
        Tuple of (sequence array, mask array)
        - sequence: shape (seq_length, n_features)
        - mask: shape (seq_length,) with 1 for valid games, 0 for padding
    """
    active_sequence_features = (
        sequence_features if sequence_features is not None else SEQUENCE_FEATURES
    )
    n_features = len(active_sequence_features)
    sequence = np.zeros((seq_length, n_features))
    mask = np.zeros(seq_length)

    if team not in team_history:
        return sequence, mask

    team_df = team_history[team]

    # Get games chronologically before this (season, week)
    prior_games = team_df[
        (team_df["season"] < season)
        | ((team_df["season"] == season) & (team_df["week"] < week))
    ]

    if len(prior_games) == 0:
        return sequence, mask

    # Take last seq_length games
    recent_games = prior_games.tail(seq_length)
    n_games = len(recent_games)

    # Fill sequence from the end (most recent games at the end)
    # Padding at the beginning if fewer than seq_length games
    start_idx = seq_length - n_games

    for i, (_, game) in enumerate(recent_games.iterrows()):
        idx = start_idx + i
        for j, feat in enumerate(active_sequence_features):
            value = game[feat]
            sequence[idx, j] = 0.0 if pd.isna(value) else float(value)
        mask[idx] = 1.0

    return sequence, mask


def _normalize_sequences(
    sequences: np.ndarray,
    masks: np.ndarray,
    feature_names: List[str],
    stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
    """
    Normalize sequence features using stats from valid (non-padded) values.

    Args:
        sequences: shape (n_samples, seq_len, n_features)
        masks: shape (n_samples, seq_len)
        stats: Optional pre-computed (mean, std) per feature

    Returns:
        Tuple of (normalized sequences, stats dict)
    """
    n_features = sequences.shape[2]
    if len(feature_names) != n_features:
        raise ValueError(
            f"Expected {n_features} sequence feature names, got {len(feature_names)}"
        )
    normalized = sequences.copy()
    computed_stats: Dict[str, Tuple[float, float]] = {}

    for f in range(n_features):
        feature_name = feature_names[f]
        feature_vals = sequences[:, :, f]

        if stats is not None and feature_name in stats:
            mean, std = stats[feature_name]
        else:
            valid_vals = feature_vals[masks > 0]
            finite_vals = valid_vals[np.isfinite(valid_vals)]
            if len(finite_vals) > 0:
                mean = float(np.mean(finite_vals))
                std = float(np.std(finite_vals))
            else:
                mean = 0.0
                std = 1.0

        computed_stats[feature_name] = (mean, std)
        safe_feature_vals = np.nan_to_num(feature_vals, nan=mean)

        if std > 0:
            normalized[:, :, f] = (safe_feature_vals - mean) / std
        else:
            normalized[:, :, f] = safe_feature_vals - mean

        # Zero out padded positions
        normalized[:, :, f] = normalized[:, :, f] * masks

    return normalized, computed_stats


def _extract_matchup_features(
    df: pd.DataFrame,
    matchup_feature_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extract matchup-level features for each game.

    Args:
        df: DataFrame with engineered features
        matchup_feature_cols: Optional list of matchup feature names.
            Defaults to MATCHUP_FEATURES (10 LSTM matchup features).

    Returns:
        Array of shape (n_samples, n_matchup_features)
    """
    feat_list = (
        matchup_feature_cols if matchup_feature_cols is not None else MATCHUP_FEATURES
    )
    missing = [feat for feat in feat_list if feat not in df.columns]
    if missing:
        raise ValueError(f"Missing matchup feature columns: {missing}")
    features = []

    for feat in feat_list:
        features.append(pd.to_numeric(df[feat], errors="coerce").values)

    return np.column_stack(features)


def build_siamese_sequences(
    df: pd.DataFrame,
    normalize: bool = True,
    seq_length: int = SEQUENCE_LENGTH,
    stats: Optional[NormalizationStats] = None,
    matchup_feature_cols: Optional[List[str]] = None,
    sequence_feature_cols: Optional[List[str]] = None,
    history_df: Optional[pd.DataFrame] = None,
) -> Tuple[SiameseLSTMData, Optional[NormalizationStats]]:
    """
    Build Siamese LSTM-ready sequences from game data.

    For each game, extracts SEPARATE sequences for:
    - Underdog's last N games
    - Favorite's last N games

    This enables true siamese processing where each team is encoded
    independently through the shared LSTM encoder.

    Args:
        df: DataFrame with game data and engineered features.
            Only rows with valid upset labels become training/prediction samples.
        normalize: Whether to normalize sequence features
        seq_length: Number of historical games per team
        stats: Optional pre-computed normalization stats (from training data).
               If None and normalize=True, stats will be computed from this data.
               If provided and normalize=True, provided stats will be applied.
        matchup_feature_cols: Optional list of matchup feature names.
            Defaults to None which uses MATCHUP_FEATURES.
        sequence_feature_cols: Optional list of per-game sequence feature names.
            Defaults to None which uses SEQUENCE_FEATURES.
        history_df: Optional separate DataFrame for building team game histories.
            Use this to include games that don't have labels (e.g. sub-3-spread
            games) in team sequences while keeping labeled data separate.
            When None, ``df`` is used for both labels and history.

    Returns:
        Tuple of (SiameseLSTMData, NormalizationStats or None).
        Stats are returned only when normalize=True and stats=None (training mode).
    """
    # Filter to games with valid targets
    valid_df = df[df["upset"].notna()].copy()

    if len(valid_df) == 0:
        raise ValueError("No games with valid upset targets found")

    missing_role_mask = (
        valid_df.get("underdog", pd.Series(np.nan, index=valid_df.index)).isna()
        | valid_df.get("favorite", pd.Series(np.nan, index=valid_df.index)).isna()
    )
    if missing_role_mask.any():
        bad_game_ids = valid_df.loc[missing_role_mask, "game_id"].astype(str).tolist()
        raise ValueError(
            "Found labeled games missing favorite/underdog metadata: "
            + ", ".join(bad_game_ids[:5])
        )

    # Build team game histories from the fullest available source
    history_source = history_df if history_df is not None else df
    team_history = _build_team_game_history(history_source)
    if len(history_source) > 0 and not team_history:
        raise ValueError(
            "No usable team history could be built from history_df. "
            "Check that score columns are present and populated."
        )

    n_samples = len(valid_df)
    active_sequence_features = (
        sequence_feature_cols
        if sequence_feature_cols is not None
        else SEQUENCE_FEATURES
    )
    n_seq_features = len(active_sequence_features)

    # Initialize SEPARATE arrays for each team
    underdog_sequences = np.zeros((n_samples, seq_length, n_seq_features))
    favorite_sequences = np.zeros((n_samples, seq_length, n_seq_features))
    underdog_masks = np.zeros((n_samples, seq_length))
    favorite_masks = np.zeros((n_samples, seq_length))

    game_ids = []

    for i, (idx, row) in enumerate(valid_df.iterrows()):
        season = row["season"]
        week = row["week"]
        underdog = row.get("underdog")
        favorite = row.get("favorite")

        game_id = row.get(
            "game_id", f"{season}_{week}_{row['home_team']}_{row['away_team']}"
        )
        game_ids.append(game_id)

        if pd.isna(underdog) or pd.isna(favorite):
            continue

        # Get SEPARATE sequences for each team
        underdog_sequences[i], underdog_masks[i] = _get_team_sequence(
            underdog,
            season,
            week,
            team_history,
            seq_length,
            sequence_features=active_sequence_features,
        )
        favorite_sequences[i], favorite_masks[i] = _get_team_sequence(
            favorite,
            season,
            week,
            team_history,
            seq_length,
            sequence_features=active_sequence_features,
        )

    # Normalize sequences (separately for each team, using combined stats)
    sequence_stats: Optional[Dict[str, Tuple[float, float]]] = None
    if normalize:
        if stats is None:
            # TRAINING: Compute stats from this data
            all_sequences = np.concatenate(
                [underdog_sequences, favorite_sequences], axis=0
            )
            all_masks = np.concatenate([underdog_masks, favorite_masks], axis=0)
            _, sequence_stats = _normalize_sequences(
                all_sequences,
                all_masks,
                feature_names=active_sequence_features,
            )
        else:
            # VALIDATION/TEST: Use provided stats
            missing_sequence_stats = [
                feature
                for feature in active_sequence_features
                if feature not in stats.sequence_stats
            ]
            if missing_sequence_stats:
                raise ValueError(
                    f"Missing sequence normalization stats: {missing_sequence_stats}"
                )
            sequence_stats = stats.sequence_stats

        # Apply stats to both team sequences
        underdog_sequences, _ = _normalize_sequences(
            underdog_sequences,
            underdog_masks,
            feature_names=active_sequence_features,
            stats=sequence_stats,
        )
        favorite_sequences, _ = _normalize_sequences(
            favorite_sequences,
            favorite_masks,
            feature_names=active_sequence_features,
            stats=sequence_stats,
        )

    # Extract matchup features
    active_matchup_features = (
        matchup_feature_cols if matchup_feature_cols is not None else MATCHUP_FEATURES
    )
    matchup_features = _extract_matchup_features(valid_df, matchup_feature_cols)

    # Normalize matchup features
    matchup_stats: Optional[Dict[str, Tuple[float, float]]] = None
    if normalize:
        matchup_stats = {}
        if stats is not None:
            missing_matchup_stats = [
                feature
                for feature in active_matchup_features
                if feature not in stats.matchup_stats
            ]
            if missing_matchup_stats:
                raise ValueError(
                    f"Missing matchup normalization stats: {missing_matchup_stats}"
                )
        for f, feat_name in enumerate(active_matchup_features):
            col = matchup_features[:, f]

            if stats is None:
                # TRAINING: Compute from this data
                finite = col[np.isfinite(col)]
                if len(finite) > 0:
                    mean, std = float(np.mean(finite)), float(np.std(finite))
                else:
                    mean, std = 0.0, 1.0
            else:
                # VALIDATION/TEST: Use provided stats
                mean, std = stats.matchup_stats[feat_name]

            matchup_stats[feat_name] = (mean, std)
            safe_col = np.nan_to_num(col, nan=mean)
            if std > 0:
                matchup_features[:, f] = (safe_col - mean) / std
            else:
                matchup_features[:, f] = safe_col - mean
    else:
        matchup_features = np.nan_to_num(matchup_features, nan=0.0)

    # Extract targets
    targets = valid_df["upset"].values.astype(np.float32)

    # Build computed stats (only when training, i.e., stats=None and normalize=True)
    computed_stats: Optional[NormalizationStats] = None
    if (
        normalize
        and stats is None
        and sequence_stats is not None
        and matchup_stats is not None
    ):
        computed_stats = NormalizationStats(
            sequence_stats=sequence_stats,
            matchup_stats=matchup_stats,
        )

    return (
        SiameseLSTMData(
            underdog_sequences=underdog_sequences.astype(np.float32),
            favorite_sequences=favorite_sequences.astype(np.float32),
            matchup_features=matchup_features.astype(np.float32),
            targets=targets,
            underdog_masks=underdog_masks.astype(np.float32),
            favorite_masks=favorite_masks.astype(np.float32),
            game_ids=np.array(game_ids),
        ),
        computed_stats,
    )


def get_sequence_feature_names() -> List[str]:
    """Get names of sequence features (per timestep)."""
    return SEQUENCE_FEATURES.copy()


def get_matchup_feature_names() -> List[str]:
    """Get names of matchup-level features."""
    return MATCHUP_FEATURES.copy()
