"""Sequence builder for Siamese LSTM model.

Converts DataFrame to separate team game history sequences for the siamese LSTM architecture.
Each team's sequence is processed independently through the shared encoder.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


# Per-game features for sequence (what we track about each historical game)
SEQUENCE_FEATURES = [
    "points_scored",
    "points_allowed",
    "point_diff",
    "win",
]

# Matchup-level features (describe THIS specific game being predicted)
MATCHUP_FEATURES = [
    "spread_magnitude",
    "home_indicator",
    "divisional_game",
    "rest_advantage",
    "week_number",
    "primetime_game",
    "is_dome",
    "cold_weather",
    "windy_game",
    "over_under_normalized",
]

SEQUENCE_LENGTH = 5


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


# Backward compatibility alias
LSTMSequenceData = SiameseLSTMData


def _build_team_game_history(df: pd.DataFrame) -> Dict[Tuple[str, int], pd.DataFrame]:
    """
    Build game history for each team in each season.

    Args:
        df: DataFrame with game data including scores

    Returns:
        Dictionary mapping (team, season) to DataFrame of team's games in order
    """
    history: Dict[Tuple[str, int], List[dict]] = {}

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

        # Record for home team
        home_key = (home_team, season)
        if home_key not in history:
            history[home_key] = []
        history[home_key].append({
            "week": week,
            "points_scored": float(home_score),
            "points_allowed": float(away_score),
            "point_diff": float(home_score - away_score),
            "win": 1.0 if home_score > away_score else 0.0,
        })

        # Record for away team
        away_key = (away_team, season)
        if away_key not in history:
            history[away_key] = []
        history[away_key].append({
            "week": week,
            "points_scored": float(away_score),
            "points_allowed": float(home_score),
            "point_diff": float(away_score - home_score),
            "win": 1.0 if away_score > home_score else 0.0,
        })

    # Convert to DataFrames sorted by week
    result = {}
    for key, games in history.items():
        team_df = pd.DataFrame(games).sort_values("week").reset_index(drop=True)
        result[key] = team_df

    return result


def _get_team_sequence(
    team: str,
    season: int,
    week: int,
    team_history: Dict[Tuple[str, int], pd.DataFrame],
    seq_length: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the last N games for a team prior to a given week.

    Args:
        team: Team abbreviation
        season: Season year
        week: Week number (games BEFORE this week)
        team_history: Dictionary of team game histories
        seq_length: Number of games to include

    Returns:
        Tuple of (sequence array, mask array)
        - sequence: shape (seq_length, n_features)
        - mask: shape (seq_length,) with 1 for valid games, 0 for padding
    """
    n_features = len(SEQUENCE_FEATURES)
    sequence = np.zeros((seq_length, n_features))
    mask = np.zeros(seq_length)

    key = (team, season)
    if key not in team_history:
        return sequence, mask

    team_df = team_history[key]

    # Get games before this week
    prior_games = team_df[team_df["week"] < week]

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
        for j, feat in enumerate(SEQUENCE_FEATURES):
            sequence[idx, j] = game[feat]
        mask[idx] = 1.0

    return sequence, mask


def _normalize_sequences(
    sequences: np.ndarray,
    masks: np.ndarray,
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
    normalized = sequences.copy()
    computed_stats: Dict[str, Tuple[float, float]] = {}

    for f in range(n_features):
        feature_name = SEQUENCE_FEATURES[f] if f < len(SEQUENCE_FEATURES) else f"feat_{f}"
        feature_vals = sequences[:, :, f]

        if stats is not None and feature_name in stats:
            mean, std = stats[feature_name]
        else:
            valid_vals = feature_vals[masks > 0]
            mean = valid_vals.mean() if len(valid_vals) > 0 else 0.0
            std = valid_vals.std() if len(valid_vals) > 0 else 1.0

        computed_stats[feature_name] = (mean, std)

        if std > 0:
            normalized[:, :, f] = (feature_vals - mean) / std
        else:
            normalized[:, :, f] = feature_vals - mean

        # Zero out padded positions
        normalized[:, :, f] = normalized[:, :, f] * masks

    return normalized, computed_stats


def _extract_matchup_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract matchup-level features for each game.

    Args:
        df: DataFrame with engineered features

    Returns:
        Array of shape (n_samples, n_matchup_features)
    """
    features = []

    for feat in MATCHUP_FEATURES:
        if feat in df.columns:
            features.append(df[feat].fillna(0).values)
        else:
            features.append(np.zeros(len(df)))

    return np.column_stack(features)


def build_siamese_sequences(
    df: pd.DataFrame,
    normalize: bool = True,
    seq_length: int = SEQUENCE_LENGTH,
) -> SiameseLSTMData:
    """
    Build Siamese LSTM-ready sequences from game data.

    For each game, extracts SEPARATE sequences for:
    - Underdog's last N games
    - Favorite's last N games

    This enables true siamese processing where each team is encoded
    independently through the shared LSTM encoder.

    Args:
        df: DataFrame with game data and engineered features
        normalize: Whether to normalize sequence features
        seq_length: Number of historical games per team

    Returns:
        SiameseLSTMData containing all arrays needed for training
    """
    # Filter to games with valid targets
    valid_df = df[df["upset"].notna()].copy()

    if len(valid_df) == 0:
        raise ValueError("No games with valid upset targets found")

    # Build team game histories
    team_history = _build_team_game_history(df)

    n_samples = len(valid_df)
    n_seq_features = len(SEQUENCE_FEATURES)

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

        game_id = row.get("game_id", f"{season}_{week}_{row['home_team']}_{row['away_team']}")
        game_ids.append(game_id)

        if pd.isna(underdog) or pd.isna(favorite):
            continue

        # Get SEPARATE sequences for each team
        underdog_sequences[i], underdog_masks[i] = _get_team_sequence(
            underdog, season, week, team_history, seq_length
        )
        favorite_sequences[i], favorite_masks[i] = _get_team_sequence(
            favorite, season, week, team_history, seq_length
        )

    # Normalize sequences (separately for each team, using combined stats)
    if normalize:
        # Combine sequences to compute shared normalization stats
        all_sequences = np.concatenate([underdog_sequences, favorite_sequences], axis=0)
        all_masks = np.concatenate([underdog_masks, favorite_masks], axis=0)
        _, stats = _normalize_sequences(all_sequences, all_masks)

        # Apply same normalization to both
        underdog_sequences, _ = _normalize_sequences(underdog_sequences, underdog_masks, stats)
        favorite_sequences, _ = _normalize_sequences(favorite_sequences, favorite_masks, stats)

    # Extract matchup features
    matchup_features = _extract_matchup_features(valid_df)

    # Normalize matchup features
    if normalize:
        for f in range(matchup_features.shape[1]):
            col = matchup_features[:, f]
            std = col.std()
            if std > 0:
                matchup_features[:, f] = (col - col.mean()) / std

    # Extract targets
    targets = valid_df["upset"].values.astype(np.float32)

    return SiameseLSTMData(
        underdog_sequences=underdog_sequences.astype(np.float32),
        favorite_sequences=favorite_sequences.astype(np.float32),
        matchup_features=matchup_features.astype(np.float32),
        targets=targets,
        underdog_masks=underdog_masks.astype(np.float32),
        favorite_masks=favorite_masks.astype(np.float32),
        game_ids=np.array(game_ids),
    )


# Backward compatibility alias
build_lstm_sequences = build_siamese_sequences


def get_sequence_feature_names() -> List[str]:
    """Get names of sequence features (per timestep)."""
    return SEQUENCE_FEATURES.copy()


def get_matchup_feature_names() -> List[str]:
    """Get names of matchup-level features."""
    return MATCHUP_FEATURES.copy()
