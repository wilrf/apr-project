# tests/models/test_sequence_builder.py
"""Tests for Siamese LSTM sequence builder."""
import pytest
import pandas as pd
import numpy as np
from src.models.sequence_builder import (
    build_siamese_sequences,
    _build_team_game_history,
    _get_team_sequence,
    SiameseLSTMData,
    SEQUENCE_FEATURES,
    MATCHUP_FEATURES,
    SEQUENCE_LENGTH,
)


@pytest.fixture
def sample_game_data():
    """Create sample game data for testing."""
    # Create 10 games over 5 weeks with 2 teams playing multiple times
    data = {
        "game_id": [f"2023_{w}_{h}_{a}" for w, h, a in [
            (2, "KC", "BUF"), (2, "PHI", "DAL"), (3, "KC", "DAL"),
            (3, "BUF", "PHI"), (4, "KC", "PHI"), (4, "BUF", "DAL"),
            (5, "DAL", "KC"), (5, "PHI", "BUF"), (6, "KC", "PHI"),
            (6, "DAL", "BUF"),
        ]],
        "season": [2023] * 10,
        "week": [2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        "home_team": ["KC", "PHI", "KC", "BUF", "KC", "BUF", "DAL", "PHI", "KC", "DAL"],
        "away_team": ["BUF", "DAL", "DAL", "PHI", "PHI", "DAL", "KC", "BUF", "PHI", "BUF"],
        "home_score": [28, 21, 31, 17, 24, 20, 14, 28, 30, 21],
        "away_score": [24, 24, 17, 21, 24, 24, 27, 21, 17, 24],
        # Betting data - KC and DAL are usually favorites
        "spread_favorite": [-7, -3, -6, 3, -4, 3, 7, -3, -5, 3],
        "underdog": ["BUF", "DAL", "DAL", "PHI", "PHI", "DAL", "KC", "BUF", "PHI", "BUF"],
        "favorite": ["KC", "PHI", "KC", "BUF", "KC", "BUF", "DAL", "PHI", "KC", "DAL"],
        "upset": [0, 1, 0, 1, 1, 1, 1, 0, 0, 1],  # Underdog wins
        # Matchup features
        "spread_magnitude": [7, 3, 6, 3, 4, 3, 7, 3, 5, 3],
        "home_indicator": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        "divisional_game": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "rest_advantage": [0, 0, 1, -1, 0, 0, 1, 0, 0, 0],
        "week_number": [w / 18.0 for w in [2, 2, 3, 3, 4, 4, 5, 5, 6, 6]],
        "primetime_game": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        "is_dome": [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        "cold_weather": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        "windy_game": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "over_under_normalized": [0.2, -0.1, 0.3, 0.0, 0.1, -0.2, 0.0, 0.1, 0.2, -0.1],
    }
    return pd.DataFrame(data)


class TestBuildTeamGameHistory:
    def test_creates_history_for_all_teams(self, sample_game_data):
        """Test that history is created for all teams."""
        history = _build_team_game_history(sample_game_data)

        teams = {"KC", "BUF", "PHI", "DAL"}
        for team in teams:
            assert (team, 2023) in history, f"Missing history for {team}"

    def test_game_count_per_team(self, sample_game_data):
        """Test that each team has correct number of games."""
        history = _build_team_game_history(sample_game_data)

        # Each team plays in 5 games across the 10-game dataset
        for team in ["KC", "BUF", "PHI", "DAL"]:
            assert len(history[(team, 2023)]) == 5, f"{team} should have 5 games"

    def test_stats_calculated_correctly(self, sample_game_data):
        """Test that game stats are calculated correctly."""
        history = _build_team_game_history(sample_game_data)

        # KC in week 2: home, scored 28, allowed 24
        kc_games = history[("KC", 2023)]
        week2_game = kc_games[kc_games["week"] == 2].iloc[0]

        assert week2_game["points_scored"] == 28
        assert week2_game["points_allowed"] == 24
        assert week2_game["point_diff"] == 4
        assert week2_game["win"] == 1.0


class TestGetTeamSequence:
    def test_returns_correct_shape(self, sample_game_data):
        """Test that sequence has correct shape."""
        history = _build_team_game_history(sample_game_data)

        seq, mask = _get_team_sequence("KC", 2023, 6, history)

        assert seq.shape == (SEQUENCE_LENGTH, len(SEQUENCE_FEATURES))
        assert mask.shape == (SEQUENCE_LENGTH,)

    def test_sequence_excludes_current_week(self, sample_game_data):
        """Test that sequence only includes games BEFORE current week."""
        history = _build_team_game_history(sample_game_data)

        # Week 4 prediction - should only include weeks 2-3 (2 games for KC)
        seq, mask = _get_team_sequence("KC", 2023, 4, history)

        # Only 2 valid games (weeks 2 and 3)
        assert mask.sum() == 2

    def test_padding_for_short_history(self, sample_game_data):
        """Test that short histories are padded correctly."""
        history = _build_team_game_history(sample_game_data)

        # Week 3: KC has only 1 prior game (week 2)
        seq, mask = _get_team_sequence("KC", 2023, 3, history)

        # Should have 1 valid game, 4 padded
        assert mask.sum() == 1
        # Valid game should be at the END of sequence
        assert mask[-1] == 1
        assert mask[:4].sum() == 0

    def test_empty_history_returns_zeros(self, sample_game_data):
        """Test that unknown team returns zero sequence."""
        history = _build_team_game_history(sample_game_data)

        seq, mask = _get_team_sequence("UNKNOWN", 2023, 5, history)

        assert mask.sum() == 0
        assert seq.sum() == 0


class TestBuildSiameseSequences:
    def test_returns_correct_structure(self, sample_game_data):
        """Test that build_siamese_sequences returns SiameseLSTMData."""
        result = build_siamese_sequences(sample_game_data, normalize=False)

        assert isinstance(result, SiameseLSTMData)
        assert hasattr(result, "underdog_sequences")
        assert hasattr(result, "favorite_sequences")
        assert hasattr(result, "matchup_features")
        assert hasattr(result, "targets")
        assert hasattr(result, "underdog_masks")
        assert hasattr(result, "favorite_masks")
        assert hasattr(result, "game_ids")

    def test_separate_sequences_for_each_team(self, sample_game_data):
        """Test that underdog and favorite have separate sequences."""
        result = build_siamese_sequences(sample_game_data, normalize=False)

        n_samples = len(sample_game_data[sample_game_data["upset"].notna()])
        n_features = len(SEQUENCE_FEATURES)

        # Each team should have their own sequence array
        assert result.underdog_sequences.shape == (n_samples, SEQUENCE_LENGTH, n_features)
        assert result.favorite_sequences.shape == (n_samples, SEQUENCE_LENGTH, n_features)

        # They should NOT be concatenated
        assert result.underdog_sequences.shape[2] == n_features  # Not n_features * 2

    def test_matchup_features_shape(self, sample_game_data):
        """Test that matchup features have correct shape."""
        result = build_siamese_sequences(sample_game_data, normalize=False)

        n_samples = len(sample_game_data[sample_game_data["upset"].notna()])

        assert result.matchup_features.shape == (n_samples, len(MATCHUP_FEATURES))

    def test_separate_masks_for_each_team(self, sample_game_data):
        """Test that underdog and favorite have separate masks."""
        result = build_siamese_sequences(sample_game_data, normalize=False)

        n_samples = len(sample_game_data[sample_game_data["upset"].notna()])

        assert result.underdog_masks.shape == (n_samples, SEQUENCE_LENGTH)
        assert result.favorite_masks.shape == (n_samples, SEQUENCE_LENGTH)

    def test_targets_match_input(self, sample_game_data):
        """Test that targets are extracted correctly."""
        result = build_siamese_sequences(sample_game_data, normalize=False)

        expected_targets = sample_game_data[sample_game_data["upset"].notna()]["upset"].values
        np.testing.assert_array_equal(result.targets, expected_targets)

    def test_normalization_applied(self, sample_game_data):
        """Test that normalization changes values."""
        result_normalized = build_siamese_sequences(sample_game_data, normalize=True)
        result_raw = build_siamese_sequences(sample_game_data, normalize=False)

        # Normalized values should be different from raw
        assert not np.allclose(
            result_normalized.underdog_sequences,
            result_raw.underdog_sequences,
            atol=1e-6,
        )

    def test_filters_invalid_targets(self):
        """Test that games without targets are filtered."""
        data = pd.DataFrame({
            "game_id": ["g1", "g2", "g3"],
            "season": [2023, 2023, 2023],
            "week": [2, 3, 4],
            "home_team": ["KC", "KC", "KC"],
            "away_team": ["BUF", "BUF", "BUF"],
            "home_score": [28, 24, 21],
            "away_score": [24, 21, 28],
            "underdog": ["BUF", "BUF", "BUF"],
            "favorite": ["KC", "KC", "KC"],
            "upset": [0, 1, np.nan],  # Third game has no target
            "spread_magnitude": [3, 3, 3],
            "home_indicator": [0, 0, 0],
            "divisional_game": [0, 0, 0],
            "rest_advantage": [0, 0, 0],
            "week_number": [0.1, 0.15, 0.2],
            "primetime_game": [0, 0, 0],
            "is_dome": [0, 0, 0],
            "cold_weather": [0, 0, 0],
            "windy_game": [0, 0, 0],
            "over_under_normalized": [0, 0, 0],
        })

        result = build_siamese_sequences(data, normalize=False)
        assert result.n_samples == 2
