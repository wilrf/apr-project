"""Tests for the canonical Siamese LSTM sequence builder."""

import numpy as np
import pandas as pd
import pytest

from src.models.sequence_builder import (
    MATCHUP_FEATURES,
    MATCHUP_FEATURES_NO_SPREAD,
    SEQUENCE_FEATURES,
    SEQUENCE_FEATURES_NO_SPREAD,
    SEQUENCE_LENGTH,
    NormalizationStats,
    SiameseLSTMData,
    _build_team_game_history,
    _get_team_sequence,
    _normalize_sequences,
    build_siamese_sequences,
)


def _with_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical matchup features used by the LSTM head."""
    result = df.copy()

    for feature in MATCHUP_FEATURES:
        if feature not in result.columns:
            result[feature] = 0.0

    result["week_number"] = result["week"].astype(float)
    result["spread_magnitude"] = result["spread_favorite"].abs()
    result["underdog_is_home"] = (result["underdog"] == result["home_team"]).astype(
        float
    )
    result["home_implied_points"] = np.linspace(24.0, 27.0, len(result))
    result["away_implied_points"] = np.linspace(20.0, 23.0, len(result))
    result["total_line"] = result["home_implied_points"] + result["away_implied_points"]
    result["underdog_elo"] = np.linspace(1450.0, 1495.0, len(result))
    result["favorite_elo"] = np.linspace(1510.0, 1540.0, len(result))
    result["elo_diff"] = result["underdog_elo"] - result["favorite_elo"]
    result["pass_epa_diff"] = np.linspace(-3.0, 3.0, len(result))
    result["rush_epa_diff"] = np.linspace(-1.5, 1.5, len(result))
    result["success_rate_diff"] = np.linspace(-0.08, 0.08, len(result))
    result["cpoe_diff"] = np.linspace(-3.0, 3.0, len(result))
    result["turnover_margin_diff"] = np.linspace(-1.0, 1.0, len(result))
    result["temperature"] = np.linspace(35.0, 70.0, len(result))
    result["wind_speed"] = np.linspace(3.0, 14.0, len(result))
    result["is_dome"] = [1.0, 0.0] * (len(result) // 2) + [1.0] * (len(result) % 2)
    return result


@pytest.fixture
def sample_game_data():
    """Create sample game data for sequence-builder tests."""
    data = pd.DataFrame(
        {
            "game_id": [
                "2023_02_KC_BUF",
                "2023_02_PHI_DAL",
                "2023_03_KC_DAL",
                "2023_03_BUF_PHI",
                "2023_04_KC_PHI",
                "2023_04_BUF_DAL",
                "2023_05_DAL_KC",
                "2023_05_PHI_BUF",
                "2023_06_KC_PHI",
                "2023_06_DAL_BUF",
            ],
            "season": [2023] * 10,
            "week": [2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "home_team": [
                "KC",
                "PHI",
                "KC",
                "BUF",
                "KC",
                "BUF",
                "DAL",
                "PHI",
                "KC",
                "DAL",
            ],
            "away_team": [
                "BUF",
                "DAL",
                "DAL",
                "PHI",
                "PHI",
                "DAL",
                "KC",
                "BUF",
                "PHI",
                "BUF",
            ],
            "home_score": [28, 21, 31, 17, 24, 20, 14, 28, 30, 21],
            "away_score": [24, 24, 17, 21, 24, 24, 27, 21, 17, 24],
            "spread_favorite": [-7.0, -3.0, -6.0, 3.0, -4.0, 3.0, 7.0, -3.0, -5.0, 3.0],
            "favorite": [
                "KC",
                "PHI",
                "KC",
                "BUF",
                "KC",
                "BUF",
                "DAL",
                "PHI",
                "KC",
                "DAL",
            ],
            "underdog": [
                "BUF",
                "DAL",
                "DAL",
                "PHI",
                "PHI",
                "DAL",
                "KC",
                "BUF",
                "PHI",
                "BUF",
            ],
            "upset": [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            "home_rest": [7, 10, 7, 8, 6, 7, 6, 10, 7, 5],
            "away_rest": [7, 10, 8, 7, 6, 7, 7, 10, 7, 5],
            "home_off_pass_epa": [
                15.2,
                8.5,
                18.3,
                6.1,
                12.0,
                7.5,
                5.0,
                14.2,
                16.8,
                8.9,
            ],
            "home_off_rush_epa": [3.4, 2.1, 5.0, 1.2, 3.1, 1.8, 0.5, 3.8, 4.2, 2.0],
            "away_off_pass_epa": [
                12.1,
                9.3,
                7.0,
                10.2,
                11.5,
                9.8,
                13.5,
                8.0,
                6.5,
                11.0,
            ],
            "away_off_rush_epa": [2.8, 3.5, 1.5, 2.5, 3.0, 3.2, 4.0, 1.8, 1.2, 2.9],
            "home_success_rate": [
                0.48,
                0.44,
                0.50,
                0.42,
                0.47,
                0.45,
                0.38,
                0.51,
                0.53,
                0.46,
            ],
            "away_success_rate": [
                0.51,
                0.47,
                0.41,
                0.49,
                0.50,
                0.48,
                0.52,
                0.43,
                0.40,
                0.50,
            ],
            "home_cpoe": [1.2, -0.8, 0.7, -1.0, 0.4, -0.3, -2.1, 1.1, 1.4, -0.1],
            "away_cpoe": [2.1, 0.5, -0.7, 1.8, 1.0, 0.3, 2.4, -1.2, -0.6, 1.3],
            "home_turnover_margin": [
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                -2.0,
                1.0,
                1.0,
                0.0,
            ],
            "away_turnover_margin": [
                1.0,
                -1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                2.0,
                -1.0,
                -1.0,
                1.0,
            ],
        }
    )
    return _with_matchup_features(data)


class TestBuildTeamGameHistory:
    def test_creates_history_for_all_teams(self, sample_game_data):
        history = _build_team_game_history(sample_game_data)
        for team in {"KC", "BUF", "PHI", "DAL"}:
            assert team in history

    def test_history_contains_derived_sequence_fields(self, sample_game_data):
        history = _build_team_game_history(sample_game_data)
        kc = history["KC"]
        kc_week2 = kc[(kc["season"] == 2023) & (kc["week"] == 2)].iloc[0]

        assert kc_week2["points_scored"] == 28
        assert kc_week2["points_allowed"] == 24
        assert kc_week2["was_home"] == 1.0
        assert kc_week2["days_since_last_game"] == 7.0
        assert kc_week2["pass_epa"] == pytest.approx(15.2)
        assert kc_week2["rush_epa"] == pytest.approx(3.4)
        assert kc_week2["total_epa"] == pytest.approx(18.6)
        assert kc_week2["success_rate"] == pytest.approx(0.48)
        assert kc_week2["cpoe"] == pytest.approx(1.2)
        assert kc_week2["turnover_margin"] == pytest.approx(-1.0)

    def test_missing_rest_defaults_to_seven_days(self):
        history = _build_team_game_history(
            pd.DataFrame(
                {
                    "season": [2023],
                    "week": [2],
                    "home_team": ["KC"],
                    "away_team": ["BUF"],
                    "home_score": [24],
                    "away_score": [21],
                }
            )
        )

        assert history["KC"].iloc[0]["days_since_last_game"] == 7.0
        assert history["BUF"].iloc[0]["days_since_last_game"] == 7.0


class TestGetTeamSequence:
    def test_returns_canonical_shape(self, sample_game_data):
        history = _build_team_game_history(sample_game_data)
        seq, mask = _get_team_sequence("KC", 2023, 6, history)

        assert seq.shape == (SEQUENCE_LENGTH, len(SEQUENCE_FEATURES))
        assert mask.shape == (SEQUENCE_LENGTH,)

    def test_padding_places_recent_games_at_end(self, sample_game_data):
        history = _build_team_game_history(sample_game_data)
        seq, mask = _get_team_sequence("KC", 2023, 3, history)

        assert mask.sum() == 1
        assert mask[-1] == 1.0
        assert mask[:-1].sum() == 0.0

        points_idx = SEQUENCE_FEATURES.index("points_scored")
        pass_epa_idx = SEQUENCE_FEATURES.index("pass_epa")
        assert seq[-1, points_idx] == pytest.approx(28.0)
        assert seq[-1, pass_epa_idx] == pytest.approx(15.2)

    def test_custom_sequence_features_change_width(self, sample_game_data):
        history = _build_team_game_history(sample_game_data)
        seq, mask = _get_team_sequence(
            "KC",
            2023,
            6,
            history,
            sequence_features=["points_scored", "was_home"],
        )

        assert seq.shape == (SEQUENCE_LENGTH, 2)
        assert mask.shape == (SEQUENCE_LENGTH,)

    def test_missing_sequence_values_are_zero_filled(self, sample_game_data):
        history = _build_team_game_history(sample_game_data)
        kc = history["KC"]
        history["KC"].loc[(kc["season"] == 2023) & (kc["week"] == 2), "cpoe"] = np.nan

        seq, _ = _get_team_sequence("KC", 2023, 3, history)
        cpoe_idx = SEQUENCE_FEATURES.index("cpoe")
        assert seq[-1, cpoe_idx] == 0.0


class TestBuildSiameseSequences:
    def test_returns_expected_container(self, sample_game_data):
        result, stats = build_siamese_sequences(sample_game_data, normalize=False)
        assert isinstance(result, SiameseLSTMData)
        assert stats is None

    def test_default_shapes_match_canonical_schema(self, sample_game_data):
        result, _ = build_siamese_sequences(sample_game_data, normalize=False)

        assert result.underdog_sequences.shape == (
            result.n_samples,
            SEQUENCE_LENGTH,
            len(SEQUENCE_FEATURES),
        )
        assert result.favorite_sequences.shape == (
            result.n_samples,
            SEQUENCE_LENGTH,
            len(SEQUENCE_FEATURES),
        )
        assert result.matchup_features.shape == (
            result.n_samples,
            len(MATCHUP_FEATURES),
        )
        assert result.matchup_features.shape[1] == 10

    def test_no_spread_matchup_variant_removes_market_columns(self, sample_game_data):
        result, stats = build_siamese_sequences(
            sample_game_data,
            normalize=True,
            matchup_feature_cols=MATCHUP_FEATURES_NO_SPREAD,
            sequence_feature_cols=SEQUENCE_FEATURES_NO_SPREAD,
        )

        assert result.matchup_features.shape[1] == len(MATCHUP_FEATURES_NO_SPREAD)
        assert result.matchup_features.shape[1] == 8
        assert stats is not None
        assert "spread_magnitude" not in stats.matchup_stats
        assert "total_line" not in stats.matchup_stats

    def test_sequences_preserve_home_flag_by_team(self, sample_game_data):
        result, _ = build_siamese_sequences(sample_game_data, normalize=False)
        was_home_idx = SEQUENCE_FEATURES.index("was_home")

        assert result.favorite_sequences[2, -1, was_home_idx] == 1.0
        assert result.underdog_sequences[2, -1, was_home_idx] == 0.0

    def test_targets_follow_input_rows(self, sample_game_data):
        result, _ = build_siamese_sequences(sample_game_data, normalize=False)
        expected = sample_game_data["upset"].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(result.targets, expected)

    def test_normalization_returns_feature_stats(self, sample_game_data):
        _, stats = build_siamese_sequences(sample_game_data, normalize=True)

        assert isinstance(stats, NormalizationStats)
        assert set(stats.sequence_stats) == set(SEQUENCE_FEATURES)
        assert set(stats.matchup_stats) == set(MATCHUP_FEATURES)

    def test_provided_stats_apply_without_recomputing(self, sample_game_data):
        train_data, train_stats = build_siamese_sequences(
            sample_game_data, normalize=True
        )
        reapplied_data, returned_stats = build_siamese_sequences(
            sample_game_data,
            normalize=True,
            stats=train_stats,
        )

        assert returned_stats is None
        np.testing.assert_allclose(
            train_data.underdog_sequences,
            reapplied_data.underdog_sequences,
        )
        np.testing.assert_allclose(
            train_data.matchup_features,
            reapplied_data.matchup_features,
        )

    def test_normalization_ignores_nan_values_when_computing_stats(self):
        sequences = np.array(
            [
                [[np.nan], [1.0]],
                [[2.0], [3.0]],
            ],
            dtype=np.float32,
        )
        masks = np.ones((2, 2), dtype=np.float32)

        normalized, stats = _normalize_sequences(sequences, masks, ["feature"])

        mean, std = stats["feature"]
        assert mean == pytest.approx(2.0)
        assert std == pytest.approx(np.std([1.0, 2.0, 3.0]))
        assert np.isfinite(mean)
        assert np.isfinite(std)
        assert np.isfinite(normalized[0, 1, 0])
        assert np.isfinite(normalized[1, 0, 0])
        assert np.isfinite(normalized[1, 1, 0])

    def test_provided_stats_must_cover_all_sequence_features(self, sample_game_data):
        incomplete = NormalizationStats(
            sequence_stats={"total_epa": (0.0, 1.0)},
            matchup_stats={name: (0.0, 1.0) for name in MATCHUP_FEATURES},
        )

        with pytest.raises(ValueError, match="Missing sequence normalization stats"):
            build_siamese_sequences(
                sample_game_data,
                normalize=True,
                stats=incomplete,
            )

    def test_provided_stats_must_cover_all_matchup_features(self, sample_game_data):
        incomplete = NormalizationStats(
            sequence_stats={name: (0.0, 1.0) for name in SEQUENCE_FEATURES},
            matchup_stats={"spread_magnitude": (0.0, 1.0)},
        )

        with pytest.raises(ValueError, match="Missing matchup normalization stats"):
            build_siamese_sequences(
                sample_game_data,
                normalize=True,
                stats=incomplete,
            )

    def test_labeled_games_with_missing_roles_raise(self, sample_game_data):
        broken = sample_game_data.copy()
        broken.loc[0, "underdog"] = np.nan

        with pytest.raises(ValueError, match="missing favorite/underdog"):
            build_siamese_sequences(broken, normalize=False)

    def test_history_source_without_usable_scores_raises(self, sample_game_data):
        broken = sample_game_data.drop(columns=["home_score", "away_score"])

        with pytest.raises(ValueError, match="No usable team history"):
            build_siamese_sequences(broken, normalize=False)

    def test_tied_games_encode_half_win_in_history(self):
        history = _build_team_game_history(
            pd.DataFrame(
                {
                    "season": [2023],
                    "week": [2],
                    "home_team": ["KC"],
                    "away_team": ["BUF"],
                    "home_score": [24],
                    "away_score": [24],
                }
            )
        )

        assert history["KC"].iloc[0]["win"] == pytest.approx(0.5)
        assert history["BUF"].iloc[0]["win"] == pytest.approx(0.5)


class TestCrossSeasonHistory:
    """H3: LSTM sequences should cross season boundaries."""

    @pytest.fixture
    def cross_season_data(self):
        """KC plays weeks 16-17 of 2022 and weeks 1-2 of 2023."""
        data = pd.DataFrame(
            {
                "game_id": [
                    "2022_16_KC_BUF",
                    "2022_17_KC_DAL",
                    "2023_01_KC_PHI",
                    "2023_02_KC_BUF",
                ],
                "season": [2022, 2022, 2023, 2023],
                "week": [16, 17, 1, 2],
                "home_team": ["KC", "KC", "KC", "KC"],
                "away_team": ["BUF", "DAL", "PHI", "BUF"],
                "home_score": [28, 31, 24, 30],
                "away_score": [24, 17, 21, 17],
                "spread_favorite": [-7.0, -6.0, -4.0, -5.0],
                "favorite": ["KC", "KC", "KC", "KC"],
                "underdog": ["BUF", "DAL", "PHI", "BUF"],
                "upset": [0.0, 0.0, 0.0, 0.0],
                "home_rest": [7, 7, 10, 7],
                "away_rest": [7, 7, 10, 7],
                "home_off_pass_epa": [15.0, 18.0, 12.0, 16.0],
                "home_off_rush_epa": [3.0, 5.0, 3.0, 4.0],
                "away_off_pass_epa": [12.0, 7.0, 11.0, 8.0],
                "away_off_rush_epa": [2.8, 1.5, 3.0, 1.8],
                "home_success_rate": [0.48, 0.50, 0.47, 0.53],
                "away_success_rate": [0.51, 0.41, 0.50, 0.43],
                "home_cpoe": [1.2, 0.7, 0.4, 1.4],
                "away_cpoe": [2.1, -0.7, 1.0, -1.2],
                "home_turnover_margin": [-1.0, 0.0, 1.0, 1.0],
                "away_turnover_margin": [1.0, 0.0, 0.0, -1.0],
            }
        )
        return _with_matchup_features(data)

    def test_week1_sees_prior_season(self, cross_season_data):
        """Week 1 of 2023 should include KC's 2022 history."""
        result, _ = build_siamese_sequences(cross_season_data, normalize=False)

        # Game index 2 is 2023_01_KC_PHI. KC is favorite.
        # Before this game KC played 2 games (weeks 16,17 of 2022).
        fav_mask = result.favorite_masks[2]
        assert fav_mask.sum() >= 2, "Week 1 should see prior-season games"

    def test_history_keyed_by_team_not_season(self, cross_season_data):
        """_build_team_game_history should key by team, not (team, season)."""
        history = _build_team_game_history(cross_season_data)
        # After H3 fix, keys are team strings, not (team, season) tuples
        assert "KC" in history
        assert isinstance(list(history.keys())[0], str)


class TestSubThreeSpreadHistory:
    """H2: Sub-3-spread games should contribute to LSTM team history."""

    @pytest.fixture
    def mixed_spread_data(self):
        """Game 1 has sub-3 spread (upset=NaN), games 2-3 are labeled."""
        data = pd.DataFrame(
            {
                "game_id": ["sub3", "labeled1", "labeled2"],
                "season": [2023, 2023, 2023],
                "week": [2, 3, 4],
                "home_team": ["KC", "KC", "KC"],
                "away_team": ["BUF", "DAL", "PHI"],
                "home_score": [24, 28, 31],
                "away_score": [21, 17, 14],
                "spread_favorite": [-2.0, -7.0, -6.0],
                "favorite": ["KC", "KC", "KC"],
                "underdog": ["BUF", "DAL", "PHI"],
                "upset": [np.nan, 0.0, 0.0],  # game 1 unlabeled
                "home_rest": [7, 7, 7],
                "away_rest": [7, 7, 7],
                "home_off_pass_epa": [15.0, 18.0, 16.0],
                "home_off_rush_epa": [3.0, 5.0, 4.0],
                "away_off_pass_epa": [12.0, 7.0, 8.0],
                "away_off_rush_epa": [2.8, 1.5, 1.8],
                "home_success_rate": [0.48, 0.50, 0.53],
                "away_success_rate": [0.51, 0.41, 0.43],
                "home_cpoe": [1.2, 0.7, 1.4],
                "away_cpoe": [2.1, -0.7, -1.2],
                "home_turnover_margin": [-1.0, 0.0, 1.0],
                "away_turnover_margin": [1.0, 0.0, -1.0],
            }
        )
        return _with_matchup_features(data)

    def test_history_df_includes_unlabeled_games(self, mixed_spread_data):
        """When history_df has sub-3 games, they appear in team sequences."""
        labeled_only = mixed_spread_data[mixed_spread_data["upset"].notna()].copy()

        # Without history_df: labeled_only has only weeks 3,4 — week 3 has no prior
        result_without, _ = build_siamese_sequences(labeled_only, normalize=False)
        mask_without = result_without.favorite_masks[0]  # week 3, KC

        # With history_df: full data includes week 2 (sub-3 spread)
        result_with, _ = build_siamese_sequences(
            labeled_only, normalize=False, history_df=mixed_spread_data
        )
        mask_with = result_with.favorite_masks[0]  # week 3, KC

        assert (
            mask_with.sum() > mask_without.sum()
        ), "history_df should provide more history via sub-3 games"

    def test_unlabeled_games_not_in_targets(self, mixed_spread_data):
        """Sub-3 games in history_df must not become training targets."""
        labeled_only = mixed_spread_data[mixed_spread_data["upset"].notna()].copy()
        result, _ = build_siamese_sequences(
            labeled_only, normalize=False, history_df=mixed_spread_data
        )
        # Only 2 labeled games should produce samples
        assert result.n_samples == 2
        assert not np.isnan(result.targets).any()


class TestCanonicalFeatureLists:
    def test_sequence_feature_schema(self):
        assert SEQUENCE_FEATURES == [
            "total_epa",
            "pass_epa",
            "rush_epa",
            "success_rate",
            "cpoe",
            "turnover_margin",
            "points_scored",
            "points_allowed",
            "point_diff",
            "opponent_elo",
            "win",
            "was_home",
            "days_since_last_game",
            "short_week",
        ]
        assert SEQUENCE_FEATURES_NO_SPREAD == SEQUENCE_FEATURES

    def test_matchup_feature_schema_sizes(self):
        assert len(MATCHUP_FEATURES) == 10
        assert len(MATCHUP_FEATURES_NO_SPREAD) == 8
        assert set(MATCHUP_FEATURES_NO_SPREAD) < set(MATCHUP_FEATURES)
