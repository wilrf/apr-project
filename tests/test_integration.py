"""Integration tests for the canonical NFL upset prediction pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_baseline_brier
from src.features.pipeline import FeatureEngineeringPipeline, get_xgb_feature_columns
from src.models.trainer import ModelTrainer
from src.models.xgboost_model import UpsetXGBoost


def create_mock_season_data(
    season: int,
    n_games_per_week: int = 8,
    n_weeks: int = 17,
) -> pd.DataFrame:
    """Create realistic-enough mock data for an NFL season."""
    teams = [
        "KC",
        "DET",
        "SF",
        "PHI",
        "BAL",
        "CLE",
        "BUF",
        "MIA",
        "DAL",
        "NYG",
        "LAR",
        "SEA",
        "DEN",
        "LV",
        "MIN",
        "GB",
    ]

    rows = []
    for week in range(1, n_weeks + 1):
        rng = np.random.default_rng(seed=season * 100 + week)
        shuffled = rng.permutation(teams)

        for i in range(0, min(len(shuffled), n_games_per_week * 2), 2):
            home = shuffled[i]
            away = shuffled[i + 1]
            home_score = int(rng.integers(10, 40))
            away_score = int(rng.integers(10, 40))
            spread_magnitude = float(np.round(rng.uniform(1, 10), 1))
            home_favorite = rng.random() < 0.6
            favorite = home if home_favorite else away
            spread = -spread_magnitude if home_favorite else spread_magnitude

            rows.append(
                {
                    "game_id": f"{season}_{week:02d}_{home}_{away}",
                    "season": season,
                    "week": week,
                    "gameday": pd.Timestamp(season, 9, 1) + pd.Timedelta(days=7 * week),
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "spread_favorite": spread,
                    "team_favorite_id": favorite,
                    "over_under_line": float(np.round(rng.uniform(38, 54), 1)),
                    "home_rest": int(rng.choice([4, 6, 7, 8, 10])),
                    "away_rest": int(rng.choice([4, 6, 7, 8, 10])),
                    "div_game": int(rng.integers(0, 2)),
                    "temp": float(np.round(rng.uniform(25, 85), 1)),
                    "wind": float(np.round(rng.uniform(0, 22), 1)),
                    "roof": rng.choice(["outdoors", "dome", "closed"]),
                    "home_off_pass_epa": float(np.round(rng.normal(8, 4), 2)),
                    "home_off_rush_epa": float(np.round(rng.normal(2, 2), 2)),
                    "away_off_pass_epa": float(np.round(rng.normal(8, 4), 2)),
                    "away_off_rush_epa": float(np.round(rng.normal(2, 2), 2)),
                    "home_success_rate": float(np.round(rng.uniform(0.35, 0.55), 3)),
                    "away_success_rate": float(np.round(rng.uniform(0.35, 0.55), 3)),
                    "home_cpoe": float(np.round(rng.normal(0, 3), 2)),
                    "away_cpoe": float(np.round(rng.normal(0, 3), 2)),
                    "home_turnover_margin": int(rng.integers(-3, 4)),
                    "away_turnover_margin": int(rng.integers(-3, 4)),
                }
            )

    return pd.DataFrame(rows)


class TestFullPipelineIntegration:
    def test_pipeline_produces_valid_features(self):
        df = pd.concat(
            [
                create_mock_season_data(2022, n_games_per_week=8, n_weeks=5),
                create_mock_season_data(2023, n_games_per_week=8, n_weeks=5),
            ]
        ).reset_index(drop=True)

        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        result = pipeline.transform(df)

        for column in [
            "upset",
            "spread_magnitude",
            "pass_epa_diff",
            "elo_diff",
            "underdog_is_home",
            "temperature_missing",
        ]:
            assert column in result.columns

        valid_targets = result[result["upset"].notna()]["upset"]
        assert valid_targets.isin([0.0, 1.0]).all()
        assert (result["spread_magnitude"] >= 0).all()

    def test_pipeline_excludes_week_1_games(self):
        df = create_mock_season_data(2023, n_games_per_week=8, n_weeks=5)

        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        result = pipeline.transform(df)

        assert not (result["week"] == 1).any()
        assert (result["week"] >= 2).all()

    def test_xgboost_model_trains_and_predicts(self):
        df = create_mock_season_data(2023, n_games_per_week=8, n_weeks=10)
        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        features_df = pipeline.transform(df)
        valid_df = features_df[features_df["upset"].notna()].copy()

        if len(valid_df) < 20:
            pytest.skip("Not enough valid samples for test")

        feature_cols = get_xgb_feature_columns()
        X = valid_df[feature_cols]
        y = valid_df["upset"].astype(int)

        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]

        model = UpsetXGBoost(n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)

        assert len(preds) == len(X_test)
        assert np.logical_and(preds >= 0, preds <= 1).all()

    def test_model_trainer_cross_validation(self):
        df = pd.concat(
            [
                create_mock_season_data(2020, n_games_per_week=8, n_weeks=8),
                create_mock_season_data(2021, n_games_per_week=8, n_weeks=8),
                create_mock_season_data(2022, n_games_per_week=8, n_weeks=8),
                create_mock_season_data(2023, n_games_per_week=8, n_weeks=8),
            ]
        ).reset_index(drop=True)

        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        valid_df = pipeline.transform(df)
        valid_df = valid_df[valid_df["upset"].notna()].copy()

        model = UpsetXGBoost(n_estimators=10, max_depth=3)
        trainer = ModelTrainer(model, n_folds=3)
        results = trainer.cross_validate(
            valid_df,
            feature_cols=pipeline.get_feature_columns(),
            target_col="upset",
        )

        assert "fold_metrics" in results
        assert "aggregated" in results
        assert "predictions" in results
        assert len(results["fold_metrics"]) == 3
        assert "auc_roc_mean" in results["aggregated"]
        assert "brier_score_mean" in results["aggregated"]

    def test_baseline_brier_calculation(self):
        df = create_mock_season_data(2023, n_games_per_week=8, n_weeks=10)
        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        features_df = pipeline.transform(df)

        upset_rate = features_df[features_df["upset"].notna()]["upset"].mean()
        baseline = calculate_baseline_brier(upset_rate)
        expected = upset_rate * (1 - upset_rate)
        assert abs(baseline - expected) < 0.001


class TestFeatureVariance:
    def test_matchup_differentials_have_variance(self):
        df = create_mock_season_data(2023, n_games_per_week=8, n_weeks=10)
        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        result = pipeline.transform(df)
        valid = result[result["underdog"].notna()]

        if len(valid) < 10:
            pytest.skip("Not enough valid games")

        assert valid["pass_epa_diff"].nunique() > 1
        assert valid["elo_diff"].nunique() > 1

    def test_spread_magnitude_varies(self):
        df = create_mock_season_data(2023, n_games_per_week=8, n_weeks=10)
        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        result = pipeline.transform(df)

        assert result["spread_magnitude"].nunique() > 1
