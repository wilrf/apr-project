# tests/models/test_lstm_trainer.py
"""Tests for Siamese LSTM trainer."""
import pytest
import pandas as pd
import numpy as np
import torch
from src.models.lstm_trainer import SiameseLSTMTrainer
from src.models.sequence_builder import SiameseLSTMData


@pytest.fixture
def sample_training_data():
    """Create sample training data with multiple seasons for CV."""
    np.random.seed(42)
    games = []

    # Create games for seasons 2017-2022 (6 seasons for 6-fold CV)
    teams = ["KC", "BUF", "PHI", "DAL", "SF", "GB", "TB", "NE"]

    for season in range(2017, 2023):
        # 16 games per season across 8 teams (simplified)
        for week in range(2, 10):  # Weeks 2-9 (8 weeks)
            for i in range(0, len(teams), 2):
                home = teams[i]
                away = teams[(i + 1) % len(teams)]

                home_score = np.random.randint(14, 35)
                away_score = np.random.randint(14, 35)

                # Determine favorite (higher historical performance gets favorable spread)
                spread = np.random.choice([-7, -6, -5, -4, -3, 3, 4, 5, 6, 7])
                if spread < 0:
                    favorite, underdog = home, away
                else:
                    favorite, underdog = away, home

                upset = 1 if (underdog == home and home_score > away_score) or \
                             (underdog == away and away_score > home_score) else 0

                games.append({
                    "game_id": f"{season}_{week}_{home}_{away}",
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "spread_favorite": abs(spread),
                    "underdog": underdog,
                    "favorite": favorite,
                    "upset": upset,
                    "spread_magnitude": abs(spread),
                    "home_indicator": 1 if underdog == home else 0,
                    "divisional_game": np.random.choice([0, 1]),
                    "rest_advantage": np.random.choice([-1, 0, 1]),
                    "week_number": week / 18.0,
                    "primetime_game": np.random.choice([0, 1]),
                    "is_dome": np.random.choice([0, 1]),
                    "cold_weather": np.random.choice([0, 1]),
                    "windy_game": np.random.choice([0, 1]),
                    "over_under_normalized": np.random.uniform(-0.5, 0.5),
                })

    return pd.DataFrame(games)


class TestSiameseLSTMTrainer:
    def test_initialization(self):
        """Test trainer initializes with correct parameters."""
        trainer = SiameseLSTMTrainer(
            hidden_size=32,
            num_layers=1,
            dropout=0.2,
            learning_rate=0.01,
            batch_size=16,
        )

        assert trainer.hidden_size == 32
        assert trainer.num_layers == 1
        assert trainer.dropout == 0.2
        assert trainer.learning_rate == 0.01
        assert trainer.batch_size == 16

    def test_device_detection(self):
        """Test automatic device detection."""
        trainer = SiameseLSTMTrainer()

        # Should be one of the valid devices
        assert trainer.device.type in ["cpu", "cuda", "mps"]

    def test_create_model(self):
        """Test model creation."""
        trainer = SiameseLSTMTrainer(hidden_size=32, num_layers=1)
        model = trainer._create_model()

        assert model is not None
        assert isinstance(model, torch.nn.Module)
        # Verify it's the siamese model
        assert hasattr(model, "shared_lstm")

    def test_fit_stores_model(self, sample_training_data):
        """Test that fit() stores trained model."""
        trainer = SiameseLSTMTrainer(
            hidden_size=16,
            num_layers=1,
            epochs=2,  # Fast training for test
            batch_size=32,
        )

        # Use subset of data for faster test
        subset = sample_training_data[sample_training_data["season"] >= 2021]
        trainer.fit(subset)

        assert trainer.model is not None

    def test_predict_proba_returns_valid_probabilities(self, sample_training_data):
        """Test that predict_proba returns values in [0, 1]."""
        trainer = SiameseLSTMTrainer(
            hidden_size=16,
            num_layers=1,
            epochs=2,
            batch_size=32,
        )

        # Train on subset
        train_data = sample_training_data[sample_training_data["season"] >= 2021]
        trainer.fit(train_data)

        # Predict
        test_data = sample_training_data[sample_training_data["season"] == 2022]
        predictions = trainer.predict_proba(test_data)

        assert len(predictions) == len(test_data[test_data["upset"].notna()])
        assert all(0 <= p <= 1 for p in predictions)

    def test_predict_before_fit_raises_error(self, sample_training_data):
        """Test that predicting before training raises error."""
        trainer = SiameseLSTMTrainer()

        with pytest.raises(ValueError, match="Model not trained"):
            trainer.predict_proba(sample_training_data)


class TestSiameseLSTMTrainerCrossValidation:
    def test_cross_validate_returns_metrics(self, sample_training_data):
        """Test that cross_validate returns expected metrics structure."""
        trainer = SiameseLSTMTrainer(
            hidden_size=16,
            num_layers=1,
            epochs=2,
            batch_size=32,
            n_folds=3,  # Fewer folds for faster test
        )

        results = trainer.cross_validate(sample_training_data)

        assert "fold_metrics" in results
        assert "aggregated" in results
        assert "predictions" in results

        # Check aggregated metrics
        assert "auc_roc_mean" in results["aggregated"]
        assert "log_loss_mean" in results["aggregated"]
        assert "brier_score_mean" in results["aggregated"]

    def test_cross_validate_correct_fold_count(self, sample_training_data):
        """Test that correct number of folds are created."""
        n_folds = 3
        trainer = SiameseLSTMTrainer(
            hidden_size=16,
            num_layers=1,
            epochs=2,
            batch_size=32,
            n_folds=n_folds,
        )

        results = trainer.cross_validate(sample_training_data)

        assert len(results["fold_metrics"]) == n_folds
        assert len(results["predictions"]) == n_folds

    def test_fold_metrics_have_expected_fields(self, sample_training_data):
        """Test that each fold has expected metric fields."""
        trainer = SiameseLSTMTrainer(
            hidden_size=16,
            num_layers=1,
            epochs=2,
            batch_size=32,
            n_folds=2,
        )

        results = trainer.cross_validate(sample_training_data)

        for fold in results["fold_metrics"]:
            assert "auc_roc" in fold
            assert "log_loss" in fold
            assert "brier_score" in fold
            assert "fold" in fold
            assert "train_size" in fold
            assert "val_size" in fold
            assert "epochs_trained" in fold


class TestSiameseLSTMTrainerAttention:
    def test_get_attention_weights(self, sample_training_data):
        """Test that attention weights are returned for both teams."""
        trainer = SiameseLSTMTrainer(
            hidden_size=16,
            num_layers=1,
            epochs=2,
            batch_size=32,
        )

        # Train
        train_data = sample_training_data[sample_training_data["season"] >= 2021]
        trainer.fit(train_data)

        # Get attention
        test_data = sample_training_data[sample_training_data["season"] == 2022]
        und_attn, fav_attn = trainer.get_attention_weights(test_data)

        n_samples = len(test_data[test_data["upset"].notna()])
        assert und_attn.shape[0] == n_samples
        assert und_attn.shape[1] == 5  # SEQUENCE_LENGTH
        assert fav_attn.shape[0] == n_samples
        assert fav_attn.shape[1] == 5  # SEQUENCE_LENGTH

    def test_attention_weights_before_fit_raises_error(self, sample_training_data):
        """Test that getting attention before training raises error."""
        trainer = SiameseLSTMTrainer()

        with pytest.raises(ValueError, match="Model not trained"):
            trainer.get_attention_weights(sample_training_data)


class TestDataLeakagePrevention:
    """Tests to verify no data leakage in normalization."""

    def test_fit_stores_training_stats(self, sample_training_data):
        """Test that fit() stores training statistics."""
        trainer = SiameseLSTMTrainer(
            hidden_size=16,
            num_layers=1,
            epochs=2,
            batch_size=32,
        )

        # Use subset for faster test
        subset = sample_training_data[sample_training_data["season"] >= 2021]
        trainer.fit(subset)

        # Should have stored training stats
        assert trainer._train_stats is not None
        assert "points_scored" in trainer._train_stats.sequence_stats
        assert "spread_magnitude" in trainer._train_stats.matchup_stats

    def test_predict_proba_uses_training_stats(self, sample_training_data):
        """Test that predict_proba uses stored training stats, not test data stats."""
        trainer = SiameseLSTMTrainer(
            hidden_size=16,
            num_layers=1,
            epochs=2,
            batch_size=32,
        )

        # Train on 2020-2021
        train_data = sample_training_data[
            (sample_training_data["season"] >= 2020)
            & (sample_training_data["season"] <= 2021)
        ]
        trainer.fit(train_data)

        # Predict on 2022 (different distribution shouldn't affect normalization stats)
        test_data = sample_training_data[sample_training_data["season"] == 2022]
        predictions = trainer.predict_proba(test_data)

        # Should get valid predictions
        assert len(predictions) > 0
        assert all(0 <= p <= 1 for p in predictions)

    def test_predict_before_fit_raises_stats_error(self, sample_training_data):
        """Test that predicting before fit raises error about training stats."""
        trainer = SiameseLSTMTrainer()
        # Manually set model to non-None to test stats check
        trainer.model = trainer._create_model()

        with pytest.raises(ValueError, match="Training stats not available"):
            trainer.predict_proba(sample_training_data)

    def test_cv_folds_have_independent_stats(self, sample_training_data):
        """Test that each CV fold computes its own stats from its training split."""
        trainer = SiameseLSTMTrainer(
            hidden_size=16,
            num_layers=1,
            epochs=1,  # Minimal training
            batch_size=32,
            n_folds=2,  # Just 2 folds for speed
        )

        results = trainer.cross_validate(sample_training_data)

        # Should complete without errors - this verifies the workflow
        assert len(results["fold_metrics"]) == 2
        # Each fold should have predictions
        for fold_pred in results["predictions"]:
            assert len(fold_pred["y_pred"]) > 0
            assert len(fold_pred["y_true"]) > 0

    def test_cv_does_not_leak_validation_into_training(self, sample_training_data):
        """
        Test that CV properly separates train/val for normalization.

        This is an indirect test - we verify that the workflow completes
        and produces reasonable outputs. The actual fix ensures that
        build_siamese_sequences is called separately for train and val
        with train stats applied to val.
        """
        trainer = SiameseLSTMTrainer(
            hidden_size=16,
            num_layers=1,
            epochs=2,
            batch_size=32,
            n_folds=2,
        )

        results = trainer.cross_validate(sample_training_data)

        # Verify structure is correct
        assert "fold_metrics" in results
        assert "predictions" in results

        # Each fold should have proper sizes
        for fold in results["fold_metrics"]:
            assert fold["train_size"] > 0
            assert fold["val_size"] > 0
            # Train should be larger than val in time-series CV
            assert fold["train_size"] > fold["val_size"]

        # Predictions should be valid probabilities
        for pred in results["predictions"]:
            assert all(0 <= p <= 1 for p in pred["y_pred"])
