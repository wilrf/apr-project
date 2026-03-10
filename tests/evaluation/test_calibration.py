"""Tests for post-hoc calibration module."""

import typing

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.calibration import (
    CalibrationResult,
    IsotonicCalibrator,
    PlattScaler,
    calibrate_models,
    generate_calibration_predictions,
)


class TestPlattScaler:
    def test_fit_transform_returns_probabilities(self):
        rng = np.random.RandomState(42)
        probs = rng.uniform(0.1, 0.9, size=200)
        y = (probs > 0.5).astype(int)

        scaler = PlattScaler().fit(probs, y)
        calibrated = scaler.transform(probs)

        assert calibrated.shape == probs.shape
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_transform_before_fit_raises(self):
        scaler = PlattScaler()
        with pytest.raises(ValueError, match="fit"):
            scaler.transform(np.array([0.5]))

    def test_monotonic_output(self):
        rng = np.random.RandomState(42)
        probs = rng.uniform(0.05, 0.95, size=500)
        y = (rng.uniform(size=500) < probs).astype(int)

        scaler = PlattScaler().fit(probs, y)
        test_probs = np.linspace(0.01, 0.99, 50)
        calibrated = scaler.transform(test_probs)

        # Platt scaling is monotonic (logistic sigmoid)
        assert np.all(np.diff(calibrated) >= -1e-10)

    def test_fit_requires_both_classes(self):
        scaler = PlattScaler()
        with pytest.raises(ValueError, match="at least two classes"):
            scaler.fit(np.array([0.2, 0.3, 0.4]), np.array([1, 1, 1]))


class TestIsotonicCalibrator:
    def test_fit_transform_returns_probabilities(self):
        rng = np.random.RandomState(42)
        probs = rng.uniform(0.1, 0.9, size=200)
        y = (probs > 0.5).astype(int)

        scaler = IsotonicCalibrator().fit(probs, y)
        calibrated = scaler.transform(probs)

        assert calibrated.shape == probs.shape
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_transform_before_fit_raises(self):
        scaler = IsotonicCalibrator()
        with pytest.raises(ValueError, match="fit"):
            scaler.transform(np.array([0.5]))


class TestCalibrateModels:
    def test_calibrates_multiple_models(self):
        rng = np.random.RandomState(42)
        y_cal = rng.binomial(1, 0.3, size=200)

        cal_probs = {
            "model_a": rng.uniform(0.1, 0.8, size=200),
            "model_b": rng.uniform(0.05, 0.9, size=200),
        }
        test_probs = {
            "model_a": rng.uniform(0.1, 0.8, size=100),
            "model_b": rng.uniform(0.05, 0.9, size=100),
        }

        results = calibrate_models(cal_probs, y_cal, test_probs, method="platt")

        assert set(results.keys()) == {"model_a", "model_b"}
        for name, res in results.items():
            assert isinstance(res, CalibrationResult)
            assert res.calibrated.shape == test_probs[name].shape
            assert res.method == "platt"
            assert np.all(res.calibrated >= 0)
            assert np.all(res.calibrated <= 1)

    def test_isotonic_method(self):
        rng = np.random.RandomState(42)
        y_cal = rng.binomial(1, 0.3, size=200)
        cal_probs = {"m": rng.uniform(0.1, 0.8, size=200)}
        test_probs = {"m": rng.uniform(0.1, 0.8, size=50)}

        results = calibrate_models(cal_probs, y_cal, test_probs, method="isotonic")
        assert results["m"].method == "isotonic"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            calibrate_models(
                {"m": np.array([0.5])},
                np.array([1]),
                {"m": np.array([0.5])},
                method="unknown",
            )

    def test_key_mismatch_raises_clear_error(self):
        with pytest.raises(ValueError, match="same model keys"):
            calibrate_models(
                {"lr": np.array([0.2, 0.3])},
                np.array([0, 1]),
                {"lr": np.array([0.1]), "xgb": np.array([0.1])},
            )


def test_generate_calibration_predictions_filters_labels_but_keeps_full_history(
    monkeypatch,
):
    """Calibration should train on labeled rows but retain full history for LSTM."""
    import src.models.sequence_builder as sequence_builder
    import src.models.unified_trainer as unified_trainer
    from src.evaluation.calibration import generate_calibration_predictions

    train_df = pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021, 2022],
            "week": [2, 3, 2, 3, 2],
            "home_team": ["KC", "KC", "KC", "KC", "KC"],
            "away_team": ["BUF", "DAL", "PHI", "BUF", "DAL"],
            "favorite": ["KC", "KC", "KC", "KC", "KC"],
            "underdog": ["BUF", "DAL", "PHI", "BUF", "DAL"],
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            "xgb_feature": [10.0, 20.0, 30.0, 40.0, 50.0],
            "upset": [0.0, np.nan, 1.0, np.nan, 0.0],
        }
    )
    captured: dict[str, pd.DataFrame] = {}

    class FakeBinaryModel:
        def predict_proba(self, X):
            return np.full(len(X), 0.25)

    class FakeLSTMModel:
        def eval(self):
            return None

        def __call__(self, und_seq, fav_seq, matchup, und_mask, fav_mask):
            return torch.full((len(und_seq), 1), 0.4)

    def fake_train_final(
        self,
        df,
        feature_cols,
        target_col="upset",
        lstm_epochs=0,
        lstm_batch_size=0,
        verbose=True,
        matchup_feature_cols=None,
        sequence_feature_cols=None,
        xgb_feature_cols=None,
        full_df=None,
    ):
        captured["fit_df"] = df.copy()
        captured["full_df"] = full_df.copy()
        return {
            "lr_model": FakeBinaryModel(),
            "xgb_model": FakeBinaryModel(),
            "lstm_model": FakeLSTMModel(),
            "lstm_stats": SimpleNamespace(sequence_stats={}, matchup_stats={}),
        }

    def fake_build_siamese_sequences(
        df,
        normalize=True,
        seq_length=8,
        stats=None,
        matchup_feature_cols=None,
        sequence_feature_cols=None,
        history_df=None,
    ):
        captured["cal_df"] = df.copy()
        captured["history_df"] = history_df.copy()
        n_samples = len(df)
        return (
            sequence_builder.SiameseLSTMData(
                underdog_sequences=np.zeros((n_samples, 8, 1), dtype=np.float32),
                favorite_sequences=np.zeros((n_samples, 8, 1), dtype=np.float32),
                matchup_features=np.zeros((n_samples, 1), dtype=np.float32),
                targets=np.zeros(n_samples, dtype=np.float32),
                underdog_masks=np.zeros((n_samples, 8), dtype=np.float32),
                favorite_masks=np.zeros((n_samples, 8), dtype=np.float32),
                game_ids=np.arange(n_samples),
            ),
            None,
        )

    from types import SimpleNamespace

    monkeypatch.setattr(unified_trainer.UnifiedTrainer, "train_final", fake_train_final)
    monkeypatch.setattr(
        sequence_builder, "build_siamese_sequences", fake_build_siamese_sequences
    )

    probs, y_cal = generate_calibration_predictions(
        train_df,
        feature_cols=["feature"],
        cal_seasons=(2021, 2022),
        xgb_feature_cols=["feature", "xgb_feature"],
    )

    assert list(captured["fit_df"]["season"]) == [2020]
    assert captured["fit_df"]["upset"].notna().all()
    pd.testing.assert_frame_equal(
        captured["full_df"].reset_index(drop=True),
        train_df.reset_index(drop=True),
    )
    assert list(captured["cal_df"]["season"]) == [2021, 2022]
    assert captured["cal_df"]["upset"].notna().all()
    pd.testing.assert_frame_equal(
        captured["history_df"].reset_index(drop=True),
        train_df.reset_index(drop=True),
    )
    assert not np.isnan(y_cal).any()
    assert set(probs) == {"lr", "xgb", "lstm"}


def test_generate_calibration_predictions_keeps_single_lstm_output_vector(
    monkeypatch,
):
    """H5: single-sample calibration predictions must stay 1D."""
    import src.models.sequence_builder as sequence_builder
    import src.models.unified_trainer as unified_trainer
    from src.evaluation.calibration import generate_calibration_predictions

    train_df = pd.DataFrame(
        {
            "season": [2020, 2021],
            "week": [2, 2],
            "home_team": ["KC", "KC"],
            "away_team": ["BUF", "PHI"],
            "favorite": ["KC", "KC"],
            "underdog": ["BUF", "PHI"],
            "feature": [1.0, 2.0],
            "xgb_feature": [10.0, 20.0],
            "upset": [0.0, 1.0],
        }
    )

    class FakeBinaryModel:
        def predict_proba(self, X):
            return np.full(len(X), 0.25)

    class FakeLSTMModel:
        def eval(self):
            return None

        def __call__(self, und_seq, fav_seq, matchup, und_mask, fav_mask):
            return torch.full((len(und_seq), 1), 0.4)

    from types import SimpleNamespace

    def fake_train_final(self, *args, **kwargs):
        return {
            "lr_model": FakeBinaryModel(),
            "xgb_model": FakeBinaryModel(),
            "lstm_model": FakeLSTMModel(),
            "lstm_stats": SimpleNamespace(sequence_stats={}, matchup_stats={}),
        }

    def fake_build_siamese_sequences(*args, **kwargs):
        return (
            sequence_builder.SiameseLSTMData(
                underdog_sequences=np.zeros((1, 8, 1), dtype=np.float32),
                favorite_sequences=np.zeros((1, 8, 1), dtype=np.float32),
                matchup_features=np.zeros((1, 1), dtype=np.float32),
                targets=np.zeros(1, dtype=np.float32),
                underdog_masks=np.zeros((1, 8), dtype=np.float32),
                favorite_masks=np.zeros((1, 8), dtype=np.float32),
                game_ids=np.arange(1),
            ),
            None,
        )

    monkeypatch.setattr(unified_trainer.UnifiedTrainer, "train_final", fake_train_final)
    monkeypatch.setattr(
        sequence_builder, "build_siamese_sequences", fake_build_siamese_sequences
    )

    probs, y_cal = generate_calibration_predictions(
        train_df,
        feature_cols=["feature"],
        cal_seasons=(2021,),
        xgb_feature_cols=["feature", "xgb_feature"],
    )

    assert y_cal.shape == (1,)
    assert probs["lstm"].shape == (1,)


def test_generate_calibration_predictions_fills_feature_nans_before_model_calls(
    monkeypatch,
):
    import src.models.sequence_builder as sequence_builder
    import src.models.unified_trainer as unified_trainer
    from src.evaluation.calibration import generate_calibration_predictions

    train_df = pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021],
            "week": [2, 3, 2, 3],
            "home_team": ["KC", "BUF", "KC", "BUF"],
            "away_team": ["BUF", "KC", "PHI", "DAL"],
            "favorite": ["KC", "BUF", "KC", "BUF"],
            "underdog": ["BUF", "KC", "PHI", "DAL"],
            "feature": [1.0, np.nan, 3.0, np.nan],
            "xgb_feature": [10.0, 20.0, np.nan, 40.0],
            "upset": [0.0, 1.0, 0.0, 1.0],
        }
    )
    checks = {"fit": False, "predict_calls": 0}

    class AssertingBinaryModel:
        def predict_proba(self, X):
            assert not X.isna().any().any()
            checks["predict_calls"] += 1
            return np.full(len(X), 0.25)

    class FakeLSTMModel:
        def eval(self):
            return None

        def __call__(self, und_seq, fav_seq, matchup, und_mask, fav_mask):
            return torch.full((len(und_seq), 1), 0.4)

    from types import SimpleNamespace

    def fake_train_final(
        self,
        df,
        feature_cols,
        target_col="upset",
        lstm_epochs=0,
        lstm_batch_size=0,
        verbose=True,
        matchup_feature_cols=None,
        sequence_feature_cols=None,
        xgb_feature_cols=None,
        full_df=None,
    ):
        all_cols = list(set(feature_cols + (xgb_feature_cols or feature_cols)))
        assert not df[all_cols].isna().any().any()
        checks["fit"] = True
        return {
            "lr_model": AssertingBinaryModel(),
            "xgb_model": AssertingBinaryModel(),
            "lstm_model": FakeLSTMModel(),
            "lstm_stats": SimpleNamespace(sequence_stats={}, matchup_stats={}),
        }

    def fake_build_siamese_sequences(*args, **kwargs):
        n_samples = len(args[0])
        return (
            sequence_builder.SiameseLSTMData(
                underdog_sequences=np.zeros((n_samples, 8, 1), dtype=np.float32),
                favorite_sequences=np.zeros((n_samples, 8, 1), dtype=np.float32),
                matchup_features=np.zeros((n_samples, 1), dtype=np.float32),
                targets=np.zeros(n_samples, dtype=np.float32),
                underdog_masks=np.zeros((n_samples, 8), dtype=np.float32),
                favorite_masks=np.zeros((n_samples, 8), dtype=np.float32),
                game_ids=np.arange(n_samples),
            ),
            None,
        )

    monkeypatch.setattr(unified_trainer.UnifiedTrainer, "train_final", fake_train_final)
    monkeypatch.setattr(
        sequence_builder, "build_siamese_sequences", fake_build_siamese_sequences
    )

    probs, y_cal = generate_calibration_predictions(
        train_df,
        feature_cols=["feature"],
        cal_seasons=(2021,),
        xgb_feature_cols=["feature", "xgb_feature"],
    )

    assert checks["fit"]
    assert checks["predict_calls"] == 2
    assert set(probs) == {"lr", "xgb", "lstm"}
    assert y_cal.shape == (2,)


def test_generate_calibration_predictions_requires_nonempty_fit_df():
    train_df = pd.DataFrame(
        {
            "season": [2021, 2022],
            "week": [2, 2],
            "home_team": ["KC", "BUF"],
            "away_team": ["BUF", "KC"],
            "favorite": ["KC", "BUF"],
            "underdog": ["BUF", "KC"],
            "feature": [1.0, 2.0],
            "upset": [0.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="No fit data"):
        generate_calibration_predictions(
            train_df,
            feature_cols=["feature"],
            cal_seasons=(2021, 2022),
        )


def test_generate_calibration_predictions_type_hints_resolve():
    hints = typing.get_type_hints(generate_calibration_predictions)
    assert hints["train_df"] is pd.DataFrame
