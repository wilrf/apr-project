"""Regression tests for test-set evaluation wiring."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import src.models.evaluate_test_set as evaluate_test_set
from src.models.sequence_builder import SiameseLSTMData


def test_main_passes_combined_history_to_lstm_predictions(monkeypatch):
    """Test-time LSTM history should include both train and test games."""
    train_df = pd.DataFrame(
        {
            "season": [2022],
            "week": [18],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "favorite": ["KC"],
            "underdog": ["BUF"],
            "spread_magnitude": [7.0],
            "feature": [1.0],
            "upset": [0.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["PHI"],
            "favorite": ["KC"],
            "underdog": ["PHI"],
            "spread_magnitude": [4.0],
            "feature": [2.0],
            "upset": [0.0],
        }
    )

    captured: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(
        evaluate_test_set,
        "load_data",
        lambda: (train_df.copy(), test_df.copy(), ["feature"], ["feature"]),
    )
    monkeypatch.setattr(evaluate_test_set, "verify_data", lambda train, test: None)

    class FakeTrainer:
        def train_final(self, *args, **kwargs):
            return {"stub": True}

    monkeypatch.setattr(evaluate_test_set, "UnifiedTrainer", FakeTrainer)

    def fake_generate_predictions(
        models,
        test_df_arg,
        feature_cols,
        xgb_feature_cols,
        full_test_df=None,
        full_history_df=None,
    ):
        history_df = full_history_df if full_history_df is not None else full_test_df
        captured["history"] = history_df.copy()
        test_valid = test_df_arg[test_df_arg["upset"].notna()].copy()
        probs = np.zeros(len(test_valid), dtype=float)
        return test_valid, test_valid["upset"].to_numpy(), probs, probs, probs

    monkeypatch.setattr(
        evaluate_test_set, "generate_predictions", fake_generate_predictions
    )
    monkeypatch.setattr(
        evaluate_test_set,
        "generate_calibration_predictions",
        lambda *args, **kwargs: (
            {"lr": np.array([0.1]), "xgb": np.array([0.1]), "lstm": np.array([0.1])},
            np.array([0]),
        ),
    )
    monkeypatch.setattr(
        evaluate_test_set,
        "calibrate_models",
        lambda cal_probs, cal_y, test_probs, method="platt": {
            name: SimpleNamespace(calibrated=probs)
            for name, probs in test_probs.items()
        },
    )
    monkeypatch.setattr(
        evaluate_test_set,
        "calculate_metrics",
        lambda y_true, y_pred: {"auc_roc": 0.5, "brier_score": 0.25, "log_loss": 0.69},
    )
    monkeypatch.setattr(
        evaluate_test_set, "build_game_predictions", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        evaluate_test_set,
        "DisagreementAnalyzer",
        lambda predictions: SimpleNamespace(predictions=predictions),
    )
    monkeypatch.setattr(
        evaluate_test_set, "print_metrics_comparison", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        evaluate_test_set, "print_calibration", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        evaluate_test_set,
        "print_disagreement_comparison",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_test_set, "print_top_k_analysis", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        evaluate_test_set, "print_probability_buckets", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        evaluate_test_set, "print_per_season_breakdown", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        evaluate_test_set, "save_predictions_csv", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        evaluate_test_set, "save_report_md", lambda *args, **kwargs: None
    )

    evaluate_test_set.main()

    expected_history = pd.concat([train_df, test_df], ignore_index=True)
    pd.testing.assert_frame_equal(
        captured["history"].reset_index(drop=True),
        expected_history.reset_index(drop=True),
    )


def test_generate_predictions_keeps_single_lstm_output_vector(monkeypatch):
    """H5: single-sample LSTM predictions must stay 1D."""

    def fake_build_siamese_sequences(*args, **kwargs):
        return (
            SiameseLSTMData(
                underdog_sequences=np.zeros((1, 8, 1), dtype=np.float32),
                favorite_sequences=np.zeros((1, 8, 1), dtype=np.float32),
                matchup_features=np.zeros((1, 1), dtype=np.float32),
                targets=np.array([0.0], dtype=np.float32),
                underdog_masks=np.zeros((1, 8), dtype=np.float32),
                favorite_masks=np.zeros((1, 8), dtype=np.float32),
                game_ids=np.array(["g1"]),
            ),
            None,
        )

    class FakeBinaryModel:
        def predict_proba(self, X):
            return np.full(len(X), 0.25)

    class FakeLSTMModel:
        def eval(self):
            return None

        def __call__(self, und_seq, fav_seq, matchup, und_mask, fav_mask):
            return torch.full((len(und_seq), 1), 0.4)

    monkeypatch.setattr(
        evaluate_test_set, "build_siamese_sequences", fake_build_siamese_sequences
    )

    test_df = pd.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "favorite": ["KC"],
            "underdog": ["BUF"],
            "feature": [1.0],
            "upset": [0.0],
        }
    )
    models = {
        "lr_model": FakeBinaryModel(),
        "xgb_model": FakeBinaryModel(),
        "lstm_model": FakeLSTMModel(),
        "lstm_stats": SimpleNamespace(sequence_stats={}, matchup_stats={}),
    }

    _, _, _, _, lstm_probs = evaluate_test_set.generate_predictions(
        models,
        test_df,
        ["feature"],
        ["feature"],
    )

    assert isinstance(lstm_probs, np.ndarray)
    assert lstm_probs.shape == (1,)


def test_load_data_raises_on_missing_csv(monkeypatch, tmp_path):
    """H14: load_data should raise FileNotFoundError with guidance."""
    monkeypatch.setattr(evaluate_test_set, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="generate_features"):
        evaluate_test_set.load_data()


def test_load_data_reads_csvs_with_low_memory_disabled(monkeypatch, tmp_path):
    """Feature CSV loads should not emit dtype warnings on mixed metadata columns."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_path.write_text("stub")
    test_path.write_text("stub")

    calls = []

    def fake_read_csv(path, *args, **kwargs):
        calls.append((path, kwargs))
        return pd.DataFrame({"season": [2023], "upset": [0.0], "week": [2]})

    monkeypatch.setattr(evaluate_test_set, "DATA_DIR", tmp_path)
    monkeypatch.setattr(evaluate_test_set.pd, "read_csv", fake_read_csv)

    train_df, test_df, _, _ = evaluate_test_set.load_data()

    assert len(calls) == 2
    assert calls[0][0] == train_path
    assert calls[1][0] == test_path
    assert all(call_kwargs.get("low_memory") is False for _, call_kwargs in calls)
    assert not train_df.empty
    assert not test_df.empty


def test_verify_data_raises_on_season_overlap():
    train_df = pd.DataFrame(
        {"season": [2022, 2023], "upset": [0.0, 1.0], "week": [2, 2]}
    )
    test_df = pd.DataFrame({"season": [2023], "upset": [0.0], "week": [2]})

    with pytest.raises(ValueError, match="Season overlap"):
        evaluate_test_set.verify_data(train_df, test_df)


def test_calculate_metrics_handles_single_class():
    metrics = evaluate_test_set.calculate_metrics(
        np.array([1, 1, 1], dtype=float),
        np.array([0.9, 0.9999999, 1.0], dtype=float),
    )

    assert np.isnan(metrics["auc_roc"])
    assert np.isfinite(metrics["log_loss"])


def test_build_game_predictions_does_not_stringify_missing_team_names():
    test_df = pd.DataFrame(
        {
            "season": [2023],
            "week": [2],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "underdog": [np.nan],
            "favorite": [np.nan],
            "spread_magnitude": [3.0],
        }
    )

    predictions = evaluate_test_set.build_game_predictions(
        test_df,
        np.array([1]),
        np.array([0.6]),
        np.array([0.4]),
        np.array([0.7]),
    )

    assert predictions[0].underdog == ""
    assert predictions[0].favorite == ""
