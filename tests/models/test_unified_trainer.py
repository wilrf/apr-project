"""Regression tests for UnifiedTrainer internals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import src.models.unified_trainer as unified_trainer_module
from src.models.sequence_builder import NormalizationStats, SiameseLSTMData
from src.models.unified_trainer import UnifiedTrainer


class _FakeLSTMModel(torch.nn.Module):
    def forward(self, und_seq, fav_seq, matchup, und_mask, fav_mask):
        return torch.full((len(und_seq), 1), 0.5, device=und_seq.device)


class _NaNOutputLSTMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, und_seq, fav_seq, matchup, und_mask, fav_mask):
        return torch.full((len(und_seq), 1), float("nan"), device=und_seq.device)


def _fake_sequence_data(n_samples: int) -> SiameseLSTMData:
    return SiameseLSTMData(
        underdog_sequences=np.zeros((n_samples, 8, 1), dtype=np.float32),
        favorite_sequences=np.zeros((n_samples, 8, 1), dtype=np.float32),
        matchup_features=np.zeros((n_samples, 1), dtype=np.float32),
        targets=np.zeros(n_samples, dtype=np.float32),
        underdog_masks=np.zeros((n_samples, 8), dtype=np.float32),
        favorite_masks=np.zeros((n_samples, 8), dtype=np.float32),
        game_ids=np.arange(n_samples),
    )


def test_train_and_predict_lstm_passes_validation_data_to_early_stopping(
    monkeypatch,
):
    """UnifiedTrainer should enable early stopping during CV predictions."""
    trainer = UnifiedTrainer()
    train_df = pd.DataFrame(
        {"upset": [0.0, 1.0], "season": [2020, 2020], "week": [2, 3]}
    )
    val_df = pd.DataFrame({"upset": [0.0], "season": [2021], "week": [2]})
    captured: dict[str, object] = {}

    def fake_build_siamese_sequences(
        df,
        normalize=True,
        seq_length=8,
        stats=None,
        matchup_feature_cols=None,
        sequence_feature_cols=None,
        history_df=None,
    ):
        if stats is None:
            return _fake_sequence_data(len(df)), NormalizationStats({}, {})
        return _fake_sequence_data(len(df)), None

    def fake_run_lstm_training(
        model, train_data, epochs, batch_size, val_data=None, patience=6
    ):
        captured["val_data"] = val_data
        return {"epochs_trained": epochs}

    monkeypatch.setattr(
        unified_trainer_module, "build_siamese_sequences", fake_build_siamese_sequences
    )
    monkeypatch.setattr(
        trainer,
        "_build_lstm_model",
        lambda *args, **kwargs: _FakeLSTMModel(),
    )
    monkeypatch.setattr(trainer, "_run_lstm_training", fake_run_lstm_training)

    trainer._train_and_predict_lstm(
        train_df,
        val_df,
        epochs=2,
        batch_size=2,
        verbose=False,
    )

    assert captured["val_data"] is not None
    assert captured["val_data"].n_samples == len(val_df)


def test_run_lstm_training_raises_clear_error_on_non_finite_output():
    trainer = UnifiedTrainer()
    model = _NaNOutputLSTMModel()
    train_data = _fake_sequence_data(4)

    with pytest.raises(RuntimeError, match="non-finite"):
        trainer._run_lstm_training(model, train_data, epochs=1, batch_size=2)


def test_calculate_metrics_handles_single_class_targets():
    trainer = UnifiedTrainer()

    metrics = trainer._calculate_metrics(
        np.array([1, 1, 1], dtype=float),
        np.array([0.9, 0.9999999, 1.0], dtype=float),
    )

    assert np.isnan(metrics["auc_roc"])
    assert np.isfinite(metrics["log_loss"])


def test_run_lstm_training_zero_epochs_returns_clean_history():
    trainer = UnifiedTrainer()
    model = _FakeLSTMModel()
    train_data = _fake_sequence_data(2)

    history = trainer._run_lstm_training(model, train_data, epochs=0, batch_size=2)

    assert history["epochs_trained"] == 0
    assert history["train_loss"] == []
