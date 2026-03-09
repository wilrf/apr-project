"""Shared tuned defaults for the Siamese LSTM."""

from __future__ import annotations

from typing import Dict

TUNED_LSTM_MODEL_PARAMS: Dict[str, float | int] = {
    "hidden_size": 64,
    "num_layers": 3,
    "dropout": 0.25,
}

TUNED_LSTM_TRAINING_PARAMS: Dict[str, float | int] = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 25,
    "patience": 6,
}
