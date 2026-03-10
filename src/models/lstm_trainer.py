"""Siamese LSTM training pipeline with PyTorch DataLoader support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader

from src.evaluation.metrics import safe_log_loss, safe_roc_auc_score
from src.models.cv_splitter import TimeSeriesCVSplitter
from src.models.lstm_config import (
    TUNED_LSTM_MODEL_PARAMS,
    TUNED_LSTM_TRAINING_PARAMS,
)
from src.models.lstm_model import SiameseLSTMDataset, SiameseUpsetLSTM
from src.models.sequence_builder import (
    MATCHUP_FEATURES,
    SEQUENCE_FEATURES,
    NormalizationStats,
    SiameseLSTMData,
    build_siamese_sequences,
)


class SiameseLSTMTrainer:
    """
    Training pipeline for Siamese LSTM model with time-series cross-validation.

    Handles:
    - Sequence building from DataFrames (separate sequences for each team)
    - PyTorch DataLoader creation
    - Training loop with early stopping
    - Metric calculation
    - Cross-validation
    """

    def __init__(
        self,
        hidden_size: int = int(TUNED_LSTM_MODEL_PARAMS["hidden_size"]),
        num_layers: int = int(TUNED_LSTM_MODEL_PARAMS["num_layers"]),
        dropout: float = float(TUNED_LSTM_MODEL_PARAMS["dropout"]),
        learning_rate: float = float(TUNED_LSTM_TRAINING_PARAMS["learning_rate"]),
        batch_size: int = int(TUNED_LSTM_TRAINING_PARAMS["batch_size"]),
        epochs: int = int(TUNED_LSTM_TRAINING_PARAMS["epochs"]),
        patience: int = int(TUNED_LSTM_TRAINING_PARAMS["patience"]),
        n_folds: int = 6,
        device: Optional[str] = None,
    ):
        """
        Initialize Siamese LSTM trainer.

        Args:
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Maximum training epochs
            patience: Early stopping patience (epochs without improvement)
            n_folds: Number of CV folds
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.n_folds = n_folds

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.cv_splitter = TimeSeriesCVSplitter(n_folds=n_folds)
        self.model: Optional[SiameseUpsetLSTM] = None
        self._train_stats: Optional[NormalizationStats] = None
        self._sequence_data: Optional[SiameseLSTMData] = None

    def _create_model(self) -> SiameseUpsetLSTM:
        """Create a new Siamese LSTM model instance."""
        model = SiameseUpsetLSTM(
            sequence_features=len(SEQUENCE_FEATURES),
            matchup_features=len(MATCHUP_FEATURES),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        return model.to(self.device)

    def _create_dataloader(
        self,
        data: SiameseLSTMData,
        indices: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create DataLoader for a subset of data."""
        dataset = SiameseLSTMDataset(
            underdog_sequences=data.underdog_sequences[indices],
            favorite_sequences=data.favorite_sequences[indices],
            matchup_features=data.matchup_features[indices],
            targets=data.targets[indices],
            underdog_masks=data.underdog_masks[indices],
            favorite_masks=data.favorite_masks[indices],
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _train_epoch(
        self,
        model: SiameseUpsetLSTM,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch, return average loss."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            und_seq, fav_seq, matchup_feats, targets, und_mask, fav_mask = batch
            und_seq = und_seq.to(self.device)
            fav_seq = fav_seq.to(self.device)
            matchup_feats = matchup_feats.to(self.device)
            targets = targets.to(self.device)
            und_mask = und_mask.to(self.device)
            fav_mask = fav_mask.to(self.device)

            optimizer.zero_grad()
            outputs = model(und_seq, fav_seq, matchup_feats, und_mask, fav_mask)
            loss = criterion(outputs.squeeze(-1), targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _evaluate(
        self,
        model: SiameseUpsetLSTM,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model on validation data.

        Returns:
            Tuple of (loss, predictions, targets)
        """
        model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                und_seq, fav_seq, matchup_feats, targets, und_mask, fav_mask = batch
                und_seq = und_seq.to(self.device)
                fav_seq = fav_seq.to(self.device)
                matchup_feats = matchup_feats.to(self.device)
                targets = targets.to(self.device)
                und_mask = und_mask.to(self.device)
                fav_mask = fav_mask.to(self.device)

                outputs = model(und_seq, fav_seq, matchup_feats, und_mask, fav_mask)
                outputs_squeezed = outputs.squeeze(-1)
                loss = criterion(outputs_squeezed, targets)

                total_loss += loss.item()
                n_batches += 1

                all_preds.extend(outputs_squeezed.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, np.array(all_preds), np.array(all_targets)

    def _train_model(
        self,
        model: SiameseUpsetLSTM,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Train model with early stopping.

        Returns:
            Training history with metrics
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            val_loss, _, _ = self._evaluate(model, val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(self.device)

        history["epochs_trained"] = epoch + 1
        history["best_val_loss"] = best_val_loss

        return history

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        return {
            "auc_roc": safe_roc_auc_score(y_true, y_pred),
            "log_loss": safe_log_loss(y_true, y_pred_clipped),
            "brier_score": brier_score_loss(y_true, y_pred),
        }

    def cross_validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run time-series cross-validation.

        Args:
            df: DataFrame with game data and features

        Returns:
            Dictionary with fold metrics and aggregated results
        """
        # Filter to valid games first
        valid_df = df[df["upset"].notna()].copy()

        fold_metrics: List[Dict[str, Any]] = []
        fold_predictions: List[Dict[str, Any]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv_splitter.split(valid_df)
        ):
            # Split data BEFORE building sequences to avoid leakage
            train_df = valid_df.iloc[train_idx].copy()
            val_df = valid_df.iloc[val_idx].copy()

            # Build train sequences and compute stats from TRAIN data only
            train_data, train_stats = build_siamese_sequences(
                train_df, normalize=True, stats=None
            )

            # Build val sequences using TRAIN stats (no leakage)
            val_data, _ = build_siamese_sequences(
                val_df, normalize=True, stats=train_stats
            )

            # Create data loaders (indices are now 0 to len-1 for each split)
            train_loader = self._create_dataloader(
                train_data, np.arange(train_data.n_samples), shuffle=True
            )
            val_loader = self._create_dataloader(
                val_data, np.arange(val_data.n_samples), shuffle=False
            )

            # Create and train model
            model = self._create_model()
            history = self._train_model(model, train_loader, val_loader)

            # Get final predictions
            _, y_pred, y_true = self._evaluate(model, val_loader, nn.BCELoss())

            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred)
            metrics["fold"] = fold_idx
            metrics["train_size"] = train_data.n_samples
            metrics["val_size"] = val_data.n_samples
            metrics["epochs_trained"] = history["epochs_trained"]

            fold_metrics.append(metrics)
            fold_predictions.append(
                {
                    "val_idx": val_idx,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "game_ids": val_data.game_ids,
                }
            )

        # Aggregate metrics
        aggregated = self._aggregate_metrics(fold_metrics)

        return {
            "fold_metrics": fold_metrics,
            "aggregated": aggregated,
            "predictions": fold_predictions,
        }

    def _aggregate_metrics(
        self, fold_metrics: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate metrics across folds."""
        metric_names = ["auc_roc", "log_loss", "brier_score"]
        aggregated: Dict[str, float] = {}

        for metric in metric_names:
            values = [f[metric] for f in fold_metrics]
            aggregated[f"{metric}_mean"] = float(np.nanmean(values))
            aggregated[f"{metric}_std"] = float(np.nanstd(values))

        return aggregated

    def fit(self, df: pd.DataFrame) -> "SiameseLSTMTrainer":
        """
        Train model on full dataset (no validation split).

        Args:
            df: DataFrame with game data and features

        Returns:
            Self for chaining
        """
        # Build sequences and store training stats for later use in predict_proba
        sequence_data, self._train_stats = build_siamese_sequences(
            df, normalize=True, stats=None
        )

        # Store for prediction
        self._sequence_data = sequence_data

        # Create model and train
        self.model = self._create_model()

        train_loader = self._create_dataloader(
            sequence_data,
            np.arange(sequence_data.n_samples),
            shuffle=True,
        )

        # No validation set - train for fixed epochs
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        for _ in range(self.epochs):
            self._train_epoch(self.model, train_loader, optimizer, criterion)

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get predictions for new data.

        Args:
            df: DataFrame with game data and features

        Returns:
            Array of upset probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        if self._train_stats is None:
            raise ValueError("Training stats not available. Call fit() first.")

        # Use TRAINING stats for normalization (no data leakage)
        sequence_data, _ = build_siamese_sequences(
            df, normalize=True, stats=self._train_stats
        )

        loader = self._create_dataloader(
            sequence_data,
            np.arange(sequence_data.n_samples),
            shuffle=False,
        )

        _, predictions, _ = self._evaluate(self.model, loader, nn.BCELoss())

        return predictions

    def get_attention_weights(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get attention weights for interpretability.

        Args:
            df: DataFrame with game data and features

        Returns:
            Tuple of (underdog_attention, favorite_attention) arrays
            Each has shape (n_samples, seq_length)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        if self._train_stats is None:
            raise ValueError("Training stats not available. Call fit() first.")

        # Use TRAINING stats for normalization (no data leakage)
        sequence_data, _ = build_siamese_sequences(
            df, normalize=True, stats=self._train_stats
        )

        und_seq = torch.FloatTensor(sequence_data.underdog_sequences).to(self.device)
        fav_seq = torch.FloatTensor(sequence_data.favorite_sequences).to(self.device)
        und_mask = torch.FloatTensor(sequence_data.underdog_masks).to(self.device)
        fav_mask = torch.FloatTensor(sequence_data.favorite_masks).to(self.device)

        self.model.eval()
        und_attn, fav_attn = self.model.get_attention_weights(
            und_seq, fav_seq, und_mask, fav_mask
        )

        return und_attn.cpu().numpy(), fav_attn.cpu().numpy()


# Backward compatibility alias
LSTMTrainer = SiameseLSTMTrainer
