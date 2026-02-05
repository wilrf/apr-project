"""Unified training pipeline for multi-model comparison.

Trains Logistic Regression, XGBoost, and LSTM on identical CV folds,
enabling disagreement analysis to understand each model's structural biases.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from src.models.cv_splitter import TimeSeriesCVSplitter
from src.models.logistic_model import UpsetLogisticRegression
from src.models.xgboost_model import UpsetXGBoost
from src.models.lstm_model import SiameseUpsetLSTM, SiameseLSTMDataset
from src.models.sequence_builder import (
    build_siamese_sequences,
    NormalizationStats,
    SEQUENCE_FEATURES,
    MATCHUP_FEATURES,
)


@dataclass
class GamePrediction:
    """Prediction record for a single game across all models."""

    game_id: str
    season: int
    week: int
    underdog: str
    favorite: str
    spread_magnitude: float
    y_true: int
    lr_prob: float
    xgb_prob: float
    lstm_prob: float

    @property
    def lr_pred(self) -> int:
        return int(self.lr_prob >= 0.5)

    @property
    def xgb_pred(self) -> int:
        return int(self.xgb_prob >= 0.5)

    @property
    def lstm_pred(self) -> int:
        return int(self.lstm_prob >= 0.5)


@dataclass
class FoldResult:
    """Results from a single CV fold."""

    fold_idx: int
    val_season: int
    train_size: int
    val_size: int
    predictions: List[GamePrediction]
    lr_metrics: Dict[str, float]
    xgb_metrics: Dict[str, float]
    lstm_metrics: Dict[str, float]


@dataclass
class UnifiedCVResults:
    """Aggregated results from unified cross-validation."""

    fold_results: List[FoldResult]
    all_predictions: List[GamePrediction] = field(default_factory=list)
    aggregated_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        # Flatten predictions from all folds
        if not self.all_predictions:
            self.all_predictions = [
                pred for fold in self.fold_results for pred in fold.predictions
            ]

        # Aggregate metrics across folds
        if not self.aggregated_metrics:
            self.aggregated_metrics = self._aggregate_metrics()

    def _aggregate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across all folds."""
        metrics = {"lr": {}, "xgb": {}, "lstm": {}}

        for model_key, attr_name in [
            ("lr", "lr_metrics"),
            ("xgb", "xgb_metrics"),
            ("lstm", "lstm_metrics"),
        ]:
            all_metrics: Dict[str, List[float]] = {}
            for fold in self.fold_results:
                fold_metrics = getattr(fold, attr_name)
                for k, v in fold_metrics.items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)

            for k, values in all_metrics.items():
                metrics[model_key][f"{k}_mean"] = float(np.mean(values))
                metrics[model_key][f"{k}_std"] = float(np.std(values))

        return metrics

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all predictions to a DataFrame."""
        return pd.DataFrame([
            {
                "game_id": p.game_id,
                "season": p.season,
                "week": p.week,
                "underdog": p.underdog,
                "favorite": p.favorite,
                "spread_magnitude": p.spread_magnitude,
                "y_true": p.y_true,
                "lr_prob": p.lr_prob,
                "xgb_prob": p.xgb_prob,
                "lstm_prob": p.lstm_prob,
                "lr_pred": p.lr_pred,
                "xgb_pred": p.xgb_pred,
                "lstm_pred": p.lstm_pred,
            }
            for p in self.all_predictions
        ])


class UnifiedTrainer:
    """
    Unified training pipeline for multi-model comparison.

    Ensures all three models (LR, XGBoost, LSTM) are trained and evaluated
    on identical CV folds, enabling fair disagreement analysis.
    """

    def __init__(
        self,
        n_folds: int = 6,
        lr_params: Optional[Dict[str, Any]] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
        lstm_params: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        random_state: int = 42,
    ):
        """
        Initialize unified trainer.

        Args:
            n_folds: Number of CV folds
            lr_params: Parameters for logistic regression
            xgb_params: Parameters for XGBoost
            lstm_params: Parameters for LSTM
            device: PyTorch device ('cpu' or 'cuda')
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.cv_splitter = TimeSeriesCVSplitter(n_folds=n_folds)
        self.device = device
        self.random_state = random_state

        # Default parameters based on experiments
        self.lr_params = lr_params or {
            "C": 0.1,
            "penalty": "l1",
            "solver": "saga",
            "random_state": random_state,
        }
        self.xgb_params = xgb_params or {
            "max_depth": 1,
            "learning_rate": 0.01,
            "n_estimators": 500,
            "random_state": random_state,
        }
        self.lstm_params = lstm_params or {
            "sequence_features": len(SEQUENCE_FEATURES),
            "matchup_features": len(MATCHUP_FEATURES),
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.3,
        }

    def cross_validate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "upset",
        lstm_epochs: int = 50,
        lstm_batch_size: int = 32,
        verbose: bool = True,
    ) -> UnifiedCVResults:
        """
        Run unified cross-validation across all three models.

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names for LR/XGBoost
            target_col: Name of target column
            lstm_epochs: Number of LSTM training epochs
            lstm_batch_size: LSTM batch size
            verbose: Whether to print progress

        Returns:
            UnifiedCVResults with predictions from all models
        """
        fold_results: List[FoldResult] = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(df)):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            val_season = val_df["season"].iloc[0]

            if verbose:
                print(f"\nFold {fold_idx + 1}/{self.n_folds} (Val: {val_season})")
                print(f"  Train: {len(train_df)} games, Val: {len(val_df)} games")

            # Train LR and XGBoost
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]

            # Logistic Regression
            lr_model = UpsetLogisticRegression(**self.lr_params)
            lr_model.fit(X_train, y_train)
            lr_probs = lr_model.predict_proba(X_val)

            # XGBoost
            xgb_model = UpsetXGBoost(**self.xgb_params)
            xgb_model.fit(X_train, y_train, verbose=False)
            xgb_probs = xgb_model.predict_proba(X_val)

            # LSTM (needs sequence data)
            lstm_probs = self._train_and_predict_lstm(
                train_df, val_df, lstm_epochs, lstm_batch_size, verbose
            )

            # Calculate metrics
            y_val_arr = y_val.values
            lr_metrics = self._calculate_metrics(y_val_arr, lr_probs)
            xgb_metrics = self._calculate_metrics(y_val_arr, xgb_probs)
            lstm_metrics = self._calculate_metrics(y_val_arr, lstm_probs)

            if verbose:
                print(f"  LR AUC: {lr_metrics['auc_roc']:.4f}")
                print(f"  XGB AUC: {xgb_metrics['auc_roc']:.4f}")
                print(f"  LSTM AUC: {lstm_metrics['auc_roc']:.4f}")

            # Build game predictions
            predictions = self._build_predictions(
                val_df, y_val_arr, lr_probs, xgb_probs, lstm_probs
            )

            fold_results.append(FoldResult(
                fold_idx=fold_idx,
                val_season=val_season,
                train_size=len(train_df),
                val_size=len(val_df),
                predictions=predictions,
                lr_metrics=lr_metrics,
                xgb_metrics=xgb_metrics,
                lstm_metrics=lstm_metrics,
            ))

        return UnifiedCVResults(fold_results=fold_results)

    def _train_and_predict_lstm(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int,
        batch_size: int,
        verbose: bool,
    ) -> np.ndarray:
        """Train LSTM and get predictions on validation set."""
        # Build sequences - compute stats from training data only
        train_data, train_stats = build_siamese_sequences(
            train_df, normalize=True, stats=None
        )

        # Apply training stats to validation data
        val_data, _ = build_siamese_sequences(
            val_df, normalize=True, stats=train_stats
        )

        # Create model
        model = SiameseUpsetLSTM(**self.lstm_params)
        model = model.to(self.device)

        # Create data loaders
        train_dataset = SiameseLSTMDataset(
            train_data.underdog_sequences,
            train_data.favorite_sequences,
            train_data.matchup_features,
            train_data.targets,
            train_data.underdog_masks,
            train_data.favorite_masks,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()

        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                und_seq, fav_seq, matchup, target, und_mask, fav_mask = batch
                und_seq = und_seq.to(self.device)
                fav_seq = fav_seq.to(self.device)
                matchup = matchup.to(self.device)
                target = target.to(self.device)
                und_mask = und_mask.to(self.device)
                fav_mask = fav_mask.to(self.device)

                optimizer.zero_grad()
                output = model(und_seq, fav_seq, matchup, und_mask, fav_mask)
                loss = criterion(output.squeeze(), target)
                loss.backward()
                optimizer.step()

        # Inference
        model.eval()
        with torch.no_grad():
            und_seq = torch.FloatTensor(val_data.underdog_sequences).to(self.device)
            fav_seq = torch.FloatTensor(val_data.favorite_sequences).to(self.device)
            matchup = torch.FloatTensor(val_data.matchup_features).to(self.device)
            und_mask = torch.FloatTensor(val_data.underdog_masks).to(self.device)
            fav_mask = torch.FloatTensor(val_data.favorite_masks).to(self.device)

            probs = model(und_seq, fav_seq, matchup, und_mask, fav_mask)
            probs = probs.squeeze().cpu().numpy()

        return probs

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            "auc_roc": float(roc_auc_score(y_true, y_pred)),
            "log_loss": float(log_loss(y_true, y_pred)),
            "brier_score": float(brier_score_loss(y_true, y_pred)),
        }

    def _build_predictions(
        self,
        val_df: pd.DataFrame,
        y_true: np.ndarray,
        lr_probs: np.ndarray,
        xgb_probs: np.ndarray,
        lstm_probs: np.ndarray,
    ) -> List[GamePrediction]:
        """Build GamePrediction objects for each validation game."""
        predictions = []

        for i, (idx, row) in enumerate(val_df.iterrows()):
            game_id = row.get(
                "game_id",
                f"{row['season']}_{row['week']}_{row['home_team']}_{row['away_team']}"
            )

            predictions.append(GamePrediction(
                game_id=game_id,
                season=int(row["season"]),
                week=int(row["week"]),
                underdog=str(row.get("underdog", "")),
                favorite=str(row.get("favorite", "")),
                spread_magnitude=float(row.get("spread_magnitude", 0)),
                y_true=int(y_true[i]),
                lr_prob=float(lr_probs[i]),
                xgb_prob=float(xgb_probs[i]),
                lstm_prob=float(lstm_probs[i]),
            ))

        return predictions

    def train_final(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "upset",
        lstm_epochs: int = 50,
        lstm_batch_size: int = 32,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train final models on full dataset.

        Args:
            df: Full training DataFrame
            feature_cols: Feature column names for LR/XGBoost
            target_col: Target column name
            lstm_epochs: LSTM training epochs
            lstm_batch_size: LSTM batch size
            verbose: Whether to print progress

        Returns:
            Dictionary with trained models and normalization stats
        """
        X = df[feature_cols]
        y = df[target_col]

        if verbose:
            print("Training final models...")

        # Logistic Regression
        lr_model = UpsetLogisticRegression(**self.lr_params)
        lr_model.fit(X, y)

        # XGBoost
        xgb_model = UpsetXGBoost(**self.xgb_params)
        xgb_model.fit(X, y, verbose=False)

        # LSTM
        lstm_data, lstm_stats = build_siamese_sequences(
            df, normalize=True, stats=None
        )

        lstm_model = SiameseUpsetLSTM(**self.lstm_params)
        lstm_model = lstm_model.to(self.device)

        train_dataset = SiameseLSTMDataset(
            lstm_data.underdog_sequences,
            lstm_data.favorite_sequences,
            lstm_data.matchup_features,
            lstm_data.targets,
            lstm_data.underdog_masks,
            lstm_data.favorite_masks,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=lstm_batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()

        lstm_model.train()
        for epoch in range(lstm_epochs):
            for batch in train_loader:
                und_seq, fav_seq, matchup, target, und_mask, fav_mask = batch
                und_seq = und_seq.to(self.device)
                fav_seq = fav_seq.to(self.device)
                matchup = matchup.to(self.device)
                target = target.to(self.device)
                und_mask = und_mask.to(self.device)
                fav_mask = fav_mask.to(self.device)

                optimizer.zero_grad()
                output = lstm_model(und_seq, fav_seq, matchup, und_mask, fav_mask)
                loss = criterion(output.squeeze(), target)
                loss.backward()
                optimizer.step()

        lstm_model.eval()

        if verbose:
            print("Final models trained.")

        return {
            "lr_model": lr_model,
            "xgb_model": xgb_model,
            "lstm_model": lstm_model,
            "lstm_stats": lstm_stats,
            "feature_cols": feature_cols,
        }
