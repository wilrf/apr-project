"""Disagreement analysis framework for multi-model comparison.

Categorizes predictions based on model agreement patterns to understand
the structural biases of each model type:
- LR: Captures spread mispricing (linear signal)
- XGBoost: Captures feature interactions (non-linear patterns)
- LSTM: Captures temporal patterns (momentum, fatigue, sequences)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import safe_probability_correlation

if TYPE_CHECKING:
    from src.models.unified_trainer import GamePrediction


class PredictionCategory(Enum):
    """
    Categories based on which models correctly predicted the outcome.

    Key insight: Each category reveals different upset mechanisms.
    """

    # All models agree on outcome
    ALL_CORRECT = "all_correct"  # Obvious upsets - clear signal all models detect
    ALL_WRONG = "all_wrong"  # True randomness - no model captures

    # Single model correct (most interesting for understanding model biases)
    ONLY_LR = "only_lr"  # Spread mispricing - linear market inefficiency
    ONLY_XGB = "only_xgb"  # Interaction-driven - non-linear feature combos
    ONLY_LSTM = "only_lstm"  # Temporal signal - momentum/fatigue patterns

    # Two models correct
    LR_XGB = "lr_xgb"  # Static models agree - non-temporal signal
    LR_LSTM = "lr_lstm"  # Linear + temporal agree
    XGB_LSTM = "xgb_lstm"  # Non-linear + temporal agree


@dataclass
class CategoryStats:
    """Statistics for a prediction category."""

    category: PredictionCategory
    count: int
    pct_of_total: float
    upset_rate: float  # Actual upset rate in this category
    avg_spread: float  # Average spread magnitude
    avg_lr_prob: float
    avg_xgb_prob: float
    avg_lstm_prob: float


@dataclass
class DisagreementInsight:
    """Insight about a specific category's games."""

    category: PredictionCategory
    description: str
    key_features: Dict[str, float]  # Feature -> avg value for this category
    example_games: List[str]  # Game IDs of notable examples


class DisagreementAnalyzer:
    """
    Analyzes disagreement patterns between LR, XGBoost, and LSTM models.

    Each model captures different structural aspects of upsets:
    - LR: When the spread systematically misprices certain linear combinations
    - XGBoost: When interactions between features signal upset (non-linear)
    - LSTM: When recent game sequences signal momentum/fatigue shifts
    """

    def __init__(
        self,
        predictions: List["GamePrediction"],
        threshold: Optional[float] = None,
    ):
        """
        Initialize analyzer with predictions from all three models.

        Args:
            predictions: List of GamePrediction objects from UnifiedTrainer
            threshold: Probability threshold for binary predictions.
                If None, uses the base upset rate from the data, so a model
                "predicts upset" when it assigns higher probability than average.
                This is the principled choice for minority-class problems.
        """
        self.predictions = predictions
        if threshold is None:
            base_rate = np.mean([p.y_true for p in predictions]) if predictions else 0.5
            self.threshold = base_rate
        else:
            self.threshold = threshold
        self._categories: Optional[Dict[PredictionCategory, List["GamePrediction"]]] = (
            None
        )

    def categorize_all(self) -> Dict[PredictionCategory, List["GamePrediction"]]:
        """
        Categorize all predictions by model agreement pattern.

        Returns:
            Dictionary mapping each category to list of matching predictions
        """
        if self._categories is not None:
            return self._categories

        categories: Dict[PredictionCategory, List["GamePrediction"]] = {
            cat: [] for cat in PredictionCategory
        }

        for pred in self.predictions:
            # Determine which models were correct
            lr_correct = (pred.lr_prob >= self.threshold) == pred.y_true
            xgb_correct = (pred.xgb_prob >= self.threshold) == pred.y_true
            lstm_correct = (pred.lstm_prob >= self.threshold) == pred.y_true

            # Categorize based on pattern
            if lr_correct and xgb_correct and lstm_correct:
                category = PredictionCategory.ALL_CORRECT
            elif not lr_correct and not xgb_correct and not lstm_correct:
                category = PredictionCategory.ALL_WRONG
            elif lr_correct and not xgb_correct and not lstm_correct:
                category = PredictionCategory.ONLY_LR
            elif not lr_correct and xgb_correct and not lstm_correct:
                category = PredictionCategory.ONLY_XGB
            elif not lr_correct and not xgb_correct and lstm_correct:
                category = PredictionCategory.ONLY_LSTM
            elif lr_correct and xgb_correct and not lstm_correct:
                category = PredictionCategory.LR_XGB
            elif lr_correct and not xgb_correct and lstm_correct:
                category = PredictionCategory.LR_LSTM
            else:  # xgb_correct and lstm_correct and not lr_correct
                category = PredictionCategory.XGB_LSTM

            categories[category].append(pred)

        self._categories = categories
        return categories

    def get_category_stats(self) -> pd.DataFrame:
        """
        Get statistics for each category.

        Returns:
            DataFrame with stats per category
        """
        categories = self.categorize_all()
        total = len(self.predictions)

        stats = []
        for cat, preds in categories.items():
            if not preds:
                continue

            stats.append(
                CategoryStats(
                    category=cat,
                    count=len(preds),
                    pct_of_total=len(preds) / total * 100 if total > 0 else 0,
                    upset_rate=np.mean([p.y_true for p in preds]) if preds else 0,
                    avg_spread=(
                        np.mean([p.spread_magnitude for p in preds]) if preds else 0
                    ),
                    avg_lr_prob=np.mean([p.lr_prob for p in preds]) if preds else 0,
                    avg_xgb_prob=np.mean([p.xgb_prob for p in preds]) if preds else 0,
                    avg_lstm_prob=np.mean([p.lstm_prob for p in preds]) if preds else 0,
                )
            )

        # Convert to DataFrame
        return pd.DataFrame(
            [
                {
                    "category": s.category.value,
                    "count": s.count,
                    "pct_of_total": s.pct_of_total,
                    "upset_rate": s.upset_rate,
                    "avg_spread": s.avg_spread,
                    "avg_lr_prob": s.avg_lr_prob,
                    "avg_xgb_prob": s.avg_xgb_prob,
                    "avg_lstm_prob": s.avg_lstm_prob,
                }
                for s in stats
            ]
        )

    def get_exclusive_insights(self) -> Dict[PredictionCategory, DisagreementInsight]:
        """
        Get insights for categories where only one model is correct.

        These are the most interesting categories for understanding model biases.

        Returns:
            Dictionary with insights for ONLY_LR, ONLY_XGB, ONLY_LSTM
        """
        categories = self.categorize_all()
        insights = {}

        exclusive_configs = [
            (
                PredictionCategory.ONLY_LR,
                "lr_prob",
                "Games where only LR was correct suggest systematic spread mispricing. "
                "The market undervalued certain linear feature combinations.",
            ),
            (
                PredictionCategory.ONLY_XGB,
                "xgb_prob",
                "Games where only XGBoost was correct suggest "
                "non-linear feature interactions. "
                "Specific combinations of factors created "
                "upset conditions.",
            ),
            (
                PredictionCategory.ONLY_LSTM,
                "lstm_prob",
                "Games where only LSTM was correct reveal temporal dynamics. "
                "Recent game sequences (momentum shifts, fatigue patterns) "
                "provided predictive signal invisible to static models.",
            ),
        ]

        for cat, prob_attr, description in exclusive_configs:
            preds = categories[cat]
            if preds:
                insights[cat] = DisagreementInsight(
                    category=cat,
                    description=description,
                    key_features={
                        "avg_spread": np.mean([p.spread_magnitude for p in preds]),
                        "upset_rate": np.mean([p.y_true for p in preds]),
                        f"avg_{prob_attr.replace('_prob', '')}_confidence": np.mean(
                            [abs(getattr(p, prob_attr) - self.threshold) for p in preds]
                        ),
                    },
                    example_games=[p.game_id for p in preds[:5]],
                )

        return insights

    def get_agreement_matrix(self) -> pd.DataFrame:
        """
        Get pairwise agreement rates between models.

        Returns:
            DataFrame with agreement percentages
        """
        n = len(self.predictions)
        if n == 0:
            return pd.DataFrame()

        lr_preds = np.array(
            [int(p.lr_prob >= self.threshold) for p in self.predictions]
        )
        xgb_preds = np.array(
            [int(p.xgb_prob >= self.threshold) for p in self.predictions]
        )
        lstm_preds = np.array(
            [int(p.lstm_prob >= self.threshold) for p in self.predictions]
        )

        lr_xgb_agree = (lr_preds == xgb_preds).mean()
        lr_lstm_agree = (lr_preds == lstm_preds).mean()
        xgb_lstm_agree = (xgb_preds == lstm_preds).mean()
        all_agree = ((lr_preds == xgb_preds) & (xgb_preds == lstm_preds)).mean()

        return pd.DataFrame(
            {
                "model_pair": ["LR-XGB", "LR-LSTM", "XGB-LSTM", "All Three"],
                "agreement_rate": [
                    lr_xgb_agree,
                    lr_lstm_agree,
                    xgb_lstm_agree,
                    all_agree,
                ],
            }
        )

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix of model probabilities.

        Returns:
            DataFrame with correlation coefficients
        """
        if not self.predictions:
            return pd.DataFrame()

        lr_probs = np.array([p.lr_prob for p in self.predictions])
        xgb_probs = np.array([p.xgb_prob for p in self.predictions])
        lstm_probs = np.array([p.lstm_prob for p in self.predictions])

        lr_xgb = safe_probability_correlation(lr_probs, xgb_probs)
        lr_lstm = safe_probability_correlation(lr_probs, lstm_probs)
        xgb_lstm = safe_probability_correlation(xgb_probs, lstm_probs)
        corr = np.array(
            [
                [1.0, lr_xgb, lr_lstm],
                [lr_xgb, 1.0, xgb_lstm],
                [lr_lstm, xgb_lstm, 1.0],
            ]
        )

        return pd.DataFrame(
            corr,
            index=["LR", "XGB", "LSTM"],
            columns=["LR", "XGB", "LSTM"],
        )

    def export_table(self, path: Path) -> None:
        """
        Export all predictions to CSV for further analysis.

        Args:
            path: Output path for CSV file
        """
        categories = self.categorize_all()

        # Invert: pred -> category (O(n) instead of O(n*k) per row)
        pred_to_cat = {
            id(pred): cat.value for cat, preds in categories.items() for pred in preds
        }

        rows = []
        for pred in self.predictions:
            category = pred_to_cat.get(id(pred), "unknown")

            rows.append(
                {
                    "game_id": pred.game_id,
                    "season": pred.season,
                    "week": pred.week,
                    "underdog": pred.underdog,
                    "favorite": pred.favorite,
                    "spread_magnitude": pred.spread_magnitude,
                    "y_true": pred.y_true,
                    "lr_prob": pred.lr_prob,
                    "xgb_prob": pred.xgb_prob,
                    "lstm_prob": pred.lstm_prob,
                    "lr_pred": int(pred.lr_prob >= self.threshold),
                    "xgb_pred": int(pred.xgb_prob >= self.threshold),
                    "lstm_pred": int(pred.lstm_prob >= self.threshold),
                    "lr_correct": int((pred.lr_prob >= self.threshold) == pred.y_true),
                    "xgb_correct": int(
                        (pred.xgb_prob >= self.threshold) == pred.y_true
                    ),
                    "lstm_correct": int(
                        (pred.lstm_prob >= self.threshold) == pred.y_true
                    ),
                    "category": category,
                }
            )

        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def summarize(self) -> str:
        """
        Generate a text summary of disagreement analysis.

        Returns:
            Formatted summary string
        """
        categories = self.categorize_all()
        total = len(self.predictions)

        lines = [
            "=" * 60,
            "MULTI-MODEL DISAGREEMENT ANALYSIS",
            "=" * 60,
            f"\nTotal predictions: {total}",
            "\n--- Category Breakdown ---\n",
        ]

        for cat in PredictionCategory:
            count = len(categories[cat])
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  {cat.value:15s}: {count:4d} ({pct:5.1f}%)")

        # Agreement matrix
        agreement = self.get_agreement_matrix()
        lines.extend(
            [
                "\n--- Pairwise Agreement ---\n",
            ]
        )
        for _, row in agreement.iterrows():
            lines.append(f"  {row['model_pair']:12s}: {row['agreement_rate']:.1%}")

        # Exclusive insights
        insights = self.get_exclusive_insights()
        if insights:
            lines.extend(
                [
                    "\n--- Key Findings ---\n",
                ]
            )
            for cat, insight in insights.items():
                lines.extend(
                    [
                        f"\n{cat.value.upper()}:",
                        f"  {insight.description[:80]}...",
                        f"  Examples: {', '.join(insight.example_games[:3])}",
                    ]
                )

        return "\n".join(lines)
