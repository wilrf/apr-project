"""Pydantic v2 response models for the dashboard API.

These are the single source of contract truth the frontend mirrors. AUC fields
are ``float | None`` because a single-class season yields a NaN AUC, surfaced as
``null`` in JSON.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel

# ── Summary ─────────────────────────────────────────────────────────────────


class ModelMetrics(BaseModel):
    """Per-model headline metrics for one dataset."""

    auc: Optional[float]
    brier: float
    log_loss: float


class DatasetSummary(BaseModel):
    """Summary for a single dataset: per-model metrics + dataset-level stats."""

    n_games: int
    upset_rate: float
    baseline_brier: float
    models: Dict[str, ModelMetrics]


class SummaryResponse(BaseModel):
    """Map of dataset name -> its summary (test, cv_with_spread, cv_without_spread)."""

    datasets: Dict[str, DatasetSummary]


# ── Predictions ─────────────────────────────────────────────────────────────


class PredictionRow(BaseModel):
    """One game's per-model probabilities, correctness, and category."""

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
    lr_correct: int
    xgb_correct: int
    lstm_correct: int
    category: str


class PredictionsResponse(BaseModel):
    """All per-game rows for one dataset."""

    dataset: str
    n_games: int
    rows: List[PredictionRow]


# ── Disagreement ────────────────────────────────────────────────────────────


class DisagreementCategory(BaseModel):
    """One disagreement category's count, share, and actual upset rate."""

    category: str
    n: int
    pct: float
    upset_rate: float


class DisagreementResponse(BaseModel):
    """Disagreement category breakdown for one dataset."""

    dataset: str
    n_games: int
    categories: List[DisagreementCategory]


# ── Features (LR coefficients) ──────────────────────────────────────────────


class FeatureCoef(BaseModel):
    """One LR coefficient, ranked by absolute magnitude."""

    rank: int
    feature: str
    coefficient: float
    direction: str  # "raises" / "lowers" / "neutral" upset odds


class FeaturesResponse(BaseModel):
    """LR coefficients for the with-spread and without-spread variants."""

    with_spread: List[FeatureCoef]
    without_spread: List[FeatureCoef]


# ── Seasons ─────────────────────────────────────────────────────────────────


class SeasonRow(BaseModel):
    """Per-season games, upset rate, and per-model AUC (None when undefined)."""

    season: int
    n: int
    upset_rate: float
    lr_auc: Optional[float]
    xgb_auc: Optional[float]
    lstm_auc: Optional[float]


class SeasonsResponse(BaseModel):
    """Per-season breakdown for one dataset."""

    dataset: str
    seasons: List[SeasonRow]


# ── Curves (ROC + calibration) ──────────────────────────────────────────────


class RocSeries(BaseModel):
    """One model's ROC curve points plus its AUC."""

    model: str
    auc: Optional[float]
    fpr: List[float]
    tpr: List[float]


class CalibrationSeries(BaseModel):
    """One model's reliability-diagram points plus its calibration error."""

    model: str
    calibration_error: float
    prob_pred: List[float]
    prob_true: List[float]


class CurvesResponse(BaseModel):
    """ROC + calibration series (one entry per model) for one dataset."""

    dataset: str
    roc: List[RocSeries]
    calibration: List[CalibrationSeries]
