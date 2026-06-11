"""Data access + metric computation over the frozen ``results/`` artifacts.

Every metric is computed by reusing ``src.evaluation.metrics`` (and
``sklearn.metrics.brier_score_loss`` for Brier, matching the legacy
``tools/dashboard/serve.py``). Nothing is recomputed a different way, so the
dashboard's numbers equal ``results/`` and the README exactly.

Categorization rules (CLAUDE.md invariants):
- ``test`` dataset: the frozen ``category``/``*_correct`` columns are
  authoritative — use them directly, never recompute.
- ``cv_with_spread`` / ``cv_without_spread``: these CSVs lack a ``category``
  column. Derive ``*_pred = prob >= base_rate`` where ``base_rate =
  y_true.mean()`` for that dataset — NEVER 0.5, and never the persisted
  ``*_pred`` columns (which hardcode 0.5). This matches ``DisagreementAnalyzer``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_curve

from src.evaluation.metrics import (
    calculate_baseline_brier,
    calculate_calibration_metrics,
    safe_log_loss,
    safe_roc_auc_score,
)

# Resolve to the repository root (.../apr-research), NOT the current working
# directory. dashboard/backend/data_access.py -> parents[2] is the repo root.
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Dataset registry: logical name -> CSV path.
_DATASET_PATHS: Dict[str, Path] = {
    "test": RESULTS / "test" / "predictions.csv",
    "cv_with_spread": RESULTS / "ab_experiment" / "predictions_with_spread.csv",
    "cv_without_spread": RESULTS / "ab_experiment" / "predictions_without_spread.csv",
}

# LR-coefficient JSONs: variant -> path.
_COEF_PATHS: Dict[str, Path] = {
    "with_spread": RESULTS / "ab_experiment" / "lr_coefs_with_spread.json",
    "without_spread": RESULTS / "ab_experiment" / "lr_coefs_without_spread.json",
}

# Models in canonical (LR, XGB, LSTM) order. Each tuple: (key, prob column).
_MODELS: List[tuple[str, str]] = [
    ("lr", "lr_prob"),
    ("xgb", "xgb_prob"),
    ("lstm", "lstm_prob"),
]

DATASETS: List[str] = list(_DATASET_PATHS.keys())
COEF_VARIANTS: List[str] = list(_COEF_PATHS.keys())

# Friendly, human-readable labels for the 8-way disagreement taxonomy. Ordered
# as the legacy dashboard presents them.
CATEGORY_ORDER: List[str] = [
    "all_correct",
    "all_wrong",
    "only_lr",
    "only_xgb",
    "only_lstm",
    "lr_xgb",
    "lr_lstm",
    "xgb_lstm",
]


# ── Typed return shapes (so the API layer needs no casts) ───────────────────


class ModelMetricsDict(TypedDict):
    auc: float
    brier: float
    log_loss: float


class SummaryDict(TypedDict):
    n_games: int
    upset_rate: float
    baseline_brier: float
    models: Dict[str, ModelMetricsDict]


class DisagreementDict(TypedDict):
    category: str
    n: int
    pct: float
    upset_rate: float


class SeasonDict(TypedDict):
    season: int
    n: int
    upset_rate: float
    lr_auc: Optional[float]
    xgb_auc: Optional[float]
    lstm_auc: Optional[float]


class RocSeriesDict(TypedDict):
    model: str
    auc: Optional[float]
    fpr: List[float]
    tpr: List[float]


class CalibrationSeriesDict(TypedDict):
    model: str
    calibration_error: float
    prob_pred: List[float]
    prob_true: List[float]


# ── Loaders ─────────────────────────────────────────────────────────────────


def load_predictions(dataset: str) -> pd.DataFrame:
    """Load a prediction CSV for one of the three known datasets.

    Raises:
        KeyError: if ``dataset`` is not a known dataset name.
        FileNotFoundError: if the frozen artifact is missing.
    """
    if dataset not in _DATASET_PATHS:
        raise KeyError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}.")
    path = _DATASET_PATHS[dataset]
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen results artifact not found: {path}. "
            "Regenerate via the documented pipeline before serving the dashboard."
        )
    return pd.read_csv(path)


def load_lr_coefs(variant: str) -> Dict[str, float]:
    """Load the LR coefficient JSON for ``with_spread`` / ``without_spread``.

    Raises:
        KeyError: if ``variant`` is not a known variant name.
        FileNotFoundError: if the frozen artifact is missing.
    """
    if variant not in _COEF_PATHS:
        raise KeyError(
            f"Unknown coef variant {variant!r}; expected one of {COEF_VARIANTS}."
        )
    path = _COEF_PATHS[variant]
    if not path.exists():
        raise FileNotFoundError(f"Frozen LR-coefficient artifact not found: {path}.")
    return json.loads(path.read_text())


# ── Categorization (the threshold invariant lives here) ─────────────────────


def _eight_way_category(lr_ok: bool, xgb_ok: bool, lstm_ok: bool) -> str:
    """Map a (lr, xgb, lstm) correctness triple to the 8-way taxonomy.

    Naming convention (matches ``DisagreementAnalyzer`` and the frozen test
    ``category`` column): the label lists the models that were CORRECT.
    """
    if lr_ok and xgb_ok and lstm_ok:
        return "all_correct"
    if not lr_ok and not xgb_ok and not lstm_ok:
        return "all_wrong"
    if lr_ok and not xgb_ok and not lstm_ok:
        return "only_lr"
    if not lr_ok and xgb_ok and not lstm_ok:
        return "only_xgb"
    if not lr_ok and not xgb_ok and lstm_ok:
        return "only_lstm"
    if lr_ok and xgb_ok and not lstm_ok:
        return "lr_xgb"
    if lr_ok and not xgb_ok and lstm_ok:
        return "lr_lstm"
    return "xgb_lstm"  # xgb_ok and lstm_ok and not lr_ok


def derive_categories(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``*_correct`` and ``category`` columns.

    For ``test`` the frozen columns are authoritative and used as-is. For the
    A/B datasets the columns are derived at the dataset base rate
    (``y_true.mean()``), NEVER 0.5, matching ``DisagreementAnalyzer``.
    """
    out = df.copy()

    if dataset == "test" and "category" in out.columns:
        # Frozen, authoritative columns already present. Ensure correctness
        # columns are present too (they are, per the test CSV schema).
        return out

    base_rate = float(out["y_true"].mean())
    y_true = out["y_true"].to_numpy()
    correctness: Dict[str, np.ndarray] = {}
    for key, prob_col in _MODELS:
        pred = (out[prob_col].to_numpy() >= base_rate).astype(int)
        correct = (pred == y_true).astype(int)
        out[f"{key}_pred"] = pred
        out[f"{key}_correct"] = correct
        correctness[key] = correct

    out["category"] = [
        _eight_way_category(
            bool(correctness["lr"][i]),
            bool(correctness["xgb"][i]),
            bool(correctness["lstm"][i]),
        )
        for i in range(len(out))
    ]
    return out


# ── Metric computation (all reuse src.evaluation.metrics / sklearn) ─────────


def _nan_to_none(value: float) -> Optional[float]:
    """Convert a NaN float to ``None`` for JSON-safe serialization."""
    return None if value is None or np.isnan(value) else float(value)


def compute_summary(df: pd.DataFrame) -> SummaryDict:
    """Per-dataset summary: per-model AUC/Brier/LogLoss + n/upset_rate/baseline.

    Brier uses ``sklearn.metrics.brier_score_loss`` (matching legacy serve.py);
    AUC/LogLoss use the ``safe_*`` helpers; baseline Brier reuses
    ``calculate_baseline_brier``.
    """
    y_true = df["y_true"].to_numpy()
    models: Dict[str, ModelMetricsDict] = {}
    for key, prob_col in _MODELS:
        probs = df[prob_col].to_numpy()
        models[key] = {
            "auc": safe_roc_auc_score(y_true, probs),
            "brier": float(brier_score_loss(y_true, probs)),
            "log_loss": safe_log_loss(y_true, probs),
        }
    upset_rate = float(y_true.mean())
    return {
        "n_games": int(len(y_true)),
        "upset_rate": upset_rate,
        "baseline_brier": calculate_baseline_brier(upset_rate),
        "models": models,
    }


def compute_disagreement(dataset: str, df: pd.DataFrame) -> List[DisagreementDict]:
    """Per-category breakdown: count, share %, and actual upset rate.

    Categories are derived per ``derive_categories`` (frozen for test, base-rate
    threshold for the A/B datasets). Returned in the canonical taxonomy order,
    skipping categories with zero games (matching the legacy presentation).
    """
    cat_df = derive_categories(dataset, df)
    total = int(len(cat_df))
    out: List[DisagreementDict] = []
    for category in CATEGORY_ORDER:
        rows = cat_df[cat_df["category"] == category]
        n = int(len(rows))
        if n == 0:
            continue
        out.append(
            {
                "category": category,
                "n": n,
                "pct": round(100.0 * n / total, 1) if total else 0.0,
                "upset_rate": float(rows["y_true"].mean()),
            }
        )
    return out


def compute_seasons(df: pd.DataFrame) -> List[SeasonDict]:
    """Per-season breakdown: n, upset rate, and per-model AUC.

    Mirrors the legacy ``_compute_seasons``. AUC may be NaN for a season with a
    single class; that is surfaced as ``None`` to the API layer.
    """
    out: List[SeasonDict] = []
    for season in sorted(df["season"].unique()):
        rows = df[df["season"] == season]
        y_true = rows["y_true"].to_numpy()
        out.append(
            {
                "season": int(season),
                "n": int(len(rows)),
                "upset_rate": float(y_true.mean()),
                "lr_auc": _nan_to_none(
                    safe_roc_auc_score(y_true, rows["lr_prob"].to_numpy())
                ),
                "xgb_auc": _nan_to_none(
                    safe_roc_auc_score(y_true, rows["xgb_prob"].to_numpy())
                ),
                "lstm_auc": _nan_to_none(
                    safe_roc_auc_score(y_true, rows["lstm_prob"].to_numpy())
                ),
            }
        )
    return out


def compute_roc_points(df: pd.DataFrame) -> List[RocSeriesDict]:
    """Per-model ROC points (fpr, tpr) plus AUC, via ``sklearn.roc_curve``."""
    y_true = df["y_true"].to_numpy()
    series: List[RocSeriesDict] = []
    for key, prob_col in _MODELS:
        probs = df[prob_col].to_numpy()
        auc = safe_roc_auc_score(y_true, probs)
        if len(np.unique(y_true)) < 2:
            fpr_list: List[float] = []
            tpr_list: List[float] = []
        else:
            fpr, tpr, _ = roc_curve(y_true, probs)
            fpr_list = [float(v) for v in fpr]
            tpr_list = [float(v) for v in tpr]
        series.append(
            {
                "model": key,
                "auc": _nan_to_none(auc),
                "fpr": fpr_list,
                "tpr": tpr_list,
            }
        )
    return series


def compute_calibration_points(df: pd.DataFrame) -> List[CalibrationSeriesDict]:
    """Per-model reliability points via ``calculate_calibration_metrics``."""
    y_true = df["y_true"].to_numpy()
    series: List[CalibrationSeriesDict] = []
    for key, prob_col in _MODELS:
        metrics = calculate_calibration_metrics(y_true, df[prob_col].to_numpy())
        series.append(
            {
                "model": key,
                "calibration_error": float(metrics["calibration_error"]),
                "prob_pred": [float(v) for v in metrics["prob_pred"]],
                "prob_true": [float(v) for v in metrics["prob_true"]],
            }
        )
    return series
