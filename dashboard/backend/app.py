"""FastAPI application for the APR Research dashboard.

Six typed ``/api/*`` endpoints serve metrics computed over the frozen
``results/`` artifacts (numbers equal ``results/`` + README; metrics reuse
``src.evaluation.metrics``). The built SPA (``frontend/dist``) is mounted at
``/`` when present; OpenAPI docs live at ``/docs``. Run on one local port
(default 8050, override via ``DASHBOARD_PORT``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from dashboard.backend import data_access as da
from dashboard.backend.schemas import (
    CalibrationSeries,
    CurvesResponse,
    DatasetSummary,
    DisagreementCategory,
    DisagreementResponse,
    FeatureCoef,
    FeaturesResponse,
    ModelMetrics,
    PredictionRow,
    PredictionsResponse,
    RocSeries,
    SeasonRow,
    SeasonsResponse,
    SummaryResponse,
)

app = FastAPI(
    title="APR Research Dashboard",
    description=(
        "Typed API over the frozen results/ artifacts for the NFL upset "
        "prediction study. Every number matches results/ and the README; "
        "metrics reuse src.evaluation.metrics."
    ),
    version="0.1.0",
)

# Path to the built SPA (Vite output). Mounted at / only when it exists.
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def _require_dataset(dataset: str) -> None:
    """Raise 404 for an unknown dataset name."""
    if dataset not in da.DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown dataset {dataset!r}; expected one of {da.DATASETS}.",
        )


def _coefs_to_features(coefs: Dict[str, float]) -> List[FeatureCoef]:
    """Rank coefficients by absolute magnitude and annotate sign direction."""
    ranked = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    out: List[FeatureCoef] = []
    for rank, (feature, coef) in enumerate(ranked, start=1):
        if coef > 0:
            direction = "raises"
        elif coef < 0:
            direction = "lowers"
        else:
            direction = "neutral"
        out.append(
            FeatureCoef(
                rank=rank,
                feature=feature,
                coefficient=float(coef),
                direction=direction,
            )
        )
    return out


# ── API endpoints ───────────────────────────────────────────────────────────


@app.get("/api/summary", response_model=SummaryResponse)
def get_summary() -> SummaryResponse:
    """Per-dataset summary metrics for all three datasets."""
    datasets: Dict[str, DatasetSummary] = {}
    for dataset in da.DATASETS:
        summary = da.compute_summary(da.load_predictions(dataset))
        models = {
            key: ModelMetrics(**metrics) for key, metrics in summary["models"].items()
        }
        datasets[dataset] = DatasetSummary(
            n_games=summary["n_games"],
            upset_rate=summary["upset_rate"],
            baseline_brier=summary["baseline_brier"],
            models=models,
        )
    return SummaryResponse(datasets=datasets)


@app.get("/api/predictions/{dataset}", response_model=PredictionsResponse)
def get_predictions(dataset: str) -> PredictionsResponse:
    """Per-game prediction rows for one dataset (404 on unknown dataset)."""
    _require_dataset(dataset)
    df = da.derive_categories(dataset, da.load_predictions(dataset))
    rows = [
        PredictionRow(
            game_id=str(r["game_id"]),
            season=int(r["season"]),
            week=int(r["week"]),
            underdog=str(r["underdog"]),
            favorite=str(r["favorite"]),
            spread_magnitude=float(r["spread_magnitude"]),
            y_true=int(r["y_true"]),
            lr_prob=float(r["lr_prob"]),
            xgb_prob=float(r["xgb_prob"]),
            lstm_prob=float(r["lstm_prob"]),
            lr_correct=int(r["lr_correct"]),
            xgb_correct=int(r["xgb_correct"]),
            lstm_correct=int(r["lstm_correct"]),
            category=str(r["category"]),
        )
        for r in df.to_dict(orient="records")
    ]
    return PredictionsResponse(dataset=dataset, n_games=len(rows), rows=rows)


@app.get("/api/disagreement/{dataset}", response_model=DisagreementResponse)
def get_disagreement(dataset: str) -> DisagreementResponse:
    """Disagreement category breakdown for one dataset (404 on unknown)."""
    _require_dataset(dataset)
    df = da.load_predictions(dataset)
    cats = da.compute_disagreement(dataset, df)
    categories = [DisagreementCategory(**c) for c in cats]
    return DisagreementResponse(
        dataset=dataset, n_games=int(len(df)), categories=categories
    )


@app.get("/api/features", response_model=FeaturesResponse)
def get_features() -> FeaturesResponse:
    """LR coefficients (ranked by |coef|) for both spread variants."""
    return FeaturesResponse(
        with_spread=_coefs_to_features(da.load_lr_coefs("with_spread")),
        without_spread=_coefs_to_features(da.load_lr_coefs("without_spread")),
    )


@app.get("/api/seasons/{dataset}", response_model=SeasonsResponse)
def get_seasons(dataset: str) -> SeasonsResponse:
    """Per-season games, upset rate, and per-model AUC (404 on unknown)."""
    _require_dataset(dataset)
    seasons = da.compute_seasons(da.load_predictions(dataset))
    rows = [SeasonRow(**s) for s in seasons]
    return SeasonsResponse(dataset=dataset, seasons=rows)


@app.get("/api/curves/{dataset}", response_model=CurvesResponse)
def get_curves(dataset: str) -> CurvesResponse:
    """ROC + calibration points (per model) for one dataset (404 on unknown)."""
    _require_dataset(dataset)
    df = da.load_predictions(dataset)
    roc_series = [RocSeries(**s) for s in da.compute_roc_points(df)]
    calib_series = [CalibrationSeries(**s) for s in da.compute_calibration_points(df)]
    return CurvesResponse(dataset=dataset, roc=roc_series, calibration=calib_series)


# ── SPA mount + entrypoint ──────────────────────────────────────────────────

if FRONTEND_DIST.exists():
    # Serve the built single-page app from the same origin/port as the API.
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="spa")
else:

    @app.get("/")
    def spa_stub() -> Dict[str, str]:
        """Stub served when the SPA has not been built yet."""
        return {
            "message": (
                "Dashboard SPA not built. Run `make dashboard` to build the "
                "frontend, or use the API directly (see /docs)."
            ),
            "docs": "/docs",
        }


def main() -> None:
    """Run the dashboard server on 127.0.0.1:8050 (override via DASHBOARD_PORT)."""
    import uvicorn

    port = int(os.environ.get("DASHBOARD_PORT", "8050"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
