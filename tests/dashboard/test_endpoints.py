"""Endpoint shape tests for the dashboard backend.

Each of the six ``/api/*`` endpoints (plus ``/openapi.json``) must return 200
and the expected key/type shape. FastAPI validates the Pydantic response model
on serialize, so these tests assert key presence and value types on the JSON the
client actually receives. An unknown ``{dataset}`` must 404.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app

client = TestClient(app)

DATASETS = ["test", "cv_with_spread", "cv_without_spread"]
MODELS = ["lr", "xgb", "lstm"]


def test_summary_shape() -> None:
    resp = client.get("/api/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"datasets"}
    datasets = body["datasets"]
    assert set(datasets.keys()) == set(DATASETS)
    for ds in DATASETS:
        summary = datasets[ds]
        assert set(summary.keys()) == {
            "n_games",
            "upset_rate",
            "baseline_brier",
            "models",
        }
        assert isinstance(summary["n_games"], int)
        assert isinstance(summary["upset_rate"], float)
        assert isinstance(summary["baseline_brier"], float)
        assert set(summary["models"].keys()) == set(MODELS)
        for model in MODELS:
            metrics = summary["models"][model]
            assert set(metrics.keys()) == {"auc", "brier", "log_loss"}
            # AUC may be null for a degenerate split; on these datasets it is finite.
            assert isinstance(metrics["auc"], float)
            assert isinstance(metrics["brier"], float)
            assert isinstance(metrics["log_loss"], float)


@pytest.mark.parametrize("dataset", DATASETS)
def test_predictions_shape(dataset: str) -> None:
    resp = client.get(f"/api/predictions/{dataset}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset"] == dataset
    assert isinstance(body["n_games"], int)
    assert body["n_games"] == len(body["rows"])
    assert body["n_games"] > 0
    row = body["rows"][0]
    expected_keys = {
        "game_id",
        "season",
        "week",
        "underdog",
        "favorite",
        "spread_magnitude",
        "y_true",
        "lr_prob",
        "xgb_prob",
        "lstm_prob",
        "lr_correct",
        "xgb_correct",
        "lstm_correct",
        "category",
    }
    assert set(row.keys()) == expected_keys
    assert isinstance(row["game_id"], str)
    assert isinstance(row["season"], int)
    assert isinstance(row["week"], int)
    assert isinstance(row["spread_magnitude"], float)
    assert row["y_true"] in (0, 1)
    for model in MODELS:
        assert isinstance(row[f"{model}_prob"], float)
        assert row[f"{model}_correct"] in (0, 1)
    assert isinstance(row["category"], str)


@pytest.mark.parametrize("dataset", DATASETS)
def test_disagreement_shape(dataset: str) -> None:
    resp = client.get(f"/api/disagreement/{dataset}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset"] == dataset
    assert isinstance(body["n_games"], int)
    categories = body["categories"]
    assert len(categories) > 0
    for cat in categories:
        assert set(cat.keys()) == {"category", "n", "pct", "upset_rate"}
        assert isinstance(cat["category"], str)
        assert isinstance(cat["n"], int)
        assert isinstance(cat["pct"], float)
        assert isinstance(cat["upset_rate"], float)


def test_features_shape() -> None:
    resp = client.get("/api/features")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"with_spread", "without_spread"}
    for variant in ("with_spread", "without_spread"):
        coefs = body[variant]
        assert len(coefs) > 0
        first = coefs[0]
        assert set(first.keys()) == {"rank", "feature", "coefficient", "direction"}
        assert isinstance(first["rank"], int)
        assert isinstance(first["feature"], str)
        assert isinstance(first["coefficient"], float)
        assert first["direction"] in ("raises", "lowers", "neutral")


@pytest.mark.parametrize("dataset", DATASETS)
def test_seasons_shape(dataset: str) -> None:
    resp = client.get(f"/api/seasons/{dataset}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset"] == dataset
    seasons = body["seasons"]
    assert len(seasons) > 0
    for season in seasons:
        assert set(season.keys()) == {
            "season",
            "n",
            "upset_rate",
            "lr_auc",
            "xgb_auc",
            "lstm_auc",
        }
        assert isinstance(season["season"], int)
        assert isinstance(season["n"], int)
        assert isinstance(season["upset_rate"], float)
        for auc_key in ("lr_auc", "xgb_auc", "lstm_auc"):
            # AUC is float or null (single-class season).
            assert season[auc_key] is None or isinstance(season[auc_key], float)


@pytest.mark.parametrize("dataset", DATASETS)
def test_curves_shape(dataset: str) -> None:
    resp = client.get(f"/api/curves/{dataset}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset"] == dataset
    assert {s["model"] for s in body["roc"]} == set(MODELS)
    assert {s["model"] for s in body["calibration"]} == set(MODELS)
    for series in body["roc"]:
        assert set(series.keys()) == {"model", "auc", "fpr", "tpr"}
        assert series["auc"] is None or isinstance(series["auc"], float)
        assert isinstance(series["fpr"], list)
        assert isinstance(series["tpr"], list)
        assert len(series["fpr"]) == len(series["tpr"])
    for series in body["calibration"]:
        assert set(series.keys()) == {
            "model",
            "calibration_error",
            "prob_pred",
            "prob_true",
        }
        assert isinstance(series["calibration_error"], float)
        assert isinstance(series["prob_pred"], list)
        assert isinstance(series["prob_true"], list)
        assert len(series["prob_pred"]) == len(series["prob_true"])


def test_openapi_returns_200() -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    body = resp.json()
    assert "openapi" in body
    assert "paths" in body
    # The six documented API endpoints are present in the schema.
    for path in (
        "/api/summary",
        "/api/predictions/{dataset}",
        "/api/disagreement/{dataset}",
        "/api/features",
        "/api/seasons/{dataset}",
        "/api/curves/{dataset}",
    ):
        assert path in body["paths"], path


@pytest.mark.parametrize(
    "url",
    [
        "/api/predictions/bogus",
        "/api/disagreement/bogus",
        "/api/seasons/bogus",
        "/api/curves/bogus",
    ],
)
def test_unknown_dataset_returns_404(url: str) -> None:
    resp = client.get(url)
    assert resp.status_code == 404
