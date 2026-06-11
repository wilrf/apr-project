"""Numbers-frozen parity guard for the dashboard backend.

Every value the dashboard serves MUST equal the canonical ``results/`` artifacts
and the README "Results At A Glance" / "Spread Ablation" tables. These tests pin
each endpoint to those frozen numbers with ``pytest.approx(..., abs=1e-3)`` (the
README rounds to four decimals; tighter where the CSV supports it).

Defense in depth: rather than trusting only the plan's hand-copied numbers, the
disagreement test independently tallies the frozen ``category`` column from
``results/test/predictions.csv``, the seasons test recomputes per-season AUC from
the CSV via ``src.evaluation.metrics.safe_roc_auc_score``, and the summary test
cross-checks against ``results/audit_computed.json`` where the keys overlap.

If a backend value disagrees with a canonical value, the canonical value wins:
these tests must FAIL (do not weaken them) until the backend is corrected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from src.evaluation.metrics import safe_roc_auc_score

client = TestClient(app)

# Repo root: tests/dashboard/test_parity.py -> parents[2].
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# README rounding tolerance (four-decimal headline numbers).
ABS = 1e-3

# README "Results At A Glance" canonical values (LR, XGB, LSTM order).
TEST_AUC = {"lr": 0.5622, "xgb": 0.5755, "lstm": 0.5263}
TEST_BRIER = {"lr": 0.2026, "xgb": 0.2013, "lstm": 0.2089}
CV_WITH_AUC = {"lr": 0.6497, "xgb": 0.6377, "lstm": 0.6372}
CV_WITHOUT_AUC = {"lr": 0.5707, "xgb": 0.5662, "lstm": 0.5682}
# README "Spread Ablation" Delta column (without - with).
DELTA = {"lr": -0.0790, "xgb": -0.0715, "lstm": -0.0690}

# Frozen disagreement category tallies for the test set (from the plan; also
# verified independently against the CSV in test_disagreement_test_categories).
TEST_CATEGORY_COUNTS = {
    "all_correct": 147,
    "all_wrong": 209,
    "only_lr": 9,
    "only_xgb": 24,
    "only_lstm": 81,
    "lr_xgb": 61,
    "lr_lstm": 12,
    "xgb_lstm": 15,
}


def _summary() -> Dict[str, dict]:
    resp = client.get("/api/summary")
    assert resp.status_code == 200
    return resp.json()["datasets"]


# ── /api/summary parity ──────────────────────────────────────────────────────


def test_summary_test_auc_brier() -> None:
    test = _summary()["test"]
    for model in ("lr", "xgb", "lstm"):
        assert test["models"][model]["auc"] == pytest.approx(
            TEST_AUC[model], abs=ABS
        ), model
        assert test["models"][model]["brier"] == pytest.approx(
            TEST_BRIER[model], abs=ABS
        ), model


def test_summary_test_dataset_stats() -> None:
    test = _summary()["test"]
    assert test["n_games"] == 558
    assert test["upset_rate"] == pytest.approx(0.2849, abs=ABS)
    assert test["baseline_brier"] == pytest.approx(0.2038, abs=ABS)


def test_summary_cv_with_spread_auc() -> None:
    cv = _summary()["cv_with_spread"]
    assert cv["n_games"] == 1162
    for model in ("lr", "xgb", "lstm"):
        assert cv["models"][model]["auc"] == pytest.approx(
            CV_WITH_AUC[model], abs=ABS
        ), model


def test_summary_cv_without_spread_auc() -> None:
    cv = _summary()["cv_without_spread"]
    assert cv["n_games"] == 1162
    for model in ("lr", "xgb", "lstm"):
        assert cv["models"][model]["auc"] == pytest.approx(
            CV_WITHOUT_AUC[model], abs=ABS
        ), model


def test_summary_spread_ablation_delta() -> None:
    datasets = _summary()
    with_auc = datasets["cv_with_spread"]["models"]
    without_auc = datasets["cv_without_spread"]["models"]
    for model in ("lr", "xgb", "lstm"):
        delta = without_auc[model]["auc"] - with_auc[model]["auc"]
        assert delta == pytest.approx(DELTA[model], abs=ABS), model


def test_summary_cross_checks_audit_computed_json() -> None:
    """Defense in depth: dashboard summary must agree with audit_computed.json.

    The audit JSON is an independent computation of the same metrics. Cross-check
    the overlapping keys (test + cv AUC/Brier, n, base rate). Any mismatch is a
    real discrepancy to flag, not to paper over.
    """
    audit = json.loads((RESULTS / "audit_computed.json").read_text())
    datasets = _summary()

    audit_test = audit["test"]
    test = datasets["test"]
    assert test["n_games"] == audit_test["n"]
    assert test["upset_rate"] == pytest.approx(audit_test["base_rate"], abs=ABS)
    for model_key, audit_key in (("lr", "LR"), ("xgb", "XGB"), ("lstm", "LSTM")):
        m = test["models"][model_key]
        am = audit_test["metrics"][audit_key]
        assert m["auc"] == pytest.approx(am["auc"], abs=ABS), model_key
        assert m["brier"] == pytest.approx(am["brier"], abs=ABS), model_key
        assert m["log_loss"] == pytest.approx(am["log_loss"], abs=ABS), model_key

    audit_cv = audit["cv"]
    cv = datasets["cv_with_spread"]
    assert cv["n_games"] == audit_cv["n"]
    for model_key, audit_key in (("lr", "LR"), ("xgb", "XGB"), ("lstm", "LSTM")):
        assert cv["models"][model_key]["auc"] == pytest.approx(
            audit_cv["metrics"][audit_key]["auc"], abs=ABS
        ), model_key


# ── /api/disagreement parity (independent CSV tally) ─────────────────────────


def test_disagreement_test_categories() -> None:
    """Endpoint counts must equal an INDEPENDENT tally of the frozen category column.

    This recomputes the tally directly from ``results/test/predictions.csv`` rather
    than trusting the plan's hand-copied numbers (defense in depth), then asserts
    both the independent tally and the endpoint agree with the frozen constants.
    """
    # Independent tally straight from the CSV's frozen ``category`` column.
    df = pd.read_csv(RESULTS / "test" / "predictions.csv")
    csv_counts = df["category"].value_counts().to_dict()
    assert csv_counts == TEST_CATEGORY_COUNTS
    assert sum(csv_counts.values()) == 558

    # Endpoint must reproduce that tally exactly.
    resp = client.get("/api/disagreement/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_games"] == 558
    endpoint_counts = {c["category"]: c["n"] for c in body["categories"]}
    assert endpoint_counts == TEST_CATEGORY_COUNTS

    # Shares sum to 100% (rounding tolerance) and counts sum to n_games.
    assert sum(c["n"] for c in body["categories"]) == 558
    assert sum(c["pct"] for c in body["categories"]) == pytest.approx(100.0, abs=0.2)

    # Each endpoint upset_rate matches a direct per-category recompute.
    for cat in body["categories"]:
        rows = df[df["category"] == cat["category"]]
        assert cat["upset_rate"] == pytest.approx(
            float(rows["y_true"].mean()), abs=1e-6
        ), cat["category"]


# ── /api/features parity ─────────────────────────────────────────────────────


def test_features_counts_and_ranking() -> None:
    resp = client.get("/api/features")
    assert resp.status_code == 200
    body = resp.json()

    assert len(body["with_spread"]) == 46
    assert len(body["without_spread"]) == 42

    # Ranking is stable: ranks are 1..N and ordered by descending |coef|.
    for variant in ("with_spread", "without_spread"):
        coefs = body[variant]
        assert [c["rank"] for c in coefs] == list(range(1, len(coefs) + 1))
        abs_coefs = [abs(c["coefficient"]) for c in coefs]
        assert abs_coefs == sorted(abs_coefs, reverse=True)
        # Direction agrees with the sign of the coefficient.
        for c in coefs:
            if c["coefficient"] > 0:
                assert c["direction"] == "raises"
            elif c["coefficient"] < 0:
                assert c["direction"] == "lowers"
            else:
                assert c["direction"] == "neutral"


def test_features_spot_check_matches_json() -> None:
    """A spot-checked coefficient must equal the raw JSON artifact value."""
    raw = json.loads(
        (RESULTS / "ab_experiment" / "lr_coefs_with_spread.json").read_text()
    )
    assert raw["favorite_turnover_margin_roll3"] == pytest.approx(-0.0694, abs=ABS)

    resp = client.get("/api/features")
    coefs = {c["feature"]: c["coefficient"] for c in resp.json()["with_spread"]}
    assert coefs["favorite_turnover_margin_roll3"] == pytest.approx(
        raw["favorite_turnover_margin_roll3"], abs=1e-9
    )


# ── /api/seasons parity (independent recompute from CSV) ──────────────────────


def test_seasons_test_recompute_from_csv() -> None:
    """Per-season AUC from the endpoint must equal a direct CSV recompute.

    Guards against a divergent grouping/ordering: the test independently groups
    ``results/test/predictions.csv`` by season and recomputes each model's AUC via
    ``safe_roc_auc_score``, then asserts the endpoint reproduces it.
    """
    df = pd.read_csv(RESULTS / "test" / "predictions.csv")

    expected = {}
    for season, rows in df.groupby("season"):
        y_true = rows["y_true"].to_numpy()
        expected[int(season)] = {
            "n": int(len(rows)),
            "upset_rate": float(y_true.mean()),
            "lr_auc": safe_roc_auc_score(y_true, rows["lr_prob"].to_numpy()),
            "xgb_auc": safe_roc_auc_score(y_true, rows["xgb_prob"].to_numpy()),
            "lstm_auc": safe_roc_auc_score(y_true, rows["lstm_prob"].to_numpy()),
        }

    resp = client.get("/api/seasons/test")
    assert resp.status_code == 200
    seasons = resp.json()["seasons"]

    # Same season set, in the same sorted order.
    endpoint_seasons = [s["season"] for s in seasons]
    assert endpoint_seasons == sorted(expected.keys())

    for s in seasons:
        exp = expected[s["season"]]
        assert s["n"] == exp["n"]
        assert s["upset_rate"] == pytest.approx(exp["upset_rate"], abs=1e-9)
        for key in ("lr_auc", "xgb_auc", "lstm_auc"):
            # safe_roc_auc_score returns NaN -> endpoint serializes None.
            if pd.isna(exp[key]):
                assert s[key] is None, (s["season"], key)
            else:
                assert s[key] == pytest.approx(exp[key], abs=1e-9), (s["season"], key)
