"""Live dashboard server. Reads results fresh on each request.

Usage:
    python3 -m frontend.serve              # http://localhost:8050
    python3 -m frontend.serve --port 9000  # custom port
"""

from __future__ import annotations

import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.evaluation.metrics import safe_log_loss, safe_roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FRONTEND = ROOT / "frontend"

DEFAULT_PORT = 8050


# ── Data loading (runs on each /api/data request) ──────────────────


def _load_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    return pd.read_csv(path).to_dict(orient="records")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _compute_metrics(records: list[dict]) -> dict:
    y = np.array([r["y_true"] for r in records])
    models: dict = {}
    for key, col in [("LR", "lr_prob"), ("XGB", "xgb_prob"), ("LSTM", "lstm_prob")]:
        probs = np.array([r[col] for r in records])
        models[key] = {
            "auc": round(safe_roc_auc_score(y, probs), 4),
            "brier": round(float(brier_score_loss(y, probs)), 4),
            "log_loss": round(safe_log_loss(y, probs), 4),
        }
    upset_rate = float(y.mean())
    return {
        "n_games": len(y),
        "upset_rate": round(upset_rate, 4),
        "baseline_brier": round(upset_rate * (1 - upset_rate), 4),
        "models": models,
    }


def _compute_categories(records: list[dict]) -> list[dict]:
    if not records or "category" not in records[0]:
        return []
    cats: dict[str, list] = {}
    for r in records:
        cats.setdefault(r.get("category", "unknown"), []).append(r["y_true"])
    total = len(records)
    return sorted(
        [
            {
                "category": cat,
                "n": len(ys),
                "pct": round(100 * len(ys) / total, 1),
                "upset_rate": round(float(np.mean(ys)), 3),
            }
            for cat, ys in cats.items()
        ],
        key=lambda x: x["category"],
    )


def _compute_seasons(records: list[dict]) -> list[dict]:
    seasons: dict[int, list] = {}
    for r in records:
        seasons.setdefault(int(r["season"]), []).append(r)
    out = []
    for s in sorted(seasons):
        rs = seasons[s]
        y = np.array([r["y_true"] for r in rs])
        row: dict = {"season": s, "n": len(rs), "upset_rate": round(float(y.mean()), 3)}
        for key, col in [("lr", "lr_prob"), ("xgb", "xgb_prob"), ("lstm", "lstm_prob")]:
            probs = np.array([r[col] for r in rs])
            row[f"{key}_auc"] = round(safe_roc_auc_score(y, probs), 3)
        out.append(row)
    return out


def build_payload() -> dict:
    """Read all results from disk and compute derived metrics."""
    predictions: dict = {}
    for key, path in [
        ("test", RESULTS / "test" / "predictions.csv"),
        ("cv_with_spread", RESULTS / "ab_experiment" / "predictions_with_spread.csv"),
        ("cv_without_spread", RESULTS / "ab_experiment" / "predictions_without_spread.csv"),
    ]:
        recs = _load_csv(path)
        if recs is not None:
            predictions[key] = recs

    coefs: dict = {}
    for key, fname in [
        ("with_spread", "lr_coefs_with_spread.json"),
        ("without_spread", "lr_coefs_without_spread.json"),
    ]:
        data = _load_json(RESULTS / "ab_experiment" / fname)
        if data is not None:
            coefs[key] = data

    summaries: dict = {}
    categories: dict = {}
    seasons: dict = {}
    for dataset_key, recs in predictions.items():
        summaries[dataset_key] = _compute_metrics(recs)
        categories[dataset_key] = _compute_categories(recs)
        seasons[dataset_key] = _compute_seasons(recs)

    return {
        "predictions": predictions,
        "summaries": summaries,
        "categories": categories,
        "seasons": seasons,
        "coefs": coefs,
    }


# ── HTTP server ─────────────────────────────────────────────────────


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def do_GET(self):
        if self.path == "/api/data":
            try:
                payload = build_payload()
                body = json.dumps(payload, default=str).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                msg = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            return

        # Serve / as index.html
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def log_message(self, format, *args):
        # Quieter logging: skip static asset noise
        if "/api/" in str(args[0]) if args else False:
            super().log_message(format, *args)


def main():
    port = DEFAULT_PORT
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1])

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Dashboard: http://localhost:{port}")
    print("Reads fresh data on each page load. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
