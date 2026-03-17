"""Regression tests for A/B experiment wiring and metrics."""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pandas as pd

import src.models.run_ab_experiment as run_ab_experiment


def test_calc_metrics_handles_single_class_targets():
    metrics = run_ab_experiment.calc_metrics(
        np.array([0, 0, 0], dtype=float),
        np.array([0.0, 1e-12, 0.1], dtype=float),
    )

    assert np.isnan(metrics["auc_roc"])
    assert np.isfinite(metrics["log_loss"])


def test_main_full_mode_skips_quick_ab(monkeypatch):
    train_df = pd.DataFrame({"season": [2023], "upset": [0.0], "feature": [1.0]})
    calls: list[str] = []

    monkeypatch.setattr(
        run_ab_experiment.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(quick=False),
    )
    monkeypatch.setattr(
        run_ab_experiment,
        "load_data",
        lambda: (train_df, ["feature"], train_df.copy()),
    )
    monkeypatch.setattr(
        run_ab_experiment,
        "run_quick_ab",
        lambda *_args, **_kwargs: calls.append("quick"),
    )
    monkeypatch.setattr(
        run_ab_experiment,
        "run_full_ab",
        lambda *_args, **_kwargs: calls.append("full") or {},
    )
    monkeypatch.setattr(
        run_ab_experiment, "print_quick_comparison", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        run_ab_experiment, "print_full_comparison", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        run_ab_experiment, "save_results", lambda *_args, **_kwargs: None
    )

    run_ab_experiment.main()

    assert calls == ["full"]


def test_pick_auc_winner_ignores_nan_scores():
    assert run_ab_experiment._pick_auc_winner(0.61, np.nan) == "LR"
    assert run_ab_experiment._pick_auc_winner(np.nan, 0.58) == "XGB"
    assert run_ab_experiment._pick_auc_winner(np.nan, np.nan) == "N/A"


def test_rank_models_by_auc_places_nan_scores_last():
    ranked = run_ab_experiment._rank_models_by_auc(
        {"lr": 0.61, "xgb": np.nan, "lstm": 0.55}
    )

    assert [name for name, _ in ranked] == ["lr", "lstm", "xgb"]
