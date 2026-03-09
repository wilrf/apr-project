"""A/B Experiment: With-Spread vs Without-Spread.

Runs identical models with and without spread features to isolate
what the betting line contributes to upset prediction.

Usage:
    python3 -m src.models.run_ab_experiment --quick    # LR + XGB only (~1 min)
    python3 -m src.models.run_ab_experiment            # All 3 models (~15 min)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.evaluation.disagreement import DisagreementAnalyzer, PredictionCategory
from src.features import pipeline
from src.features.pipeline import get_xgb_feature_columns, get_xgb_no_spread_feature_columns
from src.models.cv_splitter import TimeSeriesCVSplitter
from src.models.logistic_model import UpsetLogisticRegression
from src.models.lstm_config import TUNED_LSTM_TRAINING_PARAMS
from src.models.sequence_builder import (
    MATCHUP_FEATURES_NO_SPREAD,
    SEQUENCE_FEATURES_NO_SPREAD,
)
from src.models.unified_trainer import UnifiedCVResults, UnifiedTrainer
from src.models.xgboost_model import UpsetXGBoost

DATA_DIR = Path("data/features")
RESULTS_DIR = Path("results/ab_experiment")


@dataclass
class ExperimentConfig:
    """Configuration for one side of the A/B experiment."""

    name: str
    feature_cols: List[str]  # For LR
    xgb_feature_cols: List[str]  # For XGB (expanded)
    matchup_feature_cols: Optional[List[str]]  # None = default (with spread)
    sequence_feature_cols: Optional[List[str]]  # None = default (with spread)


@dataclass
class QuickResult:
    """Results from the quick LR+XGB sanity check."""

    config: ExperimentConfig
    lr_metrics: Dict[str, float]
    xgb_metrics: Dict[str, float]
    lr_probs: np.ndarray
    xgb_probs: np.ndarray
    y_true: np.ndarray
    lr_coefs: Optional[np.ndarray]
    lr_feature_names: List[str]


def load_data():
    """Load and prepare data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv", low_memory=False)
    xgb_cols = get_xgb_feature_columns()
    feature_cols = pipeline.get_feature_columns()

    # Filter to valid upset candidates and fill NaN features (all columns)
    train_valid = train_df[train_df["upset"].notna()].copy()
    all_cols = list(set(feature_cols + xgb_cols))
    train_valid[all_cols] = train_valid[all_cols].fillna(0)

    return train_valid, feature_cols


def get_experiment_configs():
    """Create A (with spread) and B (without spread) configurations."""
    lr_cols = pipeline.get_feature_columns()
    lr_no_spread = pipeline.get_no_spread_feature_columns()
    xgb_cols = get_xgb_feature_columns()
    xgb_no_spread = get_xgb_no_spread_feature_columns()

    config_a = ExperimentConfig(
        name="With Spread",
        feature_cols=lr_cols,
        xgb_feature_cols=xgb_cols,
        matchup_feature_cols=None,
        sequence_feature_cols=None,
    )
    config_b = ExperimentConfig(
        name="Without Spread",
        feature_cols=lr_no_spread,
        xgb_feature_cols=xgb_no_spread,
        matchup_feature_cols=MATCHUP_FEATURES_NO_SPREAD,
        sequence_feature_cols=SEQUENCE_FEATURES_NO_SPREAD,
    )

    return config_a, config_b


def calc_metrics(y_true, y_pred):
    """Calculate standard metrics."""
    return {
        "auc_roc": float(roc_auc_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_pred)),
    }


# ============================================================
# Phase 1: Quick sanity check (LR + XGB only)
# ============================================================


def run_quick_ab(train_df) -> Dict[str, QuickResult]:
    """Run LR + XGB cross-validation for both experiments."""
    config_a, config_b = get_experiment_configs()
    cv = TimeSeriesCVSplitter(n_folds=6)

    results = {}
    for config in [config_a, config_b]:
        print(f"\n{'='*60}")
        print(f"QUICK EXPERIMENT: {config.name} "
              f"(LR: {len(config.feature_cols)}, XGB: {len(config.xgb_feature_cols)} features)")
        print(f"{'='*60}")

        all_lr_probs, all_xgb_probs, all_y = [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(train_df)):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]
            val_season = fold_val["season"].iloc[0]

            # LR uses base features
            X_train_lr = fold_train[config.feature_cols]
            y_train = fold_train["upset"]
            X_val_lr = fold_val[config.feature_cols]
            y_val = fold_val["upset"]

            # XGB uses expanded features
            X_train_xgb = fold_train[config.xgb_feature_cols]
            X_val_xgb = fold_val[config.xgb_feature_cols]

            lr = UpsetLogisticRegression(
                C=0.1, penalty="l1", solver="saga", random_state=42
            )
            lr.fit(X_train_lr, y_train)
            lr_probs = lr.predict_proba(X_val_lr)

            xgb = UpsetXGBoost(
                max_depth=2, learning_rate=0.03, n_estimators=300, random_state=42
            )
            xgb.fit(X_train_xgb, y_train, verbose=False)
            xgb_probs = xgb.predict_proba(X_val_xgb)

            all_lr_probs.extend(lr_probs)
            all_xgb_probs.extend(xgb_probs)
            all_y.extend(y_val.values)

            lr_auc = roc_auc_score(y_val, lr_probs)
            xgb_auc = roc_auc_score(y_val, xgb_probs)
            print(
                f"  Fold {fold_idx+1} (val={val_season}):"
                f" LR AUC={lr_auc:.3f}, XGB AUC={xgb_auc:.3f}"
            )

        y_arr = np.array(all_y)
        lr_arr = np.array(all_lr_probs)
        xgb_arr = np.array(all_xgb_probs)

        # Train final LR on all data for feature importance (uses LR features)
        lr_final = UpsetLogisticRegression(
            C=0.1, penalty="l1", solver="saga", random_state=42
        )
        lr_final.fit(train_df[config.feature_cols], train_df["upset"])
        lr_coefs = lr_final.model.coef_[0] if hasattr(lr_final, "model") else None

        results[config.name] = QuickResult(
            config=config,
            lr_metrics=calc_metrics(y_arr, lr_arr),
            xgb_metrics=calc_metrics(y_arr, xgb_arr),
            lr_probs=lr_arr,
            xgb_probs=xgb_arr,
            y_true=y_arr,
            lr_coefs=lr_coefs,
            lr_feature_names=config.feature_cols,
        )

    return results


def print_quick_comparison(results: Dict[str, QuickResult]):
    """Print side-by-side quick results."""
    a = results["With Spread"]
    b = results["Without Spread"]

    print(f"\n{'='*60}")
    print("QUICK A/B COMPARISON: LR + XGBoost")
    print(f"{'='*60}")
    print(f"  Experiment A: LR={len(a.config.feature_cols)}, "
          f"XGB={len(a.config.xgb_feature_cols)} (with spread)")
    print(f"  Experiment B: LR={len(b.config.feature_cols)}, "
          f"XGB={len(b.config.xgb_feature_cols)} (without spread)")

    # Metrics table
    print(
        f"\n{'Model':<6} {'Metric':<14} "
        f"{'With Spread':>12} {'No Spread':>12} {'Delta':>8}"
    )
    print("-" * 54)
    for model, ma, mb in [
        ("LR", a.lr_metrics, b.lr_metrics),
        ("XGB", a.xgb_metrics, b.xgb_metrics),
    ]:
        for metric in ["auc_roc", "brier_score", "log_loss"]:
            va, vb = ma[metric], mb[metric]
            delta = vb - va
            sign = "+" if delta > 0 else ""
            print(
                f"{model:<6} {metric:<14} {va:>12.4f} {vb:>12.4f} {sign}{delta:>7.4f}"
            )
        print()

    # Model ranking
    print("Model Ranking (AUC):")
    for label, r in [("With Spread", a), ("Without Spread", b)]:
        lr_auc = r.lr_metrics["auc_roc"]
        xgb_auc = r.xgb_metrics["auc_roc"]
        winner = "LR" if lr_auc > xgb_auc else "XGB"
        print(f"  {label}: LR={lr_auc:.4f}, XGB={xgb_auc:.4f} -> {winner} wins")

    # Correlation
    corr_a = np.corrcoef(a.lr_probs, a.xgb_probs)[0, 1]
    corr_b = np.corrcoef(b.lr_probs, b.xgb_probs)[0, 1]
    print("\nLR-XGB Probability Correlation:")
    print(f"  With Spread:    {corr_a:.3f}")
    print(f"  Without Spread: {corr_b:.3f}")
    print(f"  Change:         {corr_b - corr_a:+.3f}")

    # Feature importance shift (LR coefficients)
    if a.lr_coefs is not None and b.lr_coefs is not None:
        print("\n--- LR Feature Importance (Top 10 by |coefficient|) ---")

        for label, coefs, names in [
            ("With Spread", a.lr_coefs, a.lr_feature_names),
            ("Without Spread", b.lr_coefs, b.lr_feature_names),
        ]:
            print(f"\n{label}:")
            ranked = sorted(zip(names, coefs), key=lambda x: abs(x[1]), reverse=True)
            for i, (name, coef) in enumerate(ranked[:10], 1):
                print(f"  {i:>2}. {name:<35s} {coef:>+.4f}")


# ============================================================
# Phase 2: Full experiment (all 3 models)
# ============================================================


def run_full_ab(train_df) -> Dict[str, UnifiedCVResults]:
    """Run full unified CV for both experiments."""
    config_a, config_b = get_experiment_configs()
    results = {}

    for config in [config_a, config_b]:
        print(f"\n{'='*60}")
        print(f"FULL EXPERIMENT: {config.name} "
              f"(LR: {len(config.feature_cols)}, XGB: {len(config.xgb_feature_cols)})")
        print(f"{'='*60}")

        trainer = UnifiedTrainer()
        cv_results = trainer.cross_validate(
            train_df,
            feature_cols=config.feature_cols,
            xgb_feature_cols=config.xgb_feature_cols,
            matchup_feature_cols=config.matchup_feature_cols,
            sequence_feature_cols=config.sequence_feature_cols,
            lstm_epochs=int(TUNED_LSTM_TRAINING_PARAMS["epochs"]),
            lstm_batch_size=int(TUNED_LSTM_TRAINING_PARAMS["batch_size"]),
            verbose=True,
        )
        results[config.name] = cv_results

    return results


def print_full_comparison(results: Dict[str, UnifiedCVResults]):
    """Print full A/B comparison with all 3 models."""
    a = results["With Spread"]
    b = results["Without Spread"]

    print(f"\n{'='*60}")
    print("FULL A/B COMPARISON: LR + XGBoost + LSTM")
    print(f"{'='*60}")

    # Metrics comparison
    print(
        f"\n{'Model':<6} {'Metric':<14} "
        f"{'With Spread':>12} {'No Spread':>12} {'Delta':>8}"
    )
    print("-" * 54)
    for model_key in ["lr", "xgb", "lstm"]:
        for metric_base in ["auc_roc", "brier_score", "log_loss"]:
            mean_key = f"{metric_base}_mean"
            std_key = f"{metric_base}_std"
            va = a.aggregated_metrics[model_key].get(mean_key, 0)
            vb = b.aggregated_metrics[model_key].get(mean_key, 0)
            sa = a.aggregated_metrics[model_key].get(std_key, 0)
            sb = b.aggregated_metrics[model_key].get(std_key, 0)
            delta = vb - va
            sign = "+" if delta > 0 else ""
            print(
                f"{model_key.upper():<6} {metric_base:<14} "
                f"{va:>7.4f}+/-{sa:<5.3f} "
                f"{vb:>7.4f}+/-{sb:<5.3f} "
                f"{sign}{delta:>7.4f}"
            )
        print()

    # Model ranking
    print("Model Ranking by AUC:")
    for label, r in [("With Spread", a), ("Without Spread", b)]:
        aucs = {
            k: r.aggregated_metrics[k]["auc_roc_mean"] for k in ["lr", "xgb", "lstm"]
        }
        ranked = sorted(aucs.items(), key=lambda x: x[1], reverse=True)
        ranking_str = " > ".join(f"{k.upper()}({v:.3f})" for k, v in ranked)
        print(f"  {label}: {ranking_str}")

    # Correlation matrices
    for label, r in [("With Spread", a), ("Without Spread", b)]:
        preds = r.all_predictions
        lr_p = np.array([p.lr_prob for p in preds])
        xgb_p = np.array([p.xgb_prob for p in preds])
        lstm_p = np.array([p.lstm_prob for p in preds])
        data = np.column_stack([lr_p, xgb_p, lstm_p])
        corr = np.corrcoef(data.T)
        print(f"\nCorrelation Matrix ({label}):")
        print("       LR     XGB    LSTM")
        for i, name in enumerate(["LR", "XGB", "LSTM"]):
            print(f"  {name:<4} {corr[i,0]:.3f}  {corr[i,1]:.3f}  {corr[i,2]:.3f}")

    # Disagreement analysis
    for label, r in [("With Spread", a), ("Without Spread", b)]:
        analyzer = DisagreementAnalyzer(r.all_predictions)
        categories = analyzer.categorize_all()
        total = len(r.all_predictions)
        print(f"\nDisagreement Categories ({label}):")
        for cat in PredictionCategory:
            count = len(categories[cat])
            pct = count / total * 100 if total > 0 else 0
            print(f"  {cat.value:<15} {count:>4} ({pct:>5.1f}%)")

    # Segmented AUC by spread range (from GamePrediction metadata)
    print(f"\n{'='*60}")
    print("SEGMENTED AUC BY SPREAD RANGE (Without Spread models)")
    print(f"{'='*60}")
    preds_b = b.all_predictions
    spread_bins = [
        (3, 4.5, "Small (3-4.5)"),
        (5, 7, "Medium (5-7)"),
        (7.5, 30, "Large (7.5+)"),
    ]

    header = (
        f"{'Range':<16} {'N':>5} {'Base Rate':>10} "
        f"{'LR AUC':>8} {'XGB AUC':>8} {'LSTM AUC':>9}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for lo, hi, label in spread_bins:
        bucket = [p for p in preds_b if lo <= p.spread_magnitude <= hi]
        if len(bucket) < 10:
            continue
        y = np.array([p.y_true for p in bucket])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        lr_p = np.array([p.lr_prob for p in bucket])
        xgb_p = np.array([p.xgb_prob for p in bucket])
        lstm_p = np.array([p.lstm_prob for p in bucket])
        print(
            f"{label:<16} {len(bucket):>5} {y.mean():>9.1%} "
            f"{roc_auc_score(y, lr_p):>8.3f} {roc_auc_score(y, xgb_p):>8.3f} "
            f"{roc_auc_score(y, lstm_p):>9.3f}"
        )


def save_results(quick_results=None, full_results=None):
    """Save experiment results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if quick_results:
        lines = ["# A/B Experiment: Quick Results (LR + XGB)", ""]
        for label, r in quick_results.items():
            lines.extend(
                [
                    f"## {label} ({len(r.config.feature_cols)} features)",
                    "",
                    f"- LR AUC: {r.lr_metrics['auc_roc']:.4f}",
                    f"- XGB AUC: {r.xgb_metrics['auc_roc']:.4f}",
                    f"- LR-XGB correlation: "
                    f"{np.corrcoef(r.lr_probs, r.xgb_probs)[0,1]:.3f}",
                    "",
                ]
            )
        (RESULTS_DIR / "quick_results.md").write_text("\n".join(lines))
        print(f"\nSaved quick results to {RESULTS_DIR / 'quick_results.md'}")

    if quick_results:
        # Save LR coefficients for report generation
        for label, r in quick_results.items():
            safe_name = label.lower().replace(" ", "_")
            if r.lr_coefs is not None:
                coef_dict = {
                    name: float(coef)
                    for name, coef in zip(r.lr_feature_names, r.lr_coefs)
                }
                path = RESULTS_DIR / f"lr_coefs_{safe_name}.json"
                path.write_text(json.dumps(coef_dict, indent=2))

    if full_results:
        for label, cv_result in full_results.items():
            safe_name = label.lower().replace(" ", "_")
            df = cv_result.to_dataframe()
            path = RESULTS_DIR / f"predictions_{safe_name}.csv"
            df.to_csv(path, index=False)
            print(f"Saved {label} predictions to {path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Run A/B spread experiment")
    parser.add_argument("--quick", action="store_true", help="LR+XGB only (fast)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("A/B EXPERIMENT: WITH SPREAD vs WITHOUT SPREAD")
    print("=" * 60)

    train_df, feature_cols = load_data()
    xgb_cols = get_xgb_feature_columns()
    print(f"Training data: {len(train_df)} games")
    print(f"LR features:  {len(feature_cols)} with spread / "
          f"{len(pipeline.get_no_spread_feature_columns())} without")
    print(f"XGB features: {len(xgb_cols)} with spread / "
          f"{len(get_xgb_no_spread_feature_columns())} without")

    if args.quick:
        results = run_quick_ab(train_df)
        print_quick_comparison(results)
        save_results(quick_results=results)
    else:
        quick_results = run_quick_ab(train_df)
        print_quick_comparison(quick_results)
        full_results = run_full_ab(train_df)
        print_full_comparison(full_results)
        save_results(quick_results=quick_results, full_results=full_results)

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
