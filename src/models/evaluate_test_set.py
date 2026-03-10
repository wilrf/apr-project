"""Test set evaluation for multi-model upset prediction framework.

Trains all 3 models (LR, XGBoost, LSTM) on the full 2005-2022 training set,
evaluates on the held-out 2023-2025 test set, and compares to CV results.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss

from src.evaluation.calibration import (
    calibrate_models,
    generate_calibration_predictions,
)
from src.evaluation.disagreement import DisagreementAnalyzer, PredictionCategory
from src.evaluation.metrics import (
    calculate_baseline_brier,
    calculate_calibration_metrics,
    safe_log_loss,
    safe_quantile_buckets,
    safe_roc_auc_score,
)
from src.features import pipeline
from src.features.pipeline import get_xgb_feature_columns
from src.models.lstm_config import TUNED_LSTM_TRAINING_PARAMS
from src.models.sequence_builder import build_siamese_sequences
from src.models.unified_trainer import GamePrediction, UnifiedTrainer, _safe_team_str

# Paths
DATA_DIR = Path("data/features")
RESULTS_DIR = Path("results")

# Fresh CV baselines must be regenerated with the multi-representation architecture.
CV_METRICS = None
CV_CATEGORIES = None


def load_data():
    """Load train, test, and feature column data."""
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python3 -m src.data.generate_features' first."
            )
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    feature_cols = pipeline.get_feature_columns()  # 46 for LR
    xgb_feature_cols = get_xgb_feature_columns()  # 70 for XGB
    return train_df, test_df, feature_cols, xgb_feature_cols


def filter_valid_upsets(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to games with valid upset targets."""
    return df[df["upset"].notna()].copy()


def build_prediction_history(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Build the full history available at test time for LSTM sequences."""
    return pd.concat([train_df, test_df], ignore_index=True)


def verify_data(train_df, test_df):
    """Print data verification and confirm no overlap."""
    train_seasons = sorted(train_df["season"].unique())
    test_seasons = sorted(test_df["season"].unique())
    overlap = set(train_seasons) & set(test_seasons)

    print("=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)
    n_train = len(train_seasons)
    print(f"\nTraining: {train_seasons[0]}-{train_seasons[-1]}" f" ({n_train} seasons)")
    print(
        f"Testing:  {test_seasons[0]}-{test_seasons[-1]} ({len(test_seasons)} seasons)"
    )
    print(
        f"Season overlap: {'NONE (correct)' if not overlap else f'WARNING: {overlap}'}"
    )
    print(f"\nTrain games (all):         {len(train_df):,}")
    print(f"Train upset candidates:    {len(filter_valid_upsets(train_df)):,}")
    train_rate = filter_valid_upsets(train_df)["upset"].mean()
    print(f"Train upset rate:          {train_rate:.1%}")
    print(f"\nTest games (all):          {len(test_df):,}")
    print(f"Test upset candidates:     {len(filter_valid_upsets(test_df)):,}")
    print(
        f"Test upset rate:           {filter_valid_upsets(test_df)['upset'].mean():.1%}"
    )

    # Per-season breakdown for test set
    print("\nTest set by season:")
    for season in test_seasons:
        season_df = filter_valid_upsets(test_df[test_df["season"] == season])
        rate = season_df["upset"].mean()
        print(f"  {season}: {len(season_df):3d} games, upset rate = {rate:.1%}")

    if overlap:
        raise ValueError(f"Season overlap detected: {overlap}")


def generate_predictions(
    models, test_df, feature_cols, xgb_feature_cols, full_history_df=None
):
    """Generate predictions from all 3 models on test data."""
    test_valid = filter_valid_upsets(test_df)
    X_test_lr = test_valid[feature_cols]
    X_test_xgb = test_valid[xgb_feature_cols]
    y_test = test_valid["upset"].values

    # LR predictions (base 46 features)
    lr_probs = models["lr_model"].predict_proba(X_test_lr)

    # XGBoost predictions (expanded 70 features)
    xgb_probs = models["xgb_model"].predict_proba(X_test_xgb)

    # LSTM predictions - need sequences (use full history if available)
    test_seq_data, _ = build_siamese_sequences(
        test_valid,
        normalize=True,
        stats=models["lstm_stats"],
        history_df=full_history_df,
    )

    lstm_model = models["lstm_model"]
    lstm_model.eval()
    with torch.no_grad():
        und_seq = torch.FloatTensor(test_seq_data.underdog_sequences)
        fav_seq = torch.FloatTensor(test_seq_data.favorite_sequences)
        matchup = torch.FloatTensor(test_seq_data.matchup_features)
        und_mask = torch.FloatTensor(test_seq_data.underdog_masks)
        fav_mask = torch.FloatTensor(test_seq_data.favorite_masks)

        lstm_probs = lstm_model(und_seq, fav_seq, matchup, und_mask, fav_mask)
        lstm_probs = lstm_probs.squeeze(-1).cpu().numpy()

    return test_valid, y_test, lr_probs, xgb_probs, lstm_probs


def calculate_metrics(y_true, y_pred):
    """Calculate standard evaluation metrics."""
    return {
        "auc_roc": safe_roc_auc_score(y_true, y_pred),
        "brier_score": float(brier_score_loss(y_true, y_pred)),
        "log_loss": safe_log_loss(y_true, y_pred),
    }


def build_game_predictions(test_df, y_true, lr_probs, xgb_probs, lstm_probs):
    """Build GamePrediction objects for disagreement analysis."""
    predictions = []
    threshold = float(np.mean(y_true)) if len(y_true) > 0 else 0.5

    for i, (idx, row) in enumerate(test_df.iterrows()):
        game_id = row.get(
            "game_id",
            f"{row['season']}_{row['week']}_{row['home_team']}_{row['away_team']}",
        )
        predictions.append(
            GamePrediction(
                game_id=game_id,
                season=int(row["season"]),
                week=int(row["week"]),
                underdog=_safe_team_str(row.get("underdog", "")),
                favorite=_safe_team_str(row.get("favorite", "")),
                spread_magnitude=float(row.get("spread_magnitude", 0)),
                y_true=int(y_true[i]),
                lr_prob=float(lr_probs[i]),
                xgb_prob=float(xgb_probs[i]),
                lstm_prob=float(lstm_probs[i]),
                classification_threshold=threshold,
            )
        )
    return predictions


def print_metrics_comparison(test_metrics, cv_metrics):
    """Print side-by-side CV vs test metrics."""
    if cv_metrics is None:
        print(
            "\nCV baselines are not available yet. Regenerate"
            " cross-validation baselines for the redesigned context"
            " setup before comparing to test performance."
        )
        return

    print("\n" + "=" * 60)
    print("TEST vs CV PERFORMANCE")
    print("=" * 60)

    header = f"{'Model':<6} {'Metric':<14} {'CV':>8} {'Test':>8} {'Delta':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for model_key, model_name in [("lr", "LR"), ("xgb", "XGB"), ("lstm", "LSTM")]:
        for metric in ["auc_roc", "brier_score", "log_loss"]:
            cv_val = cv_metrics[model_key][metric]
            test_val = test_metrics[model_key][metric]
            delta = test_val - cv_val
            # For AUC, higher is better; for Brier/LL, lower is better
            sign = "+" if delta > 0 else ""
            print(
                f"{model_name:<6} {metric:<14} "
                f"{cv_val:>8.4f} {test_val:>8.4f} "
                f"{sign}{delta:>7.4f}"
            )
        print()


def print_calibration(y_true, lr_probs, xgb_probs, lstm_probs):
    """Print calibration metrics and baseline comparison."""
    upset_rate = y_true.mean()
    baseline_brier = calculate_baseline_brier(upset_rate)

    print("=" * 60)
    print("CALIBRATION & BASELINES")
    print("=" * 60)
    print(f"\nTest upset rate:    {upset_rate:.1%}")
    print(
        f"Baseline Brier:    {baseline_brier:.4f} (constant prediction at upset rate)"
    )

    for name, probs in [("LR", lr_probs), ("XGB", xgb_probs), ("LSTM", lstm_probs)]:
        brier = brier_score_loss(y_true, probs)
        cal = calculate_calibration_metrics(y_true, probs)
        improvement = (baseline_brier - brier) / baseline_brier * 100
        print(f"\n{name}:")
        print(f"  Brier Score:       {brier:.4f} ({improvement:+.1f}% vs baseline)")
        print(f"  Calibration (ECE): {cal['calibration_error']:.4f}")
        print(f"  Mean prediction:   {probs.mean():.4f}")


def print_disagreement_comparison(analyzer, cv_categories):
    """Print test disagreement analysis compared to CV."""
    if cv_categories is None:
        print(
            "\nDisagreement CV baselines are not available yet."
            " Regenerate them before comparing test"
            " disagreement rates."
        )
        return

    categories = analyzer.categorize_all()

    print("\n" + "=" * 60)
    print("DISAGREEMENT ANALYSIS: TEST vs CV")
    print("=" * 60)

    total = len(analyzer.predictions)

    header = (
        f"{'Category':<15} {'Test Count':>10} "
        f"{'Test %':>8} {'CV %':>6} {'Upset Rate':>10}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for cat in PredictionCategory:
        count = len(categories[cat])
        pct = count / total * 100 if total > 0 else 0
        cv_pct = cv_categories.get(cat.value, (0, 0))[1]
        upset_rate = (
            np.mean([p.y_true for p in categories[cat]]) if categories[cat] else 0
        )
        print(
            f"{cat.value:<15} {count:>10} "
            f"{pct:>7.1f}% {cv_pct:>5.1f}% {upset_rate:>9.1%}"
        )

    # Agreement matrix
    agreement = analyzer.get_agreement_matrix()
    print("\n--- Pairwise Agreement ---")
    for _, row in agreement.iterrows():
        print(f"  {row['model_pair']:<12}: {row['agreement_rate']:.1%}")

    # Correlation matrix
    corr = analyzer.get_correlation_matrix()
    print("\n--- Probability Correlations ---")
    print(corr.to_string(float_format=lambda x: f"{x:.3f}"))


def print_top_k_analysis(predictions):
    """Analyze upset hit rate among each model's top-K most suspicious games."""
    print("\n" + "=" * 60)
    print("TOP-K ANALYSIS")
    print("=" * 60)
    print("(Games ranked by predicted upset probability, highest first)")

    pred_df = pd.DataFrame(
        [
            {
                "y_true": p.y_true,
                "lr_prob": p.lr_prob,
                "xgb_prob": p.xgb_prob,
                "lstm_prob": p.lstm_prob,
                "ensemble_prob": (p.lr_prob + p.xgb_prob + p.lstm_prob) / 3,
                "game_id": p.game_id,
                "spread_magnitude": p.spread_magnitude,
            }
            for p in predictions
        ]
    )

    base_rate = pred_df["y_true"].mean()
    n_total = len(pred_df)
    total_upsets = pred_df["y_true"].sum()
    k_values = [10, 20, 30, 50, 75, 100]

    print(
        f"\nBase rate: {base_rate:.1%} ({int(total_upsets)} upsets in {n_total} games)"
    )

    # Header
    header = f"{'K':>5}  "
    for name in ["LR", "XGB", "LSTM", "Ensemble"]:
        header += f"{'Upsets':>7} {'Rate':>6} {'Lift':>5}  "
    print(f"\n{header}")
    print("-" * len(header))

    for k in k_values:
        if k > n_total:
            continue
        row = f"{k:>5}  "
        for col, name in [
            ("lr_prob", "LR"),
            ("xgb_prob", "XGB"),
            ("lstm_prob", "LSTM"),
            ("ensemble_prob", "Ensemble"),
        ]:
            top_k = pred_df.nlargest(k, col)
            hits = int(top_k["y_true"].sum())
            hit_rate = top_k["y_true"].mean()
            lift = hit_rate / base_rate if base_rate > 0 else 0
            row += f"{hits:>4}/{k:<3} {hit_rate:>5.1%} {lift:>4.1f}x  "
        print(row)

    # Show the top 10 games per model with details
    print("\n--- Top 10 Games by Model ---")
    for col, name in [
        ("lr_prob", "LR"),
        ("xgb_prob", "XGB"),
        ("lstm_prob", "LSTM"),
        ("ensemble_prob", "Ensemble"),
    ]:
        top10 = pred_df.nlargest(10, col)
        hits = int(top10["y_true"].sum())
        print(f"\n{name} top 10 ({hits}/10 actual upsets):")
        for _, g in top10.iterrows():
            upset_marker = "UPSET" if g["y_true"] == 1 else "     "
            print(
                f"  {g['game_id']:<30s} spread={g['spread_magnitude']:>4.1f}  "
                f"p={g[col]:.3f}  {upset_marker}"
            )


def print_probability_buckets(predictions):
    """Analyze actual upset rate within predicted probability buckets."""
    print("\n" + "=" * 60)
    print("PROBABILITY BUCKET ANALYSIS")
    print("=" * 60)
    print("(Do games rated higher actually upset more often?)")

    pred_df = pd.DataFrame(
        [
            {
                "y_true": p.y_true,
                "lr_prob": p.lr_prob,
                "xgb_prob": p.xgb_prob,
                "lstm_prob": p.lstm_prob,
                "ensemble_prob": (p.lr_prob + p.xgb_prob + p.lstm_prob) / 3,
            }
            for p in predictions
        ]
    )

    for col, name in [
        ("lr_prob", "LR"),
        ("xgb_prob", "XGB"),
        ("lstm_prob", "LSTM"),
        ("ensemble_prob", "Ensemble"),
    ]:
        print(f"\n--- {name} ---")

        # Use quintiles (5 equal-sized groups) for clean presentation
        pred_df["bucket"] = safe_quantile_buckets(pred_df[col], q=5)
        n_buckets = pred_df["bucket"].nunique()

        header = (
            f"{'Quintile':<10} {'Prob Range':<18} "
            f"{'Games':>6} {'Upsets':>7} "
            f"{'Upset Rate':>10} {'Avg Prob':>9}"
        )
        print(header)
        print("-" * len(header))

        for bucket in sorted(pred_df["bucket"].dropna().unique()):
            bdf = pred_df[pred_df["bucket"] == bucket]
            prob_min = bdf[col].min()
            prob_max = bdf[col].max()
            n_games = len(bdf)
            n_upsets = int(bdf["y_true"].sum())
            upset_rate = bdf["y_true"].mean()
            avg_prob = bdf[col].mean()

            label = f"Q{bucket+1}" if n_buckets > 1 else "All"
            desc = (
                "Lowest"
                if bucket == 0
                else ("Highest" if bucket == n_buckets - 1 else "")
            )
            print(
                f"{label:<4} {desc:<5} [{prob_min:.3f}-{prob_max:.3f}]  "
                f"{n_games:>6} {n_upsets:>7} {upset_rate:>9.1%} {avg_prob:>9.3f}"
            )

        # Monotonicity check: does upset rate increase with predicted probability?
        bucket_rates = []
        for b in sorted(pred_df["bucket"].dropna().unique()):
            bucket_rates.append(pred_df[pred_df["bucket"] == b]["y_true"].mean())
        pred_df.drop(columns=["bucket"], inplace=True)

        monotonic_pairs = sum(
            1
            for i in range(len(bucket_rates) - 1)
            if bucket_rates[i] <= bucket_rates[i + 1]
        )
        total_pairs = len(bucket_rates) - 1
        spread = bucket_rates[-1] - bucket_rates[0] if bucket_rates else 0
        print(
            f"  Monotonicity: {monotonic_pairs}/{total_pairs} pairs increasing  "
            f"| Top-bottom spread: {spread:+.1%}"
        )


def print_per_season_breakdown(predictions):
    """Print per-season metrics breakdown for test set."""
    print("\n" + "=" * 60)
    print("PER-SEASON BREAKDOWN (TEST SET)")
    print("=" * 60)

    pred_df = pd.DataFrame(
        [
            {
                "season": p.season,
                "y_true": p.y_true,
                "lr_prob": p.lr_prob,
                "xgb_prob": p.xgb_prob,
                "lstm_prob": p.lstm_prob,
            }
            for p in predictions
        ]
    )

    for season in sorted(pred_df["season"].unique()):
        sdf = pred_df[pred_df["season"] == season]
        y = sdf["y_true"].values
        n = len(y)
        upset_rate = y.mean()

        print(f"\n{season} ({n} games, upset rate = {upset_rate:.1%}):")
        for model_name, col in [
            ("LR", "lr_prob"),
            ("XGB", "xgb_prob"),
            ("LSTM", "lstm_prob"),
        ]:
            probs = sdf[col].values
            auc = safe_roc_auc_score(y, probs)
            brier = brier_score_loss(y, probs)
            if np.isnan(auc):
                print(f"  {model_name}: insufficient data for AUC")
            else:
                print(f"  {model_name}: AUC={auc:.3f}, Brier={brier:.3f}")


def save_predictions_csv(predictions, output_path):
    """Save all predictions to CSV."""
    rows = []
    analyzer = DisagreementAnalyzer(predictions)
    categories = analyzer.categorize_all()

    # Build prediction -> category lookup
    pred_to_cat = {}
    for cat, preds in categories.items():
        for p in preds:
            pred_to_cat[p.game_id] = cat.value

    threshold = analyzer.threshold
    for p in predictions:
        lr_pred = int(p.lr_prob >= threshold)
        xgb_pred = int(p.xgb_prob >= threshold)
        lstm_pred = int(p.lstm_prob >= threshold)
        rows.append(
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
                "lr_pred": lr_pred,
                "xgb_pred": xgb_pred,
                "lstm_pred": lstm_pred,
                "lr_correct": int(lr_pred == p.y_true),
                "xgb_correct": int(xgb_pred == p.y_true),
                "lstm_correct": int(lstm_pred == p.y_true),
                "category": pred_to_cat.get(p.game_id, "unknown"),
            }
        )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path} ({len(df)} rows)")


def save_report_md(test_metrics, cv_metrics, analyzer, predictions, output_path):
    """Save markdown report."""
    categories = analyzer.categorize_all()
    total = len(predictions)
    agreement = analyzer.get_agreement_matrix()
    corr = analyzer.get_correlation_matrix()

    # Get test upset rate
    y_true = np.array([p.y_true for p in predictions])
    upset_rate = y_true.mean()
    baseline_brier = calculate_baseline_brier(upset_rate)

    lines = [
        "# Test Set Evaluation: 2023-2025",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Overview",
        "",
        "- **Training:** 2005-2022 (full training set, no CV split)",
        f"- **Testing:** 2023-2025 ({total} upset candidates)",
        f"- **Test upset rate:** {upset_rate:.1%}",
        f"- **Baseline Brier:** {baseline_brier:.4f}",
        "",
    ]

    if cv_metrics is None:
        lines.extend(
            [
                "## Test vs CV Performance",
                "",
                "Fresh CV baselines must be regenerated after"
                " the multi-representation architecture before this"
                " comparison can be reported.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Test vs CV Performance",
                "",
                "| Model | Metric | CV | Test | Delta |",
                "|-------|--------|----|------|-------|",
            ]
        )

        for model_key, model_name in [("lr", "LR"), ("xgb", "XGB"), ("lstm", "LSTM")]:
            for metric in ["auc_roc", "brier_score", "log_loss"]:
                cv_val = cv_metrics[model_key][metric]
                test_val = test_metrics[model_key][metric]
                delta = test_val - cv_val
                sign = "+" if delta > 0 else ""
                lines.append(
                    f"| {model_name} | {metric} "
                    f"| {cv_val:.4f} | {test_val:.4f} "
                    f"| {sign}{delta:.4f} |"
                )

    if CV_CATEGORIES is None:
        lines.extend(
            [
                "",
                "## Disagreement Analysis",
                "",
                "Fresh disagreement CV baselines must be"
                " regenerated with the multi-representation architecture"
                " before CV-vs-test disagreement comparisons"
                " can be reported.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Disagreement Analysis",
                "",
                "| Category | Count | Test % | CV % | Upset Rate |",
                "|----------|-------|--------|------|------------|",
            ]
        )

        for cat in PredictionCategory:
            count = len(categories[cat])
            pct = count / total * 100 if total > 0 else 0
            cv_pct = CV_CATEGORIES.get(cat.value, (0, 0))[1]
            ur = np.mean([p.y_true for p in categories[cat]]) if categories[cat] else 0
            lines.append(
                f"| {cat.value} | {count} | {pct:.1f}% | {cv_pct:.1f}% | {ur:.1%} |"
            )

    lines.extend(
        [
            "",
            "## Model Agreement",
            "",
            "| Model Pair | Agreement Rate |",
            "|------------|---------------|",
        ]
    )
    for _, row in agreement.iterrows():
        lines.append(f"| {row['model_pair']} | {row['agreement_rate']:.1%} |")

    lines.extend(
        [
            "",
            "## Probability Correlations",
            "",
            "|      | LR    | XGB   | LSTM  |",
            "|------|-------|-------|-------|",
        ]
    )
    for model in ["LR", "XGB", "LSTM"]:
        row_vals = [f"{corr.loc[model, col]:.3f}" for col in ["LR", "XGB", "LSTM"]]
        lines.append(f"| {model} | {' | '.join(row_vals)} |")

    # Top-K analysis
    lines.extend(["", "## Top-K Analysis", ""])
    lines.append(
        "Games ranked by predicted upset probability. Lift = hit rate / base rate."
    )
    lines.append("")

    topk_df = pd.DataFrame(
        [
            {
                "y_true": p.y_true,
                "lr_prob": p.lr_prob,
                "xgb_prob": p.xgb_prob,
                "lstm_prob": p.lstm_prob,
                "ensemble_prob": (p.lr_prob + p.xgb_prob + p.lstm_prob) / 3,
            }
            for p in predictions
        ]
    )
    base_rate_topk = topk_df["y_true"].mean()

    lines.append(f"Base rate: {base_rate_topk:.1%}")
    lines.append("")
    lines.append("| K | LR | XGB | LSTM | Ensemble |")
    lines.append("|---|-----|-----|------|----------|")

    for k in [10, 20, 30, 50, 75, 100]:
        if k > len(topk_df):
            continue
        cells = [f"| {k} "]
        for col in ["lr_prob", "xgb_prob", "lstm_prob", "ensemble_prob"]:
            top_k = topk_df.nlargest(k, col)
            hits = int(top_k["y_true"].sum())
            hit_rate = top_k["y_true"].mean()
            lift = hit_rate / base_rate_topk if base_rate_topk > 0 else 0
            cells.append(f"| {hits}/{k} ({hit_rate:.0%}, {lift:.1f}x) ")
        lines.append(" ".join(cells) + "|")

    # Probability bucket analysis
    lines.extend(["", "## Probability Bucket Analysis", ""])
    lines.append("Games grouped into quintiles by predicted probability.")
    lines.append("")

    for col, name in [
        ("lr_prob", "LR"),
        ("xgb_prob", "XGB"),
        ("lstm_prob", "LSTM"),
        ("ensemble_prob", "Ensemble"),
    ]:
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            "| Quintile | Prob Range | Games | Upsets | Upset Rate | Avg Prob |"
        )
        lines.append(
            "|----------|-----------|-------|--------|------------|----------|"
        )

        topk_df["_bucket"] = safe_quantile_buckets(topk_df[col], q=5)
        n_buckets = topk_df["_bucket"].nunique()
        for bucket in sorted(topk_df["_bucket"].dropna().unique()):
            bdf = topk_df[topk_df["_bucket"] == bucket]
            prob_min = bdf[col].min()
            prob_max = bdf[col].max()
            if bucket == 0:
                q_label = "Low"
            elif bucket == n_buckets - 1:
                q_label = "High"
            else:
                q_label = "Mid"
            label = f"Q{bucket+1} ({q_label})"
            upsets = int(bdf["y_true"].sum())
            rate = bdf["y_true"].mean()
            avg_p = bdf[col].mean()
            lines.append(
                f"| {label} | [{prob_min:.3f}-{prob_max:.3f}]"
                f" | {len(bdf)} | {upsets}"
                f" | {rate:.1%} | {avg_p:.3f} |"
            )
        topk_df.drop(columns=["_bucket"], inplace=True)
        lines.append("")

    # Per-season breakdown
    lines.extend(["## Per-Season Breakdown", ""])
    pred_df = pd.DataFrame(
        [
            {
                "season": p.season,
                "y_true": p.y_true,
                "lr_prob": p.lr_prob,
                "xgb_prob": p.xgb_prob,
                "lstm_prob": p.lstm_prob,
            }
            for p in predictions
        ]
    )

    for season in sorted(pred_df["season"].unique()):
        sdf = pred_df[pred_df["season"] == season]
        y = sdf["y_true"].values
        lines.append(f"### {season} ({len(y)} games, upset rate = {y.mean():.1%})")
        lines.append("")
        lines.append("| Model | AUC-ROC | Brier |")
        lines.append("|-------|---------|-------|")
        for model_name, col in [
            ("LR", "lr_prob"),
            ("XGB", "xgb_prob"),
            ("LSTM", "lstm_prob"),
        ]:
            probs = sdf[col].values
            auc = safe_roc_auc_score(y, probs)
            brier = brier_score_loss(y, probs)
            if np.isnan(auc):
                lines.append(f"| {model_name} | N/A | N/A |")
            else:
                lines.append(f"| {model_name} | {auc:.3f} | {brier:.3f} |")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Saved report to {output_path}")


def main():
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION: 2023-2025")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df, test_df, feature_cols, xgb_feature_cols = load_data()

    # Verify data integrity
    verify_data(train_df, test_df)

    # Filter to valid upset candidates
    train_valid = filter_valid_upsets(train_df)
    test_valid = filter_valid_upsets(test_df)

    # Fill NaN in features (some rolling stats are NaN for early-season games)
    all_cols = list(set(feature_cols + xgb_feature_cols))
    train_valid[all_cols] = train_valid[all_cols].fillna(0)
    test_valid[all_cols] = test_valid[all_cols].fillna(0)

    # Train final models on full training data
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODELS ON 2005-2022")
    print("=" * 60)
    print(f"  LR features:   {len(feature_cols)}")
    print(f"  XGB features:  {len(xgb_feature_cols)}")

    trainer = UnifiedTrainer()
    models = trainer.train_final(
        train_valid,
        feature_cols,
        target_col="upset",
        lstm_epochs=int(TUNED_LSTM_TRAINING_PARAMS["epochs"]),
        lstm_batch_size=int(TUNED_LSTM_TRAINING_PARAMS["batch_size"]),
        verbose=True,
        xgb_feature_cols=xgb_feature_cols,
        full_df=train_df,
    )

    # Generate test predictions (raw)
    print("\nGenerating test predictions...")
    prediction_history = build_prediction_history(train_df, test_df)
    test_pred_df, y_test, lr_probs, xgb_probs, lstm_probs = generate_predictions(
        models,
        test_valid,
        feature_cols,
        xgb_feature_cols,
        full_history_df=prediction_history,
    )

    # Post-hoc calibration via Platt scaling
    # Train on 2005-2020, predict on 2021-2022 to fit calibrators
    print("\n" + "=" * 60)
    print("POST-HOC CALIBRATION (Platt scaling)")
    print("=" * 60)
    print("Fitting calibrators on 2021-2022 held-out predictions...")

    cal_probs, cal_y = generate_calibration_predictions(
        train_df,
        feature_cols,
        cal_seasons=(2021, 2022),
        xgb_feature_cols=xgb_feature_cols,
    )
    cal_results = calibrate_models(
        cal_probs,
        cal_y,
        {"lr": lr_probs, "xgb": xgb_probs, "lstm": lstm_probs},
        method="platt",
    )

    lr_probs_cal = cal_results["lr"].calibrated
    xgb_probs_cal = cal_results["xgb"].calibrated
    lstm_probs_cal = cal_results["lstm"].calibrated

    # Report raw vs calibrated metrics
    print("\nRaw model metrics:")
    test_metrics_raw = {
        "lr": calculate_metrics(y_test, lr_probs),
        "xgb": calculate_metrics(y_test, xgb_probs),
        "lstm": calculate_metrics(y_test, lstm_probs),
    }
    for name in ["lr", "xgb", "lstm"]:
        m = test_metrics_raw[name]
        print(
            f"  {name.upper()}: AUC={m['auc_roc']:.4f}"
            f"  Brier={m['brier_score']:.4f}"
            f"  LogLoss={m['log_loss']:.4f}"
        )

    print("\nCalibrated model metrics:")
    test_metrics = {
        "lr": calculate_metrics(y_test, lr_probs_cal),
        "xgb": calculate_metrics(y_test, xgb_probs_cal),
        "lstm": calculate_metrics(y_test, lstm_probs_cal),
    }
    for name in ["lr", "xgb", "lstm"]:
        m = test_metrics[name]
        print(
            f"  {name.upper()}: AUC={m['auc_roc']:.4f}"
            f"  Brier={m['brier_score']:.4f}"
            f"  LogLoss={m['log_loss']:.4f}"
        )

    # Use calibrated probabilities for all downstream analysis
    lr_probs = lr_probs_cal
    xgb_probs = xgb_probs_cal
    lstm_probs = lstm_probs_cal

    # Print metrics comparison
    print_metrics_comparison(test_metrics, CV_METRICS)

    # Calibration diagnostics
    print_calibration(y_test, lr_probs, xgb_probs, lstm_probs)

    # Build GamePrediction objects for disagreement analysis
    predictions = build_game_predictions(
        test_pred_df, y_test, lr_probs, xgb_probs, lstm_probs
    )

    # Disagreement analysis
    analyzer = DisagreementAnalyzer(predictions)
    print_disagreement_comparison(analyzer, CV_CATEGORIES)

    # Top-K and probability bucket analyses
    print_top_k_analysis(predictions)
    print_probability_buckets(predictions)

    # Per-season breakdown
    print_per_season_breakdown(predictions)

    # Save outputs
    save_predictions_csv(predictions, RESULTS_DIR / "test" / "predictions.csv")
    save_report_md(
        test_metrics,
        CV_METRICS,
        analyzer,
        predictions,
        RESULTS_DIR / "test" / "report.md",
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
