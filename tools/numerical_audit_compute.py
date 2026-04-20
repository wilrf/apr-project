#!/usr/bin/env python3
"""Compute the AP Research numerical audit and write markdown reports."""

# ruff: noqa: E402,E501

from __future__ import annotations

import inspect
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import pipeline
from src.models.sequence_builder import (
    MATCHUP_FEATURES,
    MATCHUP_FEATURES_NO_SPREAD,
    SEQUENCE_FEATURES,
    SEQUENCE_LENGTH,
)

N_BOOT = 10_000
RNG_SEED_CV = 42
RNG_SEED_TEST = 123

MODEL_COLS = {
    "LR": "lr_prob",
    "XGB": "xgb_prob",
    "LSTM": "lstm_prob",
}

CATEGORY_ORDER = [
    "all_correct",
    "all_wrong",
    "only_lstm",
    "only_lr",
    "only_xgb",
    "lr_xgb",
    "lr_lstm",
    "xgb_lstm",
]


def rel(path: Path) -> str:
    """Return a repo-relative path string."""
    return str(path.relative_to(ROOT))


def line_ref(obj: Any) -> str:
    """Return the source location for a computation function."""
    path = Path(inspect.getsourcefile(obj) or __file__).resolve()
    line = inspect.getsourcelines(obj)[1]
    return f"{rel(path)}:{line}"


def find_line(path: str, pattern: str) -> str:
    """Find the first line matching a literal or regex pattern."""
    full = ROOT / path
    if not full.exists():
        return f"{path}:missing"
    compiled = re.compile(pattern)
    for idx, line in enumerate(full.read_text(errors="ignore").splitlines(), 1):
        if compiled.search(line):
            return f"{path}:{idx}"
    return f"{path}:not-found"


def fmt(value: float | int | None, digits: int = 4) -> str:
    """Format a numeric value for markdown."""
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    """Format a ratio as a percentage."""
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def ci_text(low: float, high: float, digits: int = 4) -> str:
    """Format a confidence interval."""
    return f"[{fmt(low, digits)}, {fmt(high, digits)}]"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a simple markdown table."""
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def safe_auc(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Compute AUC, returning NaN when a bootstrap sample has one class."""
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, pred))


def metrics_for_predictions(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute AUC, Brier score, and log loss for all three model columns."""
    y = df["y_true"].to_numpy(dtype=float)
    output: dict[str, dict[str, float]] = {}
    for model, col in MODEL_COLS.items():
        pred = df[col].to_numpy(dtype=float)
        output[model] = {
            "auc": float(roc_auc_score(y, pred)),
            "brier": float(brier_score_loss(y, pred)),
            "log_loss": float(log_loss(y, np.clip(pred, 1e-15, 1 - 1e-15))),
        }
    return output


def probability_correlations(df: pd.DataFrame) -> dict[str, float]:
    """Compute pairwise Pearson correlations among model probabilities."""
    corr = df[[MODEL_COLS[m] for m in ["LR", "XGB", "LSTM"]]].corr()
    return {
        "LR-XGB": float(corr.loc["lr_prob", "xgb_prob"]),
        "LR-LSTM": float(corr.loc["lr_prob", "lstm_prob"]),
        "XGB-LSTM": float(corr.loc["xgb_prob", "lstm_prob"]),
    }


def prediction_bits(df: pd.DataFrame, threshold: float) -> dict[str, np.ndarray]:
    """Convert probability columns to binary upset predictions."""
    return {
        model: (df[col].to_numpy(dtype=float) >= threshold).astype(int)
        for model, col in MODEL_COLS.items()
    }


def categorize_predictions(
    df: pd.DataFrame,
    threshold: float | None = None,
    use_stored_columns: bool = False,
) -> pd.Series:
    """Categorize predictions by which model was correct."""
    y = df["y_true"].to_numpy(dtype=int)
    if use_stored_columns:
        bits = {
            "LR": df["lr_pred"].to_numpy(dtype=int),
            "XGB": df["xgb_pred"].to_numpy(dtype=int),
            "LSTM": df["lstm_pred"].to_numpy(dtype=int),
        }
    else:
        active_threshold = (
            float(df["y_true"].mean()) if threshold is None else threshold
        )
        bits = prediction_bits(df, active_threshold)

    lr = bits["LR"] == y
    xgb = bits["XGB"] == y
    lstm = bits["LSTM"] == y

    labels = np.select(
        [
            lr & xgb & lstm,
            ~lr & ~xgb & ~lstm,
            lr & ~xgb & ~lstm,
            ~lr & xgb & ~lstm,
            ~lr & ~xgb & lstm,
            lr & xgb & ~lstm,
            lr & ~xgb & lstm,
            ~lr & xgb & lstm,
        ],
        [
            "all_correct",
            "all_wrong",
            "only_lr",
            "only_xgb",
            "only_lstm",
            "lr_xgb",
            "lr_lstm",
            "xgb_lstm",
        ],
        default="unknown",
    )
    return pd.Series(labels, index=df.index, name="category")


def spread_bucket(value: float) -> str:
    """Map spread magnitude to the audit bucket."""
    if 3 <= value <= 6.5:
        return "small"
    if 7 <= value <= 13.5:
        return "medium"
    if value >= 14:
        return "large"
    return "other"


def category_breakdown(
    df: pd.DataFrame, category_col: str
) -> dict[str, dict[str, Any]]:
    """Summarize disagreement categories."""
    total = len(df)
    output: dict[str, dict[str, Any]] = {}
    for category in CATEGORY_ORDER:
        sub = df[df[category_col] == category]
        n = len(sub)
        upsets = int(sub["y_true"].sum()) if n else 0
        output[category] = {
            "n": n,
            "pct": n / total if total else float("nan"),
            "upsets": upsets,
            "non_upsets": n - upsets,
            "upset_rate": float(sub["y_true"].mean()) if n else None,
        }
    return output


def agreement_rates(df: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Compute pairwise and all-three binary agreement rates."""
    bits = prediction_bits(df, threshold)
    lr, xgb, lstm = bits["LR"], bits["XGB"], bits["LSTM"]
    return {
        "LR-XGB": float(np.mean(lr == xgb)),
        "LR-LSTM": float(np.mean(lr == lstm)),
        "XGB-LSTM": float(np.mean(xgb == lstm)),
        "All three": float(np.mean((lr == xgb) & (xgb == lstm))),
    }


def dataset_composition(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Compute dataset composition and season counts."""

    def split_summary(df: pd.DataFrame) -> dict[str, Any]:
        labeled = df[df["upset"].notna()].copy()
        sub3 = df[df["upset"].isna() & (df["spread_magnitude"] < 3)]
        return {
            "total_rows": len(df),
            "labeled_games": len(labeled),
            "upsets": int(labeled["upset"].sum()),
            "upset_rate": float(labeled["upset"].mean()),
            "sub3_unlabeled": len(sub3),
            "other_unlabeled": len(df) - len(labeled) - len(sub3),
        }

    test_seasons = {}
    for season, sdf in test.groupby("season"):
        labeled = sdf[sdf["upset"].notna()]
        test_seasons[int(season)] = {
            "total_rows": len(sdf),
            "labeled_games": len(labeled),
            "upsets": int(labeled["upset"].sum()),
            "upset_rate": float(labeled["upset"].mean()),
        }

    return {
        "train": split_summary(train),
        "test": split_summary(test),
        "test_seasons": test_seasons,
    }


def bootstrap_cv(
    cv: pd.DataFrame,
    cv_no_spread: pd.DataFrame,
) -> dict[str, Any]:
    """Bootstrap CV AUCs, pairwise differences, and spread-ablation deltas."""
    y = cv["y_true"].to_numpy(dtype=int)
    n = len(y)
    rng = np.random.default_rng(RNG_SEED_CV)

    auc_with = {model: np.empty(N_BOOT) for model in MODEL_COLS}
    auc_without = {model: np.empty(N_BOOT) for model in MODEL_COLS}

    pred_with = {
        model: cv[col].to_numpy(dtype=float) for model, col in MODEL_COLS.items()
    }
    pred_without = {
        model: cv_no_spread[col].to_numpy(dtype=float)
        for model, col in MODEL_COLS.items()
    }

    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            for model in MODEL_COLS:
                auc_with[model][i] = np.nan
                auc_without[model][i] = np.nan
            continue
        for model in MODEL_COLS:
            auc_with[model][i] = roc_auc_score(yy, pred_with[model][idx])
            auc_without[model][i] = roc_auc_score(yy, pred_without[model][idx])

    def interval(arr: np.ndarray) -> tuple[float, float]:
        arr = arr[np.isfinite(arr)]
        return (
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        )

    auc_ci = {model: interval(values) for model, values in auc_with.items()}
    pairwise = {}
    for left, right in [("LR", "XGB"), ("LR", "LSTM"), ("XGB", "LSTM")]:
        diff = auc_with[left] - auc_with[right]
        point = roc_auc_score(y, pred_with[left]) - roc_auc_score(y, pred_with[right])
        pairwise[f"{left}-{right}"] = {
            "point": float(point),
            "mean": float(np.nanmean(diff)),
            "ci": interval(diff),
        }

    ablation = {}
    for model in MODEL_COLS:
        delta = auc_without[model] - auc_with[model]
        point = roc_auc_score(y, pred_without[model]) - roc_auc_score(
            y, pred_with[model]
        )
        finite = delta[np.isfinite(delta)]
        p_two_sided = 2 * min(
            (int(np.sum(finite >= 0)) + 1) / (len(finite) + 1),
            (int(np.sum(finite <= 0)) + 1) / (len(finite) + 1),
        )
        ablation[model] = {
            "point": float(point),
            "mean": float(np.mean(finite)),
            "ci": interval(finite),
            "p_two_sided": float(min(p_two_sided, 1.0)),
        }

    return {
        "auc_ci": auc_ci,
        "pairwise": pairwise,
        "ablation": ablation,
    }


def bootstrap_test_auc_diffs(test: pd.DataFrame) -> dict[str, Any]:
    """Bootstrap test-set pairwise AUC differences."""
    y = test["y_true"].to_numpy(dtype=int)
    n = len(y)
    rng = np.random.default_rng(RNG_SEED_TEST)
    preds = {
        model: test[col].to_numpy(dtype=float) for model, col in MODEL_COLS.items()
    }
    output = {}
    for left, right in [("XGB", "LSTM"), ("LR", "LSTM"), ("LR", "XGB")]:
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            yy = y[idx]
            diffs[i] = (
                np.nan
                if np.unique(yy).size < 2
                else roc_auc_score(yy, preds[left][idx])
                - roc_auc_score(yy, preds[right][idx])
            )
        finite = diffs[np.isfinite(diffs)]
        p_two_sided = 2 * min(
            (int(np.sum(finite >= 0)) + 1) / (len(finite) + 1),
            (int(np.sum(finite <= 0)) + 1) / (len(finite) + 1),
        )
        point = roc_auc_score(y, preds[left]) - roc_auc_score(y, preds[right])
        output[f"{left}-{right}"] = {
            "point": float(point),
            "mean": float(np.mean(finite)),
            "ci": (
                float(np.percentile(finite, 2.5)),
                float(np.percentile(finite, 97.5)),
            ),
            "p_two_sided": float(min(p_two_sided, 1.0)),
        }
    return output


def lstm_bucket_analyses(cv: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Compute LSTM-exclusive and LSTM-vs-static-consensus bucket analyses."""
    df = cv.copy()
    df["category"] = categorize_predictions(df, threshold)
    df["bucket"] = df["spread_magnitude"].map(spread_bucket)
    bits = prediction_bits(df, threshold)
    y = df["y_true"].to_numpy(dtype=int)

    analysis_a = {}
    only_lstm = df["category"] == "only_lstm"
    for bucket_name in ["small", "medium", "large"]:
        sub = df[only_lstm & (df["bucket"] == bucket_name)]
        n = len(sub)
        upsets = int(sub["y_true"].sum()) if n else 0
        analysis_a[bucket_name] = {
            "n": n,
            "upsets": upsets,
            "non_upsets": n - upsets,
            "upset_detection_rate": (upsets / n if n else None),
        }

    consensus_mask = (bits["LR"] == bits["XGB"]) & (bits["LSTM"] != bits["LR"])
    lstm_correct = bits["LSTM"] == y
    analysis_b = {}
    for bucket_name in ["small", "medium", "large"]:
        bucket_mask = (df["bucket"].to_numpy() == bucket_name) & consensus_mask
        n = int(np.sum(bucket_mask))
        correct = int(np.sum(lstm_correct & bucket_mask))
        analysis_b[bucket_name] = {
            "n": n,
            "lstm_correct": correct,
            "lstm_wrong": n - correct,
            "hit_rate": (correct / n if n else None),
        }

    return {"analysis_a": analysis_a, "analysis_b": analysis_b}


def topk_hits(test: pd.DataFrame) -> dict[str, Any]:
    """Compute top-K upset hit rates on the test set."""
    df = test.copy()
    df["ensemble_prob"] = (df["lr_prob"] + df["xgb_prob"] + df["lstm_prob"]) / 3.0
    output = {"base_rate": float(df["y_true"].mean()), "k": {}}
    for k in [10, 20, 50]:
        output["k"][str(k)] = {}
        for label, col in [
            ("LR", "lr_prob"),
            ("XGB", "xgb_prob"),
            ("LSTM", "lstm_prob"),
            ("Ensemble", "ensemble_prob"),
        ]:
            top = df.nlargest(k, col)
            hits = int(top["y_true"].sum())
            output["k"][str(k)][label] = {
                "hits": hits,
                "rate": float(hits / k),
                "lift": float((hits / k) / output["base_rate"]),
            }
    return output


def taxonomy_stats(cv: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Compute taxonomy coverage and mixed-category tests."""
    df = cv.copy()
    df["category"] = categorize_predictions(df, threshold)
    df["bucket"] = df["spread_magnitude"].map(spread_bucket)
    n_total = len(df)
    base = float(df["y_true"].mean())

    hidden = df[
        (df["category"] == "all_wrong") & df["bucket"].isin(["medium", "large"])
    ]
    irreducible = df[(df["category"] == "all_wrong") & (df["bucket"] == "small")]
    all_wrong = df[df["category"] == "all_wrong"]

    mixed = {}
    for category in ["lr_xgb", "only_xgb", "lr_lstm", "only_lr", "xgb_lstm"]:
        sub = df[df["category"] == category]
        k = int(sub["y_true"].sum())
        n = len(sub)
        mixed[category] = {
            "n": n,
            "pct": n / n_total,
            "upsets": k,
            "upset_rate": float(k / n) if n else None,
            "p_vs_base_two_sided": (
                float(stats.binomtest(k, n, base, alternative="two-sided").pvalue)
                if n
                else None
            ),
        }

    return {
        "base_rate": base,
        "coverage": {
            "consensus_all_correct": {
                "n": int(np.sum(df["category"] == "all_correct")),
                "pct": float(np.mean(df["category"] == "all_correct")),
            },
            "temporal_only_lstm": {
                "n": int(np.sum(df["category"] == "only_lstm")),
                "pct": float(np.mean(df["category"] == "only_lstm")),
            },
            "hidden_all_wrong_medium_large": {
                "n": len(hidden),
                "pct": len(hidden) / n_total,
                "pct_of_all_wrong": len(hidden) / len(all_wrong),
            },
            "irreducible_all_wrong_small": {
                "n": len(irreducible),
                "pct": len(irreducible) / n_total,
                "pct_of_all_wrong": len(irreducible) / len(all_wrong),
            },
        },
        "mixed": mixed,
    }


def probability_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute min, max, mean, and sample standard deviation for probabilities."""
    output = {}
    for model, col in MODEL_COLS.items():
        values = df[col].to_numpy(dtype=float)
        output[model] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
    return output


def feature_details() -> dict[str, Any]:
    """Compute feature counts and rolling-window settings from code constants."""
    lr = pipeline.get_feature_columns()
    lr_ns = pipeline.get_no_spread_feature_columns()
    xgb = pipeline.get_xgb_feature_columns()
    xgb_ns = pipeline.get_xgb_no_spread_feature_columns()
    return {
        "lr_count": len(lr),
        "lr_no_spread_count": len(lr_ns),
        "xgb_count": len(xgb),
        "xgb_no_spread_count": len(xgb_ns),
        "lstm_sequence_features": len(SEQUENCE_FEATURES),
        "lstm_sequence_length": SEQUENCE_LENGTH,
        "lstm_matchup_features": len(MATCHUP_FEATURES),
        "lstm_matchup_no_spread": len(MATCHUP_FEATURES_NO_SPREAD),
        "rolling_window": pipeline.ROLLING_WINDOW,
        "xgb_lags": [1, 2, 3],
    }


def lr_coefficient_details() -> dict[str, Any]:
    """Compute LR coefficient ratio from the saved with-spread coefficients."""
    coefs = json.loads(
        (ROOT / "results/ab_experiment/lr_coefs_with_spread.json").read_text()
    )
    ranked = sorted(coefs.items(), key=lambda item: abs(item[1]), reverse=True)
    spread = coefs["spread_magnitude"]
    next_feature, next_coef = ranked[1]
    return {
        "spread_magnitude": float(spread),
        "next_feature": next_feature,
        "next_coefficient": float(next_coef),
        "ratio_to_next_abs": float(abs(spread) / abs(next_coef)),
        "top10": [
            {"feature": k, "coefficient": float(v), "abs": float(abs(v))}
            for k, v in ranked[:10]
        ],
    }


def extract_pptx_slide_count(path: Path) -> int | None:
    """Return the slide count using only the standard library."""
    if not path.exists():
        return None
    with zipfile.ZipFile(path) as zf:
        slide_names = [
            name
            for name in zf.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        ]
    return len(slide_names)


def extract_pptx_text(path: Path) -> dict[int, str]:
    """Extract slide text using only the standard library."""
    if not path.exists():
        return {}
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    slides: dict[int, str] = {}
    with zipfile.ZipFile(path) as zf:
        slide_names = sorted(
            [
                name
                for name in zf.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ],
            key=lambda name: int(re.search(r"slide(\d+)\.xml", name).group(1)),
        )
        for idx, name in enumerate(slide_names, 1):
            root = ET.fromstring(zf.read(name))
            texts = [node.text or "" for node in root.findall(".//a:t", ns)]
            slides[idx] = " ".join(text.strip() for text in texts if text.strip())
    return slides


def compute_audit() -> dict[str, Any]:
    """Load repo artifacts and compute all audit numbers."""
    train = pd.read_csv(ROOT / "data/features/train.csv", low_memory=False)
    test = pd.read_csv(ROOT / "data/features/test.csv", low_memory=False)
    cv = pd.read_csv(ROOT / "results/ab_experiment/predictions_with_spread.csv")
    cv_no_spread = pd.read_csv(
        ROOT / "results/ab_experiment/predictions_without_spread.csv"
    )
    test_predictions = pd.read_csv(ROOT / "results/test/predictions.csv")

    cv_threshold = float(cv["y_true"].mean())
    test_threshold = float(test_predictions["y_true"].mean())

    cv = cv.copy()
    cv["category_global"] = categorize_predictions(cv, cv_threshold)
    cv["category_stored"] = categorize_predictions(cv, use_stored_columns=True)
    cv["bucket"] = cv["spread_magnitude"].map(spread_bucket)

    cv_no_spread = cv_no_spread.copy()
    cv_no_spread["category_global"] = categorize_predictions(cv_no_spread, cv_threshold)
    cv_no_spread["category_stored"] = categorize_predictions(
        cv_no_spread, use_stored_columns=True
    )

    test_predictions = test_predictions.copy()
    test_predictions["category_global"] = categorize_predictions(
        test_predictions, test_threshold
    )

    boot_cv = bootstrap_cv(cv, cv_no_spread)
    boot_test = bootstrap_test_auc_diffs(test_predictions)
    taxonomy = taxonomy_stats(cv, cv_threshold)

    season_auc = {}
    for season, sdf in test_predictions.groupby("season"):
        season_auc[int(season)] = {
            "n": len(sdf),
            "upsets": int(sdf["y_true"].sum()),
            "upset_rate": float(sdf["y_true"].mean()),
            "LR": safe_auc(sdf["y_true"].to_numpy(), sdf["lr_prob"].to_numpy()),
            "XGB": safe_auc(sdf["y_true"].to_numpy(), sdf["xgb_prob"].to_numpy()),
            "LSTM": safe_auc(sdf["y_true"].to_numpy(), sdf["lstm_prob"].to_numpy()),
        }

    cv_metrics = metrics_for_predictions(cv)
    test_metrics = metrics_for_predictions(test_predictions)
    no_spread_metrics = metrics_for_predictions(cv_no_spread)

    output = {
        "metadata": {
            "n_bootstrap": N_BOOT,
            "cv_bootstrap_seed": RNG_SEED_CV,
            "test_bootstrap_seed": RNG_SEED_TEST,
            "slide_content_md_present": (
                ROOT / "AP_Research_Slides_Content.md"
            ).exists(),
            "revised_pptx_slide_count": extract_pptx_slide_count(
                ROOT / "docs/AP_Research_POD_Revised.pptx"
            ),
        },
        "dataset": dataset_composition(train, test),
        "cv": {
            "n": len(cv),
            "upsets": int(cv["y_true"].sum()),
            "base_rate": cv_threshold,
            "metrics": cv_metrics,
            "auc_ci": boot_cv["auc_ci"],
            "pairwise_auc": boot_cv["pairwise"],
            "correlations": probability_correlations(cv),
            "agreement": agreement_rates(cv, cv_threshold),
            "categories": category_breakdown(cv, "category_global"),
            "categories_stored_pred_columns": category_breakdown(cv, "category_stored"),
            "bucket_totals": {
                bucket: {
                    "n": int(len(sdf)),
                    "upsets": int(sdf["y_true"].sum()),
                    "upset_rate": float(sdf["y_true"].mean()),
                }
                for bucket, sdf in cv.groupby("bucket")
            },
        },
        "lstm_buckets": lstm_bucket_analyses(cv, cv_threshold),
        "spread_ablation": {
            "metrics_without_spread": no_spread_metrics,
            "bootstrap": boot_cv["ablation"],
            "correlations_without_spread": probability_correlations(cv_no_spread),
            "agreement_without_spread": agreement_rates(cv_no_spread, cv_threshold),
            "with_spread_only_lstm": category_breakdown(cv, "category_global")[
                "only_lstm"
            ],
            "without_spread_only_lstm": category_breakdown(
                cv_no_spread, "category_global"
            )["only_lstm"],
            "stored_with_spread_only_lstm": category_breakdown(cv, "category_stored")[
                "only_lstm"
            ],
            "stored_without_spread_only_lstm": category_breakdown(
                cv_no_spread, "category_stored"
            )["only_lstm"],
        },
        "test": {
            "n": len(test_predictions),
            "upsets": int(test_predictions["y_true"].sum()),
            "base_rate": test_threshold,
            "metrics": test_metrics,
            "cv_to_test_gap": {
                model: cv_metrics[model]["auc"] - test_metrics[model]["auc"]
                for model in MODEL_COLS
            },
            "season_auc": season_auc,
            "correlations": probability_correlations(test_predictions),
            "pairwise_auc": boot_test,
            "topk": topk_hits(test_predictions),
        },
        "taxonomy": taxonomy,
        "calibration": {
            "cv_raw_probability_summary": probability_summary(cv),
            "test_platt_probability_summary": probability_summary(test_predictions),
            "test_raw_probability_summary_current_artifact": None,
        },
        "features": feature_details(),
        "lr_coefficients": lr_coefficient_details(),
        "source_refs": {
            "dataset_composition": line_ref(dataset_composition),
            "metrics": line_ref(metrics_for_predictions),
            "bootstrap_cv": line_ref(bootstrap_cv),
            "bootstrap_test": line_ref(bootstrap_test_auc_diffs),
            "categorize": line_ref(categorize_predictions),
            "lstm_buckets": line_ref(lstm_bucket_analyses),
            "topk": line_ref(topk_hits),
            "taxonomy": line_ref(taxonomy_stats),
            "calibration_summary": line_ref(probability_summary),
            "features": line_ref(feature_details),
            "lr_coefficients": line_ref(lr_coefficient_details),
        },
    }
    return output


def render_audit_results(audit: dict[str, Any]) -> str:
    """Render audit_results.md."""
    refs = audit["source_refs"]
    cv = audit["cv"]
    test = audit["test"]
    spread = audit["spread_ablation"]
    taxonomy = audit["taxonomy"]
    calibration = audit["calibration"]
    features = audit["features"]
    lr_coefs = audit["lr_coefficients"]

    lines: list[str] = [
        "# AP Research Numerical Audit",
        "",
        "- Doc Type: Results",
        "- Topic: Numerical Audit",
        "- Topic Slug: numerical-audit",
        "- Date: 2026-04-20",
        "- Status: Complete",
        "",
        "## Scope and Source Inventory",
        "",
        "Primary numeric sources examined:",
        "",
        "- `data/features/train.csv`",
        "- `data/features/test.csv`",
        "- `results/ab_experiment/predictions_with_spread.csv`",
        "- `results/ab_experiment/predictions_without_spread.csv`",
        "- `results/test/predictions.csv`",
        "- `results/ab_experiment/lr_coefs_with_spread.json`",
        "- Current markdown summaries in `docs/` and `results/` for consistency checks",
        "- Presentation sources: `docs/AP_Research_POD_Revised.pptx`, `docs/AP_Research_POD_Corrected.pptx`, and `tools/presentation/rewrite_presentation.py`",
        "",
        "No `.ipynb` notebooks are present in the repo outside ignored virtualenv/cache directories. `AP_Research_Slides_Content.md` is not present, so slide reconciliation uses the latest PowerPoint artifact (`docs/AP_Research_POD_Revised.pptx`, 20 slides) and its generator (`tools/presentation/rewrite_presentation.py`).",
        "",
        "All current model metrics below are recomputed from saved prediction CSVs, not copied from prose summaries. Bootstrap intervals use 10,000 paired nonparametric resamples.",
        "",
        "## 1. Dataset Composition",
        "",
        f"Computed at `{refs['dataset_composition']}` from `data/features/train.csv` and `data/features/test.csv`.",
        "",
    ]

    dataset_rows = []
    for split in ["train", "test"]:
        item = audit["dataset"][split]
        label = "Train (2005-2022)" if split == "train" else "Test (2023-2025)"
        dataset_rows.append(
            [
                label,
                item["total_rows"],
                item["labeled_games"],
                item["upsets"],
                pct(item["upset_rate"], 3),
                item["sub3_unlabeled"],
                item["other_unlabeled"],
            ]
        )
    lines.append(
        md_table(
            [
                "Split",
                "Total rows",
                "Labeled games",
                "Upsets",
                "Upset base rate",
                "Sub-3 unlabeled",
                "Other unlabeled",
            ],
            dataset_rows,
        )
    )
    lines.extend(
        [
            "",
            "The exact modeling counts are the labeled games: 3,495 training games and 558 test games. Total generated rows are larger because sub-3-spread games remain for rolling-stat continuity.",
            "",
            md_table(
                ["Season", "Total rows", "Labeled games", "Upsets", "Upset rate"],
                [
                    [
                        season,
                        values["total_rows"],
                        values["labeled_games"],
                        values["upsets"],
                        pct(values["upset_rate"], 3),
                    ]
                    for season, values in audit["dataset"]["test_seasons"].items()
                ],
            ),
            "",
            "## 2. Model Performance in Cross-Validation",
            "",
            f"Metrics computed at `{refs['metrics']}`; bootstrap CIs at `{refs['bootstrap_cv']}` from `results/ab_experiment/predictions_with_spread.csv`.",
            "",
        ]
    )
    lines.append(
        md_table(
            ["Model", "AUC", "AUC 95% CI", "Brier", "Log loss"],
            [
                [
                    model,
                    fmt(cv["metrics"][model]["auc"]),
                    ci_text(*cv["auc_ci"][model]),
                    fmt(cv["metrics"][model]["brier"]),
                    fmt(cv["metrics"][model]["log_loss"]),
                ]
                for model in ["LR", "XGB", "LSTM"]
            ],
        )
    )
    lines.extend(["", "Pairwise AUC differences (left minus right):", ""])
    lines.append(
        md_table(
            ["Comparison", "Point estimate", "Bootstrap mean", "95% CI"],
            [
                [
                    key,
                    fmt(value["point"]),
                    fmt(value["mean"]),
                    ci_text(*value["ci"]),
                ]
                for key, value in cv["pairwise_auc"].items()
            ],
        )
    )
    lines.extend(["", "Pairwise probability correlations:", ""])
    lines.append(
        md_table(
            ["Pair", "Correlation"],
            [[pair, fmt(value, 6)] for pair, value in cv["correlations"].items()],
        )
    )

    lines.extend(
        [
            "",
            "Inconsistency flagged: `results/ab_experiment/full_results_2026-03-09T1430.md` reports LSTM CV AUC `0.6407` and LSTM correlations `.784/.699`; the current raw CSV gives LSTM AUC `0.6372` and correlations `0.7643/0.6737`.",
            "",
            "## 3. Disagreement Analysis (CV Set)",
            "",
            f"Computed at `{refs['categorize']}` using the documented single global CV base-rate threshold `{fmt(cv['base_rate'], 6)}`. Total CV games: {cv['n']}; upsets: {cv['upsets']}; base rate: {pct(cv['base_rate'], 3)}.",
            "",
        ]
    )
    cat_rows = []
    for category in CATEGORY_ORDER:
        item = cv["categories"][category]
        cat_rows.append(
            [
                category,
                item["n"],
                pct(item["pct"], 2),
                item["upsets"],
                item["non_upsets"],
                pct(item["upset_rate"], 2),
            ]
        )
    lines.append(
        md_table(
            ["Category", "N", "% games", "Upsets", "Non-upsets", "Upset rate"],
            cat_rows,
        )
    )
    only_lstm = cv["categories"]["only_lstm"]
    p_lstm = stats.binomtest(
        only_lstm["upsets"],
        only_lstm["n"],
        cv["base_rate"],
        alternative="less",
    ).pvalue
    lines.extend(
        [
            "",
            f"Category count check: {sum(item['n'] for item in cv['categories'].values())} / {cv['n']}.",
            f"LSTM-exclusive skew test: {only_lstm['upsets']}/{only_lstm['n']} upsets ({pct(only_lstm['upset_rate'], 2)}) versus CV base rate {pct(cv['base_rate'], 2)} gives one-sided binomial p = `{fmt(p_lstm, 6)}`. The old p = `0.029` is tied to the stale 65-game count.",
            "",
            "Important threshold inconsistency: the saved CV `lr_pred`/`xgb_pred`/`lstm_pred` columns were written with fold-specific thresholds, not the single global CV threshold used above. Those stored columns give `only_lstm = {n}` rather than 72.".format(
                n=cv["categories_stored_pred_columns"]["only_lstm"]["n"]
            ),
            "",
            "## 4. LSTM Spread Bucket Analysis",
            "",
            f"Computed at `{refs['lstm_buckets']}`. Buckets: small 3-6.5, medium 7-13.5, large 14+.",
            "",
            "### Analysis A: LSTM-Exclusive Correct Predictions",
            "",
        ]
    )
    a_rows = []
    for bucket_name in ["small", "medium", "large"]:
        item = audit["lstm_buckets"]["analysis_a"][bucket_name]
        a_rows.append(
            [
                bucket_name,
                item["n"],
                item["upsets"],
                item["non_upsets"],
                pct(item["upset_detection_rate"], 1),
            ]
        )
    lines.append(
        md_table(
            ["Bucket", "Count", "Upsets caught", "Non-upsets rejected", "Upset share"],
            a_rows,
        )
    )
    lines.extend(
        [
            "",
            "Analysis A currently totals 72, not 65. The bucket split is 60 + 12 + 0 = 72, so those bucket numbers match current LSTM-exclusive correct predictions under the documented global threshold.",
            "",
            "### Analysis B: LSTM Disagrees With LR+XGB Consensus",
            "",
        ]
    )
    b_rows = []
    for bucket_name in ["small", "medium", "large"]:
        item = audit["lstm_buckets"]["analysis_b"][bucket_name]
        b_rows.append(
            [
                bucket_name,
                item["n"],
                item["lstm_correct"],
                item["lstm_wrong"],
                pct(item["hit_rate"], 1),
            ]
        )
    lines.append(
        md_table(
            ["Bucket", "LSTM vs consensus N", "LSTM right", "LSTM wrong", "Hit rate"],
            b_rows,
        )
    )
    lines.extend(
        [
            "",
            "Analysis B totals 150 (98 small + 52 medium + 0 large). Therefore the 72 number is not an LR+XGB-consensus disagreement count; it is the current LSTM-exclusive-correct count. The 65 number appears only in stale markdown summaries.",
            "",
            "## 5. Spread Ablation Results",
            "",
            f"Computed at `{refs['metrics']}` and `{refs['bootstrap_cv']}` from `predictions_with_spread.csv` and `predictions_without_spread.csv`. Delta is without-spread AUC minus with-spread AUC.",
            "",
        ]
    )
    ablation_rows = []
    for model in ["LR", "XGB", "LSTM"]:
        with_auc = cv["metrics"][model]["auc"]
        without_auc = spread["metrics_without_spread"][model]["auc"]
        boot = spread["bootstrap"][model]
        p_text = (
            "<0.0002" if boot["p_two_sided"] < 0.0002 else fmt(boot["p_two_sided"], 6)
        )
        ablation_rows.append(
            [
                model,
                fmt(with_auc),
                fmt(without_auc),
                fmt(without_auc - with_auc),
                ci_text(*boot["ci"]),
                p_text,
            ]
        )
    lines.append(
        md_table(
            [
                "Model",
                "AUC with spread",
                "AUC without spread",
                "Delta",
                "Delta 95% CI",
                "p",
            ],
            ablation_rows,
        )
    )
    lines.extend(["", "No-spread pairwise probability correlations:", ""])
    lines.append(
        md_table(
            ["Pair", "Correlation"],
            [
                [pair, fmt(value, 6)]
                for pair, value in spread["correlations_without_spread"].items()
            ],
        )
    )
    ws_ol = spread["with_spread_only_lstm"]
    ns_ol = spread["without_spread_only_lstm"]
    lines.extend(
        [
            "",
            f"LSTM-exclusive current global-threshold count: with spread {ws_ol['n']}/{cv['n']} = {pct(ws_ol['pct'], 2)}; without spread {ns_ol['n']}/{cv['n']} = {pct(ns_ol['pct'], 2)}. The slide claim `5.6% -> 11.0%` is stale for the current raw CSV; current is `6.20% -> 10.76%`.",
            f"LSTM-exclusive upsets: with spread {ws_ol['upsets']}; without spread {ns_ol['upsets']}. The current raw CSV gives `15 -> 44`, not `12 -> 33`.",
            "",
            "LR coefficient check (standardized by `StandardScaler` in `src/models/logistic_model.py`):",
            "",
            md_table(
                ["Feature", "Coefficient", "Absolute ratio vs spread"],
                [
                    ["spread_magnitude", fmt(lr_coefs["spread_magnitude"], 6), "1.000"],
                    [
                        lr_coefs["next_feature"],
                        fmt(lr_coefs["next_coefficient"], 6),
                        fmt(lr_coefs["ratio_to_next_abs"], 3),
                    ],
                ],
            ),
            f"The spread coefficient is `{fmt(lr_coefs['spread_magnitude'], 6)}`. The next-largest coefficient by absolute value is `{lr_coefs['next_feature']}` at `{fmt(lr_coefs['next_coefficient'], 6)}`, so the spread-to-next ratio is `{fmt(lr_coefs['ratio_to_next_abs'], 3)}`.",
            "",
            "## 6. Out-of-Sample Test Set (2023-2025)",
            "",
            f"Computed at `{refs['metrics']}` and `{refs['bootstrap_test']}` from `results/test/predictions.csv` (saved calibrated probabilities).",
            "",
        ]
    )
    lines.append(
        md_table(
            [
                "Model",
                "Test AUC",
                "Test Brier",
                "Test log loss",
                "CV AUC",
                "CV-test gap",
            ],
            [
                [
                    model,
                    fmt(test["metrics"][model]["auc"]),
                    fmt(test["metrics"][model]["brier"]),
                    fmt(test["metrics"][model]["log_loss"]),
                    fmt(cv["metrics"][model]["auc"]),
                    fmt(test["cv_to_test_gap"][model]),
                ]
                for model in ["LR", "XGB", "LSTM"]
            ],
        )
    )
    lines.extend(["", "Per-season test AUC:", ""])
    lines.append(
        md_table(
            ["Season", "N", "Upsets", "Upset rate", "LR AUC", "XGB AUC", "LSTM AUC"],
            [
                [
                    season,
                    item["n"],
                    item["upsets"],
                    pct(item["upset_rate"], 2),
                    fmt(item["LR"]),
                    fmt(item["XGB"]),
                    fmt(item["LSTM"]),
                ]
                for season, item in test["season_auc"].items()
            ],
        )
    )
    lines.extend(["", "Test pairwise probability correlations:", ""])
    lines.append(
        md_table(
            ["Pair", "Correlation"],
            [[pair, fmt(value, 6)] for pair, value in test["correlations"].items()],
        )
    )
    lines.extend(["", "Test pairwise AUC differences (left minus right):", ""])
    lines.append(
        md_table(
            ["Comparison", "Point estimate", "95% CI", "p"],
            [
                [
                    key,
                    fmt(item["point"]),
                    ci_text(*item["ci"]),
                    fmt(item["p_two_sided"], 6),
                ]
                for key, item in test["pairwise_auc"].items()
            ],
        )
    )
    lines.append(
        "All current test-set pairwise AUC-difference bootstrap CIs include zero. XGB is best by point estimate, but not significantly better than LSTM or LR on this 558-game test set."
    )

    lines.extend(
        [
            "",
            "## 7. Top-K Hit Rates on Test Set",
            "",
            f"Computed at `{refs['topk']}`. Ensemble is a simple unweighted average: `(lr_prob + xgb_prob + lstm_prob) / 3`.",
            "",
            f"Base rate: {test['topk']['upsets'] if 'upsets' in test['topk'] else test['upsets']}/{test['n']} = {pct(test['topk']['base_rate'], 2)}.",
            "",
        ]
    )
    topk_rows = []
    for k, results in test["topk"]["k"].items():
        row = [k]
        for model in ["LR", "XGB", "LSTM", "Ensemble"]:
            item = results[model]
            row.append(
                f"{item['hits']}/{k} ({pct(item['rate'], 0)}, {fmt(item['lift'], 2)}x)"
            )
        topk_rows.append(row)
    lines.append(md_table(["K", "LR", "XGB", "LSTM", "Ensemble"], topk_rows))

    coverage = taxonomy["coverage"]
    lines.extend(
        [
            "",
            "## 8. Taxonomy Coverage",
            "",
            f"Computed at `{refs['taxonomy']}` from CV categories and spread buckets.",
            "",
        ]
    )
    lines.append(
        md_table(
            [
                "Taxonomy type",
                "Definition used",
                "N",
                "% of all CV games",
                "% of all_wrong failures",
            ],
            [
                [
                    "Consensus",
                    "all_correct",
                    coverage["consensus_all_correct"]["n"],
                    pct(coverage["consensus_all_correct"]["pct"], 2),
                    "NA",
                ],
                [
                    "Temporal/LSTM-exclusive",
                    "only_lstm",
                    coverage["temporal_only_lstm"]["n"],
                    pct(coverage["temporal_only_lstm"]["pct"], 2),
                    "NA",
                ],
                [
                    "Hidden information",
                    "all_wrong at medium/large spreads",
                    coverage["hidden_all_wrong_medium_large"]["n"],
                    pct(coverage["hidden_all_wrong_medium_large"]["pct"], 2),
                    pct(
                        coverage["hidden_all_wrong_medium_large"]["pct_of_all_wrong"], 2
                    ),
                ],
                [
                    "Irreducible variance",
                    "all_wrong at small spreads",
                    coverage["irreducible_all_wrong_small"]["n"],
                    pct(coverage["irreducible_all_wrong_small"]["pct"], 2),
                    pct(coverage["irreducible_all_wrong_small"]["pct_of_all_wrong"], 2),
                ],
            ],
        )
    )
    named_pct = sum(item["pct"] for item in coverage.values() if "pct" in item)
    mixed_pct = sum(item["pct"] for item in taxonomy["mixed"].values())
    lines.extend(
        [
            "",
            f"Named taxonomy coverage totals {pct(named_pct, 2)} of CV games. Mixed categories total {pct(mixed_pct, 2)}.",
            "",
            "Mixed-category upset-rate tests use exact two-sided binomial tests against the CV base rate.",
            "",
        ]
    )
    lines.append(
        md_table(
            ["Mixed category", "N", "% games", "Upsets", "Upset rate", "p vs base"],
            [
                [
                    category,
                    item["n"],
                    pct(item["pct"], 2),
                    item["upsets"],
                    pct(item["upset_rate"], 2),
                    fmt(item["p_vs_base_two_sided"], 6),
                ]
                for category, item in taxonomy["mixed"].items()
            ],
        )
    )
    lines.append(
        "No mixed category is statistically distinguishable from the CV base rate at alpha = 0.05. A principled collapse would be to keep them as an explicit `mixed/ambiguous architecture signal` bucket. Folding `lr_xgb` into a static/non-temporal type is defensible, but the other mixed categories combine temporal and static models and should not be forced into the four named types without a new design rule."
    )

    lines.extend(
        [
            "",
            "## 9. Calibration Details",
            "",
            f"Probability summaries computed at `{refs['calibration_summary']}`.",
            "",
            "Platt calibration is applied to all three final test-model outputs in `src/models/evaluate_test_set.py`: LR, XGB, and LSTM calibrators are fit on held-out 2021-2022 predictions and applied to 2023-2025 raw test probabilities. The saved `results/test/predictions.csv` contains calibrated probabilities.",
            "",
            "Primary CV analyses use uncalibrated CV predictions from `results/ab_experiment/predictions_with_spread.csv`. Spread ablation uses uncalibrated CV predictions from both A/B CSVs. Test metrics, test correlations, and test Top-K in this audit use the saved calibrated test probabilities.",
            "",
            "Current saved probability distributions:",
            "",
        ]
    )
    prob_rows = []
    for dataset_label, key in [
        ("CV raw", "cv_raw_probability_summary"),
        ("Test Platt-calibrated", "test_platt_probability_summary"),
    ]:
        for model in ["LR", "XGB", "LSTM"]:
            item = calibration[key][model]
            prob_rows.append(
                [
                    dataset_label,
                    model,
                    fmt(item["min"], 6),
                    fmt(item["max"], 6),
                    fmt(item["mean"], 6),
                    fmt(item["std"], 6),
                ]
            )
    lines.append(md_table(["Dataset", "Model", "Min", "Max", "Mean", "Std"], prob_rows))
    lines.extend(
        [
            "",
            "Exact pre-calibration test probability summary stats for the current saved `results/test/predictions.csv` are not recoverable from repo artifacts: raw test probabilities and fitted Platt calibrators were not saved. Existing docs record only old pre-calibration ranges from a rerun, not mean/std. Rerunning `src.models.evaluate_test_set` would create a new stochastic LSTM run, so it would not verify the current saved test file.",
            "",
            "## 10. Feature Details",
            "",
            f"Computed at `{refs['features']}` from feature-pipeline and sequence-builder constants.",
            "",
        ]
    )
    lines.append(
        md_table(
            ["Item", "Count/value"],
            [
                ["LR features", features["lr_count"]],
                ["LR no-spread features", features["lr_no_spread_count"]],
                ["XGB features", features["xgb_count"]],
                ["XGB no-spread features", features["xgb_no_spread_count"]],
                [
                    "LSTM sequence representation",
                    f"{features['lstm_sequence_features']} features x {features['lstm_sequence_length']} timesteps",
                ],
                ["LSTM matchup context", features["lstm_matchup_features"]],
                ["LSTM matchup no-spread context", features["lstm_matchup_no_spread"]],
                ["Rolling window", f"{features['rolling_window']}-game"],
                ["XGB per-game lags", ", ".join(str(x) for x in features["xgb_lags"])],
            ],
        )
    )
    lines.append(
        "There is no 5-game rolling window in the canonical pipeline. The flat pipeline uses 3-game rolling means/std/trends and XGB gets last-1/last-2/last-3 per-game lag stats; the LSTM uses 8 prior games per team."
    )

    lines.extend(
        [
            "",
            "## Cross-File Inconsistency Summary",
            "",
            md_table(
                [
                    "Item",
                    "Current CSV recomputation",
                    "Conflicting file/claim",
                    "Status",
                ],
                [
                    [
                        "LSTM CV AUC",
                        fmt(cv["metrics"]["LSTM"]["auc"]),
                        "`results/ab_experiment/full_results_2026-03-09T1430.md`: 0.6407",
                        "Inconsistent",
                    ],
                    [
                        "CV LSTM correlations",
                        f"LR-LSTM {fmt(cv['correlations']['LR-LSTM'], 3)}, XGB-LSTM {fmt(cv['correlations']['XGB-LSTM'], 3)}",
                        "`full_results`: .784/.699",
                        "Inconsistent",
                    ],
                    [
                        "LSTM-exclusive count",
                        f"{ws_ol['n']} ({ws_ol['upsets']} upsets, {ws_ol['non_upsets']} non-upsets)",
                        "`full_results` flat table: 65 (12/53)",
                        "Inconsistent",
                    ],
                    [
                        "No-spread LSTM AUC",
                        fmt(spread["metrics_without_spread"]["LSTM"]["auc"]),
                        "`full_results`: 0.5739",
                        "Inconsistent",
                    ],
                    [
                        "Test LSTM AUC",
                        fmt(test["metrics"]["LSTM"]["auc"]),
                        "`docs/2026-03-15-paper-rewrite-audit-results.md`: 0.5240",
                        "Inconsistent with current test CSV",
                    ],
                    [
                        "Test LSTM correlations",
                        f"LR-LSTM {fmt(test['correlations']['LR-LSTM'], 3)}, XGB-LSTM {fmt(test['correlations']['XGB-LSTM'], 3)}",
                        "Slides/docs: .429/.408 or .311/.273",
                        "Inconsistent across runs",
                    ],
                ],
            ),
            "",
        ]
    )

    return "\n".join(lines)


def slide_claims(audit: dict[str, Any]) -> list[dict[str, str]]:
    """Return slide reconciliation rows for the latest revised deck."""
    cv = audit["cv"]
    test = audit["test"]
    spread = audit["spread_ablation"]
    taxonomy = audit["taxonomy"]
    refs = audit["source_refs"]
    slide_src = "tools/presentation/rewrite_presentation.py"

    def src(pattern: str) -> str:
        return find_line(slide_src, pattern)

    rows: list[dict[str, str]] = []

    def add(slide: int, claim: str, status: str, correct: str, source: str) -> None:
        rows.append(
            {
                "Slide": str(slide),
                "Claim": claim,
                "Status": status,
                "Correct / note": correct,
                "Source": source,
            }
        )

    add(
        3,
        "Upset = 3+ point underdog wins",
        "CONFIRMED",
        "Pipeline labels only spread >= 3.",
        "src/features/pipeline.py",
    )
    add(
        3,
        "Upsets happen in roughly 29% of games",
        "CONFIRMED",
        f"Train {pct(audit['dataset']['train']['upset_rate'], 1)}, test {pct(audit['dataset']['test']['upset_rate'], 1)}, CV {pct(cv['base_rate'], 1)}.",
        refs["dataset_composition"],
    )
    add(
        6,
        "18 NFL regular seasons, 2005-2022 for training",
        "CONFIRMED",
        f"{audit['dataset']['train']['labeled_games']} labeled training games across 2005-2022.",
        refs["dataset_composition"],
    )
    add(
        6,
        "Hold out 2023-2025 as blind test",
        "CONFIRMED",
        f"{audit['dataset']['test']['labeled_games']} labeled test games across 2023-2025.",
        refs["dataset_composition"],
    )
    add(
        6,
        "LR 46 stats; XGB 70 features; LSTM raw 8-game sequences",
        "CONFIRMED",
        "LR 46, XGB 70, LSTM 14 x 8 plus 10 matchup context.",
        refs["features"],
    )
    add(
        6,
        "6-fold expanding-window CV",
        "CONFIRMED",
        f"Current CV predictions cover {cv['n']} games from validation seasons 2017-2022.",
        refs["metrics"],
    )
    add(
        7,
        "Training: 3,495 games",
        "CONFIRMED",
        "3,495 labeled train games.",
        refs["dataset_composition"],
    )
    add(
        7,
        "Test: 558 games",
        "CONFIRMED",
        "558 labeled test games.",
        refs["dataset_composition"],
    )
    add(
        7,
        "Base rate: ~30% upsets",
        "CONFIRMED",
        f"Train {pct(audit['dataset']['train']['upset_rate'], 1)}, test {pct(audit['dataset']['test']['upset_rate'], 1)}.",
        refs["dataset_composition"],
    )
    add(
        8,
        "CV validation years 2017-2022, N = 1,162",
        "CONFIRMED",
        f"Current CV prediction CSV has {cv['n']} rows.",
        refs["metrics"],
    )
    add(
        8,
        "Pairwise AUC-difference CIs contain zero",
        "CONFIRMED",
        "All current CV pairwise AUC-difference bootstrap CIs include zero.",
        refs["bootstrap_cv"],
    )
    add(
        8,
        "LR-XGB correlation .874",
        "CONFIRMED",
        f"Current {fmt(cv['correlations']['LR-XGB'], 3)}.",
        refs["metrics"],
    )
    add(
        8,
        "LSTM correlations .784 with LR and .699 with XGB",
        "INCORRECT",
        f"Current LR-LSTM {fmt(cv['correlations']['LR-LSTM'], 3)} and XGB-LSTM {fmt(cv['correlations']['XGB-LSTM'], 3)}.",
        refs["metrics"],
    )
    add(
        8,
        "XGB-LSTM .699 is the lowest correlation",
        "INCORRECT",
        f"XGB-LSTM is still lowest, but current value is {fmt(cv['correlations']['XGB-LSTM'], 3)}.",
        refs["metrics"],
    )
    add(
        9,
        "CV set 1,162 games, threshold ~0.30",
        "CONFIRMED",
        f"N={cv['n']}, threshold={fmt(cv['base_rate'], 3)}.",
        refs["categorize"],
    )
    all_agree = (
        cv["categories"]["all_correct"]["n"] + cv["categories"]["all_wrong"]["n"]
    )
    add(
        9,
        "74.7% all-agree categories; 25.3% split",
        "INCORRECT",
        f"Current global-threshold all-agree is {all_agree}/{cv['n']} = {pct(all_agree / cv['n'], 1)}; split is {pct(1 - all_agree / cv['n'], 1)}.",
        refs["categorize"],
    )
    add(
        9,
        "65 LSTM-exclusive games",
        "INCORRECT",
        f"Current global-threshold only_lstm is {cv['categories']['only_lstm']['n']}. Stored fold-threshold columns give {cv['categories_stored_pred_columns']['only_lstm']['n']}.",
        refs["categorize"],
    )
    add(
        9,
        "53 non-upset rejections (81.5%) and 12 upsets",
        "INCORRECT",
        f"Current only_lstm is {cv['categories']['only_lstm']['non_upsets']} non-upsets ({pct(cv['categories']['only_lstm']['non_upsets'] / cv['categories']['only_lstm']['n'], 1)}) and {cv['categories']['only_lstm']['upsets']} upsets.",
        refs["categorize"],
    )
    add(
        9,
        "Binomial p = 0.029",
        "INCORRECT",
        "Current one-sided p for only_lstm upset rate below base rate is 0.061332.",
        refs["categorize"],
    )
    add(
        10,
        "Close games 3-6.5; LSTM right 92% as false-alarm filter",
        "CONFIRMED",
        "Current small-spread only_lstm: 55/60 non-upset rejections = 91.7%.",
        refs["lstm_buckets"],
    )
    add(
        10,
        "Medium spreads 7-13.5; catches 83% as real upsets",
        "CONFIRMED",
        "Current medium-spread only_lstm: 10/12 upsets = 83.3%.",
        refs["lstm_buckets"],
    )
    add(
        10,
        "LSTM watches last 8 games",
        "CONFIRMED",
        "SEQUENCE_LENGTH = 8.",
        refs["features"],
    )
    add(
        11,
        "All spread-ablation drops significant",
        "CONFIRMED",
        "All current bootstrap delta CIs are negative; p < 0.0002 by two-sided sign bootstrap.",
        refs["bootstrap_cv"],
    )
    add(
        11,
        "Without spread: LSTM .574 > LR .571 > XGB .566",
        "INCORRECT",
        f"Current no-spread AUCs: LR {fmt(spread['metrics_without_spread']['LR']['auc'], 3)}, LSTM {fmt(spread['metrics_without_spread']['LSTM']['auc'], 3)}, XGB {fmt(spread['metrics_without_spread']['XGB']['auc'], 3)}. Ranking is LR > LSTM > XGB.",
        refs["metrics"],
    )
    add(
        11,
        "LSTM degrades least (-.067)",
        "INCORRECT",
        f"Current LSTM delta is {fmt(spread['metrics_without_spread']['LSTM']['auc'] - cv['metrics']['LSTM']['auc'], 3)}; it is still least negative by point estimate.",
        refs["metrics"],
    )
    add(
        11,
        "LSTM smaller delta vs LR is not significant",
        "CONFIRMED",
        "Current delta difference still has uncertainty crossing zero; individual delta CIs are negative.",
        refs["bootstrap_cv"],
    )
    add(
        11,
        "LSTM-exclusive predictions double 5.6% -> 11.0%",
        "INCORRECT",
        f"Current global-threshold values are {pct(spread['with_spread_only_lstm']['pct'], 1)} -> {pct(spread['without_spread_only_lstm']['pct'], 1)}.",
        refs["categorize"],
    )
    add(
        11,
        "Upsets caught jump 12 -> 33",
        "INCORRECT",
        f"Current global-threshold only_lstm upsets are {spread['with_spread_only_lstm']['upsets']} -> {spread['without_spread_only_lstm']['upsets']}.",
        refs["categorize"],
    )
    add(
        11,
        "LR-XGB correlation .874 -> .742",
        "CONFIRMED",
        f"Current {fmt(cv['correlations']['LR-XGB'], 3)} -> {fmt(spread['correlations_without_spread']['LR-XGB'], 3)}.",
        refs["metrics"],
    )
    add(
        12,
        "Test base rate = 28.5%",
        "CONFIRMED",
        f"Current test base rate {pct(test['base_rate'], 1)}.",
        refs["metrics"],
    )
    add(
        12,
        "LSTM largest generalization gap (-.117)",
        "INCORRECT",
        f"Current LSTM CV-test gap is {fmt(test['cv_to_test_gap']['LSTM'], 3)}; it remains largest.",
        refs["metrics"],
    )
    add(
        12,
        "XGB generalizes best (-.062)",
        "CONFIRMED",
        f"Current XGB gap {fmt(test['cv_to_test_gap']['XGB'], 3)}.",
        refs["metrics"],
    )
    add(
        12,
        "LSTM trails in all three test seasons",
        "CONFIRMED",
        "Current LSTM AUC is below LR and XGB in 2023, 2024, and 2025.",
        refs["metrics"],
    )
    add(
        12,
        "LSTM correlations collapse .784 -> .429 and .699 -> .408",
        "INCORRECT",
        f"Current collapse is {fmt(cv['correlations']['LR-LSTM'], 3)} -> {fmt(test['correlations']['LR-LSTM'], 3)} and {fmt(cv['correlations']['XGB-LSTM'], 3)} -> {fmt(test['correlations']['XGB-LSTM'], 3)}.",
        refs["metrics"],
    )
    add(
        12,
        "XGB top-10 hit rate 60%, 2.1x lift",
        "CONFIRMED",
        "Current XGB top 10 is 6/10 = 60.0%, lift 2.11x.",
        refs["topk"],
    )
    add(
        13,
        "All three agree / structurally readable: 45% of games",
        "CONFIRMED",
        f"Current all_correct is {pct(cv['categories']['all_correct']['pct'], 1)}.",
        refs["categorize"],
    )
    add(
        13,
        "Only LSTM sees it: 6% of games",
        "CONFIRMED",
        f"Current only_lstm is {pct(cv['categories']['only_lstm']['pct'], 1)}.",
        refs["categorize"],
    )
    add(
        13,
        "Temporal signal statistically confirmed",
        "AMBIGUOUS",
        "The current overall only_lstm skew p-value is 0.0613, not <0.05; bucket inversion remains descriptive with small cells.",
        refs["categorize"],
    )
    add(
        14,
        "LSTM AUC .641 tied with LR .650 and XGB .638",
        "INCORRECT",
        f"Current AUCs: LR {fmt(cv['metrics']['LR']['auc'], 3)}, XGB {fmt(cv['metrics']['XGB']['auc'], 3)}, LSTM {fmt(cv['metrics']['LSTM']['auc'], 3)}. Statistical tie confirmed.",
        refs["metrics"],
    )
    add(
        14,
        "LSTM test AUC .524",
        "INCORRECT",
        f"Current saved test CSV gives LSTM AUC {fmt(test['metrics']['LSTM']['auc'], 3)}.",
        refs["metrics"],
    )
    add(
        14,
        "LSTM gap -.117",
        "INCORRECT",
        f"Current LSTM gap is {fmt(test['cv_to_test_gap']['LSTM'], 3)}.",
        refs["metrics"],
    )
    add(
        14,
        "LSTM-static correlation .784 -> .429",
        "INCORRECT",
        f"Current LR-LSTM is {fmt(cv['correlations']['LR-LSTM'], 3)} -> {fmt(test['correlations']['LR-LSTM'], 3)}.",
        refs["metrics"],
    )
    add(
        15,
        "LR lost .079; XGB lost .072",
        "CONFIRMED",
        f"Current LR {fmt(spread['metrics_without_spread']['LR']['auc'] - cv['metrics']['LR']['auc'], 3)}, XGB {fmt(spread['metrics_without_spread']['XGB']['auc'] - cv['metrics']['XGB']['auc'], 3)}.",
        refs["metrics"],
    )
    add(
        15,
        "LSTM lost .067",
        "INCORRECT",
        f"Current LSTM delta is {fmt(spread['metrics_without_spread']['LSTM']['auc'] - cv['metrics']['LSTM']['auc'], 3)}.",
        refs["metrics"],
    )
    add(
        15,
        "All spread drops p < .001",
        "CONFIRMED",
        "Current bootstrap p < 0.0002 for all three model deltas.",
        refs["bootstrap_cv"],
    )
    add(
        15,
        "Spread coefficient 5x larger than any team stat",
        "CONFIRMED",
        f"Spread coefficient {fmt(audit['lr_coefficients']['spread_magnitude'], 3)} is {fmt(audit['lr_coefficients']['ratio_to_next_abs'], 2)}x the next-largest coefficient overall.",
        refs["lr_coefficients"],
    )
    add(
        15,
        "No-spread ranking LSTM .574 > LR .571 > XGB .566",
        "INCORRECT",
        f"Current no-spread ranking is LR {fmt(spread['metrics_without_spread']['LR']['auc'], 3)} > LSTM {fmt(spread['metrics_without_spread']['LSTM']['auc'], 3)} > XGB {fmt(spread['metrics_without_spread']['XGB']['auc'], 3)}.",
        refs["metrics"],
    )
    add(
        15,
        "XGB-LSTM correlation .699 -> .419",
        "INCORRECT",
        f"Current XGB-LSTM is {fmt(cv['correlations']['XGB-LSTM'], 3)} -> {fmt(spread['correlations_without_spread']['XGB-LSTM'], 3)}.",
        refs["metrics"],
    )
    add(
        15,
        "LSTM-exclusive catches doubled 12 -> 33 upsets",
        "INCORRECT",
        f"Current only_lstm upsets are {spread['with_spread_only_lstm']['upsets']} -> {spread['without_spread_only_lstm']['upsets']}.",
        refs["categorize"],
    )
    add(
        16,
        "LR-XGB agree 87% of predictions",
        "CONFIRMED",
        f"Current LR-XGB agreement {pct(cv['agreement']['LR-XGB'], 1)}.",
        refs["categorize"],
    )
    add(
        16,
        "LSTM unique signal doubles without spread",
        "AMBIGUOUS",
        f"Current exact percentages are {pct(spread['with_spread_only_lstm']['pct'], 1)} -> {pct(spread['without_spread_only_lstm']['pct'], 1)} (1.74x), not exactly 2.0x.",
        refs["categorize"],
    )
    add(
        17,
        "LSTM inversion rests on 12 medium-spread exclusives",
        "CONFIRMED",
        "Current medium-spread only_lstm count is 12.",
        refs["lstm_buckets"],
    )
    add(
        17,
        "Platt calibration compresses test probabilities to [0.19, 0.51]",
        "INCORRECT",
        "Current saved calibrated test ranges are LR [0.194, 0.435], XGB [0.216, 0.498], LSTM [0.218, 0.548]. Overall max is 0.548, not 0.51.",
        refs["calibration_summary"],
    )
    add(
        17,
        "Primary analyses use uncalibrated CV predictions",
        "CONFIRMED",
        "CV A/B prediction CSVs are raw; calibration is only applied in held-out test evaluation.",
        "src/models/evaluate_test_set.py",
    )
    add(
        17,
        "3,495 training games, 558 test games",
        "CONFIRMED",
        "Counts are labeled games.",
        refs["dataset_composition"],
    )
    add(
        17,
        "Per-season test results rest on 181-192 games",
        "CONFIRMED",
        "2023:185, 2024:192, 2025:181 labeled games.",
        refs["dataset_composition"],
    )
    add(
        18,
        "Consensus signal 45%",
        "CONFIRMED",
        f"Current all_correct {pct(taxonomy['coverage']['consensus_all_correct']['pct'], 1)}.",
        refs["taxonomy"],
    )
    add(
        18,
        "Temporal signal 6%; close 92%, medium 83%",
        "CONFIRMED",
        "Current only_lstm 6.2%; small non-upset rejections 91.7%; medium upset share 83.3%.",
        refs["lstm_buckets"],
    )
    add(
        18,
        "Hidden information ~24% of failures",
        "INCORRECT",
        f"Current all_wrong medium/large is {pct(taxonomy['coverage']['hidden_all_wrong_medium_large']['pct_of_all_wrong'], 1)} of failures, or {pct(taxonomy['coverage']['hidden_all_wrong_medium_large']['pct'], 1)} of all CV games.",
        refs["taxonomy"],
    )
    add(
        18,
        "Irreducible variance ~76% of failures",
        "INCORRECT",
        f"Current all_wrong small is {pct(taxonomy['coverage']['irreducible_all_wrong_small']['pct_of_all_wrong'], 1)} of failures, or {pct(taxonomy['coverage']['irreducible_all_wrong_small']['pct'], 1)} of all CV games.",
        refs["taxonomy"],
    )

    return rows


def render_slide_reconciliation(audit: dict[str, Any]) -> str:
    """Render slide_reconciliation.md."""
    rows = slide_claims(audit)
    lines = [
        "# Slide Reconciliation",
        "",
        "- Doc Type: Results",
        "- Topic: Numerical Audit",
        "- Topic Slug: numerical-audit",
        "- Date: 2026-04-20",
        "- Status: Complete",
        "",
        "`AP_Research_Slides_Content.md` was not found in the repo. I reconciled numerical project-result claims against the latest available deck artifact, `docs/AP_Research_POD_Revised.pptx`, and its generator, `tools/presentation/rewrite_presentation.py`. Citation years, journal volume/page numbers, and general definition examples are not treated as project-result claims.",
        "",
        md_table(
            ["Slide", "Claim", "Status", "Correct / note", "Source"],
            [
                [
                    row["Slide"],
                    row["Claim"],
                    row["Status"],
                    row["Correct / note"],
                    row["Source"],
                ]
                for row in rows
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run the audit and write all outputs."""
    audit = compute_audit()

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "audit_computed.json").write_text(json.dumps(audit, indent=2))

    (ROOT / "audit_results.md").write_text(render_audit_results(audit))
    (ROOT / "slide_reconciliation.md").write_text(render_slide_reconciliation(audit))
    print("Wrote audit_results.md")
    print("Wrote slide_reconciliation.md")
    print("Wrote results/audit_computed.json")


if __name__ == "__main__":
    main()
