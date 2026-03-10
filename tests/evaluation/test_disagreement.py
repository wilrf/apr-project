"""Tests for DisagreementAnalyzer (C2)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.disagreement import (
    DisagreementAnalyzer,
    PredictionCategory,
)
from src.models.unified_trainer import GamePrediction


def _pred(
    lr: float,
    xgb: float,
    lstm: float,
    y: int,
    gid: str = "g",
    spread: float = 7.0,
) -> GamePrediction:
    return GamePrediction(
        game_id=gid,
        season=2023,
        week=5,
        underdog="BUF",
        favorite="KC",
        spread_magnitude=spread,
        y_true=y,
        lr_prob=lr,
        xgb_prob=xgb,
        lstm_prob=lstm,
    )


# ── Threshold computation ───────────────────────────────────────────


class TestAutoThreshold:
    def test_default_equals_base_rate(self):
        preds = [
            _pred(0.5, 0.5, 0.5, y, gid=f"g{i}") for i, y in enumerate([1, 1, 0, 0, 0])
        ]
        analyzer = DisagreementAnalyzer(preds)
        assert analyzer.threshold == pytest.approx(0.4)

    def test_explicit_override(self):
        preds = [_pred(0.5, 0.5, 0.5, 1), _pred(0.5, 0.5, 0.5, 0)]
        assert DisagreementAnalyzer(preds, threshold=0.6).threshold == 0.6

    def test_empty_defaults_to_half(self):
        assert DisagreementAnalyzer([]).threshold == 0.5


# ── Categorization ──────────────────────────────────────────────────


class TestCategorization:
    T = 0.3  # explicit threshold for all tests

    def _single(self, lr, xgb, lstm, y):
        return DisagreementAnalyzer(
            [_pred(lr, xgb, lstm, y, gid="x")], threshold=self.T
        ).categorize_all()

    def test_all_correct_upset(self):
        cats = self._single(0.4, 0.5, 0.35, y=1)
        assert len(cats[PredictionCategory.ALL_CORRECT]) == 1

    def test_all_correct_no_upset(self):
        cats = self._single(0.1, 0.2, 0.15, y=0)
        assert len(cats[PredictionCategory.ALL_CORRECT]) == 1

    def test_all_wrong(self):
        cats = self._single(0.4, 0.5, 0.35, y=0)
        assert len(cats[PredictionCategory.ALL_WRONG]) == 1

    def test_only_lr(self):
        cats = self._single(0.4, 0.2, 0.1, y=1)
        assert len(cats[PredictionCategory.ONLY_LR]) == 1

    def test_only_xgb(self):
        cats = self._single(0.2, 0.4, 0.1, y=1)
        assert len(cats[PredictionCategory.ONLY_XGB]) == 1

    def test_only_lstm(self):
        cats = self._single(0.2, 0.1, 0.4, y=1)
        assert len(cats[PredictionCategory.ONLY_LSTM]) == 1

    def test_lr_xgb(self):
        cats = self._single(0.4, 0.5, 0.1, y=1)
        assert len(cats[PredictionCategory.LR_XGB]) == 1

    def test_lr_lstm(self):
        cats = self._single(0.4, 0.1, 0.5, y=1)
        assert len(cats[PredictionCategory.LR_LSTM]) == 1

    def test_xgb_lstm(self):
        cats = self._single(0.1, 0.4, 0.5, y=1)
        assert len(cats[PredictionCategory.XGB_LSTM]) == 1

    def test_all_eight_categories(self):
        preds = [
            _pred(0.4, 0.5, 0.35, 1, gid="ac"),  # ALL_CORRECT
            _pred(0.4, 0.5, 0.35, 0, gid="aw"),  # ALL_WRONG
            _pred(0.4, 0.2, 0.1, 1, gid="lr"),  # ONLY_LR
            _pred(0.2, 0.4, 0.1, 1, gid="xgb"),  # ONLY_XGB
            _pred(0.2, 0.1, 0.4, 1, gid="lstm"),  # ONLY_LSTM
            _pred(0.4, 0.5, 0.1, 1, gid="lx"),  # LR_XGB
            _pred(0.4, 0.1, 0.5, 1, gid="ll"),  # LR_LSTM
            _pred(0.1, 0.4, 0.5, 1, gid="xl"),  # XGB_LSTM
        ]
        cats = DisagreementAnalyzer(preds, threshold=self.T).categorize_all()
        for cat in PredictionCategory:
            assert len(cats[cat]) == 1, f"{cat} should have exactly 1 prediction"

    def test_caching(self):
        a = DisagreementAnalyzer([_pred(0.4, 0.5, 0.35, 1)], threshold=self.T)
        assert a.categorize_all() is a.categorize_all()


# ── Agreement matrix ────────────────────────────────────────────────


class TestAgreementMatrix:
    def test_uses_analyzer_threshold(self):
        # At threshold 0.3: LR=upset(0.35≥0.3), XGB=no(0.25<0.3), LSTM=upset(0.35≥0.3)
        preds = [_pred(0.35, 0.25, 0.35, 1, gid="g1")]
        matrix = DisagreementAnalyzer(preds, threshold=0.3).get_agreement_matrix()

        lr_xgb = matrix.loc[matrix["model_pair"] == "LR-XGB", "agreement_rate"].iloc[0]
        lr_lstm = matrix.loc[matrix["model_pair"] == "LR-LSTM", "agreement_rate"].iloc[
            0
        ]
        assert lr_xgb == 0.0  # disagree
        assert lr_lstm == 1.0  # agree

    def test_all_three_row(self):
        preds = [
            _pred(0.5, 0.5, 0.5, 1, gid="g1"),
            _pred(0.1, 0.1, 0.1, 0, gid="g2"),
        ]
        matrix = DisagreementAnalyzer(preds, threshold=0.3).get_agreement_matrix()
        all_row = matrix.loc[
            matrix["model_pair"] == "All Three", "agreement_rate"
        ].iloc[0]
        assert all_row == 1.0

    def test_empty(self):
        assert DisagreementAnalyzer([]).get_agreement_matrix().empty


# ── Category stats ──────────────────────────────────────────────────


class TestCategoryStats:
    def test_counts_and_rates(self):
        preds = [
            _pred(0.4, 0.5, 0.35, 1, gid="g1", spread=7.0),  # ALL_CORRECT
            _pred(0.4, 0.5, 0.35, 0, gid="g2", spread=3.0),  # ALL_WRONG
            _pred(0.1, 0.2, 0.1, 0, gid="g3", spread=5.0),  # ALL_CORRECT
        ]
        stats = DisagreementAnalyzer(preds, threshold=0.3).get_category_stats()
        ac = stats[stats["category"] == "all_correct"]
        assert ac.iloc[0]["count"] == 2
        assert ac.iloc[0]["upset_rate"] == pytest.approx(0.5)

    def test_pct_sums_to_100(self):
        preds = [_pred(0.4, 0.5, 0.35, 1, gid=f"g{i}") for i in range(10)]
        stats = DisagreementAnalyzer(preds, threshold=0.3).get_category_stats()
        assert stats["pct_of_total"].sum() == pytest.approx(100.0)


# ── Correlation matrix ──────────────────────────────────────────────


class TestCorrelationMatrix:
    def test_shape_symmetry_diagonal(self):
        preds = [
            _pred(0.4, 0.5, 0.35, 1, gid="a"),
            _pred(0.2, 0.3, 0.8, 0, gid="b"),
            _pred(0.6, 0.1, 0.4, 1, gid="c"),
        ]
        corr = DisagreementAnalyzer(preds).get_correlation_matrix()
        assert corr.shape == (3, 3)
        np.testing.assert_allclose(np.diag(corr.values), 1.0)
        np.testing.assert_allclose(corr.values, corr.values.T)

    def test_empty(self):
        assert DisagreementAnalyzer([]).get_correlation_matrix().empty


# ── Exclusive insights ──────────────────────────────────────────────


class TestExclusiveInsights:
    def test_one_per_exclusive_category(self):
        preds = [
            _pred(0.4, 0.2, 0.1, 1, gid="lr_g"),
            _pred(0.2, 0.4, 0.1, 1, gid="xgb_g"),
            _pred(0.2, 0.1, 0.4, 1, gid="lstm_g"),
        ]
        ins = DisagreementAnalyzer(preds, threshold=0.3).get_exclusive_insights()
        assert set(ins.keys()) == {
            PredictionCategory.ONLY_LR,
            PredictionCategory.ONLY_XGB,
            PredictionCategory.ONLY_LSTM,
        }
        assert ins[PredictionCategory.ONLY_LR].example_games == ["lr_g"]

    def test_empty_when_no_exclusive(self):
        preds = [_pred(0.5, 0.5, 0.5, 1, gid="g")]
        assert (
            len(DisagreementAnalyzer(preds, threshold=0.3).get_exclusive_insights())
            == 0
        )


# ── CSV export ──────────────────────────────────────────────────────


class TestExportTable:
    def test_uses_analyzer_threshold(self):
        preds = [_pred(0.35, 0.25, 0.35, 1, gid="g1")]
        analyzer = DisagreementAnalyzer(preds, threshold=0.3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.csv"
            analyzer.export_table(p)
            df = pd.read_csv(p)
            assert df.iloc[0]["lr_pred"] == 1  # 0.35 >= 0.3
            assert df.iloc[0]["xgb_pred"] == 0  # 0.25 < 0.3
            assert df.iloc[0]["lstm_pred"] == 1  # 0.35 >= 0.3

    def test_expected_columns(self):
        preds = [_pred(0.5, 0.5, 0.5, 1, gid="g1")]
        analyzer = DisagreementAnalyzer(preds, threshold=0.3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.csv"
            analyzer.export_table(p)
            cols = set(pd.read_csv(p).columns)
            assert cols == {
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
                "lr_pred",
                "xgb_pred",
                "lstm_pred",
                "lr_correct",
                "xgb_correct",
                "lstm_correct",
                "category",
            }


# ── Summarize ───────────────────────────────────────────────────────


class TestSummarize:
    def test_string_output(self):
        preds = [_pred(0.4, 0.5, 0.35, 1, gid="g1"), _pred(0.1, 0.2, 0.1, 0, gid="g2")]
        s = DisagreementAnalyzer(preds, threshold=0.3).summarize()
        assert "MULTI-MODEL DISAGREEMENT ANALYSIS" in s
        assert "Total predictions: 2" in s

    def test_single_prediction(self):
        s = DisagreementAnalyzer([_pred(0.4, 0.5, 0.35, 1)], threshold=0.3).summarize()
        assert "Total predictions: 1" in s
