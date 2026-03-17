"""Tests for shared prediction formatting helpers."""

from __future__ import annotations

import numpy as np

from src.models.prediction_utils import safe_team_str


def test_safe_team_str_handles_missing_values():
    assert safe_team_str(np.nan) == ""
    assert safe_team_str(None) == ""
    assert safe_team_str("KC") == "KC"
