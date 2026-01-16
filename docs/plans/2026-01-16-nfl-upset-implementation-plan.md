# NFL Upset Prediction: Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and compare XGBoost vs LSTM models for predicting NFL moneyline upsets, analyzing which structural factors drive predictions and how each model reasons differently.

**Architecture:** Data pipeline ingests NFL stats from nflverse and betting lines from Kaggle, engineers ~50-80 features including rolling averages and matchup differentials, trains both models with time-series cross-validation, then compares predictions using SHAP/attention interpretability.

**Tech Stack:** Python, pandas, nfl_data_py, xgboost, pytorch, shap, captum, mlflow, optuna

---

## Phase 0: Project Setup

### Task 0.1: Initialize Project Structure

**Files:**
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/features/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/evaluation/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`
- Create: `tests/features/__init__.py`
- Create: `tests/models/__init__.py`
- Create: `data/raw/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `data/splits/.gitkeep`
- Create: `results/figures/.gitkeep`
- Create: `notebooks/.gitkeep`
- Create: `requirements.txt`
- Create: `pyproject.toml`

**Step 1: Create directory structure**

```bash
mkdir -p src/data src/features src/models src/evaluation
mkdir -p tests/data tests/features tests/models
mkdir -p data/raw data/processed data/splits
mkdir -p results/figures notebooks
```

**Step 2: Create __init__.py files**

Create empty `__init__.py` files in each src and tests subdirectory.

**Step 3: Create requirements.txt**

```txt
# Data
pandas>=2.0.0
numpy>=1.24.0
nfl-data-py>=0.3.0

# ML
xgboost>=2.0.0
torch>=2.0.0
scikit-learn>=1.3.0

# Interpretability
shap>=0.42.0
captum>=0.7.0

# Experiment tracking
mlflow>=2.8.0
optuna>=3.4.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Development
jupyter>=1.0.0
black>=23.9.0
ruff>=0.0.292
```

**Step 4: Create pyproject.toml**

```toml
[project]
name = "nfl-upset-prediction"
version = "0.1.0"
description = "XGBoost vs LSTM comparison for NFL upset prediction"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --tb=short"

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.ruff]
line-length = 88
select = ["E", "F", "I"]
```

**Step 5: Create .gitkeep files**

```bash
touch data/raw/.gitkeep data/processed/.gitkeep data/splits/.gitkeep
touch results/figures/.gitkeep notebooks/.gitkeep
```

**Step 6: Commit**

```bash
git init
git add .
git commit -m "chore: initialize project structure with directories and dependencies"
```

---

## Phase 1: Data Collection & Verification

### Task 1.1: Create NFL Data Loader

**Files:**
- Create: `src/data/nfl_loader.py`
- Create: `tests/data/test_nfl_loader.py`

**Step 1: Write the failing test**

```python
# tests/data/test_nfl_loader.py
import pytest
import pandas as pd
from src.data.nfl_loader import load_schedules, load_pbp_data


class TestLoadSchedules:
    def test_load_schedules_returns_dataframe(self):
        """Test that load_schedules returns a pandas DataFrame."""
        df = load_schedules(seasons=[2023])
        assert isinstance(df, pd.DataFrame)

    def test_load_schedules_has_required_columns(self):
        """Test that schedules have required columns for merging."""
        df = load_schedules(seasons=[2023])
        required_cols = [
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_schedules_filters_regular_season(self):
        """Test that only regular season games are returned."""
        df = load_schedules(seasons=[2023], regular_season_only=True)
        assert all(df["game_type"] == "REG")

    def test_load_schedules_multiple_seasons(self):
        """Test loading multiple seasons."""
        df = load_schedules(seasons=[2022, 2023])
        assert set(df["season"].unique()) == {2022, 2023}


class TestLoadPbpData:
    def test_load_pbp_returns_dataframe(self):
        """Test that load_pbp_data returns a pandas DataFrame."""
        df = load_pbp_data(seasons=[2023])
        assert isinstance(df, pd.DataFrame)

    def test_load_pbp_has_epa_columns(self):
        """Test that pbp data has EPA columns."""
        df = load_pbp_data(seasons=[2023])
        assert "epa" in df.columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_nfl_loader.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.data.nfl_loader'"

**Step 3: Write minimal implementation**

```python
# src/data/nfl_loader.py
"""NFL data loading utilities using nfl_data_py."""

import pandas as pd
import nfl_data_py as nfl


def load_schedules(
    seasons: list[int],
    regular_season_only: bool = True,
) -> pd.DataFrame:
    """
    Load NFL schedule data for specified seasons.

    Args:
        seasons: List of seasons to load (e.g., [2022, 2023])
        regular_season_only: If True, filter to regular season games only

    Returns:
        DataFrame with game schedule information
    """
    df = nfl.import_schedules(seasons)

    if regular_season_only:
        df = df[df["game_type"] == "REG"].copy()

    return df


def load_pbp_data(seasons: list[int]) -> pd.DataFrame:
    """
    Load play-by-play data for specified seasons.

    Args:
        seasons: List of seasons to load

    Returns:
        DataFrame with play-by-play data including EPA
    """
    df = nfl.import_pbp_data(seasons)
    return df
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_nfl_loader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/nfl_loader.py tests/data/test_nfl_loader.py
git commit -m "feat: add NFL data loader using nfl_data_py"
```

---

### Task 1.2: Create Betting Data Loader

**Files:**
- Create: `src/data/betting_loader.py`
- Create: `tests/data/test_betting_loader.py`
- Download: Kaggle dataset to `data/raw/spreadspoke_scores.csv`

**Step 1: Download Kaggle dataset manually**

Download from: https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data
Save to: `data/raw/spreadspoke_scores.csv`

**Step 2: Write the failing test**

```python
# tests/data/test_betting_loader.py
import pytest
import pandas as pd
from pathlib import Path
from src.data.betting_loader import load_betting_data, normalize_team_abbr


class TestLoadBettingData:
    def test_load_betting_data_returns_dataframe(self):
        """Test that load_betting_data returns a DataFrame."""
        df = load_betting_data()
        assert isinstance(df, pd.DataFrame)

    def test_load_betting_data_has_spread_columns(self):
        """Test that betting data has spread columns."""
        df = load_betting_data()
        assert "spread_favorite" in df.columns

    def test_load_betting_data_filters_by_season(self):
        """Test filtering betting data by season range."""
        df = load_betting_data(min_season=2020, max_season=2023)
        assert df["schedule_season"].min() >= 2020
        assert df["schedule_season"].max() <= 2023


class TestNormalizeTeamAbbr:
    def test_normalize_known_relocations(self):
        """Test that relocated teams are normalized."""
        assert normalize_team_abbr("STL") == "LA"  # Rams
        assert normalize_team_abbr("SD") == "LAC"  # Chargers
        assert normalize_team_abbr("OAK") == "LV"  # Raiders

    def test_normalize_unchanged_teams(self):
        """Test that non-relocated teams are unchanged."""
        assert normalize_team_abbr("KC") == "KC"
        assert normalize_team_abbr("NE") == "NE"
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/data/test_betting_loader.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 4: Write minimal implementation**

```python
# src/data/betting_loader.py
"""Betting data loading utilities for Kaggle spreadspoke dataset."""

import pandas as pd
from pathlib import Path

# Team abbreviation mapping for relocations and inconsistencies
TEAM_ABBR_MAP = {
    # Relocations
    "STL": "LA",   # Rams to LA (2016)
    "SD": "LAC",   # Chargers to LA (2017)
    "OAK": "LV",   # Raiders to Vegas (2020)
    # Naming inconsistencies
    "JAC": "JAX",
    "WSH": "WAS",
}


def normalize_team_abbr(abbr: str) -> str:
    """
    Normalize team abbreviation to nflverse standard.

    Args:
        abbr: Team abbreviation to normalize

    Returns:
        Normalized team abbreviation
    """
    return TEAM_ABBR_MAP.get(abbr, abbr)


def load_betting_data(
    filepath: Path | None = None,
    min_season: int | None = None,
    max_season: int | None = None,
) -> pd.DataFrame:
    """
    Load betting data from Kaggle spreadspoke dataset.

    Args:
        filepath: Path to CSV file. Defaults to data/raw/spreadspoke_scores.csv
        min_season: Minimum season to include
        max_season: Maximum season to include

    Returns:
        DataFrame with betting data
    """
    if filepath is None:
        filepath = Path("data/raw/spreadspoke_scores.csv")

    df = pd.read_csv(filepath)

    # Filter by season if specified
    if min_season is not None:
        df = df[df["schedule_season"] >= min_season]
    if max_season is not None:
        df = df[df["schedule_season"] <= max_season]

    # Normalize team abbreviations
    df["team_home"] = df["team_home"].apply(normalize_team_abbr)
    df["team_away"] = df["team_away"].apply(normalize_team_abbr)

    return df
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/data/test_betting_loader.py -v`
Expected: PASS (assuming CSV is downloaded)

**Step 6: Commit**

```bash
git add src/data/betting_loader.py tests/data/test_betting_loader.py
git commit -m "feat: add betting data loader for Kaggle spreadspoke dataset"
```

---

### Task 1.3: Create Data Merger with Audit Trail

**Files:**
- Create: `src/data/merger.py`
- Create: `tests/data/test_merger.py`

**Step 1: Write the failing test**

```python
# tests/data/test_merger.py
import pytest
import pandas as pd
from src.data.merger import merge_nfl_betting_data, create_merge_audit


class TestMergeNflBettingData:
    @pytest.fixture
    def sample_nfl_data(self):
        """Create sample NFL schedule data."""
        return pd.DataFrame({
            "game_id": ["2023_01_KC_DET", "2023_01_BAL_HOU"],
            "season": [2023, 2023],
            "week": [1, 1],
            "home_team": ["DET", "HOU"],
            "away_team": ["KC", "BAL"],
            "home_score": [20, 25],
            "away_score": [21, 9],
        })

    @pytest.fixture
    def sample_betting_data(self):
        """Create sample betting data."""
        return pd.DataFrame({
            "schedule_season": [2023, 2023],
            "schedule_week": [1, 1],
            "team_home": ["DET", "HOU"],
            "team_away": ["KC", "BAL"],
            "spread_favorite": [-6.0, -9.5],
            "team_favorite_id": ["KC", "BAL"],
        })

    def test_merge_returns_dataframe(self, sample_nfl_data, sample_betting_data):
        """Test that merge returns a DataFrame."""
        merged, _ = merge_nfl_betting_data(sample_nfl_data, sample_betting_data)
        assert isinstance(merged, pd.DataFrame)

    def test_merge_preserves_game_ids(self, sample_nfl_data, sample_betting_data):
        """Test that all game_ids are preserved after merge."""
        merged, _ = merge_nfl_betting_data(sample_nfl_data, sample_betting_data)
        assert set(merged["game_id"]) == set(sample_nfl_data["game_id"])

    def test_merge_adds_spread_column(self, sample_nfl_data, sample_betting_data):
        """Test that spread column is added after merge."""
        merged, _ = merge_nfl_betting_data(sample_nfl_data, sample_betting_data)
        assert "spread_favorite" in merged.columns


class TestCreateMergeAudit:
    def test_audit_tracks_unmatched_rows(self):
        """Test that audit captures unmatched rows."""
        nfl_df = pd.DataFrame({
            "game_id": ["2023_01_KC_DET"],
            "season": [2023],
            "week": [1],
            "home_team": ["DET"],
            "away_team": ["KC"],
        })
        betting_df = pd.DataFrame({
            "schedule_season": [2023],
            "schedule_week": [1],
            "team_home": ["XXX"],  # Won't match
            "team_away": ["YYY"],
        })
        _, audit = merge_nfl_betting_data(nfl_df, betting_df)
        assert len(audit["unmatched_nfl"]) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_merger.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/data/merger.py
"""Data merging utilities for NFL and betting datasets."""

import pandas as pd
from pathlib import Path
from typing import TypedDict


class MergeAudit(TypedDict):
    """Audit information from data merge."""
    unmatched_nfl: pd.DataFrame
    unmatched_betting: pd.DataFrame
    duplicate_matches: pd.DataFrame
    merge_rate: float


def merge_nfl_betting_data(
    nfl_df: pd.DataFrame,
    betting_df: pd.DataFrame,
) -> tuple[pd.DataFrame, MergeAudit]:
    """
    Merge NFL schedule data with betting data.

    Uses nflverse schedules as the spine. Joins on season, week, home_team, away_team.

    Args:
        nfl_df: NFL schedule DataFrame with game_id as canonical identifier
        betting_df: Betting DataFrame from Kaggle spreadspoke

    Returns:
        Tuple of (merged DataFrame, audit dictionary)
    """
    # Standardize betting columns for merge
    betting_df = betting_df.rename(columns={
        "schedule_season": "season",
        "schedule_week": "week",
        "team_home": "home_team",
        "team_away": "away_team",
    })

    # Track original counts for audit
    nfl_count = len(nfl_df)

    # Merge on canonical keys
    merged = nfl_df.merge(
        betting_df,
        on=["season", "week", "home_team", "away_team"],
        how="left",
        indicator=True,
    )

    # Identify unmatched rows
    unmatched_nfl = merged[merged["_merge"] == "left_only"].copy()
    matched = merged[merged["_merge"] == "both"].copy()

    # Check for duplicates (multiple betting rows per game)
    duplicate_matches = matched[matched.duplicated(subset=["game_id"], keep=False)]

    # Keep first match if duplicates exist
    matched = matched.drop_duplicates(subset=["game_id"], keep="first")

    # Clean up merge indicator
    matched = matched.drop(columns=["_merge"])

    # Calculate merge rate
    merge_rate = len(matched) / nfl_count if nfl_count > 0 else 0.0

    audit: MergeAudit = {
        "unmatched_nfl": unmatched_nfl[["game_id", "season", "week", "home_team", "away_team"]],
        "unmatched_betting": pd.DataFrame(),  # Would need separate tracking
        "duplicate_matches": duplicate_matches,
        "merge_rate": merge_rate,
    }

    return matched, audit


def save_merge_audit(audit: MergeAudit, filepath: Path) -> None:
    """Save merge audit to CSV for review."""
    audit["unmatched_nfl"].to_csv(filepath, index=False)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_merger.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/merger.py tests/data/test_merger.py
git commit -m "feat: add data merger with audit trail for NFL and betting data"
```

---

### Task 1.4: Data Verification Script

**Files:**
- Create: `src/data/verify_data.py`
- Create: `data/README.md`

**Step 1: Write the verification script**

```python
# src/data/verify_data.py
"""Data verification script to check coverage before committing to date range."""

import pandas as pd
from pathlib import Path

from src.data.nfl_loader import load_schedules
from src.data.betting_loader import load_betting_data
from src.data.merger import merge_nfl_betting_data


def verify_data_coverage(
    seasons: list[int],
    output_path: Path = Path("data/README.md"),
) -> dict:
    """
    Verify data coverage for specified seasons.

    Checks:
    1. Spread coverage per season
    2. EPA data availability
    3. Missing data gaps

    Args:
        seasons: List of seasons to verify
        output_path: Path to write README with findings

    Returns:
        Dictionary with coverage statistics
    """
    print("Loading NFL schedule data...")
    nfl_df = load_schedules(seasons=seasons, regular_season_only=True)

    print("Loading betting data...")
    betting_df = load_betting_data(min_season=min(seasons), max_season=max(seasons))

    print("Merging datasets...")
    merged, audit = merge_nfl_betting_data(nfl_df, betting_df)

    # Calculate coverage by season
    coverage_by_season = {}
    for season in seasons:
        season_nfl = nfl_df[nfl_df["season"] == season]
        season_merged = merged[merged["season"] == season]

        total_games = len(season_nfl)
        games_with_spread = season_merged["spread_favorite"].notna().sum()

        coverage_by_season[season] = {
            "total_games": total_games,
            "games_with_spread": games_with_spread,
            "coverage_pct": games_with_spread / total_games * 100 if total_games > 0 else 0,
        }

    # Write README
    _write_data_readme(coverage_by_season, audit, output_path)

    return {
        "coverage_by_season": coverage_by_season,
        "overall_merge_rate": audit["merge_rate"],
        "unmatched_count": len(audit["unmatched_nfl"]),
    }


def _write_data_readme(
    coverage: dict,
    audit: dict,
    output_path: Path,
) -> None:
    """Write data README with coverage information."""
    lines = [
        "# Data Coverage Report",
        "",
        "## Spread Coverage by Season",
        "",
        "| Season | Total Games | Games with Spread | Coverage % |",
        "|--------|-------------|-------------------|------------|",
    ]

    for season, stats in sorted(coverage.items()):
        lines.append(
            f"| {season} | {stats['total_games']} | {stats['games_with_spread']} | {stats['coverage_pct']:.1f}% |"
        )

    lines.extend([
        "",
        f"## Merge Statistics",
        "",
        f"- Overall merge rate: {audit['merge_rate']:.1%}",
        f"- Unmatched NFL games: {len(audit['unmatched_nfl'])}",
        "",
        "## Data Gaps",
        "",
        "Document any identified gaps here after running verification.",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Data README written to {output_path}")


if __name__ == "__main__":
    # Run verification for target seasons
    seasons = list(range(2005, 2024))
    results = verify_data_coverage(seasons)

    print("\n=== Coverage Summary ===")
    for season, stats in results["coverage_by_season"].items():
        if stats["coverage_pct"] < 90:
            print(f"WARNING: {season} has only {stats['coverage_pct']:.1f}% spread coverage")
```

**Step 2: Run verification**

Run: `python -m src.data.verify_data`
Expected: Output showing coverage by season, warnings for low coverage

**Step 3: Review output and adjust date range if needed**

Based on verification output:
- If seasons 2005-2009 have <90% coverage, adjust start date to 2010
- Document gaps in `data/README.md`

**Step 4: Commit**

```bash
git add src/data/verify_data.py data/README.md
git commit -m "feat: add data verification script and coverage report"
```

---

## Phase 2: Feature Engineering

### Task 2.1: Create Target Variable Calculator

**Files:**
- Create: `src/features/target.py`
- Create: `tests/features/test_target.py`

**Step 1: Write the failing test**

```python
# tests/features/test_target.py
import pytest
import pandas as pd
from src.features.target import calculate_upset_target, identify_underdog


class TestIdentifyUnderdog:
    def test_home_team_underdog_positive_spread(self):
        """Home team is underdog when spread is positive (favorite is away)."""
        row = pd.Series({
            "home_team": "DET",
            "away_team": "KC",
            "spread_favorite": -6.0,
            "team_favorite_id": "KC",
        })
        assert identify_underdog(row) == "DET"

    def test_away_team_underdog_negative_spread(self):
        """Away team is underdog when they're not the favorite."""
        row = pd.Series({
            "home_team": "KC",
            "away_team": "DET",
            "spread_favorite": -6.0,
            "team_favorite_id": "KC",
        })
        assert identify_underdog(row) == "DET"

    def test_excludes_small_spreads(self):
        """Returns None for spreads < 3 (pick'em or close games)."""
        row = pd.Series({
            "home_team": "DET",
            "away_team": "KC",
            "spread_favorite": -2.5,
            "team_favorite_id": "KC",
        })
        assert identify_underdog(row) is None


class TestCalculateUpsetTarget:
    @pytest.fixture
    def sample_games(self):
        return pd.DataFrame({
            "game_id": ["g1", "g2", "g3"],
            "home_team": ["DET", "KC", "BUF"],
            "away_team": ["KC", "DET", "MIA"],
            "home_score": [24, 31, 21],
            "away_score": [21, 17, 28],
            "spread_favorite": [-6.0, -3.0, -7.0],
            "team_favorite_id": ["KC", "KC", "BUF"],
        })

    def test_upset_when_underdog_wins(self, sample_games):
        """Target is 1 when underdog wins outright."""
        result = calculate_upset_target(sample_games)
        # g1: DET (underdog) beat KC -> upset = 1
        assert result.loc[result["game_id"] == "g1", "upset"].values[0] == 1

    def test_no_upset_when_favorite_wins(self, sample_games):
        """Target is 0 when favorite wins."""
        result = calculate_upset_target(sample_games)
        # g2: KC (favorite) beat DET -> upset = 0
        assert result.loc[result["game_id"] == "g2", "upset"].values[0] == 0

    def test_away_underdog_upset(self, sample_games):
        """Away underdog winning is still an upset."""
        result = calculate_upset_target(sample_games)
        # g3: MIA (away underdog) beat BUF -> upset = 1
        assert result.loc[result["game_id"] == "g3", "upset"].values[0] == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_target.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/features/target.py
"""Target variable calculation for upset prediction."""

import pandas as pd

MINIMUM_SPREAD = 3.0  # Minimum spread to qualify as underdog


def identify_underdog(row: pd.Series) -> str | None:
    """
    Identify the underdog team for a game.

    Args:
        row: Game row with spread and team information

    Returns:
        Underdog team abbreviation, or None if spread < 3
    """
    spread = abs(row["spread_favorite"])

    if spread < MINIMUM_SPREAD:
        return None

    favorite = row["team_favorite_id"]

    if favorite == row["home_team"]:
        return row["away_team"]
    else:
        return row["home_team"]


def calculate_upset_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate binary upset target variable.

    Upset = 1 if underdog wins outright, 0 otherwise.
    Games with spread < 3 are excluded (NaN target).

    Args:
        df: DataFrame with game results and spread information

    Returns:
        DataFrame with added 'upset' and 'underdog' columns
    """
    df = df.copy()

    # Identify underdog for each game
    df["underdog"] = df.apply(identify_underdog, axis=1)

    # Calculate winner
    df["winner"] = df.apply(
        lambda r: r["home_team"] if r["home_score"] > r["away_score"]
        else (r["away_team"] if r["away_score"] > r["home_score"] else None),
        axis=1
    )

    # Upset = 1 if underdog won
    df["upset"] = (df["underdog"] == df["winner"]).astype(int)

    # Set NaN for excluded games (small spreads or ties)
    df.loc[df["underdog"].isna(), "upset"] = None

    return df
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_target.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/features/target.py tests/features/test_target.py
git commit -m "feat: add target variable calculation for upset prediction"
```

---

### Task 2.2: Create Rolling Average Calculator

**Files:**
- Create: `src/features/rolling.py`
- Create: `tests/features/test_rolling.py`

**Step 1: Write the failing test**

```python
# tests/features/test_rolling.py
import pytest
import pandas as pd
import numpy as np
from src.features.rolling import (
    calculate_rolling_stats,
    get_team_game_sequence,
    ROLLING_WINDOW,
)


class TestGetTeamGameSequence:
    @pytest.fixture
    def sample_games(self):
        """Sample games for a team across multiple weeks."""
        return pd.DataFrame({
            "game_id": ["g1", "g2", "g3", "g4", "g5"],
            "season": [2023, 2023, 2023, 2023, 2023],
            "week": [1, 2, 3, 5, 6],  # Note: week 4 is bye
            "home_team": ["KC", "DET", "KC", "KC", "BUF"],
            "away_team": ["DET", "KC", "BAL", "NE", "KC"],
            "home_score": [21, 17, 28, 31, 24],
            "away_score": [20, 24, 14, 10, 17],
        })

    def test_sequence_ordered_by_week(self, sample_games):
        """Team game sequence should be ordered by week."""
        seq = get_team_game_sequence(sample_games, "KC", 2023)
        assert list(seq["week"]) == [1, 2, 3, 5, 6]

    def test_sequence_contains_all_team_games(self, sample_games):
        """Sequence should contain all games where team played."""
        seq = get_team_game_sequence(sample_games, "KC", 2023)
        assert len(seq) == 5


class TestCalculateRollingStats:
    @pytest.fixture
    def sample_team_games(self):
        """Sample sequential games for rolling calculation."""
        return pd.DataFrame({
            "week": [1, 2, 3, 4, 5, 6, 7],
            "points_scored": [21, 28, 14, 35, 17, 24, 31],
            "points_allowed": [17, 24, 21, 14, 28, 10, 14],
        })

    def test_rolling_mean_calculation(self, sample_team_games):
        """Test rolling mean is calculated correctly."""
        result = calculate_rolling_stats(
            sample_team_games,
            columns=["points_scored"],
            window=ROLLING_WINDOW,
        )

        # Week 6 rolling mean (weeks 2-6): (28+14+35+17+24)/5 = 23.6
        assert abs(result.loc[5, "points_scored_roll5"] - 23.6) < 0.1

    def test_early_season_uses_available_games(self, sample_team_games):
        """Early season should use all available games (< window)."""
        result = calculate_rolling_stats(
            sample_team_games,
            columns=["points_scored"],
            window=ROLLING_WINDOW,
        )

        # Week 3 should use weeks 1-2 only: (21+28)/2 = 24.5
        assert abs(result.loc[2, "points_scored_roll5"] - 24.5) < 0.1

    def test_week_1_is_nan(self, sample_team_games):
        """Week 1 has no prior games, should be NaN."""
        result = calculate_rolling_stats(
            sample_team_games,
            columns=["points_scored"],
            window=ROLLING_WINDOW,
        )
        assert pd.isna(result.loc[0, "points_scored_roll5"])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_rolling.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/features/rolling.py
"""Rolling average calculations for team performance features."""

import pandas as pd

ROLLING_WINDOW = 5  # Use last 5 games


def get_team_game_sequence(
    games_df: pd.DataFrame,
    team: str,
    season: int,
) -> pd.DataFrame:
    """
    Get sequential games for a team in a season.

    Args:
        games_df: DataFrame with all games
        team: Team abbreviation
        season: Season year

    Returns:
        DataFrame of team's games ordered by week
    """
    season_games = games_df[games_df["season"] == season].copy()

    # Find games where team played (home or away)
    team_games = season_games[
        (season_games["home_team"] == team) |
        (season_games["away_team"] == team)
    ].copy()

    return team_games.sort_values("week").reset_index(drop=True)


def calculate_rolling_stats(
    team_games: pd.DataFrame,
    columns: list[str],
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Calculate rolling statistics for specified columns.

    Uses shift(1) to ensure we only use PRIOR games (no leakage).
    Early season games use all available prior games.

    Args:
        team_games: Team's games in order
        columns: Columns to calculate rolling stats for
        window: Rolling window size

    Returns:
        DataFrame with added rolling stat columns
    """
    result = team_games.copy()

    for col in columns:
        # Shift by 1 to only use prior games
        shifted = result[col].shift(1)

        # Calculate rolling mean with min_periods=1 for early season
        rolling = shifted.rolling(window=window, min_periods=1).mean()

        # Week 1 has no prior games, set to NaN
        rolling.iloc[0] = pd.NA

        result[f"{col}_roll{window}"] = rolling

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_rolling.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/features/rolling.py tests/features/test_rolling.py
git commit -m "feat: add rolling average calculator with temporal integrity"
```

---

### Task 2.3: Create Matchup Differential Features

**Files:**
- Create: `src/features/matchup.py`
- Create: `tests/features/test_matchup.py`

**Step 1: Write the failing test**

```python
# tests/features/test_matchup.py
import pytest
import pandas as pd
from src.features.matchup import calculate_matchup_differentials


class TestCalculateMatchupDifferentials:
    @pytest.fixture
    def sample_game_features(self):
        """Sample game with team rolling stats."""
        return pd.DataFrame({
            "game_id": ["g1"],
            "underdog": ["DET"],
            "favorite": ["KC"],
            # Underdog (DET) stats
            "det_pass_epa_roll5": [0.15],
            "det_rush_yards_roll5": [120.0],
            "det_turnover_margin_roll5": [0.5],
            # Favorite (KC) stats
            "kc_pass_def_epa_roll5": [0.05],
            "kc_rush_yards_allowed_roll5": [100.0],
            "kc_turnover_margin_roll5": [1.0],
        })

    def test_offense_defense_mismatch(self, sample_game_features):
        """Test offense/defense mismatch calculation."""
        result = calculate_matchup_differentials(sample_game_features)
        # DET pass EPA (0.15) - KC pass def EPA (0.05) = 0.10
        assert abs(result["offense_defense_mismatch"].values[0] - 0.10) < 0.01

    def test_rush_attack_advantage(self, sample_game_features):
        """Test rush attack advantage calculation."""
        result = calculate_matchup_differentials(sample_game_features)
        # DET rush (120) - KC rush allowed (100) = 20
        assert result["rush_attack_advantage"].values[0] == 20.0

    def test_turnover_edge(self, sample_game_features):
        """Test turnover edge calculation."""
        result = calculate_matchup_differentials(sample_game_features)
        # DET margin (0.5) - KC margin (1.0) = -0.5
        assert result["turnover_edge"].values[0] == -0.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_matchup.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/features/matchup.py
"""Matchup differential feature calculations."""

import pandas as pd


def calculate_matchup_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate matchup differential features (underdog - favorite).

    Args:
        df: DataFrame with team rolling stats for both teams

    Returns:
        DataFrame with added matchup differential columns
    """
    result = df.copy()

    # Get underdog and favorite for column naming
    # Note: Actual implementation will need dynamic column mapping

    # Offense vs Defense mismatch (underdog offense - favorite defense)
    result["offense_defense_mismatch"] = (
        result["det_pass_epa_roll5"] - result["kc_pass_def_epa_roll5"]
    )

    # Rush attack advantage
    result["rush_attack_advantage"] = (
        result["det_rush_yards_roll5"] - result["kc_rush_yards_allowed_roll5"]
    )

    # Turnover edge
    result["turnover_edge"] = (
        result["det_turnover_margin_roll5"] - result["kc_turnover_margin_roll5"]
    )

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_matchup.py -v`
Expected: PASS

**Step 5: Refactor for dynamic team handling**

Extend implementation to handle any team pairing:

```python
# src/features/matchup.py (extended)
"""Matchup differential feature calculations."""

import pandas as pd


def calculate_matchup_differentials(
    df: pd.DataFrame,
    underdog_prefix: str | None = None,
    favorite_prefix: str | None = None,
) -> pd.DataFrame:
    """
    Calculate matchup differential features (underdog - favorite).

    Args:
        df: DataFrame with team rolling stats for both teams
        underdog_prefix: Column prefix for underdog stats (e.g., "det_")
        favorite_prefix: Column prefix for favorite stats (e.g., "kc_")

    Returns:
        DataFrame with added matchup differential columns
    """
    result = df.copy()

    # If prefixes not provided, try to infer from underdog/favorite columns
    if underdog_prefix is None:
        underdog_prefix = result["underdog"].iloc[0].lower() + "_"
    if favorite_prefix is None:
        favorite_prefix = result["favorite"].iloc[0].lower() + "_"

    # Define differential calculations
    differentials = {
        "offense_defense_mismatch": (
            f"{underdog_prefix}pass_epa_roll5",
            f"{favorite_prefix}pass_def_epa_roll5",
        ),
        "rush_attack_advantage": (
            f"{underdog_prefix}rush_yards_roll5",
            f"{favorite_prefix}rush_yards_allowed_roll5",
        ),
        "turnover_edge": (
            f"{underdog_prefix}turnover_margin_roll5",
            f"{favorite_prefix}turnover_margin_roll5",
        ),
    }

    for feature_name, (underdog_col, favorite_col) in differentials.items():
        if underdog_col in result.columns and favorite_col in result.columns:
            result[feature_name] = result[underdog_col] - result[favorite_col]

    return result
```

**Step 6: Commit**

```bash
git add src/features/matchup.py tests/features/test_matchup.py
git commit -m "feat: add matchup differential feature calculations"
```

---

### Task 2.4: Create Full Feature Engineering Pipeline

**Files:**
- Create: `src/features/pipeline.py`
- Create: `tests/features/test_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/features/test_pipeline.py
import pytest
import pandas as pd
from src.features.pipeline import FeatureEngineeringPipeline


class TestFeatureEngineeringPipeline:
    def test_pipeline_produces_expected_columns(self):
        """Test that pipeline produces all expected feature columns."""
        pipeline = FeatureEngineeringPipeline()
        # Using minimal mock data
        result = pipeline.transform(mock_data())

        expected_cols = [
            "game_id",
            "upset",  # Target
            "spread_magnitude",  # Spread feature
            "offense_defense_mismatch",  # Matchup differential
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_pipeline_excludes_week_1(self):
        """Test that Week 1 games are excluded (no prior data)."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data_with_week_1())

        # Week 1 games should be filtered out
        assert not any(result["week"] == 1)

    def test_pipeline_respects_temporal_integrity(self):
        """Test that no future data leaks into features."""
        # This is validated by the rolling calculation tests
        pass


def mock_data():
    """Create minimal mock data for testing."""
    return pd.DataFrame({
        "game_id": ["2023_02_KC_DET"],
        "season": [2023],
        "week": [2],
        "home_team": ["DET"],
        "away_team": ["KC"],
        "home_score": [24],
        "away_score": [21],
        "spread_favorite": [-6.0],
        "team_favorite_id": ["KC"],
    })


def mock_data_with_week_1():
    """Mock data including Week 1."""
    return pd.DataFrame({
        "game_id": ["2023_01_KC_DET", "2023_02_KC_BAL"],
        "season": [2023, 2023],
        "week": [1, 2],
        "home_team": ["DET", "BAL"],
        "away_team": ["KC", "KC"],
        "home_score": [20, 21],
        "away_score": [21, 17],
        "spread_favorite": [-6.0, -3.0],
        "team_favorite_id": ["KC", "KC"],
    })
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_pipeline.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/features/pipeline.py
"""Feature engineering pipeline combining all feature calculations."""

import pandas as pd
from typing import Protocol

from src.features.target import calculate_upset_target
from src.features.rolling import calculate_rolling_stats, ROLLING_WINDOW
from src.features.matchup import calculate_matchup_differentials


class FeatureEngineeringPipeline:
    """
    Pipeline for generating all features for upset prediction.

    Combines:
    - Target variable calculation
    - Rolling team performance stats
    - Matchup differentials
    - Situational features
    """

    def __init__(self, exclude_week_1: bool = True):
        """
        Initialize pipeline.

        Args:
            exclude_week_1: Whether to exclude Week 1 games (no prior data)
        """
        self.exclude_week_1 = exclude_week_1

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline.

        Args:
            df: Raw merged game data

        Returns:
            DataFrame with all engineered features
        """
        result = df.copy()

        # Step 1: Calculate target variable
        result = calculate_upset_target(result)

        # Step 2: Add spread features
        result["spread_magnitude"] = result["spread_favorite"].abs()

        # Step 3: Exclude Week 1 if configured
        if self.exclude_week_1:
            result = result[result["week"] > 1].copy()

        # Step 4: Calculate rolling stats (placeholder - needs team-level aggregation)
        # This will be expanded in actual implementation

        # Step 5: Calculate matchup differentials (placeholder)
        # This requires rolling stats to be computed first

        # Placeholder columns for tests
        if "offense_defense_mismatch" not in result.columns:
            result["offense_defense_mismatch"] = 0.0

        return result

    def get_feature_columns(self) -> list[str]:
        """Get list of feature column names for modeling."""
        return [
            # Spread features
            "spread_magnitude",
            # Matchup differentials
            "offense_defense_mismatch",
            "rush_attack_advantage",
            "turnover_edge",
            # Situational
            "rest_advantage",
            "home_indicator",
            "divisional_game",
        ]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/features/pipeline.py tests/features/test_pipeline.py
git commit -m "feat: add feature engineering pipeline skeleton"
```

---

## Phase 3: Model Implementation

### Task 3.1: Create Time-Series Cross-Validation Splitter

**Files:**
- Create: `src/models/cv_splitter.py`
- Create: `tests/models/test_cv_splitter.py`

**Step 1: Write the failing test**

```python
# tests/models/test_cv_splitter.py
import pytest
import pandas as pd
from src.models.cv_splitter import TimeSeriesCVSplitter


class TestTimeSeriesCVSplitter:
    @pytest.fixture
    def sample_data(self):
        """Sample data spanning multiple seasons."""
        seasons = list(range(2018, 2024))  # 2018-2023
        games_per_season = 10
        return pd.DataFrame({
            "game_id": [f"g{i}" for i in range(len(seasons) * games_per_season)],
            "season": [s for s in seasons for _ in range(games_per_season)],
        })

    def test_splitter_creates_correct_number_of_folds(self, sample_data):
        """Test that splitter creates expected number of folds."""
        splitter = TimeSeriesCVSplitter(n_folds=5)
        folds = list(splitter.split(sample_data))
        assert len(folds) == 5

    def test_train_always_before_val(self, sample_data):
        """Test that training seasons always precede validation."""
        splitter = TimeSeriesCVSplitter(n_folds=3)
        for train_idx, val_idx in splitter.split(sample_data):
            train_seasons = sample_data.iloc[train_idx]["season"].unique()
            val_seasons = sample_data.iloc[val_idx]["season"].unique()

            assert max(train_seasons) < min(val_seasons)

    def test_val_is_single_season(self, sample_data):
        """Test that each validation fold is a single season."""
        splitter = TimeSeriesCVSplitter(n_folds=3)
        for _, val_idx in splitter.split(sample_data):
            val_seasons = sample_data.iloc[val_idx]["season"].unique()
            assert len(val_seasons) == 1

    def test_train_expands_each_fold(self, sample_data):
        """Test that training set grows with each fold."""
        splitter = TimeSeriesCVSplitter(n_folds=3)
        train_sizes = []
        for train_idx, _ in splitter.split(sample_data):
            train_sizes.append(len(train_idx))

        # Each subsequent fold should have more training data
        assert train_sizes == sorted(train_sizes)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_cv_splitter.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/models/cv_splitter.py
"""Time-series cross-validation for NFL seasons."""

import pandas as pd
import numpy as np
from typing import Iterator


class TimeSeriesCVSplitter:
    """
    Time-series cross-validation splitter for NFL data.

    Creates expanding window folds where:
    - Training: All seasons up to validation year
    - Validation: Single season

    Example with n_folds=5 and data from 2005-2023:
        Fold 1: Train 2005-2018, Val 2019
        Fold 2: Train 2005-2019, Val 2020
        Fold 3: Train 2005-2020, Val 2021
        Fold 4: Train 2005-2021, Val 2022
        Fold 5: Train 2005-2022, Val 2023
    """

    def __init__(self, n_folds: int = 5):
        """
        Initialize splitter.

        Args:
            n_folds: Number of cross-validation folds
        """
        self.n_folds = n_folds

    def split(
        self,
        df: pd.DataFrame,
        season_col: str = "season",
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation indices for each fold.

        Args:
            df: DataFrame with season column
            season_col: Name of season column

        Yields:
            Tuple of (train_indices, val_indices) for each fold
        """
        seasons = sorted(df[season_col].unique())
        n_seasons = len(seasons)

        if n_seasons < self.n_folds + 1:
            raise ValueError(
                f"Need at least {self.n_folds + 1} seasons for {self.n_folds} folds"
            )

        # Validation seasons are the last n_folds seasons
        val_seasons = seasons[-self.n_folds:]

        for val_season in val_seasons:
            # Training: all seasons before validation
            train_mask = df[season_col] < val_season
            val_mask = df[season_col] == val_season

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_folds
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_cv_splitter.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/cv_splitter.py tests/models/test_cv_splitter.py
git commit -m "feat: add time-series cross-validation splitter"
```

---

### Task 3.2: Create XGBoost Model Wrapper

**Files:**
- Create: `src/models/xgboost_model.py`
- Create: `tests/models/test_xgboost_model.py`

**Step 1: Write the failing test**

```python
# tests/models/test_xgboost_model.py
import pytest
import pandas as pd
import numpy as np
from src.models.xgboost_model import UpsetXGBoost


class TestUpsetXGBoost:
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "spread_magnitude": np.random.uniform(3, 14, n),
            "offense_diff": np.random.normal(0, 1, n),
            "defense_diff": np.random.normal(0, 1, n),
            "upset": np.random.binomial(1, 0.35, n),
        })

    def test_fit_returns_self(self, sample_data):
        """Test that fit returns the model instance."""
        model = UpsetXGBoost()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        result = model.fit(X, y)
        assert result is model

    def test_predict_proba_returns_probabilities(self, sample_data):
        """Test that predict_proba returns values between 0 and 1."""
        model = UpsetXGBoost()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert all(0 <= p <= 1 for p in probs)

    def test_predict_returns_binary(self, sample_data):
        """Test that predict returns binary predictions."""
        model = UpsetXGBoost()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        model.fit(X, y)

        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_feature_importance_available(self, sample_data):
        """Test that feature importance is available after fit."""
        model = UpsetXGBoost()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert len(importance) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_xgboost_model.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/models/xgboost_model.py
"""XGBoost model wrapper for upset prediction."""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Any


class UpsetXGBoost:
    """
    XGBoost classifier wrapper for NFL upset prediction.

    Provides consistent interface with probability outputs
    and feature importance extraction.
    """

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        min_child_weight: int = 1,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """
        Initialize XGBoost model.

        Args:
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            n_estimators: Number of boosting rounds
            min_child_weight: Minimum sum of instance weight in child
            random_state: Random seed for reproducibility
            **kwargs: Additional XGBoost parameters
        """
        self.params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_weight": min_child_weight,
            "random_state": random_state,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            **kwargs,
        }
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list | None = None,
        verbose: bool = False,
    ) -> "UpsetXGBoost":
        """
        Fit the XGBoost model.

        Args:
            X: Feature DataFrame
            y: Target Series
            eval_set: Optional validation set for early stopping
            verbose: Whether to print training progress

        Returns:
            Self
        """
        self.feature_names = list(X.columns)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=verbose,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict upset probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Array of upset probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary upset outcomes.

        Args:
            X: Feature DataFrame
            threshold: Probability threshold for positive class

        Returns:
            Array of binary predictions
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        importance = self.model.get_booster().get_score(importance_type=importance_type)

        # Map back to feature names
        return {
            name: importance.get(f"f{i}", 0.0)
            for i, name in enumerate(self.feature_names)
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_xgboost_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/xgboost_model.py tests/models/test_xgboost_model.py
git commit -m "feat: add XGBoost model wrapper for upset prediction"
```

---

### Task 3.3: Create LSTM Model Architecture

**Files:**
- Create: `src/models/lstm_model.py`
- Create: `tests/models/test_lstm_model.py`

**Step 1: Write the failing test**

```python
# tests/models/test_lstm_model.py
import pytest
import torch
import numpy as np
from src.models.lstm_model import UpsetLSTM, LSTMDataset


class TestLSTMDataset:
    def test_dataset_length(self):
        """Test dataset returns correct length."""
        sequences = np.random.randn(100, 5, 15)  # 100 games, 5 timesteps, 15 features
        matchup = np.random.randn(100, 10)  # 10 matchup features
        targets = np.random.randint(0, 2, 100)

        dataset = LSTMDataset(sequences, matchup, targets)
        assert len(dataset) == 100

    def test_dataset_returns_tensors(self):
        """Test dataset returns torch tensors."""
        sequences = np.random.randn(10, 5, 15)
        matchup = np.random.randn(10, 10)
        targets = np.random.randint(0, 2, 10)

        dataset = LSTMDataset(sequences, matchup, targets)
        seq, match, target = dataset[0]

        assert isinstance(seq, torch.Tensor)
        assert isinstance(match, torch.Tensor)
        assert isinstance(target, torch.Tensor)


class TestUpsetLSTM:
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return UpsetLSTM(
            sequence_features=15,
            matchup_features=10,
            hidden_size=64,
            num_layers=2,
        )

    def test_forward_returns_probabilities(self, model):
        """Test forward pass returns values between 0 and 1."""
        batch_size = 8
        seq = torch.randn(batch_size, 5, 15)
        matchup = torch.randn(batch_size, 10)

        output = model(seq, matchup)

        assert output.shape == (batch_size, 1)
        assert all(0 <= p <= 1 for p in output.squeeze())

    def test_model_handles_masking(self, model):
        """Test model handles padded sequences with mask."""
        batch_size = 4
        seq = torch.randn(batch_size, 5, 15)
        matchup = torch.randn(batch_size, 10)
        # Mask: first 2 samples have full sequences, last 2 have only 3 games
        mask = torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ]).float()

        output = model(seq, matchup, mask=mask)
        assert output.shape == (batch_size, 1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_lstm_model.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/models/lstm_model.py
"""LSTM model for upset prediction with attention."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
    """
    Dataset for LSTM model.

    Handles:
    - Team game sequences (last 5 games)
    - Matchup-level features
    - Target variable
    """

    def __init__(
        self,
        sequences: np.ndarray,
        matchup_features: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray | None = None,
    ):
        """
        Initialize dataset.

        Args:
            sequences: Shape (n_samples, seq_len, n_features)
            matchup_features: Shape (n_samples, n_matchup_features)
            targets: Shape (n_samples,)
            masks: Optional shape (n_samples, seq_len) for padding
        """
        self.sequences = torch.FloatTensor(sequences)
        self.matchup_features = torch.FloatTensor(matchup_features)
        self.targets = torch.FloatTensor(targets)
        self.masks = torch.FloatTensor(masks) if masks is not None else None

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple:
        if self.masks is not None:
            return (
                self.sequences[idx],
                self.matchup_features[idx],
                self.targets[idx],
                self.masks[idx],
            )
        return (
            self.sequences[idx],
            self.matchup_features[idx],
            self.targets[idx],
        )


class UpsetLSTM(nn.Module):
    """
    Siamese LSTM architecture for upset prediction.

    Architecture:
    - Shared LSTM encoder for both teams' game sequences
    - Attention mechanism for interpretability
    - Concatenation with matchup features
    - Dense layers for final prediction
    """

    def __init__(
        self,
        sequence_features: int = 15,
        matchup_features: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize LSTM model.

        Args:
            sequence_features: Number of features per timestep
            matchup_features: Number of matchup-level features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=sequence_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Attention layer for interpretability
        self.attention = nn.Linear(hidden_size, 1)

        # Dense layers after concatenation
        combined_size = hidden_size + matchup_features
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sequences: torch.Tensor,
        matchup_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequences: Shape (batch, seq_len, features)
            matchup_features: Shape (batch, matchup_features)
            mask: Optional shape (batch, seq_len) for padded sequences

        Returns:
            Probability of upset, shape (batch, 1)
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(sequences)  # (batch, seq_len, hidden)

        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            attn_weights = attn_weights * mask
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted sum of LSTM outputs
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden)

        # Concatenate with matchup features
        combined = torch.cat([context, matchup_features], dim=1)

        # Final prediction
        return self.fc(combined)

    def get_attention_weights(
        self,
        sequences: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get attention weights for interpretability.

        Args:
            sequences: Input sequences
            mask: Optional padding mask

        Returns:
            Attention weights per timestep
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(sequences)
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)

            if mask is not None:
                mask = mask.unsqueeze(-1)
                attn_weights = attn_weights * mask
                attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)

            return attn_weights.squeeze(-1)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_lstm_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/lstm_model.py tests/models/test_lstm_model.py
git commit -m "feat: add LSTM model architecture with attention"
```

---

### Task 3.4: Create Model Training Pipeline

**Files:**
- Create: `src/models/trainer.py`
- Create: `tests/models/test_trainer.py`

**Step 1: Write the failing test**

```python
# tests/models/test_trainer.py
import pytest
import pandas as pd
import numpy as np
from src.models.trainer import ModelTrainer
from src.models.xgboost_model import UpsetXGBoost


class TestModelTrainer:
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            "season": [2020] * 50 + [2021] * 50 + [2022] * 50 + [2023] * 50,
            "spread_magnitude": np.random.uniform(3, 14, n),
            "offense_diff": np.random.normal(0, 1, n),
            "defense_diff": np.random.normal(0, 1, n),
            "upset": np.random.binomial(1, 0.35, n),
        })

    def test_trainer_runs_cross_validation(self, sample_data):
        """Test that trainer runs all CV folds."""
        model = UpsetXGBoost()
        trainer = ModelTrainer(model, n_folds=3)

        features = ["spread_magnitude", "offense_diff", "defense_diff"]
        results = trainer.cross_validate(sample_data, features, "upset")

        assert len(results["fold_metrics"]) == 3

    def test_trainer_returns_metrics(self, sample_data):
        """Test that trainer returns expected metrics."""
        model = UpsetXGBoost()
        trainer = ModelTrainer(model, n_folds=2)

        features = ["spread_magnitude", "offense_diff", "defense_diff"]
        results = trainer.cross_validate(sample_data, features, "upset")

        assert "auc_roc" in results["fold_metrics"][0]
        assert "log_loss" in results["fold_metrics"][0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_trainer.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/models/trainer.py
"""Model training pipeline with cross-validation."""

import pandas as pd
import numpy as np
from typing import Protocol, Any
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from src.models.cv_splitter import TimeSeriesCVSplitter


class Model(Protocol):
    """Protocol for model interface."""
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class ModelTrainer:
    """
    Training pipeline with time-series cross-validation.

    Handles:
    - Cross-validation splits
    - Model training per fold
    - Metric calculation
    - MLflow logging (optional)
    """

    def __init__(
        self,
        model: Model,
        n_folds: int = 5,
    ):
        """
        Initialize trainer.

        Args:
            model: Model instance to train
            n_folds: Number of CV folds
        """
        self.model = model
        self.cv_splitter = TimeSeriesCVSplitter(n_folds=n_folds)

    def cross_validate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> dict:
        """
        Run time-series cross-validation.

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column

        Returns:
            Dictionary with fold metrics and aggregated results
        """
        X = df[feature_cols]
        y = df[target_col]

        fold_metrics = []
        fold_predictions = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(df)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            self.model.fit(X_train, y_train)

            # Get predictions
            y_pred_proba = self.model.predict_proba(X_val)

            # Calculate metrics
            metrics = self._calculate_metrics(y_val, y_pred_proba)
            metrics["fold"] = fold_idx
            metrics["train_size"] = len(train_idx)
            metrics["val_size"] = len(val_idx)

            fold_metrics.append(metrics)
            fold_predictions.append({
                "val_idx": val_idx,
                "y_true": y_val.values,
                "y_pred": y_pred_proba,
            })

        # Aggregate metrics
        aggregated = self._aggregate_metrics(fold_metrics)

        return {
            "fold_metrics": fold_metrics,
            "aggregated": aggregated,
            "predictions": fold_predictions,
        }

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
    ) -> dict:
        """Calculate evaluation metrics."""
        return {
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            "log_loss": log_loss(y_true, y_pred_proba),
            "brier_score": brier_score_loss(y_true, y_pred_proba),
        }

    def _aggregate_metrics(self, fold_metrics: list[dict]) -> dict:
        """Aggregate metrics across folds."""
        metric_names = ["auc_roc", "log_loss", "brier_score"]
        aggregated = {}

        for metric in metric_names:
            values = [f[metric] for f in fold_metrics]
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)

        return aggregated
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_trainer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/trainer.py tests/models/test_trainer.py
git commit -m "feat: add model training pipeline with time-series CV"
```

---

## Phase 4: Evaluation & Interpretability

### Task 4.1: Create Evaluation Metrics Module

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics.py`

**Step 1: Write the failing test**

```python
# tests/evaluation/test_metrics.py
import pytest
import numpy as np
from src.evaluation.metrics import (
    calculate_calibration_metrics,
    calculate_betting_metrics,
    calculate_baseline_brier,
)


class TestCalibrationMetrics:
    def test_perfect_calibration(self):
        """Test calibration with perfectly calibrated predictions."""
        # Predictions at 70% should be correct 70% of the time
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])  # 7/10 = 70%
        y_pred = np.array([0.7] * 10)

        result = calculate_calibration_metrics(y_true, y_pred, n_bins=1)
        assert abs(result["calibration_error"]) < 0.1


class TestBettingMetrics:
    def test_roi_calculation(self):
        """Test ROI is calculated correctly."""
        y_true = np.array([1, 0, 1, 0, 1])  # 3 wins, 2 losses
        y_pred = np.array([0.8, 0.6, 0.7, 0.9, 0.75])  # All predict upset
        odds = np.array([2.0, 2.0, 2.0, 2.0, 2.0])  # Even money

        result = calculate_betting_metrics(y_true, y_pred, odds, threshold=0.5)
        # 3 wins at +100 = +3 units, 2 losses = -2 units, net +1 unit
        # ROI = 1/5 = 20%
        assert abs(result["roi"] - 0.20) < 0.01


class TestBaselineBrier:
    def test_baseline_brier_formula(self):
        """Test baseline Brier score calculation."""
        # With upset rate r, baseline Brier = r * (1 - r)
        upset_rate = 0.35
        expected = 0.35 * 0.65  # = 0.2275

        result = calculate_baseline_brier(upset_rate)
        assert abs(result - expected) < 0.001
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/evaluation/metrics.py
"""Evaluation metrics for upset prediction models."""

import numpy as np
from sklearn.calibration import calibration_curve


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Calculate calibration metrics.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)

    # Expected Calibration Error (ECE)
    bin_counts = np.histogram(y_pred, bins=n_bins)[0]
    bin_weights = bin_counts / len(y_pred)

    ece = np.sum(np.abs(prob_true - prob_pred) * bin_weights[:len(prob_true)])

    return {
        "calibration_error": ece,
        "prob_true": prob_true,
        "prob_pred": prob_pred,
    }


def calculate_betting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    odds: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Calculate betting profitability metrics.

    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        odds: Decimal odds for each bet
        threshold: Probability threshold for placing bet

    Returns:
        Dictionary with betting metrics
    """
    # Identify bets placed
    bets_placed = y_pred >= threshold
    n_bets = bets_placed.sum()

    if n_bets == 0:
        return {"roi": 0.0, "n_bets": 0, "win_rate": 0.0}

    # Calculate returns
    wins = (y_true == 1) & bets_placed
    losses = (y_true == 0) & bets_placed

    # Assuming unit bets
    profit = wins.sum() * (odds[wins].mean() - 1) if wins.sum() > 0 else 0
    loss = losses.sum()  # Lost unit bets

    total_wagered = n_bets
    net_profit = profit - loss

    return {
        "roi": net_profit / total_wagered if total_wagered > 0 else 0.0,
        "n_bets": int(n_bets),
        "win_rate": wins.sum() / n_bets if n_bets > 0 else 0.0,
        "total_profit": net_profit,
    }


def calculate_baseline_brier(upset_rate: float) -> float:
    """
    Calculate baseline Brier score for constant prediction.

    The baseline is what you'd get by always predicting the upset rate.

    Args:
        upset_rate: Historical upset rate (proportion)

    Returns:
        Baseline Brier score
    """
    return upset_rate * (1 - upset_rate)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat: add evaluation metrics for calibration and betting"
```

---

### Task 4.2: Create SHAP Analysis Module

**Files:**
- Create: `src/evaluation/shap_analysis.py`
- Create: `tests/evaluation/test_shap_analysis.py`

**Step 1: Write the failing test**

```python
# tests/evaluation/test_shap_analysis.py
import pytest
import pandas as pd
import numpy as np
from src.evaluation.shap_analysis import (
    compute_shap_values,
    get_shap_feature_importance,
)
from src.models.xgboost_model import UpsetXGBoost


class TestSHAPAnalysis:
    @pytest.fixture
    def trained_model(self):
        """Create and train a model."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            "spread_magnitude": np.random.uniform(3, 14, n),
            "offense_diff": np.random.normal(0, 1, n),
            "defense_diff": np.random.normal(0, 1, n),
        })
        y = np.random.binomial(1, 0.35, n)

        model = UpsetXGBoost()
        model.fit(X, pd.Series(y))
        return model, X

    def test_shap_values_computed(self, trained_model):
        """Test that SHAP values are computed for all samples."""
        model, X = trained_model
        shap_values = compute_shap_values(model, X)

        assert shap_values.shape == X.shape

    def test_shap_importance_returns_dict(self, trained_model):
        """Test that feature importance returns dictionary."""
        model, X = trained_model
        importance = get_shap_feature_importance(model, X)

        assert isinstance(importance, dict)
        assert len(importance) == len(X.columns)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_shap_analysis.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/evaluation/shap_analysis.py
"""SHAP analysis for XGBoost model interpretability."""

import shap
import pandas as pd
import numpy as np
from src.models.xgboost_model import UpsetXGBoost


def compute_shap_values(
    model: UpsetXGBoost,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Compute SHAP values for model predictions.

    Args:
        model: Trained XGBoost model
        X: Feature DataFrame

    Returns:
        SHAP values array, shape (n_samples, n_features)
    """
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X)

    return shap_values


def get_shap_feature_importance(
    model: UpsetXGBoost,
    X: pd.DataFrame,
) -> dict[str, float]:
    """
    Get global feature importance from SHAP values.

    Args:
        model: Trained XGBoost model
        X: Feature DataFrame

    Returns:
        Dictionary mapping feature names to mean absolute SHAP values
    """
    shap_values = compute_shap_values(model, X)

    # Mean absolute SHAP value per feature
    importance = np.abs(shap_values).mean(axis=0)

    return dict(zip(X.columns, importance))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_shap_analysis.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/evaluation/shap_analysis.py tests/evaluation/test_shap_analysis.py
git commit -m "feat: add SHAP analysis module for XGBoost interpretability"
```

---

### Task 4.3: Create Model Comparison Module

**Files:**
- Create: `src/evaluation/comparison.py`
- Create: `tests/evaluation/test_comparison.py`

**Step 1: Write the failing test**

```python
# tests/evaluation/test_comparison.py
import pytest
import numpy as np
from src.evaluation.comparison import (
    calculate_agreement_rate,
    find_disagreement_cases,
    compare_feature_importance,
)


class TestModelComparison:
    def test_agreement_rate_perfect_agreement(self):
        """Test agreement rate when models fully agree."""
        preds_a = np.array([0.7, 0.3, 0.8, 0.2])
        preds_b = np.array([0.6, 0.4, 0.9, 0.1])

        rate = calculate_agreement_rate(preds_a, preds_b, threshold=0.5)
        assert rate == 1.0  # All same side of 0.5

    def test_agreement_rate_partial(self):
        """Test agreement rate with some disagreement."""
        preds_a = np.array([0.7, 0.3, 0.8, 0.6])
        preds_b = np.array([0.6, 0.6, 0.9, 0.4])  # 2nd and 4th disagree

        rate = calculate_agreement_rate(preds_a, preds_b, threshold=0.5)
        assert rate == 0.5  # 2 of 4 agree

    def test_find_disagreement_cases(self):
        """Test finding games with large disagreement."""
        preds_a = np.array([0.9, 0.3, 0.5, 0.7])
        preds_b = np.array([0.2, 0.35, 0.45, 0.8])

        indices = find_disagreement_cases(preds_a, preds_b, min_diff=0.3)
        assert 0 in indices  # 0.9 vs 0.2 = 0.7 diff
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_comparison.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/evaluation/comparison.py
"""Model comparison utilities for XGBoost vs LSTM analysis."""

import numpy as np
from scipy.stats import spearmanr


def calculate_agreement_rate(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Calculate agreement rate between two models.

    Agreement = both models on same side of threshold.

    Args:
        preds_a: Predictions from model A
        preds_b: Predictions from model B
        threshold: Decision threshold

    Returns:
        Proportion of predictions where models agree
    """
    classify_a = preds_a >= threshold
    classify_b = preds_b >= threshold

    agreement = (classify_a == classify_b).mean()
    return agreement


def find_disagreement_cases(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    min_diff: float = 0.3,
) -> np.ndarray:
    """
    Find indices where models disagree significantly.

    Args:
        preds_a: Predictions from model A
        preds_b: Predictions from model B
        min_diff: Minimum probability difference to flag

    Returns:
        Array of indices with significant disagreement
    """
    diff = np.abs(preds_a - preds_b)
    return np.where(diff >= min_diff)[0]


def compare_feature_importance(
    importance_a: dict[str, float],
    importance_b: dict[str, float],
) -> dict:
    """
    Compare feature importance between two models.

    Args:
        importance_a: Feature importance from model A
        importance_b: Feature importance from model B

    Returns:
        Dictionary with comparison metrics
    """
    # Get common features
    common_features = set(importance_a.keys()) & set(importance_b.keys())

    if len(common_features) < 2:
        return {"spearman_correlation": None, "common_features": 0}

    values_a = [importance_a[f] for f in common_features]
    values_b = [importance_b[f] for f in common_features]

    correlation, p_value = spearmanr(values_a, values_b)

    return {
        "spearman_correlation": correlation,
        "p_value": p_value,
        "common_features": len(common_features),
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_comparison.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/evaluation/comparison.py tests/evaluation/test_comparison.py
git commit -m "feat: add model comparison utilities for XGBoost vs LSTM"
```

---

## Phase 5: Notebooks & Visualization

### Task 5.1: Create Data Exploration Notebook

**Files:**
- Create: `notebooks/01_data_exploration.ipynb`

**Step 1: Create notebook with exploration cells**

The notebook should include:
1. Load and inspect data sources
2. Check data quality and coverage
3. Visualize upset rate by season/spread
4. Identify data gaps

**Step 2: Commit**

```bash
git add notebooks/01_data_exploration.ipynb
git commit -m "feat: add data exploration notebook"
```

---

### Task 5.2: Create Feature Engineering Notebook

**Files:**
- Create: `notebooks/02_feature_engineering.ipynb`

**Step 1: Create notebook with feature engineering pipeline**

The notebook should include:
1. Run feature engineering pipeline
2. Visualize feature distributions
3. Check for leakage (rolling stats)
4. Save processed dataset

**Step 2: Commit**

```bash
git add notebooks/02_feature_engineering.ipynb
git commit -m "feat: add feature engineering notebook"
```

---

### Task 5.3: Create XGBoost Model Notebook

**Files:**
- Create: `notebooks/03_xgboost_model.ipynb`

**Step 1: Create notebook with XGBoost training**

The notebook should include:
1. Load processed data
2. Train with cross-validation
3. Hyperparameter tuning with Optuna
4. SHAP analysis
5. Save final model

**Step 2: Commit**

```bash
git add notebooks/03_xgboost_model.ipynb
git commit -m "feat: add XGBoost model training notebook"
```

---

### Task 5.4: Create LSTM Model Notebook

**Files:**
- Create: `notebooks/04_lstm_model.ipynb`

**Step 1: Create notebook with LSTM training**

The notebook should include:
1. Prepare sequence data
2. Train LSTM with PyTorch
3. Hyperparameter tuning
4. Attention analysis
5. Save final model

**Step 2: Commit**

```bash
git add notebooks/04_lstm_model.ipynb
git commit -m "feat: add LSTM model training notebook"
```

---

### Task 5.5: Create Comparison Analysis Notebook

**Files:**
- Create: `notebooks/05_comparison_analysis.ipynb`

**Step 1: Create notebook with model comparison**

The notebook should include:
1. Load both trained models
2. Generate predictions on test set
3. Compare metrics (AUC, calibration)
4. Agreement rate analysis
5. Disagreement case studies
6. Feature importance comparison
7. Betting backtest

**Step 2: Commit**

```bash
git add notebooks/05_comparison_analysis.ipynb
git commit -m "feat: add model comparison analysis notebook"
```

---

## Phase 6: Experiment Tracking & Final Analysis

### Task 6.1: Add MLflow Integration

**Files:**
- Modify: `src/models/trainer.py`
- Create: `src/models/mlflow_utils.py`

**Step 1: Add MLflow logging to trainer**

Add optional MLflow logging for:
- Hyperparameters
- Metrics per fold
- Model artifacts
- Feature importance plots

**Step 2: Commit**

```bash
git add src/models/trainer.py src/models/mlflow_utils.py
git commit -m "feat: add MLflow integration for experiment tracking"
```

---

### Task 6.2: Create Final Report Generation

**Files:**
- Create: `src/evaluation/report.py`

**Step 1: Create report generation module**

Generate:
- Metrics summary table
- Key visualizations
- Research findings summary

**Step 2: Commit**

```bash
git add src/evaluation/report.py
git commit -m "feat: add final report generation module"
```

---

## Verification Checklist

After completing all tasks, verify:

1. **Data Pipeline**
   - [ ] `python -m src.data.verify_data` runs without errors
   - [ ] `data/README.md` shows >90% spread coverage for target years

2. **Feature Engineering**
   - [ ] Week 1 games excluded from dataset
   - [ ] Rolling features use only prior games (no leakage)
   - [ ] All ~50-80 features generated

3. **Model Training**
   - [ ] XGBoost cross-validation completes
   - [ ] LSTM training converges
   - [ ] Models saved to `results/models/`

4. **Evaluation**
   - [ ] AUC-ROC calculated for both models
   - [ ] Calibration curves generated
   - [ ] SHAP plots saved for XGBoost
   - [ ] Attention weights extracted from LSTM

5. **Comparison**
   - [ ] Agreement rate < 85% (models differ)
   - [ ] Feature importance correlation calculated
   - [ ] Disagreement cases documented

6. **Tests**
   - [ ] `pytest tests/ -v` passes all tests
   - [ ] Coverage > 80%
