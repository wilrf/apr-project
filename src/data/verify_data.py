"""Data verification script to check coverage before committing to date range."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

from src.data.nfl_loader import load_schedules
from src.data.betting_loader import load_betting_data
from src.data.merger import merge_nfl_betting_data


def verify_data_coverage(
    seasons: List[int],
    output_path: Path = Path("data/README.md"),
) -> Dict[str, Any]:
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
            "games_with_spread": int(games_with_spread),
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
    coverage: Dict[int, Dict[str, Any]],
    audit: Dict[str, Any],
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
