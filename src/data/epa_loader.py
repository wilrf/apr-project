"""Game-level advanced stats extracted from nflverse play-by-play data."""

from __future__ import annotations

import json
import socket
from contextlib import contextmanager
from typing import Iterator, List
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

# Columns needed from PBP data — keeps memory manageable (full PBP is multi-GB)
_PBP_COLUMNS = [
    "game_id",
    "play_id",
    "home_team",
    "away_team",
    "posteam",
    "play_type",
    "epa",
    "success",
    "cpoe",
    "interception",
    "fumble_lost",
    "total_home_pass_epa",
    "total_away_pass_epa",
    "total_home_rush_epa",
    "total_away_rush_epa",
]
PBP_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
    "play_by_play_{season}.parquet"
)
_DOH_ENDPOINTS = (
    "https://dns.google/resolve?name={host}&type=A",
    "https://cloudflare-dns.com/dns-query?name={host}&type=A",
)
_GITHUB_HOSTS = ("github.com", "api.github.com")


def _resolve_host_via_doh(host: str) -> str:
    """Resolve a host via DNS-over-HTTPS when local DNS is unavailable."""
    headers = {"accept": "application/dns-json"}
    for url_template in _DOH_ENDPOINTS:
        request = Request(url_template.format(host=host), headers=headers)
        try:
            with urlopen(request, timeout=15) as response:
                payload = json.load(response)
        except Exception:
            continue

        for answer in payload.get("Answer", []):
            if answer.get("type") == 1 and answer.get("data"):
                return str(answer["data"])

    raise RuntimeError(f"Failed to resolve {host} via DNS-over-HTTPS.")


@contextmanager
def _github_dns_override() -> Iterator[None]:
    """
    Patch GitHub host resolution only when local DNS fails.

    Some environments can reach GitHub release assets but cannot resolve github.com.
    """
    original_getaddrinfo = socket.getaddrinfo
    overrides: dict[str, str] = {}

    for host in _GITHUB_HOSTS:
        try:
            original_getaddrinfo(host, 443)
        except OSError:
            overrides[host] = _resolve_host_via_doh(host)

    if not overrides:
        yield
        return

    def patched_getaddrinfo(
        host: str,
        port: int,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ):
        override_ip = overrides.get(host)
        if override_ip is not None:
            return original_getaddrinfo(override_ip, port, family, type, proto, flags)
        return original_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = patched_getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


def _load_pbp_data(seasons: List[int]) -> pd.DataFrame:
    """Load play-by-play parquet files directly from nflverse releases."""
    frames = []

    with _github_dns_override():
        for season in seasons:
            try:
                frame = pd.read_parquet(
                    PBP_URL_TEMPLATE.format(season=season),
                    columns=_PBP_COLUMNS,
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed to load play-by-play data for "
                    f"season {season} from nflverse: {e}"
                ) from e

            if frame.empty:
                raise RuntimeError(
                    "Failed to load play-by-play data for "
                    f"season {season} from nflverse: empty dataset returned."
                )

            frames.append(frame)

    if not frames:
        raise RuntimeError("Failed to load play-by-play data: no seasons were loaded.")

    pbp = pd.concat(frames, ignore_index=True)
    missing_columns = sorted(set(_PBP_COLUMNS) - set(pbp.columns))
    if missing_columns:
        raise RuntimeError(
            "Failed to load play-by-play data from nflverse. "
            f"Missing required columns: {missing_columns}"
        )

    return pbp


def _build_cumulative_epa_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """Return one row per game with the final cumulative EPA totals."""
    sort_cols = [column for column in ["game_id", "play_id"] if column in pbp.columns]
    return (
        pbp.sort_values(sort_cols)
        .groupby("game_id", sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _build_rate_stats_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate success rate, CPOE, and turnover margin to game level."""
    required = {
        "posteam",
        "play_type",
        "success",
        "cpoe",
        "interception",
        "fumble_lost",
    }
    if not required.issubset(pbp.columns):
        empty = pbp[["game_id"]].drop_duplicates().copy()
        for column in [
            "home_success_rate",
            "away_success_rate",
            "home_cpoe",
            "away_cpoe",
            "home_turnover_margin",
            "away_turnover_margin",
        ]:
            empty[column] = np.nan
        return empty

    offensive_plays = pbp[pbp["posteam"].notna()].copy()
    offense_mask = offensive_plays["play_type"].isin(["pass", "run"])
    offense_stats = offensive_plays[offense_mask].copy()

    grouped = (
        offense_stats.groupby(["game_id", "posteam"], dropna=False)
        .agg(
            success_rate=("success", "mean"),
            cpoe=("cpoe", "mean"),
            giveaways=("interception", "sum"),
            fumbles_lost=("fumble_lost", "sum"),
        )
        .reset_index()
    )
    grouped["turnovers"] = grouped["giveaways"].fillna(0) + grouped[
        "fumbles_lost"
    ].fillna(0)

    teams = offensive_plays[["game_id", "home_team", "away_team"]].drop_duplicates()
    merged = grouped.merge(teams, on="game_id", how="left")

    home = (
        merged[merged["posteam"] == merged["home_team"]][
            ["game_id", "success_rate", "cpoe", "turnovers"]
        ]
        .rename(
            columns={
                "success_rate": "home_success_rate",
                "cpoe": "home_cpoe",
                "turnovers": "home_turnovers",
            }
        )
        .reset_index(drop=True)
    )
    away = (
        merged[merged["posteam"] == merged["away_team"]][
            ["game_id", "success_rate", "cpoe", "turnovers"]
        ]
        .rename(
            columns={
                "success_rate": "away_success_rate",
                "cpoe": "away_cpoe",
                "turnovers": "away_turnovers",
            }
        )
        .reset_index(drop=True)
    )

    result = teams[["game_id"]].drop_duplicates().merge(home, on="game_id", how="left")
    result = result.merge(away, on="game_id", how="left")

    home_turnovers = result["home_turnovers"].fillna(0)
    away_turnovers = result["away_turnovers"].fillna(0)
    result["home_turnover_margin"] = away_turnovers - home_turnovers
    result["away_turnover_margin"] = home_turnovers - away_turnovers

    return result.drop(columns=["home_turnovers", "away_turnovers"])


def load_game_advanced_stats(seasons: List[int]) -> pd.DataFrame:
    """
    Load game-level EPA, success rate, CPOE, and turnover stats from PBP data.

    Args:
        seasons: List of NFL seasons to load (e.g., [2005, 2006, ...])

    Returns:
        DataFrame with one row per game and home/away advanced stats.
    """
    pbp = _load_pbp_data(seasons)

    game_epa = _build_cumulative_epa_frame(pbp).rename(
        columns={
            "total_home_pass_epa": "home_off_pass_epa",
            "total_home_rush_epa": "home_off_rush_epa",
            "total_away_pass_epa": "away_off_pass_epa",
            "total_away_rush_epa": "away_off_rush_epa",
        }
    )[
        [
            "game_id",
            "home_off_pass_epa",
            "home_off_rush_epa",
            "away_off_pass_epa",
            "away_off_rush_epa",
        ]
    ]

    rate_stats = _build_rate_stats_frame(pbp)
    return game_epa.merge(rate_stats, on="game_id", how="left")


def load_game_epa(seasons: List[int]) -> pd.DataFrame:
    """
    Load game-level EPA totals from play-by-play data.

    Args:
        seasons: List of NFL seasons to load (e.g., [2005, 2006, ...])

    Returns:
        DataFrame with one row per game and columns:
        game_id, home_off_pass_epa, home_off_rush_epa,
        away_off_pass_epa, away_off_rush_epa
    """
    advanced = load_game_advanced_stats(seasons)
    return advanced[
        [
            "game_id",
            "home_off_pass_epa",
            "home_off_rush_epa",
            "away_off_pass_epa",
            "away_off_rush_epa",
        ]
    ]
