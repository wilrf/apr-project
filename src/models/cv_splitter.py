"""Time-series cross-validation for NFL seasons."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Iterator, Tuple


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
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
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
                f"Need at least {self.n_folds + 1} seasons for {self.n_folds} folds, "
                f"but only have {n_seasons} seasons"
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
