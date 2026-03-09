# tests/models/test_cv_splitter.py
import pytest
import pandas as pd
from src.models.cv_splitter import TimeSeriesCVSplitter


class TestTimeSeriesCVSplitter:
    @pytest.fixture
    def sample_data(self):
        """Sample data spanning multiple seasons."""
        seasons = list(range(2016, 2023))  # 2016-2022 (7 seasons for 6-fold CV)
        games_per_season = 10
        return pd.DataFrame(
            {
                "game_id": [f"g{i}" for i in range(len(seasons) * games_per_season)],
                "season": [s for s in seasons for _ in range(games_per_season)],
            }
        )

    def test_splitter_creates_correct_number_of_folds(self, sample_data):
        """Test that splitter creates expected number of folds."""
        splitter = TimeSeriesCVSplitter(n_folds=6)
        folds = list(splitter.split(sample_data))
        assert len(folds) == 6

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

    def test_raises_error_insufficient_seasons(self):
        """Test that error raised if not enough seasons."""
        df = pd.DataFrame(
            {
                "game_id": ["g1", "g2"],
                "season": [2022, 2023],
            }
        )
        splitter = TimeSeriesCVSplitter(n_folds=6)
        with pytest.raises(ValueError):
            list(splitter.split(df))
