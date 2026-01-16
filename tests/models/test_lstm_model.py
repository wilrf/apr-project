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

    def test_attention_weights_sum_to_one(self, model):
        """Test that attention weights sum to approximately 1."""
        batch_size = 4
        seq = torch.randn(batch_size, 5, 15)

        weights = model.get_attention_weights(seq)

        # Each sample's weights should sum to ~1
        for i in range(batch_size):
            assert abs(weights[i].sum().item() - 1.0) < 0.01
