# tests/models/test_lstm_model.py
"""Tests for Siamese LSTM model."""
import pytest
import torch
import numpy as np
from src.models.lstm_model import SiameseUpsetLSTM, SiameseLSTMDataset
from src.models.sequence_builder import SEQUENCE_FEATURES, MATCHUP_FEATURES


class TestSiameseLSTMDataset:
    def test_dataset_length(self):
        """Test dataset returns correct length."""
        n_samples = 100
        seq_len = 5
        n_features = len(SEQUENCE_FEATURES)
        n_matchup = len(MATCHUP_FEATURES)

        und_seq = np.random.randn(n_samples, seq_len, n_features)
        fav_seq = np.random.randn(n_samples, seq_len, n_features)
        matchup = np.random.randn(n_samples, n_matchup)
        targets = np.random.randint(0, 2, n_samples).astype(float)

        dataset = SiameseLSTMDataset(und_seq, fav_seq, matchup, targets)
        assert len(dataset) == n_samples

    def test_dataset_returns_tensors(self):
        """Test dataset returns torch tensors."""
        und_seq = np.random.randn(10, 5, len(SEQUENCE_FEATURES))
        fav_seq = np.random.randn(10, 5, len(SEQUENCE_FEATURES))
        matchup = np.random.randn(10, len(MATCHUP_FEATURES))
        targets = np.random.randint(0, 2, 10).astype(float)

        dataset = SiameseLSTMDataset(und_seq, fav_seq, matchup, targets)
        result = dataset[0]

        # Should return 6 items: und_seq, fav_seq, matchup, target, und_mask, fav_mask
        assert len(result) == 6
        for item in result:
            assert isinstance(item, torch.Tensor)

    def test_dataset_with_masks(self):
        """Test dataset handles masks correctly."""
        und_seq = np.random.randn(10, 5, len(SEQUENCE_FEATURES))
        fav_seq = np.random.randn(10, 5, len(SEQUENCE_FEATURES))
        matchup = np.random.randn(10, len(MATCHUP_FEATURES))
        targets = np.random.randint(0, 2, 10).astype(float)
        und_mask = np.ones((10, 5))
        fav_mask = np.ones((10, 5))

        dataset = SiameseLSTMDataset(
            und_seq, fav_seq, matchup, targets, und_mask, fav_mask
        )
        result = dataset[0]

        assert len(result) == 6


class TestSiameseUpsetLSTM:
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return SiameseUpsetLSTM(
            sequence_features=len(SEQUENCE_FEATURES),
            matchup_features=len(MATCHUP_FEATURES),
            hidden_size=64,
            num_layers=2,
        )

    def test_forward_returns_probabilities(self, model):
        """Test forward pass returns values between 0 and 1."""
        batch_size = 8
        seq_len = 5
        n_features = len(SEQUENCE_FEATURES)
        n_matchup = len(MATCHUP_FEATURES)

        und_seq = torch.randn(batch_size, seq_len, n_features)
        fav_seq = torch.randn(batch_size, seq_len, n_features)
        matchup = torch.randn(batch_size, n_matchup)

        output = model(und_seq, fav_seq, matchup)

        assert output.shape == (batch_size, 1)
        assert all(0 <= p <= 1 for p in output.squeeze())

    def test_siamese_uses_shared_encoder(self, model):
        """Test that the same encoder is used for both teams."""
        # The model should have a single shared_lstm, not two separate ones
        assert hasattr(model, "shared_lstm")
        assert isinstance(model.shared_lstm, torch.nn.LSTM)

    def test_model_handles_masking(self, model):
        """Test model handles padded sequences with mask."""
        batch_size = 4
        seq_len = 5
        n_features = len(SEQUENCE_FEATURES)
        n_matchup = len(MATCHUP_FEATURES)

        und_seq = torch.randn(batch_size, seq_len, n_features)
        fav_seq = torch.randn(batch_size, seq_len, n_features)
        matchup = torch.randn(batch_size, n_matchup)

        # Masks: first 2 have full sequences, last 2 have only 3 games
        und_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
            ]
        ).float()
        fav_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1],
            ]
        ).float()

        output = model(und_seq, fav_seq, matchup, und_mask, fav_mask)
        assert output.shape == (batch_size, 1)

    def test_attention_weights_sum_to_one(self, model):
        """Test that attention weights sum to approximately 1 for each team."""
        batch_size = 4
        seq_len = 5
        n_features = len(SEQUENCE_FEATURES)

        und_seq = torch.randn(batch_size, seq_len, n_features)
        fav_seq = torch.randn(batch_size, seq_len, n_features)

        und_weights, fav_weights = model.get_attention_weights(und_seq, fav_seq)

        # Each sample's weights should sum to ~1
        for i in range(batch_size):
            assert abs(und_weights[i].sum().item() - 1.0) < 0.01
            assert abs(fav_weights[i].sum().item() - 1.0) < 0.01

    def test_attention_weights_with_mask(self, model):
        """Test attention weights respect masking."""
        batch_size = 2
        seq_len = 5
        n_features = len(SEQUENCE_FEATURES)

        und_seq = torch.randn(batch_size, seq_len, n_features)
        fav_seq = torch.randn(batch_size, seq_len, n_features)

        # Mask out first 3 positions for underdog
        und_mask = torch.tensor(
            [
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
            ]
        ).float()
        fav_mask = torch.ones(batch_size, seq_len)

        und_weights, fav_weights = model.get_attention_weights(
            und_seq, fav_seq, und_mask, fav_mask
        )

        # Masked positions should have ~0 attention
        for i in range(batch_size):
            assert und_weights[i, :3].sum().item() < 0.01
            # Valid positions should have non-zero attention
            assert und_weights[i, 3:].sum().item() > 0.9
