"""LSTM model for upset prediction with attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple


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
        masks: Optional[np.ndarray] = None,
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
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
        mask: Optional[torch.Tensor] = None,
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
        mask: Optional[torch.Tensor] = None,
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
