"""Siamese LSTM model for upset prediction with attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple


class SiameseLSTMDataset(Dataset):
    """
    Dataset for Siamese LSTM model.

    Handles:
    - Separate sequences for underdog and favorite (last 5 games each)
    - Matchup-level features
    - Target variable
    """

    def __init__(
        self,
        underdog_sequences: np.ndarray,
        favorite_sequences: np.ndarray,
        matchup_features: np.ndarray,
        targets: np.ndarray,
        underdog_masks: Optional[np.ndarray] = None,
        favorite_masks: Optional[np.ndarray] = None,
    ):
        """
        Initialize dataset.

        Args:
            underdog_sequences: Shape (n_samples, seq_len, n_features)
            favorite_sequences: Shape (n_samples, seq_len, n_features)
            matchup_features: Shape (n_samples, n_matchup_features)
            targets: Shape (n_samples,)
            underdog_masks: Optional shape (n_samples, seq_len) for padding
            favorite_masks: Optional shape (n_samples, seq_len) for padding
        """
        self.underdog_sequences = torch.FloatTensor(underdog_sequences)
        self.favorite_sequences = torch.FloatTensor(favorite_sequences)
        self.matchup_features = torch.FloatTensor(matchup_features)
        self.targets = torch.FloatTensor(targets)
        self.underdog_masks = (
            torch.FloatTensor(underdog_masks) if underdog_masks is not None else None
        )
        self.favorite_masks = (
            torch.FloatTensor(favorite_masks) if favorite_masks is not None else None
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        und_mask = self.underdog_masks[idx] if self.underdog_masks is not None else None
        fav_mask = self.favorite_masks[idx] if self.favorite_masks is not None else None

        return (
            self.underdog_sequences[idx],
            self.favorite_sequences[idx],
            self.matchup_features[idx],
            self.targets[idx],
            und_mask if und_mask is not None else torch.ones(self.underdog_sequences.shape[1]),
            fav_mask if fav_mask is not None else torch.ones(self.favorite_sequences.shape[1]),
        )


class SiameseUpsetLSTM(nn.Module):
    """
    True Siamese LSTM architecture for upset prediction.

    Architecture per design spec:
    - Underdog's last 5 games → Shared LSTM Encoder → Attention → Encoding A
    - Favorite's last 5 games → Shared LSTM Encoder → Attention → Encoding B
    - Concat(Encoding A, Encoding B, Matchup Features) → Dense → P(upset)

    The same LSTM encoder processes both teams' sequences independently,
    learning general team performance patterns applicable to any team.
    """

    def __init__(
        self,
        sequence_features: int = 4,
        matchup_features: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize Siamese LSTM model.

        Args:
            sequence_features: Number of features per timestep per team
            matchup_features: Number of matchup-level features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_size = hidden_size

        # Shared LSTM encoder for both teams
        self.shared_lstm = nn.LSTM(
            input_size=sequence_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Attention layer (shared for both teams)
        self.attention = nn.Linear(hidden_size, 1)

        # Dense layers after concatenation
        # Combined: underdog encoding + favorite encoding + matchup features
        combined_size = hidden_size * 2 + matchup_features
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

    def _encode_team(
        self,
        sequences: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a team's game history through the shared LSTM.

        Args:
            sequences: Shape (batch, seq_len, features)
            mask: Optional shape (batch, seq_len) for padded sequences

        Returns:
            Tuple of (context vector, attention weights)
        """
        # LSTM encoding
        lstm_out, _ = self.shared_lstm(sequences)  # (batch, seq_len, hidden)

        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            attn_weights = attn_weights * mask
            # Re-normalize after masking
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted sum of LSTM outputs
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden)

        return context, attn_weights.squeeze(-1)

    def forward(
        self,
        underdog_sequences: torch.Tensor,
        favorite_sequences: torch.Tensor,
        matchup_features: torch.Tensor,
        underdog_mask: Optional[torch.Tensor] = None,
        favorite_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with true siamese processing.

        Args:
            underdog_sequences: Shape (batch, seq_len, features)
            favorite_sequences: Shape (batch, seq_len, features)
            matchup_features: Shape (batch, matchup_features)
            underdog_mask: Optional shape (batch, seq_len)
            favorite_mask: Optional shape (batch, seq_len)

        Returns:
            Probability of upset, shape (batch, 1)
        """
        # Encode each team through the SHARED encoder
        underdog_encoding, _ = self._encode_team(underdog_sequences, underdog_mask)
        favorite_encoding, _ = self._encode_team(favorite_sequences, favorite_mask)

        # Concatenate encodings with matchup features
        combined = torch.cat(
            [underdog_encoding, favorite_encoding, matchup_features], dim=1
        )

        # Final prediction
        return self.fc(combined)

    def get_attention_weights(
        self,
        underdog_sequences: torch.Tensor,
        favorite_sequences: torch.Tensor,
        underdog_mask: Optional[torch.Tensor] = None,
        favorite_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights for interpretability.

        Args:
            underdog_sequences: Input sequences for underdog
            favorite_sequences: Input sequences for favorite
            underdog_mask: Optional padding mask for underdog
            favorite_mask: Optional padding mask for favorite

        Returns:
            Tuple of (underdog_attention, favorite_attention) weights per timestep
        """
        with torch.no_grad():
            _, underdog_attn = self._encode_team(underdog_sequences, underdog_mask)
            _, favorite_attn = self._encode_team(favorite_sequences, favorite_mask)

        return underdog_attn, favorite_attn


# Backward compatibility aliases
LSTMDataset = SiameseLSTMDataset
UpsetLSTM = SiameseUpsetLSTM
