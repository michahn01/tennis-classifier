"""
Revamped Baseline Model: Matches LRCNN architecture
"""

import torch
import torch.nn as nn


class TennisClassifier(nn.Module):
    """
    Revamped Baseline: LRCNN-like architecture.
    Input (B, T, 2048) -> Dropout -> LSTM -> Dropout -> FC -> Softmax -> Mean -> Output (B, 6)
    """

    def __init__(
        self,
        num_classes=6,
        input_dim=2048,
        lstm_hidden_size=256,
        lstm_layers=1,
        dropout=0.1,
    ):
        super(TennisClassifier, self).__init__()

        self.input_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.output_dropout = nn.Dropout(dropout)

        # Feed-forward attention
        self.attention = Attention(lstm_hidden_size)

        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Args:
            x: (B, T, 2048) - Pre-extracted features
        Returns:
            probs: (B, num_classes) - Class probabilities
        """
        x = self.input_dropout(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.output_dropout(lstm_out)

        # 1) Compute attention scores: (B, T, H) -> (B, T, 1)
        attn_scores = self.attention(lstm_out)
        
        # 2) Compute attention weights: (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # 3) Compute context vector: sum(weights * lstm_out) -> (B, H)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        # 4) Classification
        logits = self.fc(context_vector)
        probs = self.softmax(logits)

        return probs


class Attention(nn.Module):
    """
    Computes a weighted average of the LSTM output across time.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (Batch, Time, Hidden)
        # scores: (Batch, Time, 1)
        return self.attn(lstm_out)
