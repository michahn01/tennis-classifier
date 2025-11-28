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
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """
        Args:
            x: (B, T, 2048) - Pre-extracted features
        Returns:
            avg_probs: (B, num_classes) - Averaged probabilities across time
        """
        x = self.input_dropout(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.output_dropout(lstm_out)

        logits = self.fc(lstm_out)
        probs = self.softmax(logits)

        # Average probabilities across time
        avg_probs = torch.mean(probs, dim=1)

        return avg_probs
