"""Baseline LSTM-based model."""

import torch
import torch.nn as nn


class TennisClassifier(nn.Module):
    """Baseline: LSTM over features from RGB videos."""

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
        """Forward pass of the model.

        Args:
            x (B, T, 2048): Pre-extracted features of RGB videos

        Returns:
            avg_probs (B, num_classes): Prediced probabilities for each class
        """
        # Apply dropout to inputs
        # (B, T, 2048) to (B, T, 2048)
        x = self.input_dropout(x)

        # Run entire sequence through LSTM cells
        # (B, T, 2048) to (B, T, lstm_hidden_size)
        lstm_out, _ = self.lstm(x)

        # Apply dropout to hidden states
        # (B, T, lstm_hidden_size) to (B, T, lstm_hidden_size)
        lstm_out = self.output_dropout(lstm_out)

        # Pass through a fully-connected layer to get logits
        # (B, T, lstm_hidden_size) to (B, T, num_classes)
        logits = self.fc(lstm_out)

        # Softmax the class logits
        # (B, T, num_classes) to (B, T, num_classes)
        probs = self.softmax(logits)

        # Average probabilities across time
        # (B, T, num_classes) to (B, num_classes)
        avg_probs = torch.mean(probs, dim=1)

        return avg_probs
