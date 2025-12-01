"""LSTM-based model augmented with attention mechanism."""

import torch
import torch.nn as nn


class AttentionTennisClassifier(nn.Module):
    """LSTM-based model augmented with attention mechanism."""

    def __init__(
        self,
        num_classes=6,
        input_dim=2048,
        lstm_hidden_size=256,
        lstm_layers=1,
        dropout=0.1,
    ):
        super(AttentionTennisClassifier, self).__init__()

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
        """Forward pass of the model.

        Args:
            x (B, T, 2048): Pre-extracted features of RGB videos

        Returns:
            probs (B, num_classes): Prediced probabilities for each class
        """
        # (B, T, 2048) to (B, T, 2048)
        x = self.input_dropout(x)

        # (B, T, 2048) to (B, T, lstm_hidden_size)
        lstm_out, _ = self.lstm(x)

        # (B, T, lstm_hidden_size) to (B, T, lstm_hidden_size)
        lstm_out = self.output_dropout(lstm_out)

        # ********** Variation from baseline **********
        # (B, T, lstm_hidden_size) to (B, T, 1)
        attn_scores = self.attention(lstm_out)

        # (B, T, 1) to (B, T, 1)
        # We're computing a weight for each time step (i.e., frame)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # (B, T, lstm_hidden_size) to (B, lstm_hidden_size)
        # We're compressing each video into a weighted sum of all of its frames' embeds
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        # Just one last FC to wrap it up into logits
        # (B, lstm_hidden_size) to (B, num_classes)
        logits = self.fc(context_vector)

        # (B, num_classes) to (B, num_classes)
        probs = self.softmax(logits)

        return probs


class Attention(nn.Module):
    """Computes a weighted average of the LSTM output across time."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, time, hidden_size)
        # scores: (batch, time, 1) - i.e., one score per time step
        return self.attn(lstm_out)
