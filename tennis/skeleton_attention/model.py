"""LSTM-based model augmented with (attention on RGB) + (pose information)."""

import torch
import torch.nn as nn


class SkeletonAttentionTennisClassifier(nn.Module):
    """LSTM-based model augmented with (attention on RGB) + (pose information).

    Uses RGB (LSTM) + Pose (1D CNN).
    """

    def __init__(
        self,
        num_classes=6,
        input_dim=2048,
        pose_input_dim=99,
        lstm_hidden_size=256,
        lstm_layers=1,
        dropout=0.1,
    ):
        super(SkeletonAttentionTennisClassifier, self).__init__()

        # RGB things
        self.rgb_input_dropout = nn.Dropout(dropout)
        self.rgb_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.rgb_output_dropout = nn.Dropout(dropout)
        self.rgb_attention = Attention(lstm_hidden_size)

        # Pose things
        self.pose_cnn = nn.Sequential(
            nn.Conv1d(pose_input_dim, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

        # Fusion
        fusion_dim = lstm_hidden_size + 64
        self.fc = nn.Linear(fusion_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Tuple (rgb_features, pose_features) where
               rgb_features: (B, T, 2048)
               pose_features: (B, T, 99)

        Returns:
            avg_probs (B, num_classes): Prediced probabilities for each class
        """
        # RGB
        rgb_x, pose_x = x
        rgb_x = self.rgb_input_dropout(rgb_x)
        rgb_out, _ = self.rgb_lstm(rgb_x)  # (B, T, H)
        rgb_out = self.rgb_output_dropout(rgb_out)

        # Weigh the RGB frames using attention mechanism
        attn_scores = self.rgb_attention(rgb_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        rgb_feat = torch.sum(attn_weights * rgb_out, dim=1)

        # Process pose data
        pose_x = pose_x.permute(0, 2, 1)
        pose_feat = self.pose_cnn(pose_x)
        pose_feat = pose_feat.squeeze(2)

        # Concatenate them and pass them through the same FC layer
        combined = torch.cat((rgb_feat, pose_feat), dim=1)
        logits = self.fc(combined)
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
