"""LSTM-based model augmented with pose information."""

import torch
import torch.nn as nn


class SkeletonTennisClassifier(nn.Module):
    """LSTM-based model augmented with pose information.

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
        super(SkeletonTennisClassifier, self).__init__()

        # RGB Stream, same as baseline (LSTM)
        self.rgb_input_dropout = nn.Dropout(dropout)
        self.rgb_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.rgb_output_dropout = nn.Dropout(dropout)

        # Pose Stream (1D CNN)
        # Input pose data is originally in (B, T, 99).
        # But before passing into pose_cnn, we must permute to (B, 99, T) for Conv1d
        self.pose_cnn = nn.Sequential(
            # conv, norm, activation, dropout
            nn.Conv1d(pose_input_dim, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Now we repeat the above pattern. The intent is to learn features
            # hierarchically (i.e., learn higher level features this time).
            nn.Conv1d(
                32, 64, kernel_size=3, padding=1
            ),  # more output channels this time though
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # avg across time
            nn.AdaptiveAvgPool1d(1),
        )  # will output (B, 64, 1)

        # Fusion of RGB stream and Pose stream
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
        rgb_x, pose_x = x

        # ********** First, do the RGB stream **********
        # (B, T, 2048) to (B, T, 2048)
        rgb_x = self.rgb_input_dropout(rgb_x)

        # (B, T, 2048) to (B, T, lstm_hidden_size)
        rgb_out, _ = self.rgb_lstm(rgb_x)

        # (B, T, lstm_hidden_size) to (B, T, lstm_hidden_size)
        rgb_out = self.rgb_output_dropout(rgb_out)

        # Temporal average pooling for RGB
        # i.e., we average information across frames
        # (B, T, lstm_hidden_size) to (B, lstm_hidden_size)
        rgb_feat = torch.mean(rgb_out, dim=1)

        # ********** Then, do the Pose stream **********
        # Permute for Conv1d: (B, T, 99) to (B, 99, T)
        pose_x = pose_x.permute(0, 2, 1)

        # Apply the 2 x (conv, norm, act, dropout) + (pool) pipeline
        pose_feat = self.pose_cnn(pose_x)

        # (B, 64, 1) to (B, 64)
        pose_feat = pose_feat.squeeze(2)

        # ********** Finally, fuse the features **********
        combined = torch.cat((rgb_feat, pose_feat), dim=1)
        logits = self.fc(combined)
        probs = self.softmax(logits)

        return probs
