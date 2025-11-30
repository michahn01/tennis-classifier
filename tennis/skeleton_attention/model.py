import torch
import torch.nn as nn


class SkeletonAttentionTennisClassifier(nn.Module):
    """
    Multi-stream classifier: RGB (LSTM) + Pose (1D CNN).
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

        # Feed-forward attention for RGB stream
        self.rgb_attention = Attention(lstm_hidden_size)

        # Pose Stream (1D CNN)
        # Input pose data is originally in (B, T, 99).
        # But before passing into pose_cnn, we must permute to (B, 99, T) for Conv1d
        self.pose_cnn = nn.Sequential(
            # conv, norm, activation, dropout
            nn.Conv1d(pose_input_dim, 32, kernel_size=3, padding=1),
            # Note: GroupNorm(1, ...) is equivalent to LayerNorm but we use GroupNorm
            # instead because LayerNorm expects (B, T, C) but we have (B, C, T)
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            # now we repeat that pattern. the intent is to learn
            # features hierarchically.
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # avg across time
            nn.AdaptiveAvgPool1d(1),
        )  # will output (B, 64, 1)

        # Fusion
        # RGB LSTM output: (B, T, H) -> Mean pooling -> (B, H)
        fusion_dim = lstm_hidden_size + 64
        self.fc = nn.Linear(fusion_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Args:
            x: Tuple (rgb_features, pose_features)
               rgb_features: (B, T, 2048)
               pose_features: (B, T, 99)
        """
        rgb_x, pose_x = x

        # First, do the RGB Stream
        rgb_x = self.rgb_input_dropout(rgb_x)
        rgb_out, _ = self.rgb_lstm(rgb_x)  # (B, T, H)
        rgb_out = self.rgb_output_dropout(rgb_out)

        # Compute attention scores: (B, T, H) -> (B, T, 1)
        attn_scores = self.rgb_attention(rgb_out)

        # Compute attention weights: (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Compute context vector: sum(weights * lstm_out) -> (B, H)
        rgb_feat = torch.sum(attn_weights * rgb_out, dim=1)

        # Then, do the pose stream
        # Permute for Conv1d: (B, T, 99) to (B, 99, T)
        pose_x = pose_x.permute(0, 2, 1)
        pose_feat = self.pose_cnn(pose_x)  # (B, 128, 1)
        pose_feat = pose_feat.squeeze(2)  # (B, 128)

        # Finally, fuse the features
        combined = torch.cat((rgb_feat, pose_feat), dim=1)
        logits = self.fc(combined)
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
