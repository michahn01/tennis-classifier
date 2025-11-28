"""
Baseline Model: ResNet50 + LSTM
Simple implementation without attention or skeleton features.

This is the baseline model that serves as the foundation for comparison.
It uses:
1. ResNet50 (pretrained on ImageNet) to extract visual features from each frame
2. LSTM to model temporal dependencies across frames
3. A fully connected layer to classify the stroke type

The model takes a video (sequence of frames) and outputs class logits for 6 stroke types.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class TennisClassifier(nn.Module):
    """
    Baseline: ResNet50 + LSTM classifier for tennis stroke classification.
    
    Architecture Overview:
    ---------------------
    Input: Video frames (B, T, 3, 224, 224)
           B = batch size, T = number of frames (e.g., 32), 3 = RGB channels, 224x224 = image size
    
    Step 1: ResNet50 Feature Extraction
           - Extract 2048-dimensional feature vector from each frame
           - Output: (B, T, 2048)
    
    Step 2: LSTM Temporal Modeling
           - Process sequence of frame features
           - Output: Hidden states (B, T, 256) and final hidden state (B, 256)
    
    Step 3: Classification
           - Use last hidden state to predict stroke class
           - Output: (B, 6) logits for 6 classes
    
    Classes: backhand, forehand, service, bvolley, fvolley, smash
    """
    def __init__(self, num_classes=6, lstm_hidden_size=256, lstm_layers=1, dropout=0.0, pooling='mean'):
        """
        Initialize the baseline model.
        
        Args:
            num_classes (int): Number of stroke classes (default: 6)
            lstm_hidden_size (int): Dimension of LSTM hidden state (default: 256)
            lstm_layers (int): Number of LSTM layers (default: 1)
            dropout (float): Dropout probability (default: 0.0)
            pooling (str): Pooling strategy for classification ('last' or 'mean'). 
                           'last' uses the final hidden state.
                           'mean' averages logits across all time steps.
        """
        super(TennisClassifier, self).__init__()
        self.pooling = pooling
        
        # ====================================================================
        # STEP 1: ResNet50 Feature Extractor
        # ====================================================================
        # Load pretrained ResNet50 model (trained on ImageNet)
        # This gives us a strong visual feature extractor without training from scratch
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # ResNet50 structure: conv layers -> batch norm -> ReLU -> ... -> avgpool -> fc (1000 classes)
        # We only want the feature extractor part, not the final classification layer
        # Remove the last layer (the 1000-class FC layer) to get feature vectors
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze ResNet params (optional, but good for baseline)
        # This is common practice when using pretrained models:
        # - ResNet already learned good visual features from ImageNet
        # - We only train the LSTM and classifier on top
        # - Faster training and less risk of overfitting
        for param in self.resnet.parameters():
            param.requires_grad = False
            

        # ResNet50 outputs 2048-dimensional feature vectors after avgpool
        # This is the size of features extracted from each frame
        self.feature_dim = 2048
        
        # ====================================================================
        # STEP 2: LSTM for Temporal Modeling
        # ====================================================================
        # LSTM processes the sequence of frame features to capture temporal patterns
        # For example: how the racket moves, body posture changes, etc.
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # ====================================================================
        # STEP 3: Classification Head
        # ====================================================================
        # Final fully connected layer maps LSTM hidden state to class logits
        # Input: LSTM hidden state (256-dim)
        # Output: Class logits (6 classes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the baseline model.
        
        Data Flow:
        ----------
        Input:  (B, T, 3, 224, 224) - Batch of videos
        ↓
        ResNet: (B, T, 2048) - Frame features
        ↓
        LSTM:   (B, 256) - Video representation (last hidden state)
        ↓
        FC:     (B, 6) - Class logits
        
        Args:
            x: Tensor of shape (B, T, C, H, W)
               B = batch size (number of videos)
               T = time steps (number of frames per video, e.g., 32)
               C = channels (3 for RGB)
               H, W = height, width (224, 224)
        
        Returns:
            logits: Tensor of shape (B, num_classes)
                   Raw scores (logits) for each class before softmax
        """

        # Get input dimensions
        # x shape: (B, T, C, H, W)
        batch_size, time_steps, C, H, W = x.size()
        

        # ====================================================================
        # ResNet Feature Extraction
        # ====================================================================
        # ResNet expects input shape: (N, C, H, W) where N is number of images
        # We have (B, T, C, H, W), so we need to flatten batch and time dimensions
        # This allows us to process all frames through ResNet in one pass

        # Flatten B and T to feed into ResNet
        c_in = x.view(batch_size * time_steps, C, H, W)
        # Shape: (B*T, 3, 224, 224)
        # Example: (4 videos * 32 frames, 3, 224, 224) = (128, 3, 224, 224)

        # Extract features from each frame using ResNet50
        features = self.resnet(c_in) # (B*T, 2048, 1, 1) : ResNet outputs features with spatial dims 1x1

        # Reshape back to sequence format: (batch, time, features)
        features = features.view(batch_size, time_steps, -1) # (B, T, 2048)
        # Example: (4, 32, 2048) - 4 videos, each with 32 frames, each frame has 2048 features
        
        # ====================================================================
        # LSTM Temporal Modeling
        # ====================================================================
        # Process the sequence of frame features through LSTM
        # LSTM maintains hidden state that accumulates information over time

        # Apply dropout to the input features of the LSTM
        # This allows for regularization even if num_layers=1
        features = self.dropout(features)

        # out: (B, T, H), (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(features)
        # lstm_out: (B, T, H) - Hidden state at each time step
        # h_n: (num_layers, B, H) - Final hidden state for each layer
        # c_n: (num_layers, B, H) - Final cell state for each layer
        


        # ====================================================================
        # Extract Video Representation & Classification
        # ====================================================================
        
        if self.pooling == 'mean':
            # Apply classifier to ALL time steps
            # lstm_out: (B, T, H)
            # logits_seq: (B, T, num_classes)
            logits_seq = self.fc(self.dropout(lstm_out))
            
            # Average logits across time
            # logits: (B, num_classes)
            logits = torch.mean(logits_seq, dim=1)
            
        else: # 'last'
            # For baseline, we use the last hidden state from the top LSTM layer
            # This represents the entire video sequence in a single vector
            # h_n shape: (num_layers, B, H)
            # h_n[-1] gets the last layer: (B, H)

            # Use the last hidden state
            # h_n shape: (num_layers, B, H)
            last_hidden = h_n[-1] # (B, H)
            # Shape: (B, 256) - One vector per video representing the entire sequence

            
            # Map the video representation to class logits
            # The FC layer learns to associate video features with stroke types
            logits = self.fc(self.dropout(last_hidden))
            # Shape: (B, 6) - One set of logits per video

        
        return logits
