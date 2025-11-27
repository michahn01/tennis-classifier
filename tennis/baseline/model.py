import torch
import torch.nn as nn
import torchvision.models as models

class TennisClassifier(nn.Module):
    def __init__(self, num_classes=6, lstm_hidden_size=256, lstm_layers=1):
        super(TennisClassifier, self).__init__()
        
        # 1. ResNet50 Feature Extractor
        resnet = models.resnet50(pretrained=True)
        # Remove the final FC layer
        # ResNet50 structure: ... -> avgpool -> fc
        # We want everything up to avgpool
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze ResNet params (optional, but good for baseline)
        r param in self.resnet.parameteters():
            param.requires_grad = False
            
        self.feature_dim = 2048
        
        # 2. LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # 3. Classification Head
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (B, T, C, H, W)
        batch_size, time_steps, C, H, W = x.size()
        
        # Flatten B and T to feed into ResNet
        c_in = x.view(batch_size * time_steps, C, H, W)
        
        # Extract features
        features = self.resnet(c_in) # (B*T, 2048, 1, 1)
        features = features.view(batch_size, time_steps, -1) # (B, T, 2048)
        
        # LSTM
        # out: (B, T, H), (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Use the last hidden state
        # h_n shape: (num_layers, B, H)
        last_hidden = h_n[-1] # (B, H)
        
        # Classification
        logits = self.fc(last_hidden) # (B, num_classes)
        
        return logits
