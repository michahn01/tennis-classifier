import torch
from tennis.model import TennisClassifier

def verify_model():
    print("Initializing model...")
    model = TennisClassifier(num_classes=6, lstm_hidden_size=256, lstm_layers=1)
    print("Model initialized.")
    
    # Create dummy input: (Batch, Time, Channels, Height, Width)
    batch_size = 2
    time_steps = 32
    channels = 3
    height = 224
    width = 224
    
    x = torch.randn(batch_size, time_steps, channels, height, width)
    print(f"Input shape: {x.shape}")
    
    print("Running forward pass...")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 6), f"Expected output shape (2, 6), got {output.shape}"
    print("Verification successful!")

if __name__ == "__main__":
    verify_model()
