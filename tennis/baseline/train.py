import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tennis.dataset import TennisDataset
from tennis.baseline.model import TennisClassifier

def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Datasets
    train_dataset = TennisDataset(args.data_dir, split='train', num_frames=args.num_frames, feature_dir=args.feature_dir, experts_only=True)
    val_dataset = TennisDataset(args.data_dir, split='val', num_frames=args.num_frames, feature_dir=args.feature_dir, experts_only=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("No training data found.")
        return

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Model
    model = TennisClassifier(
        num_classes=6, 
        input_dim=2048,
        lstm_hidden_size=args.hidden_size, 
        lstm_layers=args.lstm_layers, 
        dropout=args.dropout
    ).to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Add epsilon for numerical stability
            loss = criterion(torch.log(outputs + 1e-8), labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(torch.log(outputs + 1e-8), labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        print(f"Epoch [{epoch+1}/{args.epochs}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Tennis Classifier')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/VIDEO_RGB', help='Path to video data')
    parser.add_argument('--feature_dir', type=str, default='data/features', help='Path to pre-extracted features')
    parser.add_argument('--checkpoint_dir', type=str, default='tennis/baseline/models', help='Directory to save checkpoints')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=8, help='Data loading workers')
    parser.add_argument('--device', type=str, default=None, help='Device (e.g., cuda:0, cpu)')

    # Model arguments
    parser.add_argument('--num_frames', type=int, default=32, help='Frames per video')
    parser.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    
    args = parser.parse_args()
    train(args)
