import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tennis.dataset import TennisDataset
from tennis.attention.model import AttentionTennisClassifier


def test(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Check for model checkpoint
    model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")

    # Dataset
    test_dataset = TennisDataset(args.feature_dir, split="test")
    print(f"Test samples: {len(test_dataset)}")

    if len(test_dataset) == 0:
        print("No test data found. Please run extract_features.py first.")
        return

    # DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Model
    model = AttentionTennisClassifier(
        num_classes=6,
        input_dim=2048,
        lstm_hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)

    # Load weights
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluation
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total if total > 0 else 0
    print(f"Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Tennis Classifier")

    # Data arguments
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="data/features",
        help="Path to pre-extracted features",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="tennis/attention/models",
        help="Directory containing best_model.pth",
    )

    # Model arguments (must match training)
    parser.add_argument("--hidden_size", type=int, default=256, help="LSTM hidden size")
    parser.add_argument(
        "--lstm_layers", type=int, default=1, help="Number of LSTM layers"
    )
    parser.add_argument(
        # for test, we use .eval() so dropout is ignored actually
        # we include it here for completeness
        "--dropout", type=float, default=0, help="Dropout probability"
    )

    # Other arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Data loading workers"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--infer_params",
        action="store_true",
        help="Infer model parameters from checkpoint_dir path",
    )

    args = parser.parse_args()

    if args.infer_params:
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir must be specified when using --infer_params")
        
        import re
        # Expecting format like .../search_h512_l2_d0.5/...
        match = re.search(r"search_h(\d+)_l(\d+)_d([\d.]+)", args.checkpoint_dir)
        if match:
            args.hidden_size = int(match.group(1))
            args.lstm_layers = int(match.group(2))
            args.dropout = float(match.group(3))
            print(f"Inferred params: hidden_size={args.hidden_size}, lstm_layers={args.lstm_layers}, dropout={args.dropout}")
        else:
            raise ValueError(f"Could not infer params from path: {args.checkpoint_dir}")

    test(args)
