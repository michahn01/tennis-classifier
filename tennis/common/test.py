import os
import argparse
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from tennis.common.model_utils import (
    get_model_class,
    get_dataset_class,
    requires_pose,
    get_model_choices,
)
import tennis.common.config as config
from tennis.dataset import TennisDataset


def test(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")



    # Set default model path if not provided
    if not args.model_path:
        # Try to find a default model in the standard location
        default_path = f"tennis/{args.model_type}/models/best_model.pth"
        if os.path.exists(default_path):
            args.model_path = default_path
        else:
             # If we can't find a default, we can't proceed without a path
             pass 

    if not args.model_path:
         raise ValueError("model_path must be provided")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}")

    # Dataset
    DatasetClass = get_dataset_class(args.model_type)
    if requires_pose(args.model_type):
        if not args.pose_dir:
            raise ValueError("pose_dir must be provided for skeleton model")
        test_dataset = DatasetClass(args.feature_dir, args.pose_dir, split="test")
    else:
        test_dataset = DatasetClass(args.feature_dir, split="test")
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
    model_class = get_model_class(args.model_type)
    model = model_class(
        num_classes=6,
        input_dim=2048,
        lstm_hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)

    # Load weights
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    class_names = list(TennisDataset.CLASS_MAPPING.keys())
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    # Save plot
    if args.save_dir:
        vis_dir = args.save_dir
    else:
        vis_dir = os.path.join("visualizations", args.model_type)
    os.makedirs(vis_dir, exist_ok=True)
    
    plot_filename = f"confusion_matrix_h{args.hidden_size}_l{args.lstm_layers}_d{args.dropout}.png"
    save_path = os.path.join(vis_dir, plot_filename)
    
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

    acc = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Tennis Classifier")

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=get_model_choices(),
        help="Type of model to test",
    )

    # Data arguments
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=config.DEFAULT_FEATURE_DIR,
        help="Path to pre-extracted features",
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default=config.DEFAULT_POSE_DIR,
        help="Path to pre-extracted pose features (required for skeleton model)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file (e.g., tennis/baseline/models/best_model_h512_l2_d0.5.pth)",
    )

    # Model arguments (must match training)
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=config.DEFAULT_HIDDEN_SIZE,
        help="LSTM hidden size",
    )
    parser.add_argument(
        "--lstm_layers",
        type=int,
        default=config.DEFAULT_LSTM_LAYERS,
        help="Number of LSTM layers",
    )
    parser.add_argument(
        # for test, we use .eval() so dropout is ignored actually
        # we include it here for completeness
        "--dropout",
        type=float,
        default=0,
        help="Dropout probability",
    )

    # Other arguments
    parser.add_argument(
        "--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=config.DEFAULT_NUM_WORKERS,
        help="Data loading workers",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--infer_params",
        action="store_true",
        help="Infer model parameters from model_path filename",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save visualizations and results",
    )

    args = parser.parse_args()

    if args.infer_params:
        if not args.model_path:
            raise ValueError(
                "--model_path must be specified when using --infer_params"
            )

        import re

        # Expecting format like .../best_model_h512_l2_d0.5.pth
        # Or even just finding the pattern in the string
        match = re.search(r"h(\d+)_l(\d+)_d([\d.]+)", args.model_path)
        if match:
            args.hidden_size = int(match.group(1))
            args.lstm_layers = int(match.group(2))
            args.dropout = float(match.group(3))
            print(
                f"Inferred params: hidden_size={args.hidden_size}, lstm_layers={args.lstm_layers}, dropout={args.dropout}"
            )
        else:
            raise ValueError(f"Could not infer params from path: {args.model_path}")

    test(args)
