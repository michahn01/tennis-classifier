import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tennis.dataset import TennisDataset, TennisPoseDataset
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend


def get_model_class(model_type):
    if model_type == "baseline":
        from tennis.baseline.model import TennisClassifier

        return TennisClassifier
    elif model_type == "attention":
        from tennis.attention.model import AttentionTennisClassifier

        return AttentionTennisClassifier
    elif model_type == "skeleton":
        from tennis.skeleton.model import SkeletonTennisClassifier

        return SkeletonTennisClassifier
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Datasets
    if args.k_folds > 0:
        if args.fold_index < 0 or args.fold_index >= args.k_folds:
            raise ValueError(
                f"Fold index {args.fold_index} out of range for {args.k_folds} folds."
            )

        print(f"Running K-Fold CV: Fold {args.fold_index + 1}/{args.k_folds}")

        # Load all samples from train_val (or train + val)
        # We use TennisDataset helper to load raw samples without initializing the full dataset yet
        if args.model_type == "skeleton":
            if not args.pose_dir:
                raise ValueError("pose_dir must be provided for skeleton model")
            temp_dataset = TennisPoseDataset(args.feature_dir, args.pose_dir, split="train_val")
        else:
            temp_dataset = TennisDataset(args.feature_dir, split="train_val")

        all_samples = temp_dataset.samples

        if len(all_samples) == 0:
            print("No training data found. Please run extract_features.py first.")
            return

        # Group by Player ID
        player_samples = {}
        for sample_data, label in all_samples:
            if isinstance(sample_data, tuple):
                # TennisPoseDataset returns (rgb_path, pose_path)
                path = sample_data[0]
            else:
                path = sample_data

            filename = os.path.basename(path)
            # Extract player ID (e.g., p10 -> 10)
            if filename.startswith("p") and "_" in filename:
                player_id = filename.split("_")[0]
            else:
                player_id = filename  # Fallback

            if player_id not in player_samples:
                player_samples[player_id] = []
            player_samples[player_id].append((sample_data, label))

        # k-fold logic
        player_ids = sorted(list(player_samples.keys()))
        random.seed(5)
        random.shuffle(player_ids)

        n_players = len(player_ids)
        fold_size = n_players // args.k_folds
        remainder = n_players % args.k_folds

        folds = []
        start = 0
        for i in range(args.k_folds):
            # Distribute remainder among first few folds
            count = fold_size + (1 if i < remainder else 0)
            folds.append(player_ids[start : start + count])
            start += count

        for i, fold in enumerate(folds):
            print(f"fold {i}: len = {len(fold)}")
            print(fold)

        val_players = set(folds[args.fold_index])
        train_players = set()
        for i in range(args.k_folds):
            if i != args.fold_index:
                train_players.update(folds[i])

        train_samples = []
        val_samples = []

        for pid, samples in player_samples.items():
            if pid in train_players:
                train_samples.extend(samples)
            elif pid in val_players:
                val_samples.extend(samples)

        if args.model_type == "skeleton":
            train_dataset = TennisPoseDataset(
                args.feature_dir, args.pose_dir, split="custom", samples=train_samples
            )
            val_dataset = TennisPoseDataset(
                args.feature_dir, args.pose_dir, split="custom", samples=val_samples
            )
        else:
            train_dataset = TennisDataset(
                args.feature_dir, split="custom", samples=train_samples
            )
            val_dataset = TennisDataset(
                args.feature_dir, split="custom", samples=val_samples
            )

    else:
        # Standard split
        if args.model_type == "skeleton":
            if not args.pose_dir:
                raise ValueError("pose_dir must be provided for skeleton model")
            train_dataset = TennisPoseDataset(
                args.feature_dir, args.pose_dir, split="train"
            )
            val_dataset = TennisPoseDataset(
                args.feature_dir, args.pose_dir, split="val"
            )
        else:
            train_dataset = TennisDataset(args.feature_dir, split="train")
            val_dataset = TennisDataset(args.feature_dir, split="val")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("No training data found. Please run extract_features.py first.")
        return

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
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

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 25

    # History tracking
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Set default checkpoint dir if not provided
    if not args.checkpoint_dir:
        args.checkpoint_dir = f"tennis/{args.model_type}/models"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                # Handle multi-stream inputs (e.g., (rgb, pose))
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)

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
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%"
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    inputs = [x.to(device) for x in inputs]
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(torch.log(outputs + 1e-8), labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(
                model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth")
            )
            print("Saved best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered after {epochs_no_improve} epochs without improvement."
                )
                break

    print("-" * 30)
    print("Best Model Validation Statistics:")
    print(f"Loss: {best_val_loss:.4f}")
    print(f"Accuracy: {best_val_acc:.2f}%")
    print("-" * 30)

    # Plotting
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        epochs_range = range(1, len(history["train_loss"]) + 1)

        # Accuracy Plot
        plt.figure()
        plt.plot(epochs_range, history["train_acc"], label="Train Acc")
        plt.plot(epochs_range, history["val_acc"], label="Val Acc")
        plt.title("Accuracy Learning Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig(os.path.join(args.plot_dir, "accuracy.png"))
        plt.close()

        # Loss Plot
        plt.figure()
        plt.plot(epochs_range, history["train_loss"], label="Train Loss")
        plt.plot(epochs_range, history["val_loss"], label="Val Loss")
        plt.title("Loss Learning Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(args.plot_dir, "loss.png"))
        plt.close()

        # Combined Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(epochs_range, history["train_acc"], label="Train Acc")
        ax1.plot(epochs_range, history["val_acc"], label="Val Acc")
        ax1.set_title("Accuracy")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy (%)")
        ax1.legend()

        ax2.plot(epochs_range, history["train_loss"], label="Train Loss")
        ax2.plot(epochs_range, history["val_loss"], label="Val Loss")
        ax2.set_title("Loss")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(args.plot_dir, "combined.png"))
        plt.close()

        print(f"Plots saved to {args.plot_dir}")

    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tennis Classifier")

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["baseline", "attention", "skeleton"],
        help="Type of model to train",
    )

    # Data arguments
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="data/features",
        help="Path to pre-extracted features",
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default="data/pose_features",
        help="Path to pre-extracted pose features (required for skeleton model)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: tennis/<model_type>/models)",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory to save learning curve plots",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Data loading workers"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (e.g., cuda:0, cpu)"
    )

    # Model arguments
    parser.add_argument("--hidden_size", type=int, default=256, help="LSTM hidden size")
    parser.add_argument(
        "--lstm_layers", type=int, default=1, help="Number of LSTM layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )

    # K-Fold arguments
    parser.add_argument(
        "--k_folds", type=int, default=0, help="Number of folds for CV (0 to disable)"
    )
    parser.add_argument(
        "--fold_index",
        type=int,
        default=0,
        help="Index of the fold to use as validation (0 to k_folds-1)",
    )

    args = parser.parse_args()
    train(args)
