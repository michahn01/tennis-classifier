import os
import random
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse

from tennis.dataset import VideoDataset, TennisDataset


def get_samples(data_dir, only_experts=False):
    """
    Scans the data directory and returns a list of (video_path, label_idx) tuples.
    """
    samples = []
    subdir_to_class = {}
    for cls, subdirs in TennisDataset.CLASS_MAPPING.items():
        for subdir in subdirs:
            subdir_to_class[subdir] = cls

    classes = list(TennisDataset.CLASS_MAPPING.keys())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        if subdir not in subdir_to_class:
            continue

        target_class = subdir_to_class[subdir]
        class_idx = class_to_idx[target_class]

        files = sorted(
            [f for f in os.listdir(subdir_path) if f.endswith((".mp4", ".avi", ".mov"))]
        )

        for f in files:
            if only_experts:
                try:
                    # Filename format: p<player_id>_<action>_s<shot_id>.avi
                    # Experts are p32 to p55
                    pid_part = f.split('_')[0]
                    if pid_part.startswith('p'):
                        pid = int(pid_part[1:])
                        if not (32 <= pid <= 55):
                            continue
                    else:
                        continue
                except (ValueError, IndexError):
                    continue

            samples.append((os.path.join(subdir_path, f), class_idx))

    return samples, classes


def extract_features(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    modules = list(resnet.children())[:-1]  # Remove FC layer
    model = nn.Sequential(*modules).to(device)
    model.eval()

    transform = transforms.Compose(
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    # Clean output directory to prevent leakage
    for split in ["train", "val", "test"]:
        split_path = os.path.join(args.output_dir, split)
        if os.path.exists(split_path):
            print(f"Cleaning existing split directory: {split_path}")
            import shutil
            shutil.rmtree(split_path)

    # Load all samples
    all_samples, classes = get_samples(args.data_dir, only_experts=args.only_experts)
    print(f"Found {len(all_samples)} total samples.")

    # Split samples by Player ID to avoid contamination
    # Filename format: p<player_id>_<action>_s<shot_id>.avi
    player_samples = {}
    for path, label in all_samples:
        filename = os.path.basename(path)
        # Extract player ID (e.g., p10 -> 10)
        if filename.startswith("p") and "_" in filename:
            player_id = filename.split("_")[0]
        else:
            # Fallback for non-conforming filenames: use filename as unique ID
            print(f"Warning: Filename {filename} does not match expected format p<ID>_... Using filename as ID.")
            player_id = filename
        
        if player_id not in player_samples:
            player_samples[player_id] = []
        player_samples[player_id].append((path, label))

    player_ids = sorted(list(player_samples.keys()))
    random.seed(1)
    random.shuffle(player_ids)

    n_players = len(player_ids)

    # Split ratios: 80% Train, 10% Val, 10% Test
    train_end = int(round(0.8 * n_players))
    print("train_end =", train_end)
    val_end = train_end + int(round(0.1 * n_players))
    print("val_end =", val_end)

    train_players = set(player_ids[:train_end])
    val_players = set(player_ids[train_end:val_end])
    test_players = set(player_ids[val_end:])

    train_samples = []
    val_samples = []
    test_samples = []

    for pid, samples in player_samples.items():
        if pid in train_players:
            train_samples.extend(samples)
        elif pid in val_players:
            val_samples.extend(samples)
        else:
            test_samples.extend(samples)

    print(
        f"Split sizes (by player): Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}"
    )
    print(f"Player counts: Train={len(train_players)}, Val={len(val_players)}, Test={len(test_players)}")

    splits = {"train": train_samples, "val": val_samples, "test": test_samples}

    for split_name, samples in splits.items():
        print(f"Processing split: {split_name}")
        if not samples:
            continue

        dataset = VideoDataset(samples, num_frames=args.num_frames, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        with torch.no_grad():
            for i, (frames, labels, paths) in enumerate(loader):
                if args.limit and i * args.batch_size >= args.limit:
                    print(f"Reached limit of {args.limit} samples. Stopping.")
                    break

                if i % 10 == 0:
                    print(f"Processing batch {i}/{len(loader)}")

                frames = frames.to(device)
                B, T, C, H, W = frames.size()

                # Flatten for ResNet
                frames_flat = frames.view(B * T, C, H, W)

                # Extract features
                features_flat = model(frames_flat)
                features_flat = features_flat.view(B * T, -1)

                # Reshape back to sequence
                features = features_flat.view(B, T, -1)

                # Save features
                for b in range(B):
                    video_path = paths[b]
                    label_idx = labels[b].item()
                    class_name = classes[label_idx]

                    # Output structure: output_dir/split/class_name/filename.pt
                    base_name = os.path.splitext(os.path.basename(video_path))[0]
                    save_dir = os.path.join(args.output_dir, split_name, class_name)
                    save_path = os.path.join(save_dir, base_name + ".pt")

                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(features[b].cpu(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features using ResNet50")
    parser.add_argument(
        "--data_dir", type=str, default="data/VIDEO_RGB", help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/features",
        help="Path to output features directory",
    )
    parser.add_argument(
        "--num_frames", type=int, default=32, help="Number of frames per video"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Data loading workers"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples to process"
    )
    parser.add_argument(
        "--only_experts", action="store_true", help="Only sample experts (p32-p55)"
    )

    args = parser.parse_args()
    extract_features(args)
