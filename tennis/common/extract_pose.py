import os
import cv2
import torch
import argparse
import mediapipe as mp
import numpy as np
from tennis.dataset import TennisDataset

def get_samples(data_dir):
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
            samples.append((os.path.join(subdir_path, f), class_idx))

    return samples, classes

def extract_pose(args):
    print("Initializing MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Clean output directory
    splits_to_clean = ["train", "val", "test"]
    if args.merge_train_val:
        splits_to_clean = ["train_val", "test"]

    for split in splits_to_clean:
        split_path = os.path.join(args.output_dir, split)
        if os.path.exists(split_path):
            print(f"Cleaning existing split directory: {split_path}")
            import shutil
            shutil.rmtree(split_path)

    # Load all samples
    all_samples, classes = get_samples(args.data_dir)
    print(f"Found {len(all_samples)} total samples.")

    # Split samples by Player ID (Reuse logic from extract_features.py for consistency)
    player_samples = {}
    for path, label in all_samples:
        filename = os.path.basename(path)
        if filename.startswith("p") and "_" in filename:
            player_id = filename.split("_")[0]
        else:
            player_id = filename
        
        if player_id not in player_samples:
            player_samples[player_id] = []
        player_samples[player_id].append((path, label))

    player_ids = sorted(list(player_samples.keys()))
    import random
    random.seed(1)
    random.shuffle(player_ids)

    n_players = len(player_ids)

    if args.merge_train_val:
        train_val_end = int(round(0.9 * n_players))
        train_val_players = set(player_ids[:train_val_end])
        test_players = set(player_ids[train_val_end:])

        train_val_samples = []
        test_samples = []

        for pid, samples in player_samples.items():
            if pid in train_val_players:
                train_val_samples.extend(samples)
            else:
                test_samples.extend(samples)
        
        splits = {"train_val": train_val_samples, "test": test_samples}
    else:
        train_end = int(round(0.8 * n_players))
        val_end = train_end + int(round(0.1 * n_players))

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

        splits = {"train": train_samples, "val": val_samples, "test": test_samples}

    # Processing
    for split_name, samples in splits.items():
        print(f"Processing split: {split_name}")
        for i, (video_path, label_idx) in enumerate(samples):
            if args.limit and i >= args.limit:
                break
            
            if i % 10 == 0:
                print(f"Processing video {i}/{len(samples)}")

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)
            
            pose_features = []
            
            current_frame = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < args.num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if current_frame in indices:
                    # Convert to RGB for MediaPipe
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)
                    
                    if results.pose_landmarks:
                        # Extract 33 landmarks (x, y, z) -> 99 features
                        landmarks = []
                        for lm in results.pose_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        pose_features.append(landmarks)
                    else:
                        # If no pose detected, use zeros
                        pose_features.append([0.0] * 99)
                    
                    extracted_count += 1
                
                current_frame += 1
            
            cap.release()
            
            # Pad if needed
            while len(pose_features) < args.num_frames:
                pose_features.append([0.0] * 99)
            
            # Convert to tensor (T, 99)
            pose_tensor = torch.tensor(pose_features, dtype=torch.float32)
            
            # Save
            class_name = classes[label_idx]
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            save_dir = os.path.join(args.output_dir, split_name, class_name)
            save_path = os.path.join(save_dir, base_name + ".pt")
            
            os.makedirs(save_dir, exist_ok=True)
            torch.save(pose_tensor, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pose features using MediaPipe")
    parser.add_argument("--data_dir", type=str, default="data/VIDEO_RGB", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="data/pose_features", help="Path to output pose features directory")
    parser.add_argument("--num_frames", type=int, default=32, help="Number of frames per video")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--merge_train_val", action="store_true", help="Merge train and val splits")
    
    args = parser.parse_args()
    extract_pose(args)
