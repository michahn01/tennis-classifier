import os
import cv2
import torch
import argparse
import mediapipe as mp
import numpy as np
import warnings
import urllib.request
import tennis.common.config as config
from tennis.dataset import TennisDataset

# Suppress Protobuf warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.symbol_database"
)


def get_samples(data_dir, only_experts=False):
    """Scans the data directory and returns a list of (video_path, label_idx) tuples."""
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
                    pid_part = f.split("_")[0]
                    if pid_part.startswith("p"):
                        pid = int(pid_part[1:])
                        if not (32 <= pid <= 55):
                            continue
                    else:
                        continue
                except (ValueError, IndexError):
                    continue

            samples.append((os.path.join(subdir_path, f), class_idx))

    return samples, classes


def download_model(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")


def extract_pose_from_video(video_path, num_frames, landmarker=None):
    """
    Extracts pose landmarks from a single video.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample.
        landmarker (mp.tasks.vision.PoseLandmarker, optional): Pre-initialized landmarker.
                                                               If None, initializes a new one.

    Returns:
        pose_tensor (torch.Tensor): (T, 99) tensor of pose landmarks.
    """
    should_close_landmarker = False
    if landmarker is None:
        model_path = "pose_landmarker_full.task"
        download_model(model_path)
        
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path, delegate=BaseOptions.Delegate.GPU
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = PoseLandmarker.create_from_options(options)
        should_close_landmarker = True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if should_close_landmarker:
            landmarker.close()
        raise ValueError(f"Error opening video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle edge case where video might be shorter than num_frames or empty
    if total_frames <= 0:
         # Try to read one frame to see if it works, or just fail
         # For robustness, let's just try to read all frames
         frames_list = []
         while True:
             ret, frame = cap.read()
             if not ret:
                 break
             frames_list.append(frame)
         total_frames = len(frames_list)
         cap.release()
         
         if total_frames == 0:
             if should_close_landmarker:
                 landmarker.close()
             raise ValueError(f"Video has 0 frames: {video_path}")
         
         cap = cv2.VideoCapture(video_path)

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    pose_features = []
    current_frame = 0
    extracted_count = 0

    while cap.isOpened() and extracted_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in indices:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            detection_result = landmarker.detect(mp_image)

            if detection_result.pose_landmarks:
                # Extract 33 landmarks (x, y, z) -> 99 features
                landmarks = []
                for lm in detection_result.pose_landmarks[0]:
                    landmarks.extend([lm.x, lm.y, lm.z])
                pose_features.append(landmarks)
            else:
                # If no pose detected, use zeros
                pose_features.append([0.0] * 99)

            extracted_count += 1

        current_frame += 1

    cap.release()
    
    if should_close_landmarker:
        landmarker.close()

    # Pad if needed
    while len(pose_features) < num_frames:
        pose_features.append([0.0] * 99)

    # Convert to tensor (T, 99)
    pose_tensor = torch.tensor(pose_features, dtype=torch.float32)
    return pose_tensor


def extract_pose(args):
    model_path = "pose_landmarker_full.task"
    download_model(model_path)

    print("Initializing MediaPipe Pose Landmarker (GPU)...")
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a landmarker instance with the GPU delegate
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path, delegate=BaseOptions.Delegate.GPU
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
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
    all_samples, classes = get_samples(args.data_dir, only_experts=args.only_experts)
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

    random.seed(config.DEFAULT_EXTRACT_SEED)
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
            elif pid in test_players:
                test_samples.extend(samples)
            else:
                raise ValueError(f"Unknown player ID: {pid}")

        splits = {"train": train_samples, "val": val_samples, "test": test_samples}

    # Processing
    with PoseLandmarker.create_from_options(options) as landmarker:
        for split_name, samples in splits.items():
            print(f"Processing split: {split_name}")
            for i, (video_path, label_idx) in enumerate(samples):
                if args.limit and i >= args.limit:
                    break

                if i % 10 == 0:
                    print(f"Processing video {i}/{len(samples)}")

                try:
                    pose_tensor = extract_pose_from_video(
                        video_path, args.num_frames, landmarker
                    )
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    continue

                # Save
                class_name = classes[label_idx]
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                save_dir = os.path.join(args.output_dir, split_name, class_name)
                save_path = os.path.join(save_dir, base_name + ".pt")

                os.makedirs(save_dir, exist_ok=True)
                torch.save(pose_tensor, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pose features using MediaPipe"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=config.DEFAULT_DATA_DIR,
        help="Path to data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.DEFAULT_POSE_DIR,
        help="Path to output pose features directory",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=config.DEFAULT_NUM_FRAMES,
        help="Number of frames per video",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples to process"
    )
    parser.add_argument(
        "--only_experts", action="store_true", help="Only sample experts (p32-p55)"
    )
    parser.add_argument(
        "--merge_train_val", action="store_true", help="Merge train and val splits"
    )

    args = parser.parse_args()
    extract_pose(args)
