import os
import cv2
import torch
import argparse
import numpy as np
import mediapipe as mp


def get_pose_connections():
    try:
        return mp.solutions.pose.POSE_CONNECTIONS
    except AttributeError:
        # Fallback if mp.solutions.pose is not available directly
        # Standard MediaPipe Pose connections
        return frozenset(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 7),
                (0, 4),
                (4, 5),
                (5, 6),
                (6, 8),
                (9, 10),
                (11, 12),
                (11, 13),
                (13, 15),
                (15, 17),
                (15, 19),
                (15, 21),
                (17, 19),
                (12, 14),
                (14, 16),
                (16, 18),
                (16, 20),
                (16, 22),
                (18, 20),
                (11, 23),
                (12, 24),
                (23, 24),
                (23, 25),
                (24, 26),
                (25, 27),
                (26, 28),
                (27, 29),
                (28, 30),
                (29, 31),
                (30, 32),
                (27, 31),
                (28, 32),
            ]
        )


def draw_pose_on_image(image, landmarks, connections):
    h, w, _ = image.shape

    # Draw connections
    for start_idx, end_idx in connections:
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]

        # Skip if any point is (0,0,0) which indicates no detection
        if np.all(start_point == 0) or np.all(end_point == 0):
            continue

        x1, y1 = int(start_point[0] * w), int(start_point[1] * h)
        x2, y2 = int(end_point[0] * w), int(end_point[1] * h)

        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw landmarks
    for i, lm in enumerate(landmarks):
        if np.all(lm == 0):
            continue
        x, y = int(lm[0] * w), int(lm[1] * h)
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)


def visualize_pose(pt_file, output_dir):
    try:
        data = torch.load(pt_file)
    except Exception as e:
        print(f"Error loading {pt_file}: {e}")
        return

    # Data shape is (T, 99) -> reshape to (T, 33, 3)
    if data.dim() == 2 and data.shape[1] == 99:
        landmarks_seq = data.reshape(-1, 33, 3).numpy()
    else:
        print(f"Unexpected shape for {pt_file}: {data.shape}")
        return

    filename = os.path.splitext(os.path.basename(pt_file))[0]
    output_path = os.path.join(output_dir, filename + ".mp4")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Video settings
    width, height = 640, 480
    fps = 10  # Adjust as needed, maybe 30? extract_pose samples 32 frames, original video might be longer.
    # If we assume the 32 frames represent the whole action, 10fps makes it ~3 seconds.

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    connections = get_pose_connections()

    for frame_idx, landmarks in enumerate(landmarks_seq):
        # Create a black image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw pose
        draw_pose_on_image(image, landmarks, connections)

        # Add frame number text
        cv2.putText(
            image,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        out.write(image)

    out.release()
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize skeleton pose from .pt files"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a .pt file or a directory containing .pt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations",
    )

    args = parser.parse_args()

    if os.path.isfile(args.input_path):
        if args.input_path.endswith(".pt"):
            visualize_pose(args.input_path, args.output_dir)
        else:
            print("Input file must be a .pt file")
    elif os.path.isdir(args.input_path):
        pt_files = []
        for root, dirs, files in os.walk(args.input_path):
            for file in files:
                if file.endswith(".pt"):
                    pt_files.append(os.path.join(root, file))

        if not pt_files:
            print(f"No .pt files found in {args.input_path}")
            return

        print(f"Found {len(pt_files)} .pt files. Processing...")
        for pt_file in pt_files:
            visualize_pose(pt_file, args.output_dir)
    else:
        print(f"Invalid input path: {args.input_path}")


if __name__ == "__main__":
    main()
