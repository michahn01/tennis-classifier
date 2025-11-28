from tennis.dataset import TennisDataset
from torchvision import transforms


def verify_data():
    print("Initializing dataset...")
    data_dir = "data/VIDEO_RGB"

    # Transform
    transform = transforms.Compose(
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    dataset = TennisDataset(data_dir, split="train", num_frames=32, transform=transform)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    print("Loading one sample...")
    frames, label = dataset[0]

    print(f"Frames shape: {frames.shape}")
    print(f"Label: {label}")

    # Expected shape: (T, C, H, W)
    expected_shape = (32, 3, 224, 224)
    assert frames.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {frames.shape}"
    )

    print("Verification successful!")


if __name__ == "__main__":
    verify_data()
