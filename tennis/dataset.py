import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    """
    Dataset for loading raw video frames.
    Used by extract_features.py.
    """
    def __init__(self, samples, num_frames=32, transform=None):
        """
        Args:
            samples: List of (video_path, label_idx) tuples.
            num_frames: Number of frames to sample.
            transform: Optional transform to be applied on a sample.
        """
        self.samples = samples
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video(video_path)
        
        if self.transform:
            frames = self.transform(frames)
            
        return frames, label, video_path

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return torch.zeros((self.num_frames, 3, 224, 224)) 

        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
                if len(frames) == self.num_frames:
                    break
        
        cap.release()
        
        # Pad if not enough frames
        while len(frames) < self.num_frames:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
        frames = np.array(frames)
        
        # Return (T, C, H, W) tensor
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        return frames

class TennisDataset(Dataset):
    """
    Dataset for Tennis Stroke Classification.
    Loads pre-extracted features from disk.
    """
    
    CLASS_MAPPING = {
        "backhand": ["backhand", "backhand_slice", "backhand2hands"],
        "forehand": ["forehand_flat", "forehand_openstands", "forehand_slice"],
        "service": ["flat_service", "kick_service", "slice_service"],
        "bvolley": ["backhand_volley"],
        "fvolley": ["forehand_volley"],
        "smash": ["smash"]
    }

    def __init__(self, feature_dir, split='train'):
        """
        Args:
            feature_dir: Directory containing processed features.
            split: 'train', 'val', or 'test'.
        """
        self.feature_dir = feature_dir
        self.split = split
        
        self.classes = list(self.CLASS_MAPPING.keys())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        split_dir = os.path.join(self.feature_dir, self.split)
        
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory not found: {split_dir}")
            return samples

        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for f in os.listdir(class_dir):
                if f.endswith('.pt'):
                    samples.append((os.path.join(class_dir, f), class_idx))
                    
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path, label = self.samples[idx]
        features = torch.load(feature_path)
        return features, label
