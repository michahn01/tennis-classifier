import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class TennisDataset(Dataset):
    """
    Dataset for Tennis Stroke Classification.
    Reads videos from a directory structure and maps them to 6 classes.
    """
    
    CLASS_MAPPING = {
        "backhand": ["backhand", "backhand_slice", "backhand2hands"],
        "forehand": ["forehand_flat", "forehand_openstands", "forehand_slice"],
        "service": ["flat_service", "kick_service", "slice_service"],
        "bvolley": ["backhand_volley"],
        "fvolley": ["forehand_volley"],
        "smash": ["smash"]
    }

    def __init__(self, root_dir, split='train', num_frames=32, transform=None):
        """
        Args:
            root_dir (str): Path to the VIDEO_RGB directory.
            split (str): 'train', 'val', or 'test'.
            num_frames (int): Number of frames to sample per video.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.transform = transform
        
        self.classes = list(self.CLASS_MAPPING.keys())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # Reverse mapping to find class for a given subdirectory
        subdir_to_class = {}
        for cls, subdirs in self.CLASS_MAPPING.items():
            for subdir in subdirs:
                subdir_to_class[subdir] = cls

        # Walk through directories
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            if subdir not in subdir_to_class:
                continue
                
            target_class = subdir_to_class[subdir]
            class_idx = self.class_to_idx[target_class]
            
            files = sorted([f for f in os.listdir(subdir_path) if f.endswith(('.mp4', '.avi', '.mov'))])
            
            # Simple split logic based on file index (e.g., 80/10/10)
            # Since we don't have explicit split files, we'll hash the filename or just use index
            # Using a deterministic shuffle based on filename could be better, but let's stick to simple index for now
            # Actually, the overview mentions p1-p31 beginners, p32-p55 experts. 
            # But the filenames might not strictly follow pXX. 
            # Let's use a simple random split with seed for reproducibility if needed, 
            # or just split by file list index for simplicity as a starting point.
            
            # Let's use a deterministic split based on sorting
            num_files = len(files)
            if num_files == 0:
                continue
                
            # 80% train, 10% val, 10% test
            train_end = int(0.8 * num_files)
            val_end = int(0.9 * num_files)
            
            if self.split == 'train':
                selected_files = files[:train_end]
            elif self.split == 'val':
                selected_files = files[train_end:val_end]
            elif self.split == 'test':
                selected_files = files[val_end:]
            else:
                raise ValueError(f"Invalid split: {self.split}")
                
            for f in selected_files:
                samples.append((os.path.join(subdir_path, f), class_idx))
                
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video(video_path)
        
        if self.transform:
            # Apply transform to each frame
            # frames is (T, H, W, C) -> (T, C, H, W) for torch
            frames = self.transform(frames)
            
        return frames, label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            # Fallback or error handling
            cap.release()
            # Return zeros if video is broken? Or raise error?
            # Let's return a zero tensor of correct shape to avoid crashing training immediately
            return torch.zeros((self.num_frames, 3, 224, 224)) 

        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to 224x224
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
                if len(frames) == self.num_frames:
                    break
        
        cap.release()
        
        # Pad if not enough frames
        while len(frames) < self.num_frames:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
        frames = np.array(frames) # (T, H, W, C)
        
        # Convert to tensor and rearrange to (T, C, H, W)
        # But usually transforms expect PIL or Tensor (C, H, W)
        # We will handle this in transform or here.
        # Let's return (T, C, H, W) float tensor normalized to [0, 1] if no transform provided
        # But standard torchvision transforms expect (C, H, W).
        
        # Let's keep it simple: return (T, C, H, W) tensor
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        return frames
