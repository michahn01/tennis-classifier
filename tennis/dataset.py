import os
import cv2
import torch
import numpy as np
import tempfile
import shutil
import hashlib
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

    def __init__(self, root_dir, split='train', num_frames=32, transform=None, experts_only=True, cache_dir=None, feature_dir=None):
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.transform = transform
        self.experts_only = experts_only
        self.return_paths = False
        self.feature_dir = feature_dir
        
        self.classes = list(self.CLASS_MAPPING.keys())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = self._load_samples()

        # Caching
        if cache_dir:
            self.cache_dir = cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cleanup_cache = False
        else:
            self.cache_dir = tempfile.mkdtemp()
            self.cleanup_cache = True
            
        self.main_pid = os.getpid()
        print(f"Dataset cache directory: {self.cache_dir}")

    def _load_samples(self):
        samples = []
        subdir_to_class = {}
        for cls, subdirs in self.CLASS_MAPPING.items():
            for subdir in subdirs:
                subdir_to_class[subdir] = cls

        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            if subdir not in subdir_to_class:
                continue
                
            target_class = subdir_to_class[subdir]
            class_idx = self.class_to_idx[target_class]
            
            files = sorted([f for f in os.listdir(subdir_path) if f.endswith(('.mp4', '.avi', '.mov'))])

            if self.experts_only:
                # Filter for experts only (p32-p55)
                expert_files = []
                for f in files:
                    try:
                        player_id_str = f.split('_')[0]
                        if player_id_str.startswith('p'):
                            player_id = int(player_id_str[1:])
                            if 32 <= player_id <= 55:
                                expert_files.append(f)
                    except ValueError:
                        continue
                files = expert_files

            num_files = len(files)
            if num_files == 0:
                continue
                
            # Split: 80% train, 10% val, 10% test
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
        
        if self.feature_dir:
            # Load pre-extracted features
            rel_path = os.path.relpath(video_path, self.root_dir)
            feature_path = os.path.join(self.feature_dir, os.path.splitext(rel_path)[0] + ".pt")
            
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"Feature file not found: {feature_path}")
                
            features = torch.load(feature_path)
            return features, label

        # Generate cache path
        video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{video_hash}.pt")
        
        if os.path.exists(cache_path):
            try:
                frames = torch.load(cache_path)
            except Exception as e:
                print(f"Failed to load cache for {video_path}: {e}")
                frames = self._load_video(video_path)
                torch.save(frames, cache_path)
        else:
            frames = self._load_video(video_path)
            try:
                torch.save(frames, cache_path)
            except Exception:
                pass
        
        if self.transform:
            frames = self.transform(frames)
            
        if self.return_paths:
            return frames, label, video_path
            
        return frames, label

    def __del__(self):
        if hasattr(self, 'cache_dir') and os.path.exists(self.cache_dir):
            if os.getpid() == self.main_pid and getattr(self, 'cleanup_cache', False):
                try:
                    shutil.rmtree(self.cache_dir)
                    print(f"Cleaned up dataset cache: {self.cache_dir}")
                except Exception as e:
                    print(f"Error cleaning up cache: {e}")

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
