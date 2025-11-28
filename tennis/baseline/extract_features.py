import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import random

from tennis.dataset import VideoDataset, TennisDataset

def get_samples(data_dir, experts_only=True):
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
        
        files = sorted([f for f in os.listdir(subdir_path) if f.endswith(('.mp4', '.avi', '.mov'))])

        if experts_only:
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
            
        for f in files:
            samples.append((os.path.join(subdir_path, f), class_idx))
            
    return samples, classes

def extract_features(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    modules = list(resnet.children())[:-1] # Remove FC layer
    model = nn.Sequential(*modules).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load all samples
    all_samples, classes = get_samples(args.data_dir, experts_only=True)
    print(f"Found {len(all_samples)} total samples.")
    
    # Split samples
    class_samples = {i: [] for i in range(len(classes))}
    for path, label in all_samples:
        class_samples[label].append(path)
        
    train_samples = []
    val_samples = []
    test_samples = []
    
    for label, paths in class_samples.items():
        paths.sort()
        
        n = len(paths)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        train_paths = paths[:train_end]
        val_paths = paths[train_end:val_end]
        test_paths = paths[val_end:]
        
        for p in train_paths: train_samples.append((p, label))
        for p in val_paths: val_samples.append((p, label))
        for p in test_paths: test_samples.append((p, label))
        
    print(f"Split sizes: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    for split_name, samples in splits.items():
        print(f"Processing split: {split_name}")
        if not samples:
            continue
            
        dataset = VideoDataset(samples, num_frames=args.num_frames, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
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
    parser = argparse.ArgumentParser(description='Extract features using ResNet50')
    parser.add_argument('--data_dir', type=str, default='data/VIDEO_RGB', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='data/features', help='Path to output features directory')
    parser.add_argument('--num_frames', type=int, default=32, help='Number of frames per video')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loading workers')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples to process')
    
    args = parser.parse_args()
    extract_features(args)
