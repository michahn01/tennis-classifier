import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse

from tennis.dataset import TennisDataset

def extract_features(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    modules = list(resnet.children())[:-1] # Remove FC layer
    model = nn.Sequential(*modules)
    model = model.to(device)
    model.eval()

    # Transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    # We want to process ALL videos, so we might need to iterate over all splits or just point to the root
    # The dataset class splits based on 'split' arg. We can just instantiate for train, val, test.
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"Processing split: {split}")
        dataset = TennisDataset(args.data_dir, split=split, num_frames=args.num_frames, transform=transform, experts_only=True)
        dataset.return_paths = True # Enable path return
        
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        with torch.no_grad():
            for i, (frames, labels, paths) in enumerate(loader):
                if args.limit and i * args.batch_size >= args.limit:
                    print(f"Reached limit of {args.limit} samples. Stopping.")
                    break
                    
                if i % 10 == 0:
                    print(f"Processing batch {i}/{len(loader)}")
                # frames: (B, T, C, H, W)
                frames = frames.to(device)
                B, T, C, H, W = frames.size()
                
                # Flatten for ResNet
                frames_flat = frames.view(B * T, C, H, W)
                
                # Extract features
                features_flat = model(frames_flat) # (B*T, 2048, 1, 1)
                features_flat = features_flat.view(B * T, -1) # (B*T, 2048)
                
                # Reshape back to sequence
                features = features_flat.view(B, T, -1) # (B, T, 2048)
                
                # Save features
                for i in range(B):
                    video_path = paths[i]
                    # Create output path
                    # Structure: output_dir/class_name/filename.pt
                    
                    # Get relative path from data_dir to preserve structure
                    # video_path is absolute. 
                    # dataset.root_dir is absolute.
                    
                    rel_path = os.path.relpath(video_path, args.data_dir)
                    # rel_path e.g. "backhand/p1_backhand_s1.avi"
                    
                    # Replace extension with .pt
                    base_name = os.path.splitext(rel_path)[0]
                    save_path = os.path.join(args.output_dir, base_name + ".pt")
                    
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    torch.save(features[i].cpu(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/VIDEO_RGB')
    parser.add_argument('--output_dir', type=str, default='data/features')
    parser.add_argument('--num_frames', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples to process')
    args = parser.parse_args()
    
    extract_features(args)
