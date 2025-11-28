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
    model = nn.Sequential(*modules).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"Processing split: {split}")
        dataset = TennisDataset(args.data_dir, split=split, num_frames=args.num_frames, transform=transform, experts_only=True)
        dataset.return_paths = True
        
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
                for i in range(B):
                    video_path = paths[i]
                    rel_path = os.path.relpath(video_path, args.data_dir)
                    base_name = os.path.splitext(rel_path)[0]
                    save_path = os.path.join(args.output_dir, base_name + ".pt")
                    
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(features[i].cpu(), save_path)

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
