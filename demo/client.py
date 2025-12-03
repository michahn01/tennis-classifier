import argparse
import requests
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Client for Tennis Classifier Inference Server")
    parser.add_argument("video_path", help="Path to the video file (.mp4 or .avi)")
    parser.add_argument("--url", default="http://localhost:8000/predict", help="Server URL")
    
    args = parser.parse_args()
    
    video_path = os.path.abspath(args.video_path)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    payload = {'video_path': video_path}
    
    print(f"Sending request to {args.url} for video: {video_path}")
    try:
        response = requests.post(args.url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Result:")
            print(f"Predicted Class: {result['predicted_class']}")
            print("Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
        else:
            print(f"Error: Server returned status code {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    main()
