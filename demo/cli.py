import argparse
import cv2
import os
import time
import sys
from datetime import datetime


def record_video(output_dir):
    """Records video from webcam and saves it to output_dir."""
    
    # Create class directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.avi"
    filepath = os.path.join(output_dir, filename)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define codec and create VideoWriter object
    # MJPG is usually safe for .avi
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # We'll set resolution to 640x480 for now, or whatever the camera supports
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0
    
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    print(f"Recording to {filepath}")
    print("Press 'q' to stop recording.")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break
            
            # Write the frame
            out.write(frame)
            
            # Display the resulting frame
            cv2.imshow('Tennis Recorder', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\nSaved video to {filepath}")

def run_command(args):
    """Handler for the 'run' command."""
    print("Starting webcam...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "videos")

    record_video(output_dir)

def main():
    parser = argparse.ArgumentParser(description="Tennis Classifier CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    run_parser = subparsers.add_parser("run", help="Start video recording")
    run_parser.set_defaults(func=run_command)
    
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
