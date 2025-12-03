import argparse
import os
import subprocess
import glob

def convert_mp4_to_avi(input_path, output_path=None):
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + '.avi'
    
    # Check if output file already exists to avoid ffmpeg asking for overwrite confirmation interactively
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping.")
        return

    cmd = ['ffmpeg', '-i', input_path, output_path]
    print(f"Converting {input_path} to {output_path}...")
    try:
        subprocess.run(cmd, check=True)
        print("Conversion successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error converting file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert .mp4 files to .avi")
    parser.add_argument("input", help="Input .mp4 file or directory containing .mp4 files")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        mp4_files = glob.glob(os.path.join(args.input, "*.mp4"))
        if not mp4_files:
            print(f"No .mp4 files found in {args.input}")
            return
        
        for mp4_file in mp4_files:
            convert_mp4_to_avi(mp4_file)
    elif os.path.isfile(args.input):
        if not args.input.lower().endswith('.mp4'):
            print("Input file is not an .mp4 file")
            return
        convert_mp4_to_avi(args.input)
    else:
        print(f"Input {args.input} not found")

if __name__ == "__main__":
    main()
