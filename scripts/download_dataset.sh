#!/bin/bash
set -e

# Directory where the dataset will be stored
DATA_DIR="data"
TARGET_DIR="$DATA_DIR/VIDEO_RGB"
REPO_URL="https://github.com/THETIS-dataset/dataset.git"
TEMP_DIR="temp_dataset_repo"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Check if target directory already exists
if [ -d "$TARGET_DIR" ]; then
    echo "Dataset directory $TARGET_DIR already exists. Skipping download."
    exit 0
fi

echo "Downloading VIDEO_RGB from THETIS dataset..."

# Create a temporary directory for the repo
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Initialize git repo
git init
git remote add origin "$REPO_URL"

# Configure sparse-checkout
git config core.sparseCheckout true
echo "VIDEO_RGB/" >> .git/info/sparse-checkout

# Pull the dataset (only the specified directory)
git pull origin main --depth 1

# Move the directory to the target location
cd ..
mv "$TEMP_DIR/VIDEO_RGB" "$DATA_DIR/"

# Cleanup
rm -rf "$TEMP_DIR"

echo "Download complete. Dataset stored in $TARGET_DIR"
