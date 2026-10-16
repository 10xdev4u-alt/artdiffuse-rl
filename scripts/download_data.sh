#!/bin/bash
# Download and prepare training data

set -e

DATA_DIR="${1:-./data}"
DATASET="succinctlyai/midjourney-texttoimage"

echo "Downloading dataset: $DATASET"
kaggle datasets download "$DATASET" -p "$DATA_DIR" --unzip

echo "Dataset downloaded to $DATA_DIR"
echo "Files:"
ls -la "$DATA_DIR" | head -10
