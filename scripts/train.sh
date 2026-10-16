#!/bin/bash
# Train the diffusion model

set -e

CONFIG="${1:-configs/base.yaml}"
OUTPUT_DIR="${2:-models}"

echo "Training with config: $CONFIG"
python -m src.train --config "$CONFIG" --output "$OUTPUT_DIR"
