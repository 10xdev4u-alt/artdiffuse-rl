#!/bin/bash
# Evaluate the trained model

set -e

MODEL_PATH="${1:-models/final.pt}"
OUTPUT_DIR="${2:-results}"

echo "Evaluating model: $MODEL_PATH"
python -m src.evaluate --model "$MODEL_PATH" --output "$OUTPUT_DIR"
