#!/bin/bash
# Training script for K-League pass prediction

# Set working directory
cd "$(dirname "$0")/.."

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run training
python train.py \
    --config configs/config.yaml \
    "$@"

