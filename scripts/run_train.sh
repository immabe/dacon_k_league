#!/bin/bash
# Training script for K-League pass prediction

# GPU 설정 (기본값: 0)
GPU_ID=${1:-0}

# Set working directory
cd "$(dirname "$0")/.."

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run training
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --config configs/config.yaml \
    "${@:2}"
