#!/bin/bash
# Training script for K-League pass prediction

# GPU 설정 (기본값: 0)
GPU_ID=0
TRAIN_ARGS=()

# 첫 번째 인자가 숫자이면 GPU_ID로 사용
if [[ $1 =~ ^[0-9]+$ ]]; then
    GPU_ID=$1
    TRAIN_ARGS=("${@:2}")
else
    # 숫자가 아니면 모든 인자를 학습 파라미터로 사용
    TRAIN_ARGS=("$@")
fi

# Set working directory
cd "$(dirname "$0")/.."

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run training
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --config configs/config.yaml \
    "${TRAIN_ARGS[@]}"
