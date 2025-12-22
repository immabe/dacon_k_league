#!/bin/bash
# Setup script for K-League pass prediction project

# Set working directory
cd "$(dirname "$0")/.."

echo "Setting up K-League Pass Prediction project..."

# Create virtual environment with uv
echo "Creating virtual environment with uv..."
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate

# Install dependencies from requirements.txt
echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Setup complete!"
echo "To activate the environment, run: source .venv/bin/activate"

