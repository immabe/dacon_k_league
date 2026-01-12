# K-League Pass Coordinate Prediction AI

**Competition Rank:** 151 / 938 (Top 16%)
https://dacon.io/competitions/official/236647/overview/description

## 1. Challenge Overview

### Background
Modern football analysis often relies on simple event recording. However, true insight requires interpreting "tactical intent"—understanding the decision-making process within a complex game context. This challenge aims to train an AI to learn these contexts from K-League match data and predict the actual destination coordinates for the subsequent pass.

### Goal
**Develop an AI model that predicts the destination coordinates (X, Y) of the final pass in a given play sequence.**

- **Input:** Sequence of play events in a K-League match.
- **Model Output:** The displacement `(dx, dy)` from the last event's location.
- **Final Target:** The absolute `(X, Y)` coordinates of the pass destination (calculated as `Last Event (X, Y) + Prediction (dx, dy)`).
- **Coordinate System:** Relative coordinates mapped to a standard FIFA-recommended pitch size of **105 x 68**.
- **Evaluation Metric:** **Euclidean Distance** between the predicted coordinates and the actual coordinates.

---

## 2. Solution Approach

### Model Architecture: Transformer Sequence Encoder
I implemented a **Transformer-based Sequence Encoder** to capture the temporal dependencies and spatial context of football play sequences.

- **Embedding Layer:**
  - **Categorical Features:** Event types (Pass, Carry, Shot, etc.) and results (Successful, Failed, etc.) are mapped to dense vectors.
  - **Numerical Features:** Spatial coordinates (Start X/Y, End X/Y) and time information are normalized and projected.
  - **Positional Encoding:** Added to preserve the sequential order of events.
- **Encoder Blocks:** Stacked Transformer Encoder layers with Multi-Head Self-Attention to learn interactions between different events in the sequence.
- **Pooling Strategy:** Attention-based pooling to aggregate the sequence information into a single context vector.
- **Prediction Head:** A Multi-Layer Perceptron (MLP) that predicts the displacement `(dx, dy)` from the last event's position.

### Key Features
The model utilizes a rich set of features defined in `configs/config.yaml`:
- **Raw Features:** Event Type, Outcome, Home/Away status, Coordinates (Start/End X, Y), Time.
- **Derived Features:**
  - **Movement:** Delta X/Y, Euclidean Distance, Angle, Speed.
  - **Zonal Info:** Discretized start/end zones.
  - **Tactical:** `is_forward_pass`, `is_progressive` (significant forward movement).

### Training Strategy
- **Loss Function:** Direct optimization of **Euclidean Distance** (aligning with the competition metric).
- **Data Augmentation:** **Y-axis Mirroring** (Randomly flipping the Y-coordinates) to simulate symmetrical field situations and double the training data.
- **Validation Split:** Group-based splitting by `game_id` to prevent data leakage from the same match appearing in both train and validation sets.
- **Test Time Augmentation (TTA):** Averaging predictions from the original input and its Y-mirrored version during inference for robustness.
- **Post-processing:** Clipping predictions to pitch boundaries (0-105, 0-68) and capping unrealistic pass distances based on training distribution.

---

## 3. Project Structure

```
k-league/
├── configs/
│   └── config.yaml          # Main configuration (Model, Training, Features)
├── src/
│   ├── data/
│   │   ├── dataset.py       # Custom Dataset classes
│   │   └── datamodule.py    # PyTorch Lightning DataModule
│   ├── models/
│   │   ├── transformer.py   # Transformer Encoder implementation
│   │   └── lightning_module.py  # Lightning Module for training loop
│   └── utils/
│       ├── features.py      # Feature engineering & selection logic
│       └── metrics.py       # Euclidean Distance metric implementation
├── scripts/
│   ├── setup.sh             # Environment setup script
│   ├── run_train.sh         # Training execution script
│   └── run_inference.sh     # Inference execution script
├── train.py                 # Training entry point (MLflow + Early Stopping)
├── inference.py             # Inference entry point (Auto model loading)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 4. Usage

### Prerequisites
- Python 3.8+
- PyTorch
- PyTorch Lightning
- MLflow

### 1. Environment Setup
Run the setup script to create a virtual environment and install dependencies.
```bash
bash scripts/setup.sh
source .venv/bin/activate
```

### 2. Training
Train the model using the configuration file. The script automatically handles MLflow logging and checkpointing.
```bash
python train.py --config configs/config.yaml
```
*Note: You can override parameters via CLI:*
```bash
python train.py --override training.batch_size=128 training.max_epochs=50
```

### 3. Inference
Run inference on the test set. The script automatically detects and loads the best model from the `checkpoints/` directory.
```bash
python inference.py --config configs/config.yaml
```
This will generate a submission file in the `outputs/` directory.

---

## 5. Experiment Tracking
All training runs, hyperparameters, and metrics are tracked using **MLflow**.
- Artifacts (models, configs) are stored in `mlruns/`.
- Best models are saved in `checkpoints/`.
