"""Inference script for K-League pass prediction."""

import os
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data import KLeagueDataModule
from src.models import KLeagueLightningModule


def load_best_model(
    config: OmegaConf,
    model_dir: str,
    checkpoint_path: Optional[str] = None
) -> KLeagueLightningModule:
    """Load the best trained model.
    
    Args:
        config: Configuration object.
        model_dir: Directory containing checkpoints.
        checkpoint_path: Explicit checkpoint path (optional).
        
    Returns:
        Loaded model.
    """
    # If the provided checkpoint_path is a directory, treat it as model_dir and auto-detect ckpt inside.
    if checkpoint_path is not None and Path(checkpoint_path).is_dir():
        model_dir = str(checkpoint_path)
        checkpoint_path = None

    # Find checkpoint path
    if checkpoint_path is None:
        # Try to read from best_model_path.txt
        best_model_file = Path(model_dir) / "best_model_path.txt"
        if best_model_file.exists():
            with open(best_model_file, 'r') as f:
                checkpoint_path = f.read().strip()
        else:
            # Find the best checkpoint in directory (recursive fallback)
            ckpt_files = list(Path(model_dir).rglob("*.ckpt"))
            if not ckpt_files:
                raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
            # Sort by modification time and take the latest
            checkpoint_path = str(sorted(ckpt_files, key=os.path.getmtime)[-1])
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Initialize data module to get dimensions
    data_module = KLeagueDataModule(config)
    data_module.setup('fit')
    
    feature_dim = data_module.get_feature_dim()
    num_type_classes = data_module.get_num_type_classes()
    num_result_classes = data_module.get_num_result_classes()
    
    # Load model from checkpoint
    # NOTE: PyTorch 2.6 changed torch.load default `weights_only=True`, which can break
    # loading older Lightning checkpoints that include non-tensor objects (e.g., pathlib paths).
    # This project loads checkpoints produced by our own training runs, so we explicitly allow
    # full checkpoint loading here.
    model = KLeagueLightningModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        feature_dim=feature_dim,
        num_type_classes=num_type_classes,
        num_result_classes=num_result_classes,
        weights_only=False,
    )
    model.eval()
    
    return model, data_module


def run_inference(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Run inference on test data.
    
    Args:
        config_path: Path to configuration file.
        checkpoint_path: Path to model checkpoint (optional).
        output_path: Path to save submission file (optional).
        
    Returns:
        DataFrame with predictions.
    """
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Set random seed
    pl.seed_everything(config.seed, workers=True)
    
    # Load model
    # If user passes a directory via --checkpoint, use it as model_dir and auto-detect the ckpt inside.
    model_dir = config.paths.model_dir
    if checkpoint_path is not None and Path(checkpoint_path).is_dir():
        model_dir = checkpoint_path
        checkpoint_path = None

    model, data_module = load_best_model(config, model_dir, checkpoint_path)
    
    # Setup test data
    data_module.setup('test')
    test_dataloader = data_module.test_dataloader()
    
    # Field dimensions for denormalization
    field_length = config.field.length
    field_width = config.field.width
    
    # Run inference
    print("Running inference...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Inference")):
            # Move batch to device
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass
            pred = model(batch_device)  # (batch, 2) in [0, 1]
            
            # Denormalize coordinates
            pred_x = pred[:, 0].cpu().numpy() * field_length
            pred_y = pred[:, 1].cpu().numpy() * field_width
            
            # Store predictions
            batch_size = pred.shape[0]
            episodes = batch_device.get('game_episode')
            if episodes is None:
                raise ValueError("Batch does not contain 'game_episode'. Please ensure collate_fn keeps it.")
            if len(episodes) != batch_size:
                raise ValueError(f"Mismatch: episodes({len(episodes)}) != batch_size({batch_size})")

            for i in range(batch_size):
                all_predictions.append({
                    'game_episode': episodes[i],
                    'end_x': pred_x[i],
                    'end_y': pred_y[i]
                })
    
    # Create DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # Load sample submission for correct order
    sample_submission_path = os.path.join(config.paths.data_dir, config.paths.submission_file)
    sample_submission = pd.read_csv(sample_submission_path)
    
    # Merge to ensure correct order
    submission = sample_submission[['game_episode']].merge(
        predictions_df,
        on='game_episode',
        how='left'
    )
    
    # Fill any missing predictions with field center
    submission['end_x'] = submission['end_x'].fillna(field_length / 2)
    submission['end_y'] = submission['end_y'].fillna(field_width / 2)
    
    # Save submission
    if output_path is None:
        output_dir = Path(config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "submission.csv"
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    
    # Print statistics
    print(f"\nPrediction statistics:")
    print(f"  end_x: mean={submission['end_x'].mean():.2f}, "
          f"std={submission['end_x'].std():.2f}, "
          f"min={submission['end_x'].min():.2f}, "
          f"max={submission['end_x'].max():.2f}")
    print(f"  end_y: mean={submission['end_y'].mean():.2f}, "
          f"std={submission['end_y'].std():.2f}, "
          f"min={submission['end_y'].min():.2f}, "
          f"max={submission['end_y'].max():.2f}")
    
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for K-League pass prediction")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.ckpt) OR a directory containing checkpoints (optional, will auto-detect if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save submission file"
    )
    
    args = parser.parse_args()
    run_inference(args.config, args.checkpoint, args.output)

