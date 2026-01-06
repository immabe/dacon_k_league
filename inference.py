"""Inference script for K-League pass prediction."""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data import KLeagueDataModule
from src.models import KLeagueLightningModule
from src.utils.postprocess import stabilize_end_coordinates


def apply_y_mirror_to_features(
    features: torch.Tensor,
    config: OmegaConf
) -> torch.Tensor:
    """Apply Y-mirror transformation to features tensor.
    
    Args:
        features: (batch, seq_len, feature_dim) tensor
        config: Configuration object
        
    Returns:
        Y-mirrored features tensor
    """
    features = features.clone()
    
    idx = 0
    
    # Numerical features: start_x(0), start_y(1), end_x(2), end_y(3), time_seconds(4)
    if config.features.use_numerical:
        features[:, :, 1] = 1.0 - features[:, :, 1]  # start_y: mirror
        features[:, :, 3] = 1.0 - features[:, :, 3]  # end_y: mirror
        idx += 5
    
    # Derived features
    if config.features.use_derived:
        derived_list = list(config.features.derived)
        for i, feat_name in enumerate(derived_list):
            feat_idx = idx + i
            if feat_name == 'delta_y':
                features[:, :, feat_idx] = -features[:, :, feat_idx]
            elif feat_name == 'angle':
                # angle = (atan2(dy, dx) + pi) / (2*pi), mirror: 1 - angle
                features[:, :, feat_idx] = 1.0 - features[:, :, feat_idx]
            # zone_start, zone_end: leave as is (slight inaccuracy acceptable)
            # touchline_distance, center_distance: symmetric, no change needed
    
    return features


def predict_with_tta(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    config: OmegaConf,
    device: torch.device
) -> torch.Tensor:
    """Predict with Test Time Augmentation (Y-mirror).
    
    Args:
        model: The model
        batch: Batch dictionary with features, type_ids, result_ids, mask
        config: Configuration object
        device: Device to run on
        
    Returns:
        Averaged predictions (batch, 2)
    """
    # Original prediction
    pred_original = model(batch)  # (batch, 2): (dx_norm, dy_norm)
    
    # Y-mirrored prediction
    batch_mirrored = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    batch_mirrored['features'] = apply_y_mirror_to_features(batch['features'], config)
    
    pred_mirrored = model(batch_mirrored)  # (batch, 2)
    
    # Mirror back the dy prediction: dy_original = -dy_mirrored
    pred_mirrored_corrected = pred_mirrored.clone()
    pred_mirrored_corrected[:, 1] = -pred_mirrored[:, 1]
    
    # Average
    pred_avg = (pred_original + pred_mirrored_corrected) / 2.0
    
    return pred_avg


def _infer_checkpoint_config_path(
    config_path: Optional[str],
    checkpoint_path: Optional[str],
    default_config_path: str = "configs/config.yaml",
) -> str:
    """Resolve which config path to use for inference.

    Priority:
    - If user explicitly provided config_path (not None): use it.
    - Else, if checkpoint_path is provided:
      - If it's a directory: use <dir>/config.yaml if it exists.
      - If it's a file: use <parent>/config.yaml if it exists.
    - Else: fall back to default_config_path.
    """
    if config_path is not None:
        return config_path

    if checkpoint_path:
        p = Path(checkpoint_path)
        ckpt_dir = p if p.is_dir() else p.parent
        candidate = ckpt_dir / "config.yaml"
        if candidate.exists():
            return str(candidate)

    return default_config_path


def load_best_model(
    config: OmegaConf,
    model_dir: str,
    checkpoint_path: Optional[str] = None
) -> Tuple[KLeagueLightningModule, KLeagueDataModule, str]:
    """Load the best trained model.
    
    Args:
        config: Configuration object.
        model_dir: Directory containing checkpoints.
        checkpoint_path: Explicit checkpoint path (optional).
        
    Returns:
        (model, data_module, checkpoint_path)
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
    
    return model, data_module, checkpoint_path


def _infer_checkpoint_label(checkpoint_path: Optional[str]) -> Optional[str]:
    """Infer a human-friendly label from a checkpoint path.

    Rule (requested): use the *last folder name* of the given path.
    - If checkpoint_path is a directory: use its folder name.
    - If checkpoint_path is a file (e.g., .ckpt): use its parent folder name.
    """
    if not checkpoint_path:
        return None

    p = Path(checkpoint_path)
    # For file paths, use the parent folder. For directory paths, use the directory name.
    # (Don't depend on file existence: allow paths that don't exist yet.)
    return p.name if p.suffix == "" else p.parent.name


def run_inference(
    config_path: Optional[str],
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
    # Resolve config path (default: use checkpoint's config.yaml if present)
    resolved_config_path = _infer_checkpoint_config_path(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        default_config_path="configs/config.yaml",
    )

    # Load configuration
    print(f"Loading config from: {resolved_config_path}")
    config = OmegaConf.load(resolved_config_path)
    
    # Set random seed
    pl.seed_everything(config.seed, workers=True)
    
    # Load model
    # If user passes a directory via --checkpoint, use it as model_dir and auto-detect the ckpt inside.
    model_dir = config.paths.model_dir
    if checkpoint_path is not None and Path(checkpoint_path).is_dir():
        model_dir = checkpoint_path
        checkpoint_path = None

    model, data_module, checkpoint_path = load_best_model(config, model_dir, checkpoint_path)
    
    # Setup test data
    data_module.setup('test')
    test_dataloader = data_module.test_dataloader()
    
    # Field dimensions for denormalization
    field_length = config.field.length
    field_width = config.field.width

    # Postprocess config (optional)
    pp_cfg = getattr(getattr(config, "inference", None), "postprocess", None)
    pp_enabled = bool(getattr(pp_cfg, "enabled", False)) if pp_cfg is not None else False
    pp_clip = bool(getattr(pp_cfg, "clip_to_pitch", True)) if pp_cfg is not None else True
    pp_max_dist = getattr(pp_cfg, "max_pass_distance_m", 72.0) if pp_cfg is not None else 72.0
    pp_fallback = str(getattr(pp_cfg, "fallback", "last_start")) if pp_cfg is not None else "last_start"
    
    # TTA config
    tta_cfg = getattr(getattr(config, "inference", None), "tta", None)
    tta_enabled = bool(getattr(tta_cfg, "enabled", False)) if tta_cfg is not None else False
    tta_y_mirror = bool(getattr(tta_cfg, "y_mirror", False)) if tta_cfg is not None else False
    use_tta = tta_enabled and tta_y_mirror
    
    # Run inference
    print("Running inference...")
    if use_tta:
        print("  TTA enabled: Y-mirror averaging")
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
            
            # Forward pass (with optional TTA)
            # Model outputs normalized deltas (dx_norm, dy_norm) in [-1, 1] range.
            if use_tta:
                pred = predict_with_tta(model, batch_device, config, device)
            else:
                pred = model(batch_device)  # (batch, 2) in [-1, 1]

            # We must reconstruct end = last_start + delta.
            features = batch_device.get("features")
            mask = batch_device.get("mask")
            if not isinstance(features, torch.Tensor) or not isinstance(mask, torch.Tensor):
                raise ValueError("Inference requires 'features' and 'mask' tensors in batch.")

            batch_size = features.shape[0]
            # last valid index per sample
            seq_lens = mask.sum(dim=1) - 1
            seq_lens = seq_lens.clamp(min=0).long()
            batch_idx_t = torch.arange(batch_size, device=features.device)

            last_start_x = (features[batch_idx_t, seq_lens, 0].detach().cpu().numpy()) * field_length
            last_start_y = (features[batch_idx_t, seq_lens, 1].detach().cpu().numpy()) * field_width

            # Decode delta from [-1,1] to meters
            dx_norm = pred[:, 0].detach().cpu().numpy()
            dy_norm = pred[:, 1].detach().cpu().numpy()
            dx = dx_norm * field_length
            dy = dy_norm * field_width

            pred_x = last_start_x + dx
            pred_y = last_start_y + dy

            # Optional: rule-based stabilization using the last event's start position.
            if pp_enabled:
                pred_x, pred_y = stabilize_end_coordinates(
                    pred_x,
                    pred_y,
                    field_length=field_length,
                    field_width=field_width,
                    last_start_x=last_start_x,
                    last_start_y=last_start_y,
                    clip_to_pitch=pp_clip,
                    max_pass_distance_m=None if pp_max_dist in (None, "null") else float(pp_max_dist),
                    fallback=pp_fallback,
                )
            
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
        ckpt_label = _infer_checkpoint_label(checkpoint_path)
        output_path = output_dir / f"{ckpt_label}_submission.csv" if ckpt_label else (output_dir / "submission.csv")
    
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
        default=None,
        help="Path to configuration file (optional; if omitted, uses <checkpoint_dir>/config.yaml when available)"
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

