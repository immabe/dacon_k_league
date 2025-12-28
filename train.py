"""Training script for K-League pass prediction."""

import os
import argparse
from pathlib import Path
import re
from typing import Optional
from datetime import datetime
import uuid

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import OmegaConf

from src.data import KLeagueDataModule
from src.models import KLeagueLightningModule


def _safe_dirname(name: str, max_len: int = 80) -> str:
    """Make a filesystem-friendly directory name."""
    name = (name or "").strip()
    if not name:
        return "mlflow-run"
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:max_len].strip("._-") or "mlflow-run"


class MLflowCheckpointDirCallback(pl.Callback):
    """Route checkpoints into checkpoints/<run_name>/ after MLflow run starts."""

    def __init__(self, base_dir: Path, checkpoint_callback: ModelCheckpoint):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.checkpoint_callback = checkpoint_callback
        self.run_dir: Optional[Path] = None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger = trainer.logger
        if not isinstance(logger, MLFlowLogger):
            return

        run_id = logger.run_id
        # Resolve run_name from MLflow (preferred) so it matches what's recorded in mlruns/meta.yaml
        run_name = None
        try:
            run = logger.experiment.get_run(run_id)
            run_name = getattr(run.info, "run_name", None)
        except Exception:
            run_name = getattr(logger, "run_name", None)

        safe_name = _safe_dirname(run_name or "")
        # User requested: use run_name only (no run_id nesting)
        run_dir = self.base_dir / safe_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Update checkpoint destination directory
        self.checkpoint_callback.dirpath = run_dir
        self.run_dir = run_dir

        # Small breadcrumb for humans
        info_file = run_dir / "run_info.txt"
        info_file.write_text(f"run_name: {run_name}\nrun_id: {run_id}\n")

        # Record config in the run folder (if available on the module)
        try:
            cfg = getattr(pl_module, "config", None)
            if cfg is not None:
                (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
        except Exception as e:
            print(f"Warning: failed to write config.yaml: {e}")

        print(f"[checkpoints] MLflow run_name={run_name} run_id={run_id}")
        print(f"[checkpoints] Saving checkpoints under: {run_dir}")


def main(config_path: str, overrides: list = None):
    """Main training function.
    
    Args:
        config_path: Path to configuration YAML file.
        overrides: List of config overrides in format "key=value".
    """
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Apply overrides
    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, override_conf)
    
    # Set random seed
    pl.seed_everything(config.seed, workers=True)
    
    # Create output directories
    output_dir = Path(config.paths.output_dir)
    model_dir = Path(config.paths.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    print("Initializing data module...")
    data_module = KLeagueDataModule(config)
    data_module.setup('fit')
    
    # Get dimensions from data module
    feature_dim = data_module.get_feature_dim()
    num_type_classes = data_module.get_num_type_classes()
    num_result_classes = data_module.get_num_result_classes()
    
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of type classes: {num_type_classes}")
    print(f"Number of result classes: {num_result_classes}")
    
    # Initialize model
    print("Initializing model...")
    model = KLeagueLightningModule(
        config=config,
        feature_dim=feature_dim,
        num_type_classes=num_type_classes,
        num_result_classes=num_result_classes
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint - save best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename=config.training.checkpoint.filename,
        monitor=config.training.checkpoint.monitor,
        mode=config.training.checkpoint.mode,
        save_top_k=config.training.checkpoint.save_top_k,
        save_last=False,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Route checkpoints under a subfolder matching MLflow run name / id
    callbacks.append(MLflowCheckpointDirCallback(model_dir, checkpoint_callback))
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=config.training.early_stopping.monitor,
        patience=config.training.early_stopping.patience,
        mode=config.training.early_stopping.mode,
        min_delta=config.training.early_stopping.min_delta,
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    # Setup MLflow logger
    # Ensure we always have a deterministic run_name to build checkpoint folder names and record config.
    # If not provided, create one (timestamp + short uuid) to avoid collisions.
    if getattr(config.mlflow, "run_name", None) in (None, "null", ""):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        config.mlflow.run_name = f"run-{ts}-{suffix}"

    mlflow_logger = MLFlowLogger(
        experiment_name=config.mlflow.experiment_name,
        run_name=getattr(config.mlflow, "run_name", None),
        tracking_uri=config.mlflow.tracking_uri,
        log_model=True
    )
    
    # Log hyperparameters
    mlflow_logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        max_epochs=config.training.max_epochs,
        callbacks=callbacks,
        logger=mlflow_logger,
        gradient_clip_val=config.training.gradient_clip_val,
        precision=config.training.precision,
        log_every_n_steps=config.mlflow.log_every_n_steps,
        deterministic=True,
        enable_progress_bar=True
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, data_module)
    
    # Print best model path
    print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    
    # Save best model path:
    # - per-run folder (keeps history)
    # - root folder (backward-compatible for inference auto-load)
    best_model_paths = [
        Path(checkpoint_callback.dirpath) / "best_model_path.txt",
        model_dir / "best_model_path.txt",
    ]
    for p in best_model_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(checkpoint_callback.best_model_path)
        print(f"Best model path saved to: {p}")

    # Record config under the checkpoint run folder (run_name-only)
    try:
        run_cfg_path = Path(checkpoint_callback.dirpath) / "config.yaml"
        run_cfg_path.write_text(OmegaConf.to_yaml(config))
        print(f"Config saved to: {run_cfg_path}")
    except Exception as e:
        print(f"Warning: failed to write config.yaml: {e}")
    
    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train K-League pass prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in format key=value"
    )
    
    args = parser.parse_args()
    main(args.config, args.override)

