"""PyTorch Lightning module for K-League pass prediction."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple
from omegaconf import DictConfig

from .transformer import TransformerEncoder


class KLeagueLightningModule(pl.LightningModule):
    """Lightning module for training and inference."""
    
    def __init__(
        self,
        config: DictConfig,
        feature_dim: int,
        num_type_classes: int,
        num_result_classes: int
    ):
        """Initialize Lightning module.
        
        Args:
            config: Configuration object.
            feature_dim: Dimension of numerical features.
            num_type_classes: Number of event type classes.
            num_result_classes: Number of result classes.
        """
        super().__init__()
        
        self.config = config
        self.save_hyperparameters(ignore=['config'])
        
        # Field dimensions for denormalization
        self.field_length = config.field.length
        self.field_width = config.field.width
        
        # Build model
        model_name = config.model.name
        if model_name == "transformer":
            self.model = TransformerEncoder.from_config(
                config, feature_dim, num_type_classes, num_result_classes
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Loss function (MSE on normalized coordinates)
        self.loss_fn = nn.MSELoss()
        
        # Learning rate
        self.learning_rate = config.training.learning_rate
        
        # Store predictions for validation
        self.validation_step_outputs = []
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            batch: Dictionary containing features, type_ids, result_ids, mask.
            
        Returns:
            Predicted coordinates (batch, 2).
        """
        return self.model(
            features=batch['features'],
            type_ids=batch['type_ids'],
            result_ids=batch['result_ids'],
            mask=batch['mask']
        )
    
    def _compute_loss(
        self,
        pred: torch.Tensor,
        target_dx: torch.Tensor,
        target_dy: torch.Tensor,
        batch: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute training loss.
        
        Args:
            pred: Predicted deltas normalized to [-1, 1] (batch, 2).
            target_dx: Target dx normalized to [-1, 1].
            target_dy: Target dy normalized to [-1, 1].
            batch: Optional batch dict; required for euclidean loss (needs last_start).
            
        Returns:
            Loss value.
        """
        target = torch.stack([target_dx, target_dy], dim=1)

        loss_name = 'mse'
        if hasattr(self.config, 'training') and hasattr(self.config.training, 'loss'):
            loss_name = str(getattr(self.config.training.loss, 'name', 'mse'))

        # Leaderboard metric is Euclidean distance between end coordinates in original scale.
        # With delta targets, we reconstruct end = last_start + delta (both for pred and true).
        if loss_name == 'euclidean' or loss_name == 'euclidean_sq':
            if batch is None:
                raise ValueError("euclidean loss requires the full batch to reconstruct last_start.")

            pred_end_x, pred_end_y = self._decode_to_end_coordinates(pred, batch)
            true_end_x, true_end_y = self._decode_to_end_coordinates(target, batch)

            dist_sq = (pred_end_x - true_end_x) ** 2 + (pred_end_y - true_end_y) ** 2
            if loss_name == 'euclidean_sq':
                return dist_sq.mean()
            return torch.sqrt(dist_sq + 1e-12).mean()

        # Legacy: MSE on normalized coordinates
        return self.loss_fn(pred, target)
    
    def _get_last_start_xy_m(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get last valid start_x/start_y in meters for each sample from batch features/mask."""
        features = batch["features"]
        mask = batch.get("mask")

        if mask is None:
            last_start_x_norm = features[:, -1, 0]
            last_start_y_norm = features[:, -1, 1]
        else:
            seq_lens = mask.sum(dim=1) - 1
            seq_lens = seq_lens.clamp(min=0).long()
            batch_idx = torch.arange(features.size(0), device=features.device)
            last_start_x_norm = features[batch_idx, seq_lens, 0]
            last_start_y_norm = features[batch_idx, seq_lens, 1]

        last_start_x = last_start_x_norm * self.field_length
        last_start_y = last_start_y_norm * self.field_width
        return last_start_x, last_start_y

    def _decode_delta(self, encoded_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode normalized delta in [-1,1] to dx/dy in meters."""
        dx_norm = encoded_delta[:, 0]
        dy_norm = encoded_delta[:, 1]
        dx = dx_norm * self.field_length
        dy = dy_norm * self.field_width
        return dx, dy

    def _decode_to_end_coordinates(
        self, encoded_delta: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode normalized deltas to end_x/end_y in meters using last_start from batch."""
        last_start_x, last_start_y = self._get_last_start_xy_m(batch)
        dx, dy = self._decode_delta(encoded_delta)
        end_x = last_start_x + dx
        end_y = last_start_y + dy
        return end_x, end_y

    def _decode_to_end_coordinates_with_last_start(
        self,
        encoded_delta: torch.Tensor,
        last_start_x_m: torch.Tensor,
        last_start_y_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode encoded deltas to end_x/end_y in meters given last_start (meters)."""
        dx, dy = self._decode_delta(encoded_delta)
        end_x = last_start_x_m + dx
        end_y = last_start_y_m + dy
        return end_x, end_y

    def _compute_euclidean_distance(
        self,
        pred: torch.Tensor,
        target_dx: torch.Tensor,
        target_dy: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute Euclidean distance in original scale.
        
        Args:
            pred: Predicted deltas normalized to [-1, 1] (batch, 2).
            target_dx: Target dx normalized to [-1, 1].
            target_dy: Target dy normalized to [-1, 1].
            batch: Batch dict (needed to get last_start).
            
        Returns:
            Mean Euclidean distance.
        """
        target = torch.stack([target_dx, target_dy], dim=1)
        pred_end_x, pred_end_y = self._decode_to_end_coordinates(pred, batch)
        true_end_x, true_end_y = self._decode_to_end_coordinates(target, batch)
        # euclidean_distance expects original-scale coordinates already
        dist_sq = (pred_end_x - true_end_x) ** 2 + (pred_end_y - true_end_y) ** 2
        return torch.sqrt(dist_sq + 1e-12).mean()
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Loss value.
        """
        pred = self(batch)
        loss = self._compute_loss(pred, batch['target_dx'], batch['target_dy'], batch=batch)
        batch_size = int(batch["features"].size(0))
        
        # Compute metrics
        with torch.no_grad():
            euclidean_dist = self._compute_euclidean_distance(
                pred, batch['target_dx'], batch['target_dy'], batch=batch
            )
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_euclidean', euclidean_dist, on_step=False, on_epoch=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Dictionary with predictions and targets.
        """
        pred = self(batch)
        loss = self._compute_loss(pred, batch['target_dx'], batch['target_dy'], batch=batch)
        batch_size = int(batch["features"].size(0))
        euclidean_dist = self._compute_euclidean_distance(
            pred, batch['target_dx'], batch['target_dy'], batch=batch
        )
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_euclidean', euclidean_dist, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        # Store for epoch-end aggregation
        last_start_x_m, last_start_y_m = self._get_last_start_xy_m(batch)
        output = {
            'pred': pred.detach(),
            'target_dx': batch['target_dx'].detach(),
            'target_dy': batch['target_dy'].detach(),
            'last_start_x_m': last_start_x_m.detach(),
            'last_start_y_m': last_start_y_m.detach(),
            'loss': loss.detach()
        }
        self.validation_step_outputs.append(output)
        
        return output
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics at epoch end."""
        if not self.validation_step_outputs:
            return
        
        # Concatenate all predictions / targets / last_start
        all_pred = torch.cat([x['pred'] for x in self.validation_step_outputs])
        all_target_dx = torch.cat([x['target_dx'] for x in self.validation_step_outputs])
        all_target_dy = torch.cat([x['target_dy'] for x in self.validation_step_outputs])
        all_last_start_x_m = torch.cat([x['last_start_x_m'] for x in self.validation_step_outputs])
        all_last_start_y_m = torch.cat([x['last_start_y_m'] for x in self.validation_step_outputs])

        all_target = torch.stack([all_target_dx, all_target_dy], dim=1)
        pred_end_x, pred_end_y = self._decode_to_end_coordinates_with_last_start(
            all_pred, all_last_start_x_m, all_last_start_y_m
        )
        true_end_x, true_end_y = self._decode_to_end_coordinates_with_last_start(
            all_target, all_last_start_x_m, all_last_start_y_m
        )
        dist_sq = (pred_end_x - true_end_x) ** 2 + (pred_end_y - true_end_y) ** 2
        overall_dist = torch.sqrt(dist_sq + 1e-12).mean()
        self.log('val_euclidean_overall', overall_dist, batch_size=int(all_pred.size(0)))
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Prediction step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Dictionary with predictions and game episodes.
        """
        pred = self(batch)
        # Decode deltas to end coordinates in meters
        pred_x, pred_y = self._decode_to_end_coordinates(pred, batch)
        
        return {
            'pred_x': pred_x.cpu().numpy(),
            'pred_y': pred_y.cpu().numpy(),
            'game_episode': batch.get('game_episode', None)
        }
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler.
        
        Returns:
            Optimizer and scheduler configuration.
        """
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Scheduler
        scheduler_config = self.config.training.scheduler
        scheduler_name = scheduler_config.name
        
        if scheduler_name == "cosine":
            # Cosine annealing with warmup
            warmup_epochs = scheduler_config.warmup_epochs
            total_epochs = self.config.training.max_epochs
            min_lr = scheduler_config.min_lr
            
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                    return min_lr / self.learning_rate + (1 - min_lr / self.learning_rate) * (
                        0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
                    ).item()
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5
            )
            
        elif scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=scheduler_config.min_lr
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

