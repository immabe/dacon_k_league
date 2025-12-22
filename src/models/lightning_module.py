"""PyTorch Lightning module for K-League pass prediction."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from .baller2vec import Baller2Vec
from ..utils.metrics import normalized_euclidean_distance


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
        if model_name == "baller2vec":
            self.model = Baller2Vec.from_config(
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
        target_x: torch.Tensor,
        target_y: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss.
        
        Args:
            pred: Predicted coordinates (batch, 2) in [0, 1].
            target_x: Target x coordinates in [0, 1].
            target_y: Target y coordinates in [0, 1].
            
        Returns:
            Loss value.
        """
        target = torch.stack([target_x, target_y], dim=1)

        loss_name = 'mse'
        if hasattr(self.config, 'training') and hasattr(self.config.training, 'loss'):
            loss_name = str(getattr(self.config.training.loss, 'name', 'mse'))

        # Leaderboard metric is Euclidean distance in original scale.
        if loss_name == 'euclidean' or loss_name == 'euclidean_sq':
            pred_x = pred[:, 0] * self.field_length
            pred_y = pred[:, 1] * self.field_width
            true_x = target[:, 0] * self.field_length
            true_y = target[:, 1] * self.field_width
            dist_sq = (pred_x - true_x) ** 2 + (pred_y - true_y) ** 2
            if loss_name == 'euclidean_sq':
                return dist_sq.mean()
            return torch.sqrt(dist_sq + 1e-12).mean()

        # Legacy: MSE on normalized coordinates
        return self.loss_fn(pred, target)
    
    def _compute_euclidean_distance(
        self,
        pred: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor
    ) -> torch.Tensor:
        """Compute Euclidean distance in original scale.
        
        Args:
            pred: Predicted coordinates (batch, 2) in [0, 1].
            target_x: Target x coordinates in [0, 1].
            target_y: Target y coordinates in [0, 1].
            
        Returns:
            Mean Euclidean distance.
        """
        return normalized_euclidean_distance(
            pred[:, 0], pred[:, 1],
            target_x, target_y,
            field_length=self.field_length,
            field_width=self.field_width,
            reduction='mean'
        )
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Loss value.
        """
        pred = self(batch)
        loss = self._compute_loss(pred, batch['target_x'], batch['target_y'])
        
        # Compute metrics
        with torch.no_grad():
            euclidean_dist = self._compute_euclidean_distance(
                pred, batch['target_x'], batch['target_y']
            )
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_euclidean', euclidean_dist, on_step=False, on_epoch=True)
        
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
        loss = self._compute_loss(pred, batch['target_x'], batch['target_y'])
        euclidean_dist = self._compute_euclidean_distance(
            pred, batch['target_x'], batch['target_y']
        )
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_euclidean', euclidean_dist, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store for epoch-end aggregation
        output = {
            'pred': pred.detach(),
            'target_x': batch['target_x'].detach(),
            'target_y': batch['target_y'].detach(),
            'loss': loss.detach()
        }
        self.validation_step_outputs.append(output)
        
        return output
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics at epoch end."""
        if not self.validation_step_outputs:
            return
        
        # Concatenate all predictions and targets
        all_pred = torch.cat([x['pred'] for x in self.validation_step_outputs])
        all_target_x = torch.cat([x['target_x'] for x in self.validation_step_outputs])
        all_target_y = torch.cat([x['target_y'] for x in self.validation_step_outputs])
        
        # Compute overall Euclidean distance
        overall_dist = self._compute_euclidean_distance(all_pred, all_target_x, all_target_y)
        self.log('val_euclidean_overall', overall_dist)
        
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
        
        # Denormalize coordinates
        pred_x = pred[:, 0] * self.field_length
        pred_y = pred[:, 1] * self.field_width
        
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

