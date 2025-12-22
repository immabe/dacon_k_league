"""Metrics for K-League pass prediction."""

import numpy as np
import torch
from typing import Union


def euclidean_distance(
    pred_x: Union[np.ndarray, torch.Tensor],
    pred_y: Union[np.ndarray, torch.Tensor],
    true_x: Union[np.ndarray, torch.Tensor],
    true_y: Union[np.ndarray, torch.Tensor],
    reduction: str = 'mean'
) -> Union[float, np.ndarray, torch.Tensor]:
    """Calculate Euclidean distance between predicted and true coordinates.
    
    Note: Coordinates should be in original scale (105 x 68).
    
    Args:
        pred_x: Predicted x coordinates.
        pred_y: Predicted y coordinates.
        true_x: True x coordinates.
        true_y: True y coordinates.
        reduction: 'mean', 'sum', or 'none'.
        
    Returns:
        Euclidean distance(s).
    """
    if isinstance(pred_x, torch.Tensor):
        dist = torch.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        
        if reduction == 'mean':
            return dist.mean()
        elif reduction == 'sum':
            return dist.sum()
        else:
            return dist
    else:
        dist = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        
        if reduction == 'mean':
            return float(np.mean(dist))
        elif reduction == 'sum':
            return float(np.sum(dist))
        else:
            return dist


def normalized_euclidean_distance(
    pred_x: Union[np.ndarray, torch.Tensor],
    pred_y: Union[np.ndarray, torch.Tensor],
    true_x: Union[np.ndarray, torch.Tensor],
    true_y: Union[np.ndarray, torch.Tensor],
    field_length: float = 105.0,
    field_width: float = 68.0,
    reduction: str = 'mean'
) -> Union[float, np.ndarray, torch.Tensor]:
    """Calculate Euclidean distance for normalized coordinates [0, 1].
    
    Converts back to original scale before computing distance.
    
    Args:
        pred_x: Predicted normalized x coordinates [0, 1].
        pred_y: Predicted normalized y coordinates [0, 1].
        true_x: True normalized x coordinates [0, 1].
        true_y: True normalized y coordinates [0, 1].
        field_length: Field length for denormalization.
        field_width: Field width for denormalization.
        reduction: 'mean', 'sum', or 'none'.
        
    Returns:
        Euclidean distance(s) in original scale.
    """
    # Denormalize
    pred_x_denorm = pred_x * field_length
    pred_y_denorm = pred_y * field_width
    true_x_denorm = true_x * field_length
    true_y_denorm = true_y * field_width
    
    return euclidean_distance(
        pred_x_denorm, pred_y_denorm,
        true_x_denorm, true_y_denorm,
        reduction=reduction
    )

