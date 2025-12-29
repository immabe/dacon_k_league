"""Rule-based postprocessing for inference-time stabilization.

This module intentionally keeps the logic simple and dependency-free.
It does NOT use any external/open-source code; rules are customized for this project.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


def stabilize_end_coordinates(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    *,
    field_length: float,
    field_width: float,
    last_start_x: Optional[np.ndarray] = None,
    last_start_y: Optional[np.ndarray] = None,
    clip_to_pitch: bool = True,
    max_pass_distance_m: Optional[float] = 72.0,
    fallback: str = "last_start",
) -> Tuple[np.ndarray, np.ndarray]:
    """Stabilize predicted end coordinates with simple physical/range constraints.

    Args:
        pred_x/pred_y: Predicted end coordinates in original scale (meters).
        field_length/field_width: Pitch dimensions (meters).
        last_start_x/last_start_y: Last event's start coordinates (meters). If provided,
            will be used for fallback and distance limiting.
        clip_to_pitch: Clamp coordinates to [0, field_length] x [0, field_width].
        max_pass_distance_m: If not None, cap distance from (last_start_x, last_start_y)
            to (pred_x, pred_y) by projecting onto a circle of radius max_pass_distance_m.
        fallback: What to do for non-finite preds: "last_start" or "center".

    Returns:
        (pred_x_stable, pred_y_stable) as float arrays.
    """
    pred_x = np.asarray(pred_x, dtype=np.float64)
    pred_y = np.asarray(pred_y, dtype=np.float64)

    if pred_x.shape != pred_y.shape:
        raise ValueError(f"Shape mismatch: pred_x{pred_x.shape} vs pred_y{pred_y.shape}")

    if last_start_x is not None:
        last_start_x = np.asarray(last_start_x, dtype=np.float64)
    if last_start_y is not None:
        last_start_y = np.asarray(last_start_y, dtype=np.float64)

    if (last_start_x is None) != (last_start_y is None):
        raise ValueError("last_start_x and last_start_y must be both provided or both None")

    # 1) Handle non-finite predictions
    finite = np.isfinite(pred_x) & np.isfinite(pred_y)
    if not finite.all():
        if fallback not in ("last_start", "center"):
            raise ValueError(f"Unknown fallback: {fallback}")

        if fallback == "center" or last_start_x is None:
            fx = field_length / 2.0
            fy = field_width / 2.0
            pred_x = pred_x.copy()
            pred_y = pred_y.copy()
            pred_x[~finite] = fx
            pred_y[~finite] = fy
        else:
            pred_x = pred_x.copy()
            pred_y = pred_y.copy()
            pred_x[~finite] = last_start_x[~finite]
            pred_y[~finite] = last_start_y[~finite]

    # 2) Clip to pitch bounds first (keeps later math stable)
    if clip_to_pitch:
        pred_x = np.clip(pred_x, 0.0, float(field_length))
        pred_y = np.clip(pred_y, 0.0, float(field_width))

    # 3) Cap distance from last known ball position (last start)
    if max_pass_distance_m is not None and last_start_x is not None:
        r = float(max_pass_distance_m)
        if r <= 0:
            raise ValueError("max_pass_distance_m must be > 0 or None")

        dx = pred_x - last_start_x
        dy = pred_y - last_start_y
        dist = np.sqrt(dx * dx + dy * dy)

        # Where dist > r, project onto circle radius r.
        # Add eps to avoid divide-by-zero for dist==0.
        eps = 1e-12
        scale = r / (dist + eps)
        mask = dist > r
        if mask.any():
            pred_x = pred_x.copy()
            pred_y = pred_y.copy()
            pred_x[mask] = last_start_x[mask] + dx[mask] * scale[mask]
            pred_y[mask] = last_start_y[mask] + dy[mask] * scale[mask]

        if clip_to_pitch:
            pred_x = np.clip(pred_x, 0.0, float(field_length))
            pred_y = np.clip(pred_y, 0.0, float(field_width))

    return pred_x.astype(np.float32), pred_y.astype(np.float32)


