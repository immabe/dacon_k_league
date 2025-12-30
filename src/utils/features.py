"""Feature extraction module for K-League pass prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig


class FeatureExtractor:
    """Extract and process features from event sequences."""
    
    def __init__(self, config: DictConfig):
        """Initialize feature extractor.
        
        Args:
            config: Configuration object containing feature settings.
        """
        self.config = config
        self.field_length = config.field.length  # 105
        self.field_width = config.field.width    # 68
        
        # Load vocabulary from config
        self.type_names = list(config.vocabulary.type_names)
        self.result_names = list(config.vocabulary.result_names)
        
        # Build vocabulary mappings
        self.type_to_idx = {name: idx for idx, name in enumerate(self.type_names)}
        self.result_to_idx = {name: idx for idx, name in enumerate(self.result_names)}
        
        # Feature configuration
        self.use_categorical = config.features.use_categorical
        self.use_numerical = config.features.use_numerical
        self.use_derived = config.features.use_derived
        
        # Get enabled derived features
        self.derived_features = list(config.features.derived) if self.use_derived else []
    
    @property
    def num_type_classes(self) -> int:
        """Number of event type classes."""
        return len(self.type_names)
    
    @property
    def num_result_classes(self) -> int:
        """Number of result classes."""
        return len(self.result_names)
    
    def get_feature_dim(self) -> int:
        """Calculate total feature dimension.
        
        Returns:
            Total number of features per event.
        """
        dim = 0
        
        # Numerical features (normalized coordinates + time)
        if self.use_numerical:
            # start_x, start_y, end_x, end_y, time_seconds (normalized)
            dim += 5
        
        # Derived features
        if self.use_derived:
            # Each derived feature adds 1 dimension
            dim += len(self.derived_features)
        
        # Categorical: is_home (binary)
        if self.use_categorical:
            dim += 1
        
        return dim
    
    def encode_type_name(self, type_name: str) -> int:
        """Encode event type to index.
        
        Args:
            type_name: Event type string.
            
        Returns:
            Integer index.
        """
        return self.type_to_idx.get(type_name, self.type_to_idx['<UNK>'])
    
    def encode_result_name(self, result_name: Optional[str]) -> int:
        """Encode result to index.
        
        Args:
            result_name: Result string or None.
            
        Returns:
            Integer index.
        """
        if pd.isna(result_name) or result_name is None:
            return self.result_to_idx['<PAD>']
        return self.result_to_idx.get(result_name, self.result_to_idx['<UNK>'])
    
    def normalize_coordinates(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize coordinates to [0, 1] range.
        
        Args:
            x: X coordinates.
            y: Y coordinates.
            
        Returns:
            Tuple of normalized (x, y) arrays.
        """
        x_norm = np.clip(x / self.field_length, 0, 1)
        y_norm = np.clip(y / self.field_width, 0, 1)
        return x_norm, y_norm
    
    def denormalize_coordinates(
        self,
        x_norm: np.ndarray,
        y_norm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Denormalize coordinates from [0, 1] to field dimensions.
        
        Args:
            x_norm: Normalized X coordinates.
            y_norm: Normalized Y coordinates.
            
        Returns:
            Tuple of denormalized (x, y) arrays.
        """
        x = x_norm * self.field_length
        y = y_norm * self.field_width
        return x, y
    
    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived features from raw data.
        
        Args:
            df: DataFrame with raw event data.
            
        Returns:
            DataFrame with added derived features.
        """
        df = df.copy()

        # Handle missing end coordinates (common in test for the last event)
        # - Replace missing end with start so that derived features remain finite.
        # - This also aligns train-time features with inference-time features when we mask the
        #   last event's end_x/end_y during training.
        missing_end = df['end_x'].isna() | df['end_y'].isna()
        if missing_end.any():
            df.loc[missing_end, 'end_x'] = df.loc[missing_end, 'start_x']
            df.loc[missing_end, 'end_y'] = df.loc[missing_end, 'start_y']
        
        # Basic position differences
        df['delta_x'] = df['end_x'] - df['start_x']
        df['delta_y'] = df['end_y'] - df['start_y']
        
        # Euclidean distance
        df['distance'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
        
        # Angle of movement (in radians, normalized to [0, 1])
        df['angle'] = (np.arctan2(df['delta_y'], df['delta_x']) + np.pi) / (2 * np.pi)
        
        # Time delta for speed calculation
        # This function is called per-episode (single game_episode), so groupby is unnecessary
        df['time_delta'] = df['time_seconds'].diff().fillna(0.1)
        df['time_delta'] = df['time_delta'].clip(lower=0.1)  # Avoid division by zero
        
        # Speed (distance / time)
        df['speed'] = df['distance'] / df['time_delta']
        df['speed'] = df['speed'].clip(upper=100)  # Cap extreme values
        
        # Zone features (divide field into 6x4 grid)
        df['zone_start_x'] = (df['start_x'] / self.field_length * 6).astype(int).clip(0, 5)
        df['zone_start_y'] = (df['start_y'] / self.field_width * 4).astype(int).clip(0, 3)
        df['zone_start'] = df['zone_start_x'] * 4 + df['zone_start_y']
        
        df['zone_end_x'] = (df['end_x'] / self.field_length * 6).astype(int).clip(0, 5)
        df['zone_end_y'] = (df['end_y'] / self.field_width * 4).astype(int).clip(0, 3)
        df['zone_end'] = df['zone_end_x'] * 4 + df['zone_end_y']
        
        # Forward pass indicator (x increases means forward)
        df['is_forward_pass'] = (df['delta_x'] > 0).astype(float)
        
        # Progressive pass (moves ball forward significantly - more than 10m)
        df['is_progressive'] = (df['delta_x'] > 10).astype(float)
        
        return df
    
    def extract_sequence_features(
        self,
        df: pd.DataFrame,
        include_target: bool = True
    ) -> Dict[str, np.ndarray]:
        """Extract features from an event sequence.
        
        Args:
            df: DataFrame with events for a single episode.
            include_target: Whether to include target coordinates.
            
        Returns:
            Dictionary containing:
                - features: (seq_len, feature_dim) array
                - type_ids: (seq_len,) array of type indices
                - result_ids: (seq_len,) array of result indices
                - target_dx: target dx (normalized to [-1, 1]) (if include_target)
                - target_dy: target dy (normalized to [-1, 1]) (if include_target)
                - seq_len: sequence length
        """
        # Sort by action_id to ensure correct order
        df = df.sort_values('action_id').reset_index(drop=True)

        # If training, take target from the last row but do NOT let the model see
        # the last event's end_x/end_y (to match inference where it can be missing).
        #
        # Target is delta (dx, dy) from last event start -> end, normalized to [-1, 1]:
        #   dx_norm = clip(dx / field_length, -1, 1)
        #   dy_norm = clip(dy / field_width,  -1, 1)
        target_dx = None
        target_dy = None
        if include_target:
            last_row = df.iloc[-1]
            dx = float(last_row['end_x']) - float(last_row['start_x'])
            dy = float(last_row['end_y']) - float(last_row['start_y'])

            dx_norm = np.clip(dx / self.field_length, -1.0, 1.0)
            dy_norm = np.clip(dy / self.field_width, -1.0, 1.0)

            target_dx = dx_norm
            target_dy = dy_norm

            # Mask the last event's end coordinates to prevent leakage
            df = df.copy()
            df.loc[df.index[-1], ['end_x', 'end_y']] = np.nan

        # Compute derived features (also handles missing end coords)
        df = self.compute_derived_features(df)
        
        seq_len = len(df)
        
        # Encode categorical features
        type_ids = np.array([self.encode_type_name(t) for t in df['type_name']])
        result_ids = np.array([self.encode_result_name(r) for r in df['result_name']])
        
        # Build numerical feature array
        features_list = []
        
        if self.use_numerical:
            # Normalized coordinates
            start_x_norm, start_y_norm = self.normalize_coordinates(
                df['start_x'].values, df['start_y'].values
            )
            end_x_norm, end_y_norm = self.normalize_coordinates(
                df['end_x'].values, df['end_y'].values
            )
            
            # Normalized time (assume max time ~60 min = 3600 sec per half)
            time_norm = df['time_seconds'].values / 3600.0
            
            features_list.extend([
                start_x_norm.reshape(-1, 1),
                start_y_norm.reshape(-1, 1),
                end_x_norm.reshape(-1, 1),
                end_y_norm.reshape(-1, 1),
                time_norm.reshape(-1, 1)
            ])
        
        if self.use_derived:
            for feat_name in self.derived_features:
                if feat_name in df.columns:
                    feat_vals = df[feat_name].values
                    
                    # Normalize based on feature type
                    if feat_name == 'delta_x':
                        feat_vals = feat_vals / self.field_length
                    elif feat_name == 'delta_y':
                        feat_vals = feat_vals / self.field_width
                    elif feat_name == 'distance':
                        # Max distance is diagonal of field
                        max_dist = np.sqrt(self.field_length**2 + self.field_width**2)
                        feat_vals = feat_vals / max_dist
                    elif feat_name == 'speed':
                        feat_vals = feat_vals / 50.0  # Normalize to reasonable range
                    elif feat_name in ['zone_start', 'zone_end']:
                        feat_vals = feat_vals / 24.0  # 6x4 = 24 zones
                    # angle, is_forward_pass, is_progressive already in [0, 1]
                    
                    features_list.append(feat_vals.reshape(-1, 1))
        
        if self.use_categorical:
            # is_home as binary feature
            is_home = df['is_home'].astype(float).values
            features_list.append(is_home.reshape(-1, 1))
        
        # Stack all features
        if features_list:
            features = np.hstack(features_list).astype(np.float32)
        else:
            features = np.zeros((seq_len, 1), dtype=np.float32)
        
        result = {
            'features': features,
            'type_ids': type_ids.astype(np.int64),
            'result_ids': result_ids.astype(np.int64),
            'seq_len': seq_len
        }
        
        # Get target from last row
        if include_target:
            result['target_dx'] = np.float32(target_dx)
            result['target_dy'] = np.float32(target_dy)
        
        return result
    
    def collate_fn(
        self,
        batch: List[Dict[str, np.ndarray]],
        max_seq_len: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Collate batch of sequences with padding.
        
        Args:
            batch: List of feature dictionaries from extract_sequence_features.
            max_seq_len: Maximum sequence length (for truncation).
            
        Returns:
            Dictionary with batched and padded tensors.
        """
        import torch
        
        batch_size = len(batch)
        
        # Get max sequence length in batch
        seq_lens = [item['seq_len'] for item in batch]
        max_len = max(seq_lens)
        
        if max_seq_len is not None:
            max_len = min(max_len, max_seq_len)
        
        feature_dim = batch[0]['features'].shape[1]
        
        # Initialize padded arrays
        features = np.zeros((batch_size, max_len, feature_dim), dtype=np.float32)
        type_ids = np.zeros((batch_size, max_len), dtype=np.int64)
        result_ids = np.zeros((batch_size, max_len), dtype=np.int64)
        mask = np.zeros((batch_size, max_len), dtype=np.bool_)
        
        has_target = 'target_dx' in batch[0]
        if has_target:
            target_dx = np.zeros(batch_size, dtype=np.float32)
            target_dy = np.zeros(batch_size, dtype=np.float32)
        
        for i, item in enumerate(batch):
            seq_len = min(item['seq_len'], max_len)
            
            # Take last seq_len events (most recent are most relevant)
            if item['seq_len'] > max_len:
                start_idx = item['seq_len'] - max_len
                features[i, :seq_len] = item['features'][start_idx:]
                type_ids[i, :seq_len] = item['type_ids'][start_idx:]
                result_ids[i, :seq_len] = item['result_ids'][start_idx:]
            else:
                features[i, :seq_len] = item['features']
                type_ids[i, :seq_len] = item['type_ids']
                result_ids[i, :seq_len] = item['result_ids']
            
            mask[i, :seq_len] = True
            
            if has_target:
                target_dx[i] = item['target_dx']
                target_dy[i] = item['target_dy']
        
        result = {
            'features': torch.from_numpy(features),
            'type_ids': torch.from_numpy(type_ids),
            'result_ids': torch.from_numpy(result_ids),
            'mask': torch.from_numpy(mask),
            'seq_lens': torch.tensor(seq_lens, dtype=torch.long),
            # Keep episode identifiers for stable inference mapping
            'game_episode': [item.get('game_episode') for item in batch]
        }
        
        if has_target:
            result['target_dx'] = torch.from_numpy(target_dx)
            result['target_dy'] = torch.from_numpy(target_dy)
        
        return result

