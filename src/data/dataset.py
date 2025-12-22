"""Dataset classes for K-League pass prediction."""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from omegaconf import DictConfig

from ..utils.features import FeatureExtractor


class KLeagueDataset(Dataset):
    """Dataset for K-League pass sequence data."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_extractor: FeatureExtractor,
        game_episodes: Optional[List[str]] = None,
        include_target: bool = True
    ):
        """Initialize dataset.
        
        Args:
            data: DataFrame with all event data.
            feature_extractor: Feature extraction module.
            game_episodes: List of game_episode IDs to include. If None, use all.
            include_target: Whether to include target coordinates.
        """
        # Keep a single shared DataFrame and slice by precomputed episode ranges.
        # This avoids O(#episodes) DataFrame filtering/copying at init time.
        self.data = data
        self.feature_extractor = feature_extractor
        self.include_target = include_target

        # Sort once for stable slicing
        self.data_sorted = (
            self.data.sort_values(['game_episode', 'action_id'])
            .reset_index(drop=True)
        )

        # Build episode -> (start, end) slice index in O(N)
        episode_arr = self.data_sorted['game_episode'].to_numpy()
        if len(episode_arr) == 0:
            self.episode_to_slice = {}
            self.all_episodes = []
        else:
            change = np.flatnonzero(episode_arr[1:] != episode_arr[:-1]) + 1
            starts = np.concatenate(([0], change))
            ends = np.concatenate((change, [len(episode_arr)]))
            unique_eps = episode_arr[starts]
            self.episode_to_slice = {
                str(ep): (int(s), int(e)) for ep, s, e in zip(unique_eps, starts, ends)
            }
            self.all_episodes = [str(ep) for ep in unique_eps]
        
        # Get unique episodes
        if game_episodes is not None:
            self.game_episodes = game_episodes
        else:
            self.game_episodes = self.all_episodes

        # Validate requested episodes exist
        missing = [ep for ep in self.game_episodes if ep not in self.episode_to_slice]
        if missing:
            raise KeyError(f"Unknown game_episode(s) in dataset: {missing[:5]} (and {max(0, len(missing)-5)} more)")
    
    def __len__(self) -> int:
        """Return number of episodes."""
        return len(self.game_episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get features for a single episode.
        
        Args:
            idx: Index of episode.
            
        Returns:
            Dictionary with features, type_ids, result_ids, and optionally targets.
        """
        episode = self.game_episodes[idx]
        start, end = self.episode_to_slice[episode]
        df = self.data_sorted.iloc[start:end]
        
        result = self.feature_extractor.extract_sequence_features(
            df, include_target=self.include_target
        )
        result['game_episode'] = episode
        
        return result


class KLeagueTestDataset(Dataset):
    """Dataset for K-League test data (loads from individual CSV files)."""
    
    def __init__(
        self,
        test_csv_path: str,
        test_dir: str,
        feature_extractor: FeatureExtractor
    ):
        """Initialize test dataset.
        
        Args:
            test_csv_path: Path to test.csv with file references.
            test_dir: Directory containing test episode CSV files.
            feature_extractor: Feature extraction module.
        """
        self.feature_extractor = feature_extractor
        
        # Load test file listing
        self.test_df = pd.read_csv(test_csv_path)
        self.test_dir = test_dir
        
        # Build paths
        self.game_episodes = self.test_df['game_episode'].tolist()
        self.paths = self.test_df['path'].tolist()
    
    def __len__(self) -> int:
        """Return number of test episodes."""
        return len(self.game_episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get features for a single test episode.
        
        Args:
            idx: Index of episode.
            
        Returns:
            Dictionary with features, type_ids, result_ids (no targets).
        """
        episode = self.game_episodes[idx]
        
        # Load episode data from CSV
        path = self.paths[idx]
        # Handle relative path
        if path.startswith('./'):
            path = path[2:]
        full_path = os.path.join(os.path.dirname(self.test_dir), path)
        
        df = pd.read_csv(full_path)
        
        result = self.feature_extractor.extract_sequence_features(
            df, include_target=False
        )
        result['game_episode'] = episode
        
        return result

