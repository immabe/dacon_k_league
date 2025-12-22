"""PyTorch Lightning DataModule for K-League data."""

import os
import pandas as pd
import numpy as np
from typing import Optional, List
from functools import partial

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from omegaconf import DictConfig

from .dataset import KLeagueDataset, KLeagueTestDataset
from ..utils.features import FeatureExtractor


class KLeagueDataModule(pl.LightningDataModule):
    """Lightning DataModule for K-League pass prediction."""
    
    def __init__(self, config: DictConfig):
        """Initialize DataModule.
        
        Args:
            config: Configuration object.
        """
        super().__init__()
        self.config = config
        
        # Paths
        self.data_dir = config.paths.data_dir
        self.train_file = config.paths.train_file
        self.test_file = config.paths.test_file
        self.test_dir = config.paths.test_dir
        
        # Training params
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.val_split = config.training.val_split
        
        # Max sequence length
        self.max_seq_len = config.model.baller2vec.max_seq_len
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(config)
        
        # Data
        self.train_data: Optional[pd.DataFrame] = None
        self.train_dataset: Optional[KLeagueDataset] = None
        self.val_dataset: Optional[KLeagueDataset] = None
        self.test_dataset: Optional[KLeagueTestDataset] = None
    
    def prepare_data(self):
        """Download or prepare data (called only on main process)."""
        # Data is already available, nothing to download
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'.
        """
        if stage == 'fit' or stage is None:
            # Load training data
            train_path = os.path.join(self.data_dir, self.train_file)
            self.train_data = pd.read_csv(train_path)
            
            # Get unique episodes (with group id)
            episode_df = (
                self.train_data[['game_episode', 'game_id']]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            all_episodes = episode_df['game_episode'].tolist()

            # Split into train/val
            # Prefer group split by game_id to avoid leakage across the same match.
            use_group_split = bool(getattr(self.config.training, 'use_game_id_split', True))
            if use_group_split:
                splitter = GroupShuffleSplit(
                    n_splits=1,
                    test_size=self.val_split,
                    random_state=self.config.seed
                )
                idx = np.arange(len(episode_df))
                train_idx, val_idx = next(splitter.split(idx, groups=episode_df['game_id'].values))
                train_episodes = episode_df.loc[train_idx, 'game_episode'].tolist()
                val_episodes = episode_df.loc[val_idx, 'game_episode'].tolist()
            else:
                train_episodes, val_episodes = train_test_split(
                    all_episodes,
                    test_size=self.val_split,
                    random_state=self.config.seed
                )
            
            # Create datasets
            self.train_dataset = KLeagueDataset(
                data=self.train_data,
                feature_extractor=self.feature_extractor,
                game_episodes=train_episodes,
                include_target=True
            )
            
            self.val_dataset = KLeagueDataset(
                data=self.train_data,
                feature_extractor=self.feature_extractor,
                game_episodes=val_episodes,
                include_target=True
            )
            
            print(f"Train episodes: {len(train_episodes)}")
            print(f"Val episodes: {len(val_episodes)}")
        
        if stage == 'test' or stage == 'predict' or stage is None:
            # Load test data
            test_csv_path = os.path.join(self.data_dir, self.test_file)
            test_dir = os.path.join(self.data_dir, self.test_dir)
            
            self.test_dataset = KLeagueTestDataset(
                test_csv_path=test_csv_path,
                test_dir=test_dir,
                feature_extractor=self.feature_extractor
            )
            
            print(f"Test episodes: {len(self.test_dataset)}")
    
    def _collate_fn(self, batch):
        """Custom collate function with feature extractor."""
        return self.feature_extractor.collate_fn(batch, max_seq_len=self.max_seq_len)
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.inference.batch_size,
            shuffle=False,
            num_workers=self.config.inference.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Get prediction dataloader (same as test)."""
        return self.test_dataloader()
    
    def get_feature_dim(self) -> int:
        """Get feature dimension from extractor."""
        return self.feature_extractor.get_feature_dim()
    
    def get_num_type_classes(self) -> int:
        """Get number of event type classes."""
        return self.feature_extractor.num_type_classes
    
    def get_num_result_classes(self) -> int:
        """Get number of result classes."""
        return self.feature_extractor.num_result_classes

