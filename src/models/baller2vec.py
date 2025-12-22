"""Baller2Vec model for K-League pass prediction.

Inspired by: Alcorn & Nguyen (2021). Baller2Vec: A Multi-Entity Transformer 
For Multi-Agent Spatiotemporal Representation Learning.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict
from omegaconf import DictConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Baller2Vec(nn.Module):
    """Baller2Vec-style transformer for pass coordinate prediction.
    
    Architecture:
    1. Input embedding: Combines numerical features, type embedding, and result embedding
    2. Positional encoding: Adds sequence position information
    3. Transformer encoder: Processes the sequence
    4. Output head: Predicts (x, y) coordinates with sigmoid activation
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_type_classes: int,
        num_result_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 300,
        type_embed_dim: int = 32,
        result_embed_dim: int = 16,
        use_positional_encoding: bool = True
    ):
        """Initialize Baller2Vec model.
        
        Args:
            feature_dim: Dimension of numerical features.
            num_type_classes: Number of event type classes.
            num_result_classes: Number of result classes.
            d_model: Transformer model dimension.
            nhead: Number of attention heads.
            num_encoder_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of feedforward network.
            dropout: Dropout rate.
            max_seq_len: Maximum sequence length.
            type_embed_dim: Dimension of type embedding.
            result_embed_dim: Dimension of result embedding.
            use_positional_encoding: Whether to use positional encoding.
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        
        # Embedding layers for categorical features
        self.type_embedding = nn.Embedding(num_type_classes, type_embed_dim)
        self.result_embedding = nn.Embedding(num_result_classes, result_embed_dim)
        
        # Input projection to d_model
        input_dim = feature_dim + type_embed_dim + result_embed_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False
        )
        
        # Output head for coordinate prediction
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),  # Output (x, y)
            nn.Sigmoid()  # Ensure output in [0, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        features: torch.Tensor,
        type_ids: torch.Tensor,
        result_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Numerical features (batch, seq_len, feature_dim).
            type_ids: Event type indices (batch, seq_len).
            result_ids: Result indices (batch, seq_len).
            mask: Boolean mask (batch, seq_len), True for valid positions.
            
        Returns:
            Predicted coordinates (batch, 2) in [0, 1] range.
        """
        batch_size, seq_len, _ = features.shape
        
        # Get embeddings
        type_emb = self.type_embedding(type_ids)  # (batch, seq_len, type_embed_dim)
        result_emb = self.result_embedding(result_ids)  # (batch, seq_len, result_embed_dim)
        
        # Concatenate all features
        x = torch.cat([features, type_emb, result_emb], dim=-1)
        
        # Project to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        
        # Create attention mask (True = ignore, False = attend)
        if mask is not None:
            src_key_padding_mask = ~mask  # Invert: True where padded
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Get representation for the last valid position
        if mask is not None:
            # Find last valid position for each sequence
            seq_lens = mask.sum(dim=1) - 1  # (batch,)
            seq_lens = seq_lens.clamp(min=0)
            batch_indices = torch.arange(batch_size, device=x.device)
            last_hidden = x[batch_indices, seq_lens]  # (batch, d_model)
        else:
            last_hidden = x[:, -1]  # Take last position
        
        # Predict coordinates
        coords = self.output_head(last_hidden)  # (batch, 2)
        
        return coords
    
    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        feature_dim: int,
        num_type_classes: int,
        num_result_classes: int
    ) -> "Baller2Vec":
        """Create model from config.
        
        Args:
            config: Configuration object.
            feature_dim: Dimension of numerical features.
            num_type_classes: Number of event type classes.
            num_result_classes: Number of result classes.
            
        Returns:
            Initialized model.
        """
        model_config = config.model.baller2vec
        
        return cls(
            feature_dim=feature_dim,
            num_type_classes=num_type_classes,
            num_result_classes=num_result_classes,
            d_model=model_config.d_model,
            nhead=model_config.nhead,
            num_encoder_layers=model_config.num_encoder_layers,
            dim_feedforward=model_config.dim_feedforward,
            dropout=model_config.dropout,
            max_seq_len=model_config.max_seq_len,
            type_embed_dim=model_config.type_embed_dim,
            result_embed_dim=model_config.result_embed_dim,
            use_positional_encoding=model_config.use_positional_encoding
        )

