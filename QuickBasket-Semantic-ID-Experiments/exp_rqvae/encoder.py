"""
Experimental RQ-VAE Encoder Module.
Projects high-dimensional continuous product embeddings into a compact latent space.
Architecture: Input (384) -> 512 -> 256 -> 64
"""

import torch
import torch.nn as nn


class RQVAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dims: list[int] = None,
        latent_dim: int = 64
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
            
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(h_dim))
            curr_dim = h_dim
            
        layers.append(nn.Linear(curr_dim, latent_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, input_dim]
        Returns:
            z: Latent tensor of shape [batch_size, latent_dim]
        """
        return self.net(x)

