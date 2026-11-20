"""
Experimental RQ-VAE Decoder Module.
Reconstructs original product embeddings from quantized latent representations.
Architecture: 64 -> 256 -> 512 -> Output (384)
"""

import torch
import torch.nn as nn


class RQVAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: list[int] = None,
        output_dim: int = 384
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512]
            
        layers = []
        curr_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(h_dim))
            curr_dim = h_dim
            
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: Quantized latent tensor of shape [batch_size, latent_dim]
        Returns:
            x_hat: Reconstructed embedding of shape [batch_size, output_dim]
        """
        return self.net(z_q)

