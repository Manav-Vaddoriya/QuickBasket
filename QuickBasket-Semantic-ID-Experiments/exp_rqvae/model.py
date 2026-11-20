"""
Modular RQ-VAE Model Wrapper for Experimental Workflows.
Integrates RQVAEEncoder, ResidualQuantizer, and RQVAEDecoder.
"""

import torch
import torch.nn as nn
from .encoder import RQVAEEncoder
from .decoder import RQVAEDecoder
from .quantizer import ResidualQuantizer


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        latent_dim: int = 64,
        codebook_size: int = 128,
        num_levels: int = 3,
        encoder_hidden: list[int] = None,
        decoder_hidden: list[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.num_levels = num_levels

        self.encoder = RQVAEEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden or [512, 256],
            latent_dim=latent_dim
        )
        self.quantizer = ResidualQuantizer(
            num_levels=num_levels,
            codebook_size=codebook_size,
            latent_dim=latent_dim
        )
        self.decoder = RQVAEDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden or [256, 512],
            output_dim=input_dim
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Projects continuous input embeddings to latent space z."""
        return self.encoder(x)

    def quantize(self, z: torch.Tensor):
        """Passes latent vector z through multi-level residual quantization."""
        return self.quantizer(z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Reconstructs continuous embeddings from quantized representation."""
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor):
        """End-to-end forward pass."""
        z = self.encode(x)
        z_q, q_sum, codes, q_list, r_list = self.quantize(z)
        x_hat = self.decode(z_q)
        return x_hat, z_q, codes, q_list, r_list

    @torch.no_grad()
    def get_semantic_ids(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extracts discrete semantic ID codes across all levels."""
        z = self.encode(x)
        _, _, codes, _, _ = self.quantizer(z)
        return codes

