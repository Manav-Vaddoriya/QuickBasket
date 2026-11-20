"""
RQ-VAE End-to-End Model Assembly.

Combines:
- Multi-layer MLP Encoder (Input -> 512 -> 256 -> 64)
- 3-Level Residual Quantizer with STE
- Multi-layer MLP Decoder (64 -> 256 -> 512 -> Input)
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

        if encoder_hidden is None:
            encoder_hidden = [512, 256]
        if decoder_hidden is None:
            decoder_hidden = [256, 512]

        self.encoder = RQVAEEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden,
            latent_dim=latent_dim
        )
        
        self.quantizer = ResidualQuantizer(
            num_levels=num_levels,
            codebook_size=codebook_size,
            latent_dim=latent_dim
        )
        
        self.decoder = RQVAEDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden,
            output_dim=input_dim
        )

    def forward(
        self, 
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Full autoencoder forward pass.
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            x_hat: Reconstructed embeddings [batch_size, input_dim]
            quantized_st: Straight-through quantized vector [batch_size, latent_dim]
            codes: List of discrete codes [idx_0, idx_1, idx_2]
            quantized_list: List of code vectors selected at each level
            residuals_list: List of residuals at each level
        """
        # 1. Encode continuous embedding to latent vector z
        z = self.encoder(x)
        
        # 2. Residual Quantization with STE
        quantized_st, q_sum, codes, quantized_list, residuals_list = self.quantizer(z)
        
        # 3. Decode quantized vector to reconstruct original embedding
        x_hat = self.decoder(quantized_st)
        
        return x_hat, quantized_st, codes, quantized_list, residuals_list

    @torch.no_grad()
    def get_semantic_ids(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts discrete Semantic IDs for inference / evaluation.
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            semantic_ids: Integer tensor of shape [batch_size, num_levels]
        """
        z = self.encoder(x)
        _, _, codes, _, _ = self.quantizer(z)
        # Stack codes: [batch_size, num_levels]
        stacked = torch.stack(codes, dim=1)
        return stacked

