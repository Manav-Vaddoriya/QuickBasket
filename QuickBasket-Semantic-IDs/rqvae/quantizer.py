"""
Residual Quantizer Module for RQ-VAE.

Implements multi-level residual vector quantization with Straight-Through Estimation (STE),
efficient squared-Euclidean nearest neighbor assignment, and codebook utilization tracking.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        num_levels: int = 3,
        codebook_size: int = 128,
        latent_dim: int = 64,
        init_scale: float = 0.02
    ):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim

        # Learnable codebooks: [num_levels, codebook_size, latent_dim]
        # Initialized with small Gaussian noise
        self.codebooks = nn.Parameter(
            torch.randn(num_levels, codebook_size, latent_dim) * init_scale
        )

    def forward(
        self, 
        z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass for residual quantization.
        
        Args:
            z: Latent vector from encoder [batch_size, latent_dim]
            
        Returns:
            quantized_st: Straight-through quantized output [batch_size, latent_dim]
            q_sum: Direct sum of selected code vectors [batch_size, latent_dim]
            codes: List of code indices for each level, each [batch_size]
            quantized_list: List of chosen code vectors for each level, each [batch_size, latent_dim]
            residuals_list: List of residuals before quantization at each level [batch_size, latent_dim]
        """
        batch_size = z.shape[0]
        residual = z
        
        q_sum = torch.zeros_like(z)
        codes = []
        quantized_list = []
        residuals_list = []
        
        for level in range(self.num_levels):
            c_l = self.codebooks[level]  # [codebook_size, latent_dim]
            residuals_list.append(residual)
            
            # Efficient squared Euclidean distance:
            # ||r - c||^2 = ||r||^2 - 2 * r @ c.T + ||c||^2
            r_sq = torch.sum(residual ** 2, dim=-1, keepdim=True)        # [B, 1]
            c_sq = torch.sum(c_l ** 2, dim=-1, keepdim=True).t()          # [1, K]
            r_dot_c = torch.matmul(residual, c_l.t())                     # [B, K]
            distances = r_sq - 2.0 * r_dot_c + c_sq                       # [B, K]
            
            # 3. Find nearest codebook entry
            idx = torch.argmin(distances, dim=-1)                         # [B]
            codes.append(idx)
            
            # 4. Select corresponding code vector
            q_l = F.embedding(idx, c_l)                                   # [B, D]
            quantized_list.append(q_l)
            
            # 5. Subtract selected code vector from residual
            residual = residual - q_l
            
            # Accumulate sum of selected code vectors
            q_sum = q_sum + q_l

        # 6. Straight-Through Estimator (STE)
        # In forward pass: quantized_st == q_sum
        # In backward pass: grad propagates to z as identity
        quantized_st = z + (q_sum - z).detach()
        
        return quantized_st, q_sum, codes, quantized_list, residuals_list

    @torch.no_grad()
    def compute_codebook_metrics(
        self, 
        all_codes: list[torch.Tensor]
    ) -> list[dict]:
        """
        Calculates active codes, utilization percentage, and perplexity per quantization level.
        
        Args:
            all_codes: List of tensors [num_samples] for each level.
        """
        metrics = []
        for level, codes_tensor in enumerate(all_codes):
            counts = torch.bincount(codes_tensor.view(-1), minlength=self.codebook_size).float()
            active_codes = int((counts > 0).sum().item())
            utilization_pct = round((active_codes / self.codebook_size) * 100.0, 2)
            
            # Probability distribution over codebook
            total_counts = counts.sum().item()
            if total_counts > 0:
                probs = counts / total_counts
                # Entropy = - sum(p * log(p))
                nonzero_probs = probs[probs > 0]
                entropy = -float(torch.sum(nonzero_probs * torch.log(nonzero_probs)).item())
                perplexity = round(math.exp(entropy), 2)
            else:
                perplexity = 0.0
                
            metrics.append({
                "level": level,
                "codebook_size": self.codebook_size,
                "active_codes": active_codes,
                "utilization_pct": utilization_pct,
                "perplexity": perplexity
            })
        return metrics

