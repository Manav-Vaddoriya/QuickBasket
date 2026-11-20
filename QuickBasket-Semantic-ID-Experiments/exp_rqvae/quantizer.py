"""
Experimental Residual Quantizer Module for RQ-VAE Experiments.

Features:
- Parametric number of levels (3, 4, etc.)
- Parametric codebook sizes (64, 128, etc.)
- Strict gradient isolation: residual = residual - q_l.detach()
  prevents commitment loss from pushing prior codebooks in reverse
- Straight-Through Estimation (STE)
- Built-in K-Means codebook initialization
- Built-in dead-codebook entry tracking and reinitialization
- Codebook perplexity, active count, and utilization metrics
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


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
        self.codebooks = nn.Parameter(
            torch.randn(num_levels, codebook_size, latent_dim) * init_scale
        )

        # Track usage frequencies for dead code detection (persistent=False keeps state_dict clean)
        self.register_buffer(
            "code_usage_counts", 
            torch.zeros(num_levels, codebook_size, dtype=torch.long),
            persistent=False
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
            
            # Find nearest codebook entry
            idx = torch.argmin(distances, dim=-1)                         # [B]
            codes.append(idx)
            
            if self.training:
                # Update usage counts
                with torch.no_grad():
                    b_counts = torch.bincount(idx, minlength=self.codebook_size)
                    self.code_usage_counts[level] += b_counts
            
            # Select corresponding code vector
            q_l = F.embedding(idx, c_l)                                   # [B, D]
            quantized_list.append(q_l)
            
            # SUBTRACT DETACHED selected code vector from residual:
            # Crucial: q_l.detach() ensures level l's commitment loss only updates encoder z,
            # avoiding backward gradient leakage into previous level codebooks!
            residual = residual - q_l.detach()
            
            # Accumulate sum of selected code vectors for reconstruction
            q_sum = q_sum + q_l

        # Straight-Through Estimator (STE)
        quantized_st = z + (q_sum - z).detach()
        
        return quantized_st, q_sum, codes, quantized_list, residuals_list

    def reset_usage_counts(self):
        """Resets codebook usage tracking counters."""
        self.code_usage_counts.zero_()

    @torch.no_grad()
    def reinitialize_dead_codes(
        self,
        level: int,
        active_residuals: torch.Tensor,
        min_usage_threshold: int = 1
    ) -> int:
        """
        Reinitializes dead codebook entries that have usage <= min_usage_threshold
        with randomly sampled residuals from the current batch/dataset.
        
        Returns:
            num_reinitialized: Number of dead entries replaced.
        """
        counts = self.code_usage_counts[level]
        dead_indices = torch.where(counts <= min_usage_threshold)[0]
        num_dead = len(dead_indices)
        
        if num_dead == 0 or active_residuals.shape[0] == 0:
            return 0
            
        # Sample replacement vectors with replacement if necessary
        sample_size = min(num_dead, active_residuals.shape[0])
        perm = torch.randperm(active_residuals.shape[0])[:sample_size]
        sampled_vectors = active_residuals[perm].to(self.codebooks.device)
        
        # In-place assignment to dead codebook entries
        self.codebooks.data[level, dead_indices[:sample_size]] = sampled_vectors
        self.code_usage_counts[level, dead_indices[:sample_size]] = min_usage_threshold + 1
        
        return sample_size

    @torch.no_grad()
    def init_codebooks_kmeans(
        self,
        level: int,
        residual_vectors: torch.Tensor,
        num_iters: int = 30,
        random_state: int = 42
    ):
        """
        Initializes codebook for a specific level using fast pure PyTorch K-Means centroids.
        
        Args:
            level: Quantization level index (0, 1, 2, ...)
            residual_vectors: Tensor of residual vectors from training set [N, latent_dim]
            num_iters: K-Means iterations
            random_state: Random seed
        """
        print(f"  [K-Means Init] Fitting pure PyTorch KMeans(k={self.codebook_size}) on {len(residual_vectors):,} residuals for Level {level}...")
        torch.manual_seed(random_state)
        N, D = residual_vectors.shape
        perm = torch.randperm(N)
        centroids = residual_vectors[perm[:self.codebook_size]].clone().to(self.codebooks.device)
        
        for it in range(num_iters):
            x_sq = torch.sum(residual_vectors ** 2, dim=-1, keepdim=True)
            c_sq = torch.sum(centroids ** 2, dim=-1, keepdim=True).t()
            dot = torch.matmul(residual_vectors, centroids.t())
            dists = x_sq - 2.0 * dot + c_sq
            labels = torch.argmin(dists, dim=-1)
            
            new_centroids = centroids.clone()
            counts = torch.bincount(labels, minlength=self.codebook_size)
            
            for k in range(self.codebook_size):
                if counts[k] > 0:
                    new_centroids[k] = residual_vectors[labels == k].mean(dim=0)
                else:
                    rand_idx = torch.randint(0, N, (1,)).item()
                    new_centroids[k] = residual_vectors[rand_idx]
                    
            diff = torch.norm(new_centroids - centroids)
            centroids = new_centroids
            if diff < 1e-4:
                break

        self.codebooks.data[level] = centroids
        print(f"  [K-Means Init] Level {level} codebook initialized with {len(centroids)} centroids (shape: {centroids.shape}).")

    @torch.no_grad()
    def compute_codebook_metrics(
        self, 
        all_codes: list[torch.Tensor]
    ) -> list[dict]:
        """
        Calculates active codes, utilization percentage, and perplexity per quantization level.
        """
        metrics = []
        for level in range(self.num_levels):
            codes_lvl = all_codes[level].cpu().numpy()
            counts = np.bincount(codes_lvl, minlength=self.codebook_size)
            active_codes = int(np.sum(counts > 0))
            dead_codes = int(self.codebook_size - active_codes)
            utilization_pct = (active_codes / self.codebook_size) * 100.0

            probs = counts / np.sum(counts)
            nonzero_probs = probs[probs > 0]
            entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
            perplexity = float(math.exp(entropy))

            metrics.append({
                "level": level,
                "codebook_size": self.codebook_size,
                "active_codes": active_codes,
                "dead_codes": dead_codes,
                "utilization_pct": round(utilization_pct, 2),
                "perplexity": round(perplexity, 2),
                "perplexity_ratio": round(perplexity / self.codebook_size, 4)
            })
        return metrics

