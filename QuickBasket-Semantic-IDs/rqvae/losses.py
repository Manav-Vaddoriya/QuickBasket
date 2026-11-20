"""
RQ-VAE Multi-Task Loss Functions.

Implements:
1. Reconstruction Loss: MSE(original_embedding, reconstructed_embedding)
2. Codebook Loss: Pulls codebook entries towards their assigned residuals
3. Commitment Loss: Restrains encoder residuals from wandering away from code vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RQVAELoss(nn.Module):
    def __init__(self, beta: float = 0.25):
        """
        Args:
            beta: Commitment loss weighting coefficient (default: 0.25)
        """
        super().__init__()
        self.beta = beta

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        residuals_list: list[torch.Tensor],
        quantized_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Computes total RQ-VAE loss and individual loss components.
        
        Args:
            x: Original continuous embeddings [batch_size, input_dim]
            x_hat: Reconstructed continuous embeddings [batch_size, input_dim]
            residuals_list: List of residuals at each level [batch_size, latent_dim]
            quantized_list: List of selected code vectors at each level [batch_size, latent_dim]
            
        Returns:
            total_loss: Scalar torch.Tensor for backpropagation
            loss_dict: Dictionary of float values for monitoring
        """
        # 1. Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(x_hat, x)
        
        # 2. Codebook and Commitment Losses accumulated across quantization levels
        codebook_loss = torch.tensor(0.0, device=x.device)
        commitment_loss = torch.tensor(0.0, device=x.device)
        
        level_losses = {}
        for level, (r_l, q_l) in enumerate(zip(residuals_list, quantized_list)):
            # Codebook loss: update codebook vector towards residual (freeze residual via detach)
            c_loss_l = F.mse_loss(q_l, r_l.detach())
            
            # Commitment loss: pull encoder residual towards codebook vector (freeze codebook via detach)
            commit_loss_l = F.mse_loss(r_l, q_l.detach())
            
            codebook_loss = codebook_loss + c_loss_l
            commitment_loss = commitment_loss + commit_loss_l
            
            level_losses[f"codebook_loss_l{level}"] = float(c_loss_l.item())
            level_losses[f"commit_loss_l{level}"] = float(commit_loss_l.item())

        # Total combined loss
        total_loss = recon_loss + codebook_loss + self.beta * commitment_loss
        
        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_recon": float(recon_loss.item()),
            "loss_codebook": float(codebook_loss.item()),
            "loss_commitment": float(commitment_loss.item()),
            **level_losses
        }
        
        return total_loss, loss_dict

