"""
RQ-VAE Training and Validation Pipeline.

Features:
- Reproducible 90:10 train-validation split.
- Mini-batch training with Adam optimizer and gradient clipping.
- Dual evaluation: reconstruction loss (MSE, RMSE, Cosine Sim) and codebook perplexity/utilization.
- Checkpoint persistence for the best validation reconstruction loss.
- Early stopping mechanism to prevent overfitting and training instability.
- History logging to CSV for post-training audit.
"""

import os
import time
import math
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .model import RQVAE
from .losses import RQVAELoss


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RQVAETrainer:
    def __init__(
        self,
        model: RQVAE,
        device: str = "cpu",
        learning_rate: float = 0.001,
        commitment_beta: float = 0.25,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "QuickBasket-Semantic-IDs/checkpoints",
        patience: int = 15
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_checkpoint_path = os.path.join(checkpoint_dir, "rqvae_best.pt")
        
        self.criterion = RQVAELoss(beta=commitment_beta)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.history = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 150
    ) -> pd.DataFrame:
        print(f"\nStarting RQ-VAE Training on {self.device.upper()}...")
        print(f"Max Epochs: {max_epochs} | Early Stopping Patience: {self.patience}")
        print(f"Codebooks: {self.model.num_levels} levels x {self.model.codebook_size} codes | Latent: {self.model.latent_dim}d")
        print("-" * 80)
        
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            
            # --- Training Epoch ---
            self.model.train()
            train_recon_losses = []
            train_codebook_losses = []
            train_commit_losses = []
            train_total_losses = []
            
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                self.optimizer.zero_grad()
                
                x_hat, _, codes, quantized_list, residuals_list = self.model(batch_x)
                
                loss, loss_dict = self.criterion(
                    x=batch_x,
                    x_hat=x_hat,
                    residuals_list=residuals_list,
                    quantized_list=quantized_list
                )
                
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                self.optimizer.step()
                
                train_recon_losses.append(loss_dict["loss_recon"])
                train_codebook_losses.append(loss_dict["loss_codebook"])
                train_commit_losses.append(loss_dict["loss_commitment"])
                train_total_losses.append(loss_dict["loss_total"])

            # --- Validation Epoch ---
            self.model.eval()
            val_recon_losses = []
            val_total_losses = []
            all_val_codes = [[] for _ in range(self.model.num_levels)]
            
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(self.device)
                    x_hat, _, codes, quantized_list, residuals_list = self.model(batch_x)
                    
                    val_loss, val_dict = self.criterion(
                        x=batch_x,
                        x_hat=x_hat,
                        residuals_list=residuals_list,
                        quantized_list=quantized_list
                    )
                    val_recon_losses.append(val_dict["loss_recon"])
                    val_total_losses.append(val_dict["loss_total"])
                    
                    for lvl in range(self.model.num_levels):
                        all_val_codes[lvl].append(codes[lvl].cpu())

            # Aggregate epoch metrics
            epoch_time = time.time() - t0
            avg_train_recon = float(np.mean(train_recon_losses))
            avg_train_total = float(np.mean(train_total_losses))
            avg_val_recon = float(np.mean(val_recon_losses))
            avg_val_total = float(np.mean(val_total_losses))
            
            # Codebook utilization metrics on validation set
            concat_val_codes = [torch.cat(codes_list) for codes_list in all_val_codes]
            cb_metrics = self.model.quantizer.compute_codebook_metrics(concat_val_codes)
            
            epoch_record = {
                "epoch": epoch,
                "train_recon_mse": round(avg_train_recon, 6),
                "train_total_loss": round(avg_train_total, 6),
                "val_recon_mse": round(avg_val_recon, 6),
                "val_total_loss": round(avg_val_total, 6),
                "epoch_time_seconds": round(epoch_time, 2)
            }
            
            for m in cb_metrics:
                lvl = m["level"]
                epoch_record[f"cb_l{lvl}_active"] = m["active_codes"]
                epoch_record[f"cb_l{lvl}_utilization_pct"] = m["utilization_pct"]
                epoch_record[f"cb_l{lvl}_perplexity"] = m["perplexity"]
                
            self.history.append(epoch_record)
            
            # Console log
            l0_act = epoch_record.get("cb_l0_active", 0)
            l1_act = epoch_record.get("cb_l1_active", 0)
            l2_act = epoch_record.get("cb_l2_active", 0)
            l0_perp = epoch_record.get("cb_l0_perplexity", 0.0)
            l1_perp = epoch_record.get("cb_l1_perplexity", 0.0)
            l2_perp = epoch_record.get("cb_l2_perplexity", 0.0)
            
            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE: {avg_train_recon:.5f} | "
                f"Val MSE: {avg_val_recon:.5f} | "
                f"Active [L0:{l0_act}, L1:{l1_act}, L2:{l2_act}] | "
                f"Perplexity [L0:{l0_perp:.1f}, L1:{l1_perp:.1f}, L2:{l2_perp:.1f}] | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Checkpoint best model on validation reconstruction MSE
            if avg_val_recon < self.best_val_loss:
                self.best_val_loss = avg_val_recon
                self.patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_recon_mse": avg_val_recon,
                    "model_config": {
                        "input_dim": self.model.input_dim,
                        "latent_dim": self.model.latent_dim,
                        "codebook_size": self.model.codebook_size,
                        "num_levels": self.model.num_levels
                    }
                }, self.best_checkpoint_path)
                print(f"    --> Saved best checkpoint: Val MSE = {avg_val_recon:.6f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\n[!] Early stopping triggered at Epoch {epoch} (No improvement for {self.patience} epochs).")
                    break

        history_df = pd.DataFrame(self.history)
        return history_df

