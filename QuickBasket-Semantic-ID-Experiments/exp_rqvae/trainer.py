"""
High-Performance Modular Trainer for RQ-VAE Experiments.

Supports:
- Clean gradient isolation
- Configurable early stopping and learning rate
- Optional dead-code reinitialization hook (Experiment E3)
- Detailed per-epoch history logging and best-checkpoint saving
"""

import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
        checkpoint_path: str = "QuickBasket-Semantic-ID-Experiments/checkpoints/experiment_rqvae.pt",
        patience: int = 15,
        dead_code_reset_every: int = 0,     # If > 0, reinitializes dead codes every N epochs (Experiment E3)
        dead_code_min_usage: int = 1
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.dead_code_reset_every = dead_code_reset_every
        self.dead_code_min_usage = dead_code_min_usage

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        self.criterion = RQVAELoss(beta=commitment_beta)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.history = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 150,
        verbose: bool = True
    ) -> pd.DataFrame:
        if verbose:
            print(f"\nStarting RQ-VAE Training on {self.device.upper()}...")
            print(f"Max Epochs: {max_epochs} | Early Stopping Patience: {self.patience}")
            print(f"Codebooks: {self.model.num_levels} levels x {self.model.codebook_size} codes | Latent: {self.model.latent_dim}d")
            print("-" * 80)

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()

            # --- Training Epoch ---
            self.model.train()
            train_recon_losses = []
            train_total_losses = []
            all_train_residuals = []

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
                train_total_losses.append(loss_dict["loss_total"])

                # Store sample residuals for dead-code reinitialization if enabled
                if self.dead_code_reset_every > 0:
                    with torch.no_grad():
                        all_train_residuals.append(residuals_list)

            # --- Dead-Code Reinitialization Check (Experiment E3) ---
            dead_code_reinit_counts = {}
            if self.dead_code_reset_every > 0 and (epoch % self.dead_code_reset_every == 0) and (epoch < max_epochs - 10):
                with torch.no_grad():
                    for lvl in range(self.model.num_levels):
                        # Gather residuals for this level across batches
                        lvl_resids = torch.cat([res_list[lvl] for res_list in all_train_residuals], dim=0)
                        num_reinit = self.model.quantizer.reinitialize_dead_codes(
                            level=lvl,
                            active_residuals=lvl_resids,
                            min_usage_threshold=self.dead_code_min_usage
                        )
                        dead_code_reinit_counts[lvl] = num_reinit
                        if verbose and num_reinit > 0:
                            print(f"  [Epoch {epoch}] Reinitialized {num_reinit} dead entries in Level {lvl} codebook.")
                # Reset counters for the next tracking window
                self.model.quantizer.reset_usage_counts()

            # --- Validation Epoch ---
            self.model.eval()
            val_recon_losses = []
            val_total_losses = []
            all_val_codes = [[] for _ in range(self.model.num_levels)]

            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(self.device)
                    x_hat, _, codes, quantized_list, residuals_list = self.model(batch_x)

                    loss, loss_dict = self.criterion(
                        x=batch_x,
                        x_hat=x_hat,
                        residuals_list=residuals_list,
                        quantized_list=quantized_list
                    )

                    val_recon_losses.append(loss_dict["loss_recon"])
                    val_total_losses.append(loss_dict["loss_total"])

                    for lvl, c in enumerate(codes):
                        all_val_codes[lvl].append(c)

            epoch_time = time.time() - t0
            mean_train_recon = float(np.mean(train_recon_losses))
            mean_train_total = float(np.mean(train_total_losses))
            mean_val_recon = float(np.mean(val_recon_losses))
            mean_val_total = float(np.mean(val_total_losses))

            # Codebook diagnostics on validation set
            stacked_val_codes = [torch.cat(all_val_codes[lvl]) for lvl in range(self.model.num_levels)]
            cb_metrics = self.model.quantizer.compute_codebook_metrics(stacked_val_codes)

            # Record epoch log
            row = {
                "epoch": epoch,
                "train_recon_mse": round(mean_train_recon, 6),
                "train_total_loss": round(mean_train_total, 6),
                "val_recon_mse": round(mean_val_recon, 6),
                "val_total_loss": round(mean_val_total, 6),
                "epoch_time_seconds": round(epoch_time, 2)
            }
            for m in cb_metrics:
                lvl = m["level"]
                row[f"cb_l{lvl}_active"] = m["active_codes"]
                row[f"cb_l{lvl}_utilization_pct"] = m["utilization_pct"]
                row[f"cb_l{lvl}_perplexity"] = m["perplexity"]

            self.history.append(row)

            # Display progress every 10 epochs or on best validation
            is_best = mean_val_recon < self.best_val_loss
            if is_best:
                self.best_val_loss = mean_val_recon
                self.patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_recon_mse": mean_val_recon,
                    "model_config": {
                        "input_dim": self.model.input_dim,
                        "latent_dim": self.model.latent_dim,
                        "codebook_size": self.model.codebook_size,
                        "num_levels": self.model.num_levels
                    }
                }, self.checkpoint_path)
            else:
                self.patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == 1 or is_best):
                status_mark = " [*BEST]" if is_best else ""
                cb_summary = " | ".join([f"L{m['level']}: {m['active_codes']}/{m['codebook_size']} (perp {m['perplexity']:.1f})" for m in cb_metrics])
                print(f"Epoch {epoch:03d}/{max_epochs:03d} [{epoch_time:.1f}s] | Train MSE: {mean_train_recon:.6f} | Val MSE: {mean_val_recon:.6f}{status_mark} | {cb_summary}")

            # Early Stopping Check
            if self.patience_counter >= self.patience:
                if verbose:
                    print(f"\n[!] Early stopping triggered at epoch {epoch}. No validation improvement for {self.patience} epochs.")
                break

        history_df = pd.DataFrame(self.history)
        if verbose:
            print(f"\nTraining Complete. Best Checkpoint saved -> {self.checkpoint_path} (Val MSE = {self.best_val_loss:.6f})")
        return history_df

