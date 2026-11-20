"""
RQ-VAE Training Runner Script.

Executes Stage 5:
- 90:10 reproducible train/val split (seed=42).
- Mini-batch Adam optimization with gradient clipping.
- Codebook collapse monitoring (active codes, perplexity per level).
- Checkpoint persistence and training history export.
"""

import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

# Ensure local imports work cleanly
sys.path.append(os.path.abspath("QuickBasket-Semantic-IDs"))

from rqvae.model import RQVAE
from rqvae.train import RQVAETrainer, set_seed


def run_training_experiment(
    embeddings_path: str = "QuickBasket-Semantic-IDs/embeddings/product_embeddings.npy",
    output_dir: str = "QuickBasket-Semantic-IDs",
    batch_size: int = 256,
    learning_rate: float = 0.001,
    commitment_beta: float = 0.25,
    max_epochs: int = 150,
    patience: int = 15,
    grad_clip: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    print("=" * 70)
    print("STAGE 5: RQ-VAE MODEL TRAINING & STABILITY MONITORING")
    print("=" * 70)

    # 1. Reproducibility & Device Setup
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1] Environment & Hardware:")
    print(f"    - Device: {device.upper()}")
    print(f"    - Random Seed: {seed}")

    # 2. Ingestion & Split
    assert os.path.exists(embeddings_path), f"Embeddings not found at {embeddings_path}"
    embeddings = np.load(embeddings_path)
    total_samples, input_dim = embeddings.shape
    print(f"\n[2] Ingestion & Train-Val Partitioning:")
    print(f"    - Total Catalogue Embeddings: {total_samples:,} (dim={input_dim})")
    assert total_samples == 50015, f"Expected 50,015 records, got {total_samples}"

    tensor_embs = torch.tensor(embeddings, dtype=torch.float32)
    dataset = TensorDataset(tensor_embs)
    
    val_size = int(round(total_samples * 0.10))
    train_size = total_samples - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    print(f"    - 90% Training Set:   {len(train_dataset):,} items ({len(train_dataset)/total_samples*100:.1f}%)")
    print(f"    - 10% Validation Set: {len(val_dataset):,} items ({len(val_dataset)/total_samples*100:.1f}%)")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    print(f"    - Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # 3. Model Initialization
    latent_dim = 64
    codebook_size = 128
    num_levels = 3

    model = RQVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        codebook_size=codebook_size,
        num_levels=num_levels
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[3] Model Specifications:")
    print(f"    - Architecture: Input({input_dim}) -> 512 -> 256 -> Latent({latent_dim}) -> 3 Levels x {codebook_size} -> Output({input_dim})")
    print(f"    - Total Trainable Parameters: {total_params:,}")
    print(f"    - Hyperparameters: LR={learning_rate}, Beta={commitment_beta}, GradClip={grad_clip}, Patience={patience}")

    # 4. Trainer
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    outputs_dir = os.path.join(output_dir, "outputs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    trainer = RQVAETrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        commitment_beta=commitment_beta,
        grad_clip=grad_clip,
        checkpoint_dir=checkpoints_dir,
        patience=patience
    )

    # 5. Fit Model
    history_df = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=max_epochs
    )

    # 6. Save Training History
    history_path = os.path.join(outputs_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"\n[+] Saved Training History: {history_path} ({len(history_df)} epochs recorded)")

    # 7. Summary Training Report
    best_epoch_idx = history_df["val_recon_mse"].idxmin()
    best_record = history_df.loc[best_epoch_idx]
    
    print("\n" + "=" * 70)
    print("TRAINING STABILITY & CONVERGENCE SUMMARY:")
    print(f"    - Total Epochs Completed: {len(history_df)}")
    print(f"    - Best Validation Epoch: {int(best_record['epoch'])}")
    print(f"    - Initial Val Recon MSE: {history_df['val_recon_mse'].iloc[0]:.6f}")
    print(f"    - Best Val Recon MSE:    {best_record['val_recon_mse']:.6f} (Reduced by {((history_df['val_recon_mse'].iloc[0] - best_record['val_recon_mse'])/history_df['val_recon_mse'].iloc[0]*100):.1f}%)")
    print(f"    - Final Train Recon MSE: {best_record['train_recon_mse']:.6f}")
    print(f"    - Final Total Loss:      {best_record['val_total_loss']:.6f}")
    print("\nCODEBOOK UTILIZATION AT BEST CHECKPOINT:")
    for lvl in range(num_levels):
        act = int(best_record[f"cb_l{lvl}_active"])
        util = best_record[f"cb_l{lvl}_utilization_pct"]
        perp = best_record[f"cb_l{lvl}_perplexity"]
        print(f"    - Level {lvl}: Active {act}/{codebook_size} ({util}%) | Perplexity: {perp:.2f} (Perplexity Ratio: {perp/codebook_size:.2f})")

    print("\n" + "=" * 70)
    print("STAGE 5 TRAINING COMPLETE AND VALIDATED!")
    print("=" * 70)

    return history_df


if __name__ == "__main__":
    run_training_experiment()

