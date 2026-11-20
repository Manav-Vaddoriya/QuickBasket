"""
Experiment E2 (Variant B): K-Means Codebook Initialization using Trained Encoder Representations.

Implementation:
1. Extract latent vectors z_train from the trained baseline encoder on 45,013 training items.
2. Fit PyTorch K-Means (k=128) on z_train to initialize Codebook Level 0.
3. Compute residuals r1 = z_train - q0.
4. Fit PyTorch K-Means (k=128) on r1 to initialize Codebook Level 1.
5. Compute residuals r2 = r1 - q1.
6. Fit PyTorch K-Means (k=128) on r2 to initialize Codebook Level 2.
7. Initialize RQ-VAE with trained encoder/decoder and K-Means codebooks.
8. Train with fine-tuning learning rate (0.0002) to allow codebooks and encoder to co-adapt without collapsing.
9. Evaluate and compare against Baseline (E1) and Cold K-Means (E2-Cold).
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from exp_rqvae.model import RQVAE
from exp_rqvae.trainer import RQVAETrainer, set_seed
from exp_rqvae.evaluation import evaluate_model_full, compute_reconstruction_metrics


def run_experiment_e2_warm(
    orig_emb_path: str = "QuickBasket-Semantic-IDs/embeddings/product_embeddings.npy",
    orig_cat_path: str = "QuickBasket-Semantic-IDs/data/cleaned_catalogue.csv",
    baseline_ckpt: str = "QuickBasket-Semantic-ID-Experiments/checkpoints/baseline_rqvae.pt",
    output_base: str = "QuickBasket-Semantic-ID-Experiments"
):
    print("=" * 80)
    print("EXPERIMENT E2 (WARM ENCODER): K-MEANS CODEBOOK INITIALIZATION")
    print("=" * 80)

    set_seed(42)
    device = "cpu"

    # 1. Load Data
    embeddings = np.load(orig_emb_path)
    df = pd.read_csv(orig_cat_path)
    total_records, emb_dim = embeddings.shape

    indices = np.arange(total_records)
    train_idx, val_idx = train_test_split(indices, test_size=0.10, random_state=42)
    
    train_tensor = torch.tensor(embeddings[train_idx], dtype=torch.float32)
    val_tensor = torch.tensor(embeddings[val_idx], dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=256, shuffle=False)
    print(f"Loaded {total_records:,} items. Train: {len(train_idx):,} | Val: {len(val_idx):,}")

    # 2. Load Trained Baseline Encoder & Decoder
    ckpt = torch.load(baseline_ckpt, map_location=device)
    model = RQVAE(input_dim=384, latent_dim=64, codebook_size=128, num_levels=3)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)

    # 3. Extract Latent Vectors on Training Split
    model.eval()
    with torch.no_grad():
        z_train = model.encode(train_tensor.to(device))
    print(f"Extracted z_train: shape={z_train.shape}, norm={torch.norm(z_train, dim=-1).mean():.4f}")

    # 4. Sequential PyTorch K-Means on Residuals
    print("\n[Step 1] Initializing Codebook 0 via K-Means on trained z_train...")
    model.quantizer.init_codebooks_kmeans(level=0, residual_vectors=z_train, num_iters=30, random_state=42)

    with torch.no_grad():
        c0 = model.quantizer.codebooks[0]
        dists0 = (z_train**2).sum(-1, keepdim=True) - 2.0 * torch.matmul(z_train, c0.t()) + (c0**2).sum(-1, keepdim=True).t()
        idx0 = torch.argmin(dists0, dim=-1)
        q0 = torch.nn.functional.embedding(idx0, c0)
        r1_train = z_train - q0
        print(f"  Level 0 Active immediately after KMeans: {len(torch.unique(idx0))}/128 | Residual 1 norm: {torch.norm(r1_train, dim=-1).mean():.4f}")

    print("\n[Step 2] Initializing Codebook 1 via K-Means on Residual 1...")
    model.quantizer.init_codebooks_kmeans(level=1, residual_vectors=r1_train, num_iters=30, random_state=42)

    with torch.no_grad():
        c1 = model.quantizer.codebooks[1]
        dists1 = (r1_train**2).sum(-1, keepdim=True) - 2.0 * torch.matmul(r1_train, c1.t()) + (c1**2).sum(-1, keepdim=True).t()
        idx1 = torch.argmin(dists1, dim=-1)
        q1 = torch.nn.functional.embedding(idx1, c1)
        r2_train = r1_train - q1
        print(f"  Level 1 Active immediately after KMeans: {len(torch.unique(idx1))}/128 | Residual 2 norm: {torch.norm(r2_train, dim=-1).mean():.4f}")

    print("\n[Step 3] Initializing Codebook 2 via K-Means on Residual 2...")
    model.quantizer.init_codebooks_kmeans(level=2, residual_vectors=r2_train, num_iters=30, random_state=42)

    with torch.no_grad():
        c2 = model.quantizer.codebooks[2]
        dists2 = (r2_train**2).sum(-1, keepdim=True) - 2.0 * torch.matmul(r2_train, c2.t()) + (c2**2).sum(-1, keepdim=True).t()
        idx2 = torch.argmin(dists2, dim=-1)
        print(f"  Level 2 Active immediately after KMeans: {len(torch.unique(idx2))}/128")

    # 5. Train with fine-tuning learning rate (0.0002)
    exp_dir = os.path.join(output_base, "experiments", "kmeans_initialization")
    ckpt_path = os.path.join(output_base, "checkpoints", "kmeans_warm_rqvae.pt")

    t_start = time.time()
    trainer = RQVAETrainer(
        model=model,
        device=device,
        learning_rate=0.0002,   # Stable fine-tuning LR so codebooks adapt without collapse
        commitment_beta=0.25,
        grad_clip=1.0,
        checkpoint_path=ckpt_path,
        patience=15
    )

    history_df = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=50,
        verbose=True
    )
    total_train_time = (time.time() - t_start) / 60.0

    # Save history
    hist_csv = os.path.join(exp_dir, "training_history_warm.csv")
    history_df.to_csv(hist_csv, index=False)
    print(f"[+] Saved training history: {hist_csv}")

    # 6. Evaluate Best Checkpoint
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        val_recons = []
        for (bx,) in val_loader:
            x_hat, _, _, _, _ = model(bx.to(device))
            val_recons.append(x_hat.cpu().numpy())
        val_recons = np.concatenate(val_recons, axis=0)
        val_recon_metrics = compute_reconstruction_metrics(embeddings[val_idx], val_recons)

    full_eval, cb_df, bins_df, codes_array, recons_array = evaluate_model_full(
        model=model,
        embeddings=embeddings,
        catalogue_df=df,
        batch_size=512,
        device=device
    )
    full_eval["best_epoch"] = best_ckpt["epoch"]
    full_eval["val_recon_mse"] = val_recon_metrics["mse"]
    full_eval["train_time_min"] = round(total_train_time, 2)

    # Save metrics JSON
    metrics_json_path = os.path.join(exp_dir, "metrics_warm.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(full_eval, f, indent=2)
    print(f"[+] Saved E2-Warm metrics: {metrics_json_path}")

    # Save Semantic IDs Parquet
    out_df = df[["product_id", "item", "category", "item_cleaned"]].copy()
    out_df["code_0"] = codes_array[:, 0]
    out_df["code_1"] = codes_array[:, 1]
    out_df["code_2"] = codes_array[:, 2]
    out_df["semantic_id"] = [f"{c0}_{c1}_{c2}" for c0, c1, c2 in zip(out_df["code_0"], out_df["code_1"], out_df["code_2"])]
    parquet_path = os.path.join(exp_dir, "semantic_ids_kmeans_warm.parquet")
    out_df.to_parquet(parquet_path, index=False)
    print(f"[+] Saved E2-Warm Semantic IDs Parquet: {parquet_path}")

    # Comparison summary
    print("\n" + "=" * 80)
    print("EXPERIMENT E2 (WARM K-MEANS) RESULTS SUMMARY")
    print("=" * 80)
    print(f"Validation Recon MSE:       {val_recon_metrics['mse']:.6f}")
    print(f"Mean Cosine Similarity:     {full_eval['reconstruction']['cosine_similarity_mean']:.4f}")
    print(f"Median Cosine Similarity:   {full_eval['reconstruction']['cosine_similarity_median']:.4f}")
    print(f"Active Codes Level 0:       {full_eval['codebook_diagnostics'][0]['active_codes']}/128 (Perp: {full_eval['codebook_diagnostics'][0]['perplexity']:.2f})")
    print(f"Active Codes Level 1:       {full_eval['codebook_diagnostics'][1]['active_codes']}/128 (Perp: {full_eval['codebook_diagnostics'][1]['perplexity']:.2f})")
    print(f"Active Codes Level 2:       {full_eval['codebook_diagnostics'][2]['active_codes']}/128 (Perp: {full_eval['codebook_diagnostics'][2]['perplexity']:.2f})")
    print(f"Total Unique Semantic IDs:  {full_eval['collision_analysis']['unique_semantic_ids']:,} ({full_eval['collision_analysis']['uniqueness_percentage']}%)")
    print(f"Collision Percentage:       {full_eval['collision_analysis']['collision_percentage']}%")
    print(f"Category Purity:            {full_eval['collision_analysis']['overall_category_purity_pct']}%")

    return full_eval


if __name__ == "__main__":
    run_experiment_e2_warm()

