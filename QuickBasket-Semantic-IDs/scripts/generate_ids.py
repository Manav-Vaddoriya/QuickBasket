"""
Stage 6 & 8: Semantic ID Generation & Pipeline Export Script.

Loads the best trained RQ-VAE checkpoint, batches through all 50,015 product embeddings,
extracts discrete code indices across all 3 quantization levels, and exports
the finalized Semantic ID catalogue in CSV and Parquet formats.
"""

import os
import sys
import json

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from rqvae.model import RQVAE
from evaluation.reconstruction import evaluate_reconstruction_quality
from evaluation.codebook_metrics import compute_codebook_diagnostics
from evaluation.collision_analysis import analyze_collisions
from evaluation.category_analysis import analyze_category_semantic_ids


def generate_all_semantic_ids(
    embeddings_path: str = "QuickBasket-Semantic-IDs/embeddings/product_embeddings.npy",
    catalogue_path: str = "QuickBasket-Semantic-IDs/data/cleaned_catalogue.csv",
    checkpoint_path: str = "QuickBasket-Semantic-IDs/checkpoints/rqvae_best.pt",
    output_dir: str = "QuickBasket-Semantic-IDs/outputs",
    batch_size: int = 512,
    device: str = "cpu"
) -> tuple[pd.DataFrame, dict]:
    """
    Generates 3-level discrete Semantic IDs for all 50,015 products.
    """
    print("=" * 70)
    print("STAGE 6 & 7: SEMANTIC ID GENERATION AND EVALUATION")
    print("=" * 70)

    # 1. Load Catalogue and Embeddings
    assert os.path.exists(catalogue_path), f"Catalogue file missing: {catalogue_path}"
    assert os.path.exists(embeddings_path), f"Embeddings array missing: {embeddings_path}"
    assert os.path.exists(checkpoint_path), f"Checkpoint missing: {checkpoint_path}"

    df = pd.read_csv(catalogue_path)
    embeddings = np.load(embeddings_path)
    total_records, emb_dim = embeddings.shape
    assert len(df) == total_records, f"Row count mismatch: df={len(df)}, emb={total_records}"
    print(f"Loaded {total_records:,} product embeddings (dim={emb_dim}) and catalogue records.")

    # 2. Load RQ-VAE Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("model_config", {
        "input_dim": emb_dim,
        "latent_dim": 64,
        "codebook_size": 128,
        "num_levels": 3
    })
    
    model = RQVAE(
        input_dim=cfg["input_dim"],
        latent_dim=cfg["latent_dim"],
        codebook_size=cfg["codebook_size"],
        num_levels=cfg["num_levels"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded best RQ-VAE checkpoint (trained at epoch {checkpoint.get('epoch', 'N/A')}, Val MSE = {checkpoint.get('val_recon_mse', 'N/A'):.6f}).")

    # 3. Batch Inference for Semantic IDs & Reconstructed Embeddings
    tensor_data = torch.tensor(embeddings, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=False)

    all_codes = []
    all_reconstructions = []

    print(f"Encoding catalogue in batches of {batch_size} on {device.upper()}...")
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            # Reconstruct and get codes
            x_hat, _, codes, _, _ = model(batch_x)
            
            # Stack codes: [batch, num_levels]
            stacked_codes = torch.stack(codes, dim=1).cpu().numpy()
            all_codes.append(stacked_codes)
            all_reconstructions.append(x_hat.cpu().numpy())

    codes_array = np.concatenate(all_codes, axis=0)  # [50015, 3]
    reconstructions_array = np.concatenate(all_reconstructions, axis=0)  # [50015, 384]
    assert codes_array.shape == (50015, 3), f"Unexpected codes shape: {codes_array.shape}"
    assert reconstructions_array.shape == embeddings.shape, "Reconstruction shape mismatch!"

    # 4. Assemble Final Semantic ID Catalogue
    out_df = df[["product_id", "item", "category", "item_cleaned"]].copy()
    out_df["code_0"] = codes_array[:, 0]
    out_df["code_1"] = codes_array[:, 1]
    out_df["code_2"] = codes_array[:, 2]
    out_df["semantic_id"] = [
        f"{c0}_{c1}_{c2}" for c0, c1, c2 in zip(out_df["code_0"], out_df["code_1"], out_df["code_2"])
    ]

    print(f"\nSemantic IDs generated successfully. Sample:")
    print(out_df[["product_id", "item", "category", "code_0", "code_1", "code_2", "semantic_id"]].head(5))

    # 5. Run Stage 7 Comprehensive Evaluations
    print("\n" + "=" * 70)
    print("STAGE 7: COMPREHENSIVE EVALUATIONS")
    print("=" * 70)

    # A. Reconstruction Quality
    recon_metrics = evaluate_reconstruction_quality(embeddings, reconstructions_array)
    print(f"\n[A] Reconstruction Quality:")
    print(f"    - MSE: {recon_metrics['mse']:.6f} | RMSE: {recon_metrics['rmse']:.6f}")
    print(f"    - Cosine Similarity (Mean): {recon_metrics['cosine_similarity_mean']:.4f}")
    print(f"    - Cosine Similarity (Median): {recon_metrics['cosine_similarity_median']:.4f}")

    # B. Codebook Utilization & Collapse Diagnostics
    cb_df, cb_diagnostics = compute_codebook_diagnostics(codes_array, codebook_size=cfg["codebook_size"])
    print(f"\n[B] Codebook Utilization & Perplexity:")
    for d in cb_diagnostics:
        print(f"    - Level {d['level']}: Active {d['active_codes']}/{d['codebook_size']} ({d['utilization_pct']}%) | Perplexity: {d['perplexity']:.1f} | Top-1 Share: {d['top1_code_share_pct']}%")

    # C. Collision Analysis
    collision_metrics = analyze_collisions(codes_array)
    print(f"\n[C] Collision & Uniqueness Metrics:")
    print(f"    - Total Products: {collision_metrics['total_products']:,}")
    print(f"    - Unique Semantic IDs: {collision_metrics['unique_semantic_ids']:,} ({collision_metrics['uniqueness_percentage']}%)")
    print(f"    - Collision Count: {collision_metrics['collision_count']:,} ({collision_metrics['collision_percentage']}%)")
    print(f"    - Mean Products per Semantic ID: {collision_metrics['mean_products_per_id']}")

    # D. Category Distribution Analysis
    cat_report_df, cross_cat_summary = analyze_category_semantic_ids(out_df, codes_array)
    print(f"\n[D] Category Distribution & Cross-Category Overlap:")
    print(f"    - Cross-Category Shared IDs: {cross_cat_summary['cross_category_shared_ids']} ({cross_cat_summary['cross_category_shared_pct']}%)")
    print(f"    - Single-Category Pure IDs: {cross_cat_summary['single_category_pure_ids']}")

    # 6. Save All Final Artifacts (Stage 8)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. quickbasket_semantic_ids.csv
    csv_path = os.path.join(output_dir, "quickbasket_semantic_ids.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved Final CSV: {csv_path} ({os.path.getsize(csv_path):,} bytes)")

    # 2. quickbasket_semantic_ids.parquet
    parquet_path = os.path.join(output_dir, "quickbasket_semantic_ids.parquet")
    out_df.to_parquet(parquet_path, index=False)
    print(f"[+] Saved Final Parquet: {parquet_path} ({os.path.getsize(parquet_path):,} bytes)")

    # 3. codebook_utilization.csv
    cb_path = os.path.join(output_dir, "codebook_utilization.csv")
    cb_df.to_csv(cb_path, index=False)
    print(f"[+] Saved Codebook Utilization Report: {cb_path}")

    # 4. category_semantic_id_report.csv
    cat_path = os.path.join(output_dir, "category_semantic_id_report.csv")
    cat_report_df.to_csv(cat_path, index=False)
    print(f"[+] Saved Category Semantic ID Report: {cat_path}")

    # 5. semantic_id_evaluation.json
    eval_summary = {
        "reconstruction_quality": recon_metrics,
        "codebook_diagnostics": cb_diagnostics,
        "collision_analysis": collision_metrics,
        "cross_category_analysis": cross_cat_summary
    }
    eval_json_path = os.path.join(output_dir, "semantic_id_evaluation.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"[+] Saved Comprehensive Evaluation JSON: {eval_json_path}")

    return out_df, eval_summary


if __name__ == "__main__":
    generate_all_semantic_ids()

