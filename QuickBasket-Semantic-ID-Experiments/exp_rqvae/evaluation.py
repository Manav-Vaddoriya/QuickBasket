"""
Unified Evaluation and Diagnostics Engine for RQ-VAE Experiments.

Computes:
1. Reconstruction Quality: MSE, RMSE, MAE, Mean/Median/P10/P90 Cosine Sim, Relative Error
2. Codebook Health: Active/Dead Codes, Perplexity, Gini Inequality, Top-1/Top-5 shares
3. Collision Metrics: Uniqueness %, Collision %, Cluster-Size Bins (1, 2, 3, 4, 5, 6-10, 11-20, >20)
4. Category Purity: Overall Purity, Collision Cluster Purity, Cross-Category Overlaps
5. Prefix Progression: Branching factors across quantization levels
"""

import math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def compute_reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Computes exact reconstruction error metrics between original and reconstructed embeddings."""
    assert original.shape == reconstructed.shape, "Shape mismatch in reconstruction evaluation!"
    diff = original - reconstructed
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    orig_norm = np.linalg.norm(original, axis=-1, keepdims=True)
    recon_norm = np.linalg.norm(reconstructed, axis=-1, keepdims=True)
    dot_prod = np.sum(original * reconstructed, axis=-1, keepdims=True)
    cosine_sims = dot_prod / (orig_norm * recon_norm + 1e-10)

    mean_cosine = float(np.mean(cosine_sims))
    median_cosine = float(np.median(cosine_sims))
    min_cosine = float(np.min(cosine_sims))
    p10_cosine = float(np.percentile(cosine_sims, 10))
    p90_cosine = float(np.percentile(cosine_sims, 90))
    rel_error = np.linalg.norm(diff, axis=-1) / (orig_norm.squeeze(-1) + 1e-10)

    return {
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "cosine_similarity_mean": round(mean_cosine, 4),
        "cosine_similarity_median": round(median_cosine, 4),
        "cosine_similarity_min": round(min_cosine, 4),
        "cosine_similarity_p10": round(p10_cosine, 4),
        "cosine_similarity_p90": round(p90_cosine, 4),
        "relative_error_mean": round(float(np.mean(rel_error)), 4)
    }


def compute_codebook_diagnostics(codes_array: np.ndarray, codebook_size: int) -> tuple[pd.DataFrame, list[dict]]:
    """Calculates active codes, perplexity, and Gini inequality independently for each level."""
    num_samples, num_levels = codes_array.shape
    diagnostics = []

    for level in range(num_levels):
        lvl_codes = codes_array[:, level]
        counts = np.bincount(lvl_codes, minlength=codebook_size)
        
        active_codes = int(np.sum(counts > 0))
        dead_codes = int(codebook_size - active_codes)
        utilization_pct = round((active_codes / codebook_size) * 100.0, 2)
        
        probs = counts / num_samples
        nonzero_p = probs[probs > 0]
        entropy = -float(np.sum(nonzero_p * np.log(nonzero_p)))
        perplexity = round(math.exp(entropy), 2)
        perplexity_ratio = round(perplexity / codebook_size, 4)
        
        sorted_counts = np.sort(counts)
        n = len(counts)
        idx = np.arange(1, n + 1)
        gini = float((np.sum((2 * idx - n - 1) * sorted_counts)) / (n * np.sum(sorted_counts)))
        
        top1_share = round(float(np.max(counts) / num_samples * 100.0), 2)
        top5_share = round(float(np.sum(sorted_counts[-5:]) / num_samples * 100.0), 2)
        
        diagnostics.append({
            "level": level,
            "codebook_size": codebook_size,
            "active_codes": active_codes,
            "dead_codes": dead_codes,
            "utilization_pct": utilization_pct,
            "perplexity": perplexity,
            "perplexity_ratio": perplexity_ratio,
            "gini_inequality": round(gini, 4),
            "top1_code_share_pct": top1_share,
            "top5_codes_share_pct": top5_share,
            "min_frequency": int(np.min(counts[counts > 0])) if active_codes > 0 else 0,
            "max_frequency": int(np.max(counts)),
            "median_frequency": float(np.median(counts[counts > 0])) if active_codes > 0 else 0.0
        })

    summary_df = pd.DataFrame(diagnostics)
    return summary_df, diagnostics


def compute_collision_and_cluster_metrics(codes_array: np.ndarray, categories: list[str]) -> tuple[dict, pd.DataFrame]:
    """Calculates cluster-size distributions, category purity, and collision rates."""
    total_products, num_levels = codes_array.shape
    id_strings = ["_".join(map(str, row)) for row in codes_array]
    
    id_counts = Counter(id_strings)
    unique_ids = len(id_counts)
    collision_count = total_products - unique_ids
    collision_pct = round((collision_count / total_products) * 100.0, 2)
    uniqueness_pct = round((unique_ids / total_products) * 100.0, 2)
    
    cluster_sizes = list(id_counts.values())
    max_cluster_size = max(cluster_sizes)
    mean_cluster_size = round(float(np.mean(cluster_sizes)), 2)
    median_cluster_size = float(np.median(cluster_sizes))
    singletons = sum(1 for sz in cluster_sizes if sz == 1)
    singleton_pct = round((singletons / unique_ids) * 100.0, 2)
    
    bins = {
        "1 product (singleton)": sum(1 for sz in cluster_sizes if sz == 1),
        "2 products": sum(1 for sz in cluster_sizes if sz == 2),
        "3 products": sum(1 for sz in cluster_sizes if sz == 3),
        "4 products": sum(1 for sz in cluster_sizes if sz == 4),
        "5 products": sum(1 for sz in cluster_sizes if sz == 5),
        "6-10 products": sum(1 for sz in cluster_sizes if 6 <= sz <= 10),
        "11-20 products": sum(1 for sz in cluster_sizes if 11 <= sz <= 20),
        ">20 products": sum(1 for sz in cluster_sizes if sz > 20)
    }

    # Category Purity
    id_to_cats = defaultdict(set)
    for sid, cat in zip(id_strings, categories):
        id_to_cats[sid].add(cat)
        
    pure_ids_count = sum(1 for sid, cats in id_to_cats.items() if len(cats) == 1)
    overall_purity_pct = round((pure_ids_count / unique_ids) * 100.0, 2)
    
    shared_ids = {sid: cats for sid, cats in id_to_cats.items() if id_counts[sid] > 1}
    pure_shared_count = sum(1 for sid, cats in shared_ids.items() if len(cats) == 1)
    collision_purity_pct = round((pure_shared_count / len(shared_ids)) * 100.0, 2) if len(shared_ids) > 0 else 100.0
    cross_cat_count = len(shared_ids) - pure_shared_count

    # Prefix progression
    prefix_metrics = {}
    prev_unique = 1
    for lvl in range(num_levels):
        prefix_tuples = [tuple(row[:lvl + 1]) for row in codes_array]
        prefix_unique = len(set(prefix_tuples))
        branching = round(prefix_unique / prev_unique, 2) if lvl > 0 else prefix_unique
        prev_unique = prefix_unique
        prefix_metrics[f"level_0_to_{lvl}"] = {
            "unique_prefixes": prefix_unique,
            "branching_factor": branching
        }

    metrics = {
        "total_products": total_products,
        "unique_semantic_ids": unique_ids,
        "uniqueness_percentage": uniqueness_pct,
        "collision_count": collision_count,
        "collision_percentage": collision_pct,
        "singleton_ids_count": singletons,
        "singleton_ids_percentage": singleton_pct,
        "shared_ids_count": len(shared_ids),
        "mean_products_per_id": mean_cluster_size,
        "median_products_per_id": median_cluster_size,
        "max_products_per_id": max_cluster_size,
        "overall_category_purity_pct": overall_purity_pct,
        "collision_cluster_purity_pct": collision_purity_pct,
        "cross_category_shared_ids_count": cross_cat_count,
        "cluster_size_distribution": bins,
        "prefix_progression": prefix_metrics
    }

    # Bins DataFrame
    bin_rows = []
    for b_name, count in bins.items():
        bin_rows.append({
            "cluster_size_bin": b_name,
            "unique_semantic_ids_count": count,
            "percentage_of_all_ids": round((count / unique_ids) * 100.0, 2)
        })
    bins_df = pd.DataFrame(bin_rows)

    return metrics, bins_df


def evaluate_model_full(
    model: torch.nn.Module,
    embeddings: np.ndarray,
    catalogue_df: pd.DataFrame,
    batch_size: int = 512,
    device: str = "cpu"
) -> tuple[dict, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Runs batch inference and computes the full suite of metrics for any model."""
    model.eval()
    model.to(device)
    
    loader = DataLoader(
        TensorDataset(torch.tensor(embeddings, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False
    )
    
    all_recons = []
    all_codes = []
    
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            x_hat, _, codes, _, _ = model(batch_x)
            all_recons.append(x_hat.cpu().numpy())
            stacked_codes = torch.stack(codes, dim=1).cpu().numpy()
            all_codes.append(stacked_codes)
            
    recons_array = np.concatenate(all_recons, axis=0)
    codes_array = np.concatenate(all_codes, axis=0)
    
    recon_metrics = compute_reconstruction_metrics(embeddings, recons_array)
    cb_df, cb_diagnostics = compute_codebook_diagnostics(codes_array, codebook_size=model.codebook_size)
    collision_metrics, bins_df = compute_collision_and_cluster_metrics(codes_array, catalogue_df['category'].tolist())
    
    full_eval = {
        "reconstruction": recon_metrics,
        "codebook_diagnostics": cb_diagnostics,
        "collision_analysis": collision_metrics
    }
    
    return full_eval, cb_df, bins_df, codes_array, recons_array

