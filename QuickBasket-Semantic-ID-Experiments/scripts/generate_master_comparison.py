"""
Master Comparison & Best Model Export
Compiles outputs/experiment_comparison.csv across all experimental configurations
and exports outputs/semantic_ids_best.parquet with disambiguation identifiers.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)


def build_master_comparison(exp_dir="QuickBasket-Semantic-ID-Experiments"):
    configs = [
        ("E1_Baseline (3L x 128, beta=0.25)", os.path.join(exp_dir, "baseline", "metrics.json")),
        ("E2_Cold_KMeans (3L x 128)", os.path.join(exp_dir, "experiments", "kmeans_initialization", "metrics.json")),
        ("E2_Warm_KMeans (3L x 128)", os.path.join(exp_dir, "experiments", "kmeans_initialization", "metrics_warm.json")),
        ("E3_Dead_Code_Reset (3L x 128)", os.path.join(exp_dir, "experiments", "dead_code_reinitialization", "metrics.json")),
        ("E4_Four_Level (4L x 128)", os.path.join(exp_dir, "experiments", "four_level_rqvae", "metrics.json")),
        ("E5A_Codebook_3Lx64", os.path.join(exp_dir, "experiments", "codebook_comparison", "e5a_3l_64", "metrics.json")),
        ("E5C_Codebook_4Lx64", os.path.join(exp_dir, "experiments", "codebook_comparison", "e5c_4l_64", "metrics.json")),
        ("E6_Beta_0.10", os.path.join(exp_dir, "experiments", "commitment_tuning", "beta_0_1", "metrics.json")),
        ("E6_Beta_0.50", os.path.join(exp_dir, "experiments", "commitment_tuning", "beta_0_5", "metrics.json")),
        ("E6_Beta_1.00", os.path.join(exp_dir, "experiments", "commitment_tuning", "beta_1_0", "metrics.json")),
    ]

    rows = []
    for exp_label, mpath in configs:
        if not os.path.exists(mpath):
            continue
        with open(mpath, "r") as f:
            m = json.load(f)

        recon = m.get("reconstruction", {})
        coll = m.get("collision_analysis", {})
        cb = m.get("codebook_diagnostics", [])

        levels = len(cb)
        cb_size = cb[0]["codebook_size"] if cb else 128
        total_codes = levels * cb_size
        total_active = sum(c["active_codes"] for c in cb)
        active_str = "/".join(str(c["active_codes"]) for c in cb)
        ppl_str = "/".join(f"{c['perplexity']:.1f}" for c in cb)

        rows.append({
            "Experiment": exp_label,
            "Levels": levels,
            "Codebook_Size": cb_size,
            "Theoretical_Capacity": cb_size ** levels,
            "Val_Recon_MSE": m.get("val_recon_mse", recon.get("mse")),
            "Cosine_Mean": recon.get("cosine_similarity_mean"),
            "Cosine_Median": recon.get("cosine_similarity_median"),
            "Cosine_P10": recon.get("cosine_similarity_p10"),
            "Cosine_P90": recon.get("cosine_similarity_p90"),
            "Unique_IDs": coll.get("unique_semantic_ids"),
            "Uniqueness_Pct": coll.get("uniqueness_percentage"),
            "Singletons": coll.get("singleton_ids_count"),
            "Collision_Rate_Pct": coll.get("collision_percentage"),
            "Max_Cluster_Size": coll.get("max_products_per_id"),
            "Category_Purity_Pct": coll.get("overall_category_purity_pct"),
            "Active_Codes": active_str,
            "Total_Active_Pct": f"{total_active}/{total_codes} ({total_active/total_codes*100:.1f}%)" if total_codes else "N/A",
            "Perplexity": ppl_str,
            "Best_Epoch": m.get("best_epoch", "N/A"),
            "Train_Time_Min": m.get("train_time_min", "N/A")
        })

    master_df = pd.DataFrame(rows)
    out_csv = os.path.join(exp_dir, "outputs", "experiment_comparison.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    master_df.to_csv(out_csv, index=False)
    print("Master comparison saved to:", out_csv)
    print(master_df.to_string())
    return master_df


def generate_disambiguated_best_ids(exp_dir="QuickBasket-Semantic-ID-Experiments"):
    """
    Exports the best model Semantic IDs with disambiguation:
    - Primary Best 3-Level model: E2-Warm (highest fidelity, 100% code utilization)
    - Primary Best 4-Level model: E4 (maximum raw uniqueness)
    - Adds disambiguation suffixes:
      semantic_id_disambiguated = f"{semantic_id}_{rank_in_cluster}"
    """
    warm_parquet = os.path.join(exp_dir, "experiments", "kmeans_initialization", "semantic_ids_kmeans_warm.parquet")
    if not os.path.exists(warm_parquet):
        print(f"Warning: {warm_parquet} not found!")
        return

    df = pd.read_parquet(warm_parquet)
    # Assign deterministic disambiguation rank within each semantic_id cluster
    # Sort deterministically by product_id
    df = df.sort_values(by=["semantic_id", "product_id"]).reset_index(drop=True)
    df["disambiguation_index"] = df.groupby("semantic_id").cumcount()
    df["cluster_size"] = df.groupby("semantic_id")["product_id"].transform("count")
    df["is_singleton"] = df["cluster_size"] == 1

    # Full disambiguated tuple: e.g. 42_89_11_0
    df["semantic_id_disambiguated"] = df["semantic_id"] + "_" + df["disambiguation_index"].astype(str)

    out_parquet = os.path.join(exp_dir, "outputs", "semantic_ids_best.parquet")
    df.to_parquet(out_parquet, index=False)
    print("Exported best disambiguated semantic IDs to:", out_parquet)
    print(f"Total products: {len(df)}")
    print(f"Unique base Semantic IDs: {df['semantic_id'].nunique()}")
    print(f"Unique disambiguated Semantic IDs: {df['semantic_id_disambiguated'].nunique()} (100% uniqueness)")
    print(df[["product_id", "item", "category", "semantic_id", "disambiguation_index", "semantic_id_disambiguated"]].head(10))


if __name__ == "__main__":
    build_master_comparison()
    generate_disambiguated_best_ids()

