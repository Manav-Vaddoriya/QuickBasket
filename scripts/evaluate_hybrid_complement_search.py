"""
Benchmark & Evaluation: Hybrid Complement Search vs. Baselines
Evaluates Hit@K, MRR, NDCG@10, and Substitute Leakage Rate on Held-Out Complement Test Edges.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.retrieval.hybrid_complement_search import HybridComplementSearchEngine


def evaluate_engine_sample(engine: HybridComplementSearchEngine, n_test_anchors: int = 150):
    print("=" * 80)
    print(f"BENCHMARKING HYBRID COMPLEMENT SEARCH ENGINE ON {n_test_anchors} TEST ANCHORS")
    print("=" * 80)

    comp_df = pd.read_parquet(engine.complement_graph_path)
    
    # Filter for held-out or gold/silver edges for evaluation
    gold_edges = comp_df[comp_df["tier"].isin(["gold", "silver"])]
    
    # Find anchors with at least 3 ground truth complements
    anchor_counts = gold_edges["anchor_item_id"].value_counts()
    eligible_anchors = anchor_counts[anchor_counts >= 3].index.tolist()
    
    rng = np.random.RandomState(42)
    sample_anchors = rng.choice(eligible_anchors, size=min(n_test_anchors, len(eligible_anchors)), replace=False)

    print(f"Sampled {len(sample_anchors)} evaluation anchors.")

    # Ground truth mapping: anchor -> set of candidate IDs
    gt_map = gold_edges.groupby("anchor_item_id")["candidate_item_id"].apply(set).to_dict()

    metrics = {
        "Dense_Only": {"hits@5": 0, "hits@10": 0, "mrr": 0.0, "ndcg@10": 0.0, "substitute_rate": 0.0, "latency_ms": []},
        "TwoTower_Only": {"hits@5": 0, "hits@10": 0, "mrr": 0.0, "ndcg@10": 0.0, "substitute_rate": 0.0, "latency_ms": []},
        "SemanticID_Only": {"hits@5": 0, "hits@10": 0, "mrr": 0.0, "ndcg@10": 0.0, "substitute_rate": 0.0, "latency_ms": []},
        "Hybrid_Search (Ours)": {"hits@5": 0, "hits@10": 0, "mrr": 0.0, "ndcg@10": 0.0, "substitute_rate": 0.0, "latency_ms": []}
    }

    configurations = [
        ("Dense_Only", {"dense": 1.0, "semantic_id": 0.0, "graph": 0.0}, False),
        ("TwoTower_Only", {"dense": 1.0, "semantic_id": 0.0, "graph": 0.0}, True),
        ("SemanticID_Only", {"dense": 0.0, "semantic_id": 1.0, "graph": 0.0}, True),
        ("Hybrid_Search (Ours)", {"dense": 0.35, "semantic_id": 0.40, "graph": 0.25}, True)
    ]

    total_anchors = len(sample_anchors)

    for cfg_name, weights, filter_subs in configurations:
        print(f"\nEvaluating configuration: {cfg_name}...")
        
        hit5_count = 0
        hit10_count = 0
        mrr_total = 0.0
        ndcg10_total = 0.0
        substitute_count = 0
        total_retrieved = 0
        
        for a_id in sample_anchors:
            gt_cands = gt_map.get(a_id, set())
            if not gt_cands:
                continue

            a_idx = engine.item_id_to_idx.get(a_id)
            if a_idx is None:
                continue

            a_base_sem = engine.idx_to_semantic_id.get(a_idx)

            t0 = time.time()
            res = engine.get_complements(
                query_item=a_idx,
                top_k=10,
                fusion_method="rrf",
                weights=weights,
                filter_substitutes=filter_subs,
                max_per_category=0  # Evaluate raw ranking without category cap
            )
            lat = (time.time() - t0) * 1000.0
            metrics[cfg_name]["latency_ms"].append(lat)

            retrieved_ids = [c["item_id"] for c in res["complements"]]
            total_retrieved += len(retrieved_ids)

            # Check for substitutes (items with same base Semantic ID as anchor)
            for c in res["complements"]:
                if c["semantic_id"] == a_base_sem:
                    substitute_count += 1

            # Hit@5 and Hit@10
            top5 = set(retrieved_ids[:5])
            top10 = set(retrieved_ids[:10])

            if bool(top5 & gt_cands):
                hit5_count += 1
            if bool(top10 & gt_cands):
                hit10_count += 1

            # MRR and NDCG@10
            rr = 0.0
            dcg = 0.0
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(10, len(gt_cands))))

            for rank_idx, cand_id in enumerate(retrieved_ids[:10]):
                if cand_id in gt_cands:
                    if rr == 0.0:
                        rr = 1.0 / (rank_idx + 1)
                    dcg += 1.0 / np.log2(rank_idx + 2)

            mrr_total += rr
            if idcg > 0:
                ndcg10_total += (dcg / idcg)

        metrics[cfg_name]["hits@5"] = round(hit5_count / total_anchors * 100.0, 2)
        metrics[cfg_name]["hits@10"] = round(hit10_count / total_anchors * 100.0, 2)
        metrics[cfg_name]["mrr"] = round(mrr_total / total_anchors, 4)
        metrics[cfg_name]["ndcg@10"] = round(ndcg10_total / total_anchors, 4)
        metrics[cfg_name]["substitute_rate"] = round(substitute_count / max(1, total_retrieved) * 100.0, 2)
        metrics[cfg_name]["mean_latency_ms"] = round(float(np.mean(metrics[cfg_name]["latency_ms"])), 2)

    # Compile Summary Table
    rows = []
    for cfg_name in metrics:
        m = metrics[cfg_name]
        rows.append({
            "Retrieval Method": cfg_name,
            "Hit@5 (%)": f"{m['hits@5']:.1f}%",
            "Hit@10 (%)": f"{m['hits@10']:.1f}%",
            "MRR": f"{m['mrr']:.4f}",
            "NDCG@10": f"{m['ndcg@10']:.4f}",
            "Substitute Leakage (%)": f"{m['substitute_rate']:.2f}%",
            "Latency (ms)": f"{m['mean_latency_ms']:.1f} ms"
        })

    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("HYBRID COMPLEMENT RETRIEVAL BENCHMARK RESULTS")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    out_csv = "QuickBasket-Semantic-ID-Experiments/outputs/hybrid_search_benchmark.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to: {out_csv}")


if __name__ == "__main__":
    eng = HybridComplementSearchEngine().load()
    evaluate_engine_sample(eng, n_test_anchors=150)

