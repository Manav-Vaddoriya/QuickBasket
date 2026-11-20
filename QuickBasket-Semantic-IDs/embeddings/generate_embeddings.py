"""
Stage 3: Pretrained Product Embeddings Generation & Representation Validation.

This module uses a pretrained Sentence Transformer (all-MiniLM-L6-v2) to encode
all 50,015 QuickBasket catalogue items into dense, L2-normalized semantic vectors.
It maintains exact row-to-product mapping, benchmarks inference time and memory,
and performs rigorous semantic similarity evaluations across key product pairs.
"""

import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd
import torch

# Suppress unnecessary framework overheads
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sentence_transformers import SentenceTransformer


def load_config(config_path: str = "QuickBasket-Semantic-IDs/configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def compute_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two L2-normalized vectors (dot product)."""
    return float(np.dot(v1, v2))


def evaluate_semantic_similarity(
    df: pd.DataFrame, 
    embeddings: np.ndarray, 
    id_to_idx: dict[str, int]
) -> list[dict]:
    """
    Evaluates cosine similarity behavior across representative catalogue product pairs.
    Tests:
    1. Direct substitutes / category counterparts (e.g. Amul Milk vs Mother Dairy Milk)
    2. Variant variations within same product line (e.g. Britannia Whole Wheat vs Britannia Bread)
    3. Cross-brand functional equivalents (e.g. Amul Butter vs Mother Dairy Butter)
    4. Brand cross-category divergence (e.g. Amul Milk vs Amul Chocolate)
    5. Complete semantic opposites / cross-category pairs (e.g. Milk vs Dog Food, Milk vs Pencil)
    6. Single-word medical items (e.g. Eno vs Crocin)
    """
    test_queries = [
        # (item_a_name, item_b_name, description, expected_relationship)
        ("Amul Milk", "Mother Dairy Milk", "Direct Dairy Substitutes (Milk)", "High (>0.80)"),
        ("Amul Milk", "Mother Dairy Toned Milk", "Dairy Variant (Whole vs Toned)", "High (>0.80)"),
        ("Britannia Whole Wheat Bread", "Britannia Bread", "Brand Variant (Bread)", "High (>0.85)"),
        ("Amul Butter", "Nandini Milk", "Same Category Different Item", "Moderate-High (>0.60)"),
        ("Eno", "Crocin", "Single-Word Pharma Items (Eno vs Crocin)", "Moderate (>0.40)"),
        ("Amul Milk", "Pedigree Puppy Small Breed Dog Food", "Dairy vs Pet Care", "Low (<0.35)"),
        ("Amul Milk", "Yellow Spark Doraemon Medium Size Plastic Lunch Box", "Dairy vs Stationery/Office", "Low (<0.30)"),
        ("Aashirvaad Select Atta", "Tata Tea", "Atta/Rice/Dal vs Beverages", "Low (<0.30)")
    ]

    results = []
    for item_a, item_b, desc, rel in test_queries:
        match_a = df[df["item"] == item_a]
        match_b = df[df["item"] == item_b]
        
        if len(match_a) == 0 or len(match_b) == 0:
            # Fallback search by substring if exact match differs
            match_a = df[df["item"].str.contains(item_a, regex=False)]
            match_b = df[df["item"].str.contains(item_b, regex=False)]
            
        if len(match_a) > 0 and len(match_b) > 0:
            idx_a = match_a.index[0]
            idx_b = match_b.index[0]
            sim = compute_similarity(embeddings[idx_a], embeddings[idx_b])
            results.append({
                "item_a": match_a.iloc[0]["item"],
                "category_a": match_a.iloc[0]["category"],
                "item_b": match_b.iloc[0]["item"],
                "category_b": match_b.iloc[0]["category"],
                "cosine_similarity": round(sim, 4),
                "relationship": desc,
                "expected": rel
            })
    return results


def run_embedding_generation(
    input_csv: str = "QuickBasket-Semantic-IDs/data/product_texts.csv",
    output_dir: str = "QuickBasket-Semantic-IDs",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> dict:
    """
    Executes batched product embedding generation with rigorous validation and metadata export.
    """
    print("=" * 70)
    print("STAGE 3: PRETRAINED PRODUCT EMBEDDING GENERATION")
    print(f"Model: {model_name}")
    print("=" * 70)

    # 1. Hardware & Environment Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_cpus = os.cpu_count() or 4
    torch.set_num_threads(num_cpus)
    print(f"\n[1] Runtime Hardware Configuration:")
    print(f"    - Execution Device: {device.upper()}")
    print(f"    - CPU Thread Pool: {num_cpus} threads")
    print(f"    - Batch Size: {batch_size}")

    # 2. Ingestion
    assert os.path.exists(input_csv), f"Product text file not found at: {input_csv}"
    df = pd.read_csv(input_csv)
    total_records = len(df)
    print(f"\n[2] Catalogue Input Data:")
    print(f"    - Total Products: {total_records:,}")
    print(f"    - Columns: {list(df.columns)}")
    assert total_records == 50015, f"Expected 50,015 records, found {total_records}."

    texts = df["product_text"].tolist()
    product_ids = df["product_id"].tolist()

    # 3. Model Loading
    print(f"\n[3] Loading Pretrained Sentence Transformer: {model_name}...")
    t_load_start = time.time()
    model = SentenceTransformer(model_name, device=device)
    load_time = round(time.time() - t_load_start, 2)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"    - Model loaded in: {load_time}s")
    print(f"    - Embedding Dimension: {embedding_dim}")
    assert embedding_dim == 384, f"Expected 384 dimensions for MiniLM, found {embedding_dim}"

    # 4. Batched Embedding Generation
    print(f"\n[4] Generating Normalized Embeddings for {total_records:,} Products...")
    t_gen_start = time.time()
    
    # SentenceTransformer handles batching, progress bar, and L2 normalization natively
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    generation_time = round(time.time() - t_gen_start, 2)
    throughput = round(total_records / generation_time, 1)
    
    print(f"    - Total Generation Time: {generation_time}s ({generation_time / 60:.2f} minutes)")
    print(f"    - Throughput: {throughput} items/second")

    # 5. Numerical and Shape Validation
    print(f"\n[5] Embedding Validation Checks:")
    print(f"    - Embedding Shape: {embeddings.shape}")
    assert embeddings.shape == (50015, 384), f"Unexpected shape: {embeddings.shape}"
    assert embeddings.dtype == np.float32, f"Unexpected dtype: {embeddings.dtype}"
    
    # Check L2 normalization: norm of each vector must be 1.0 ± 1e-4
    norms = np.linalg.norm(embeddings, axis=1)
    min_norm = float(norms.min())
    max_norm = float(norms.max())
    mean_norm = float(norms.mean())
    print(f"    - Vector Norm Range: [{min_norm:.6f}, {max_norm:.6f}] (Mean: {mean_norm:.6f})")
    assert np.allclose(norms, 1.0, atol=1e-4), "L2 normalization check failed!"

    # Check for NaN / Inf
    assert not np.isnan(embeddings).any(), "Found NaN values in embeddings!"
    assert not np.isinf(embeddings).any(), "Found Inf values in embeddings!"
    print("    - Zero NaN or Inf anomalies detected.")

    # Memory footprint calculation
    memory_mb = round(embeddings.nbytes / (1024 * 1024), 2)
    print(f"    - In-Memory Size: {memory_mb} MB")

    # 6. Save Deliverables
    emb_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    
    # A. product_embeddings.npy
    emb_path = os.path.join(emb_dir, "product_embeddings.npy")
    np.save(emb_path, embeddings)
    file_size_mb = round(os.path.getsize(emb_path) / (1024 * 1024), 2)
    print(f"\n[+] Saved Embeddings Array: {emb_path} ({file_size_mb} MB)")

    # B. Product ID Mapping File (row index -> product_id)
    id_mapping = {int(idx): pid for idx, pid in enumerate(product_ids)}
    id_to_idx = {pid: int(idx) for idx, pid in enumerate(product_ids)}
    mapping_path = os.path.join(emb_dir, "product_id_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(id_mapping, f)
    print(f"[+] Saved ID Mapping File: {mapping_path} ({os.path.getsize(mapping_path):,} bytes)")

    # 7. Semantic Similarity Evaluation
    print(f"\n[6] Evaluating Semantic Similarity on Representative Pairs:")
    sim_eval = evaluate_semantic_similarity(df, embeddings, id_to_idx)
    for res in sim_eval:
        print(f"    * {res['item_a']} <-> {res['item_b']}")
        print(f"      Similarity: {res['cosine_similarity']:.4f} | Category: [{res['category_a']}] vs [{res['category_b']}]")
        print(f"      Relationship: {res['relationship']} | Expected: {res['expected']}\n")

    # C. Embedding Metadata JSON
    metadata = {
        "model_name": model_name,
        "embedding_dimension": embedding_dim,
        "total_records": total_records,
        "device": device,
        "batch_size": batch_size,
        "load_time_seconds": load_time,
        "generation_time_seconds": generation_time,
        "throughput_items_per_second": throughput,
        "memory_size_mb": memory_mb,
        "l2_normalized": True,
        "norm_min": min_norm,
        "norm_max": max_norm,
        "norm_mean": mean_norm,
        "output_shape": list(embeddings.shape),
        "representative_similarities": sim_eval,
        "model_comparison": {
            "all-MiniLM-L6-v2": {
                "dimension": 384,
                "parameters_millions": 22.7,
                "speed_relative": "1.0x (Baseline - Fast)",
                "memory_mb_50k": 76.8,
                "context_window_tokens": 256,
                "status": "Generated & Validated as Primary Foundation"
            },
            "all-mpnet-base-v2": {
                "dimension": 768,
                "parameters_millions": 109.5,
                "speed_relative": "~0.25x (~4-5x slower)",
                "memory_mb_50k": 153.6,
                "context_window_tokens": 384,
                "status": "High-Capacity Alternative (Dimension 768 requires doubling RQ-VAE input layer)"
            }
        }
    }
    
    metadata_path = os.path.join(emb_dir, "embedding_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved Metadata: {metadata_path} ({os.path.getsize(metadata_path):,} bytes)")

    print("\n" + "=" * 70)
    print("STAGE 3 EMBEDDING GENERATION AND VALIDATION COMPLETE!")
    print("=" * 70)
    return metadata


if __name__ == "__main__":
    run_embedding_generation()

