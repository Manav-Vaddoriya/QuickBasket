import os
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Set, Tuple, Optional

class Evaluator:
    """
    Evaluation Engine for Retrieval Metrics against Full Item Catalog.
    Computes Hit Ratio@K, Recall@K, and MRR@K for K in [5, 10, 20].
    Loads precomputed embeddings from item_embeddings_50k.pkl if available for instant evaluation.
    """
    def __init__(self, test_df: pd.DataFrame, catalog_df: pd.DataFrame, pkl_path: str = os.path.join("sample datasets", "item_embeddings_50k.pkl")) :
        self.test_df = test_df
        self.catalog_df = catalog_df.reset_index(drop=True)
        self.catalog_items = list(self.catalog_df['item_name'])
        self.catalog_size = len(self.catalog_items)
        self.item_to_catalog_idx = {name: idx for idx, name in enumerate(self.catalog_items)}

        # Load cached embeddings if file exists
        self.cached_matrix = None
        self.cached_item_to_idx = None
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    cached_data = pickle.load(f)
                self.cached_matrix = cached_data.get('embeddings')
                self.cached_item_to_idx = cached_data.get('item_to_idx')
                print(f"Evaluator loaded cached catalog embeddings from '{pkl_path}'.")
            except Exception as e:
                print(f"Could not load cached embeddings: {e}")

        pos_test_df = test_df[test_df['label'] == 1]
        self.ground_truth = {}
        for anchor, group in pos_test_df.groupby('anchor_item'):
            pos_items = set(group['candidate_item'].unique()) - {anchor}
            if pos_items:
                self.ground_truth[anchor] = pos_items

        self.test_anchors = sorted(list(self.ground_truth.keys()))
        print(f"Evaluator initialized with {len(self.test_anchors)} test anchors and {self.catalog_size} total catalog items.")

    def evaluate(self, model, device: torch.device = None, batch_size: int = 256, k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        model.eval()
        if device is None:
            device = next(model.parameters()).device

        # 1. Obtain Catalog Matrix (Use cached embeddings if available, else encode dynamically)
        if self.cached_matrix is not None and self.cached_item_to_idx is not None:
            # Reorder cached matrix to match catalog_df order
            catalog_matrix = np.zeros((self.catalog_size, self.cached_matrix.shape[1]), dtype='float32')
            for i, name in enumerate(self.catalog_items):
                idx = self.cached_item_to_idx.get(name)
                if idx is not None:
                    catalog_matrix[i] = self.cached_matrix[idx]
        else:
            catalog_embeddings = []
            with torch.no_grad():
                for i in range(0, self.catalog_size, batch_size):
                    batch_texts = self.catalog_items[i : i + batch_size]
                    if hasattr(model, 'encode_candidate'):
                        emb = model.encode_candidate(batch_texts, device=device)
                    else:
                        emb = model.encode(batch_texts, device=device)
                    catalog_embeddings.append(emb.cpu().numpy())
            catalog_matrix = np.vstack(catalog_embeddings)

        # 2. Obtain Test Anchors Matrix
        anchor_embeddings = []
        with torch.no_grad():
            for i in range(0, len(self.test_anchors), batch_size):
                batch_texts = self.test_anchors[i : i + batch_size]
                if hasattr(model, 'encode_anchor'):
                    emb = model.encode_anchor(batch_texts, device=device)
                else:
                    emb = model.encode(batch_texts, device=device)
                anchor_embeddings.append(emb.cpu().numpy())
        anchor_matrix = np.vstack(anchor_embeddings)

        # 3. Compute Metrics
        max_k = max(k_values)
        results = {f"{metric}@{k}": [] for k in k_values for metric in ["Hit", "Recall", "MRR"]}

        for a_idx, anchor_item in enumerate(self.test_anchors):
            anchor_vec = anchor_matrix[a_idx]
            sims = np.dot(catalog_matrix, anchor_vec)

            if anchor_item in self.item_to_catalog_idx:
                sims[self.item_to_catalog_idx[anchor_item]] = -np.inf

            top_indices = np.argpartition(sims, -max_k - 1)[-max_k - 1:]
            top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
            
            top_items = [self.catalog_items[idx] for idx in top_indices if self.catalog_items[idx] != anchor_item][:max_k]
            gt_positives = self.ground_truth[anchor_item]

            for k in k_values:
                retrieved_k = top_items[:k]
                hits = [1 if item in gt_positives else 0 for item in retrieved_k]
                n_hits = sum(hits)

                hit_k = 1.0 if n_hits > 0 else 0.0
                recall_k = n_hits / len(gt_positives) if gt_positives else 0.0

                mrr_k = 0.0
                for rank_idx, h in enumerate(hits):
                    if h == 1:
                        mrr_k = 1.0 / (rank_idx + 1.0)
                        break

                results[f"Hit@{k}"].append(hit_k)
                results[f"Recall@{k}"].append(recall_k)
                results[f"MRR@{k}"].append(mrr_k)

        return {metric_name: float(np.mean(vals)) for metric_name, vals in results.items()}
