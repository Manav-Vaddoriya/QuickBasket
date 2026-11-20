"""
QuickBasket Hybrid Complement Search Engine
============================================
Combines:
1. Discrete Semantic ID Complement Graph & Anti-Substitute Constraint (RQ-VAE)
2. Continuous Two-Tower Complementary Neural Projections (v5 InfoNCE)
3. Empirical Behavioral Co-Purchase / Complement Graph Priors
4. Multi-modal Fusion (RRF / Weighted Convex) with Category Diversity Throttling
"""

import os
import sys
import time
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def safe_pickle_load(f):
    try:
        return pickle.load(f)
    except (ModuleNotFoundError, AttributeError):
        f.seek(0)
        return SafeUnpickler(f).load()


class TwoTowerItemRetrievalModel(nn.Module):
    """
    Two-Tower Neural Projection Architecture for Item-to-Item Complement Retrieval.
    Transforms generic dense embeddings (384d) into an asymmetric complementary retrieval space (256d).
    """
    def __init__(self, in_features: int = 384, proj_dim: int = 256):
        super().__init__()
        self.anchor_res = nn.Linear(in_features, proj_dim)
        self.anchor_mlp = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.cand_res = nn.Linear(in_features, proj_dim)
        self.cand_mlp = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
    def encode_anchor(self, x: torch.Tensor) -> torch.Tensor:
        res = self.anchor_res(x)
        out = self.anchor_mlp(x) + res
        return F.normalize(out, p=2, dim=-1)
        
    def encode_candidate(self, x: torch.Tensor) -> torch.Tensor:
        res = self.cand_res(x)
        out = self.cand_mlp(x) + res
        return F.normalize(out, p=2, dim=-1)


class HybridComplementSearchEngine:
    """
    Production-grade Hybrid Search Engine for retrieving complementary grocery products.
    Integrates discrete Semantic IDs with continuous neural embeddings and behavioral co-purchase graphs.
    """
    def __init__(
        self,
        catalog_path: str = "data/reference/catalog.parquet",
        semantic_ids_path: str = "QuickBasket-Semantic-ID-Experiments/outputs/semantic_ids_best.parquet",
        embeddings_path: str = "sample datasets/item_embeddings_50k.pkl",
        retrieval_model_path: str = "artifacts/item_retrieval_v5/best_item_retrieval_model_v5.pt",
        complement_graph_path: str = "data/world/complement_graph.parquet",
        device: str = "cpu"
    ):
        self.catalog_path = catalog_path
        self.semantic_ids_path = semantic_ids_path
        self.embeddings_path = embeddings_path
        self.retrieval_model_path = retrieval_model_path
        self.complement_graph_path = complement_graph_path
        self.device = torch.device(device)

        # Catalog & Metadata
        self.catalog_df: Optional[pd.DataFrame] = None
        self.item_ids: List[str] = []
        self.item_names: List[str] = []
        self.item_id_to_idx: Dict[str, int] = {}
        self.item_name_to_idx: Dict[str, int] = {}
        self.idx_to_meta: Dict[int, Dict[str, Any]] = {}
        self.catalog_size: int = 0

        # Semantic IDs
        self.semantic_df: Optional[pd.DataFrame] = None
        self.idx_to_semantic_id: Dict[int, str] = {}
        self.idx_to_disambiguated_id: Dict[int, str] = {}
        self.idx_to_c01: Dict[int, str] = {}
        self.c01_array: Optional[np.ndarray] = None
        self.base_sem_array: Optional[np.ndarray] = None
        self.code_transition_affinities: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.c01_to_indices: Dict[str, List[int]] = defaultdict(list)

        # Continuous Two-Tower Projections
        self.raw_embeddings: Optional[torch.Tensor] = None
        self.two_tower_model: Optional[TwoTowerItemRetrievalModel] = None
        self.cand_projections: Optional[torch.Tensor] = None  # Shape: (N, 256)

        # Behavioral Graph Priors
        self.graph_adj: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.graph_tier: Dict[str, Dict[str, str]] = defaultdict(dict)

        self.is_loaded: bool = False

    def load(self) -> "HybridComplementSearchEngine":
        """Loads and aligns all retrieval assets into memory."""
        t0 = time.time()
        print("=" * 80)
        print("INITIALIZING QUICKBASKET HYBRID COMPLEMENT SEARCH ENGINE")
        print("=" * 80)

        # 1. Load Catalog
        print(f"\n[1/5] Loading reference catalog from '{self.catalog_path}'...")
        self.catalog_df = pd.read_parquet(self.catalog_path)
        self.item_ids = list(self.catalog_df["item_id"].astype(str))
        self.item_names = list(self.catalog_df["item_name"].astype(str))
        self.catalog_size = len(self.item_ids)

        for idx, row in enumerate(self.catalog_df.itertuples()):
            iid = str(row.item_id)
            iname = str(row.item_name)
            self.item_id_to_idx[iid] = idx
            self.item_name_to_idx[iname] = idx
            self.idx_to_meta[idx] = {
                "item_id": iid,
                "item_name": iname,
                "category_l1": getattr(row, "category_l1", "Unknown"),
                "category_l2": getattr(row, "category_l2", "Unknown"),
                "brand": getattr(row, "brand", "Unknown"),
                "price_inr": getattr(row, "price_inr", 0.0),
                "pack_size": getattr(row, "pack_size", "Standard")
            }
        print(f" -> Catalog indexed: {self.catalog_size:,} items.")

        # 2. Load Semantic IDs (generated by best RQ-VAE model)
        print(f"\n[2/5] Loading best generated Semantic IDs from '{self.semantic_ids_path}'...")
        self.semantic_df = pd.read_parquet(self.semantic_ids_path)
        
        # Build mappings
        item_to_sem = dict(zip(self.semantic_df["item"], self.semantic_df["semantic_id"]))
        item_to_full = dict(zip(self.semantic_df["item"], self.semantic_df["semantic_id_disambiguated"]))
        item_to_c01 = dict(zip(self.semantic_df["item"], self.semantic_df["code_0"].astype(str) + "_" + self.semantic_df["code_1"].astype(str)))

        base_list = []
        c01_list = []
        for idx in range(self.catalog_size):
            iname = self.item_names[idx]
            sem_id = item_to_sem.get(iname, "UNKNOWN")
            full_id = item_to_full.get(iname, "UNKNOWN")
            c01 = item_to_c01.get(iname, "UNKNOWN")
            
            self.idx_to_semantic_id[idx] = sem_id
            self.idx_to_disambiguated_id[idx] = full_id
            self.idx_to_c01[idx] = c01
            base_list.append(sem_id)
            c01_list.append(c01)
            self.c01_to_indices[c01].append(idx)
            
            # Enrich metadata
            self.idx_to_meta[idx]["semantic_id"] = sem_id
            self.idx_to_meta[idx]["semantic_id_disambiguated"] = full_id
            self.idx_to_meta[idx]["c01"] = c01

        self.base_sem_array = np.array(base_list)
        self.c01_array = np.array(c01_list)
        print(f" -> Mapped Semantic IDs across all catalog items (Unique base tuples: {len(set(base_list)):,}).")

        # 3. Load Behavioral Complement Graph & Compute Semantic Code Transitions
        print(f"\n[3/5] Loading complement graph and building Semantic ID transition index from '{self.complement_graph_path}'...")
        comp_df = pd.read_parquet(self.complement_graph_path)
        
        # Direct graph adjacency
        for row in comp_df[["anchor_item_id", "candidate_item_id", "base_affinity", "tier"]].itertuples(index=False):
            a_id, c_id, aff, tier = str(row[0]), str(row[1]), float(row[2]), str(row[3])
            self.graph_adj[a_id][c_id] = aff
            self.graph_tier[a_id][c_id] = tier

        # Learn discrete Semantic Code Transitions: Count(C_A -> C_B)
        pair_counts = defaultdict(lambda: defaultdict(int))
        code_totals = defaultdict(int)

        for row in comp_df[["anchor_item_id", "candidate_item_id"]].itertuples(index=False):
            a_idx = self.item_id_to_idx.get(str(row[0]))
            c_idx = self.item_id_to_idx.get(str(row[1]))
            if a_idx is not None and c_idx is not None:
                c_a = self.idx_to_c01[a_idx]
                c_b = self.idx_to_c01[c_idx]
                if c_a != "UNKNOWN" and c_b != "UNKNOWN":
                    pair_counts[c_a][c_b] += 1
                    code_totals[c_a] += 1

        # Compute normalized transition affinity: P(C_B | C_A) / sqrt(P(C_B))
        for c_a, c_b_dict in pair_counts.items():
            tot_a = code_totals[c_a]
            for c_b, cnt in c_b_dict.items():
                tot_b = code_totals.get(c_b, 1)
                # Symmetric Normalized Affinity
                affinity = cnt / np.sqrt(tot_a * tot_b)
                self.code_transition_affinities[c_a][c_b] = float(affinity)

        print(f" -> Indexed {len(comp_df):,} complement graph edges.")
        print(f" -> Built Semantic Code transition index: {len(self.code_transition_affinities):,} active code hubs.")

        # 4. Load Raw Embeddings
        print(f"\n[4/5] Loading dense embeddings from '{self.embeddings_path}'...")
        with open(self.embeddings_path, "rb") as f:
            cache = safe_pickle_load(f)
        raw_embs = cache.get("embeddings")
        item_to_raw_idx = cache.get("item_to_idx", {})
        
        aligned_embs = np.zeros((self.catalog_size, raw_embs.shape[1]), dtype=np.float32)
        for i, iname in enumerate(self.item_names):
            ridx = item_to_raw_idx.get(iname)
            if ridx is not None:
                vec = raw_embs[ridx]
                norm = np.linalg.norm(vec)
                aligned_embs[i] = vec / norm if norm > 0 else vec

        self.raw_embeddings = torch.from_numpy(aligned_embs).float().to(self.device)
        print(f" -> Aligned embeddings matrix: {self.raw_embeddings.shape}.")

        # 5. Load Two-Tower Neural Projection Model & Pre-project Candidates
        print(f"\n[5/5] Loading Two-Tower Neural Retrieval Model from '{self.retrieval_model_path}'...")
        if os.path.exists(self.retrieval_model_path):
            self.two_tower_model = TwoTowerItemRetrievalModel(in_features=384, proj_dim=256)
            ckpt = torch.load(self.retrieval_model_path, map_location=self.device)
            self.two_tower_model.load_state_dict(ckpt["model_state_dict"])
            self.two_tower_model.to(self.device)
            self.two_tower_model.eval()

            with torch.no_grad():
                self.cand_projections = self.two_tower_model.encode_candidate(self.raw_embeddings)
            print(f" -> Two-Tower Model loaded and candidates pre-projected: {self.cand_projections.shape}.")
        else:
            print(f" [!] Two-Tower checkpoint not found at '{self.retrieval_model_path}'. Fallback to unit embeddings.")
            self.cand_projections = F.normalize(self.raw_embeddings, p=2, dim=-1)

        self.is_loaded = True
        print(f"\n[+] Hybrid Complement Search Engine successfully initialized in {time.time() - t0:.2f}s!")
        return self

    def _resolve_item_index(self, item_query: Union[str, int]) -> int:
        """Resolves item name, SKU ID, or integer index to catalogue index."""
        if isinstance(item_query, int):
            if 0 <= item_query < self.catalog_size:
                return item_query
            raise ValueError(f"Index {item_query} out of bounds.")
        if item_query in self.item_name_to_idx:
            return self.item_name_to_idx[item_query]
        if item_query in self.item_id_to_idx:
            return self.item_id_to_idx[item_query]
        # Case-insensitive exact search
        q_low = item_query.lower().strip()
        for name, idx in self.item_name_to_idx.items():
            if name.lower() == q_low:
                return idx

        # Substring prefix / fuzzy fallback
        matches = [idx for name, idx in self.item_name_to_idx.items() if q_low in name.lower()]
        if matches:
            # Pick shortest name (least specific / most representative staple)
            matches.sort(key=lambda i: len(self.item_names[i]))
            return matches[0]

        raise KeyError(f"Item '{item_query}' not found in catalogue.")

    def compute_dense_scores(self, anchor_idx: int) -> np.ndarray:
        """Computes complementary similarity via Two-Tower projection."""
        with torch.no_grad():
            if self.two_tower_model is not None:
                q_emb = self.raw_embeddings[anchor_idx].unsqueeze(0)
                q_proj = self.two_tower_model.encode_anchor(q_emb)
                sims = torch.mm(q_proj, self.cand_projections.t()).squeeze(0)
            else:
                q_proj = self.cand_projections[anchor_idx].unsqueeze(0)
                sims = torch.mm(q_proj, self.cand_projections.t()).squeeze(0)
        return sims.cpu().numpy()

    def compute_semantic_id_scores(self, anchor_idx: int, filter_substitutes: bool = True) -> np.ndarray:
        """
        Computes discrete complement affinity using Semantic ID transition pathways.
        Optionally applies strict anti-substitute penalty for items sharing the exact base tuple.
        """
        anchor_c01 = self.idx_to_c01[anchor_idx]
        anchor_base_id = self.idx_to_semantic_id[anchor_idx]
        
        affinities_map = self.code_transition_affinities.get(anchor_c01, {})
        scores = np.zeros(self.catalog_size, dtype=np.float32)

        # Vectorized lookup across c01 clusters
        for c_cand, aff in affinities_map.items():
            cand_indices = self.c01_to_indices.get(c_cand)
            if cand_indices:
                scores[cand_indices] = aff

        # Enforce Anti-Substitute Constraint using Semantic IDs
        if filter_substitutes and anchor_base_id != "UNKNOWN":
            substitute_mask = (self.base_sem_array == anchor_base_id)
            scores[substitute_mask] = -np.inf

        return scores

    def compute_graph_scores(self, anchor_idx: int) -> np.ndarray:
        """Extracts direct behavioral / complement graph affinities."""
        anchor_id = self.item_ids[anchor_idx]
        direct_edges = self.graph_adj.get(anchor_id, {})
        scores = np.zeros(self.catalog_size, dtype=np.float32)
        
        for cand_id, aff in direct_edges.items():
            cand_idx = self.item_id_to_idx.get(cand_id)
            if cand_idx is not None:
                scores[cand_idx] = aff
        return scores

    def get_complements(
        self,
        query_item: Union[str, int],
        top_k: int = 10,
        fusion_method: str = "rrf",
        weights: Optional[Dict[str, float]] = None,
        filter_substitutes: bool = True,
        max_per_category: int = 2,
        excluded_items: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieves top-K complementary items for a single anchor product.

        Args:
            query_item: Name or SKU of anchor item (e.g. 'Amul Milk').
            top_k: Number of complementary items to return.
            fusion_method: 'rrf' (Reciprocal Rank Fusion) or 'weighted' (Normalized score sum).
            weights: Dict of weights {'dense': 0.35, 'semantic_id': 0.40, 'graph': 0.25}.
            filter_substitutes: If True, uses Semantic IDs to strictly ban items sharing base ID (c0_c1_c2).
            max_per_category: Maximum items from the same category_l2 (enforces basket diversity).
            excluded_items: Optional list of item names or IDs to suppress (e.g. existing cart items).

        Returns:
            Dict containing anchor metadata and ordered list of top complementary items with scoring breakdown.
        """
        if not self.is_loaded:
            self.load()

        anchor_idx = self._resolve_item_index(query_item)
        anchor_meta = self.idx_to_meta[anchor_idx]
        anchor_id = anchor_meta["item_id"]

        if weights is None:
            # Default production weights
            weights = {"dense": 0.35, "semantic_id": 0.40, "graph": 0.25}

        # 1. Compute Modality Scores
        dense_scores = self.compute_dense_scores(anchor_idx)
        sem_scores = self.compute_semantic_id_scores(anchor_idx, filter_substitutes=filter_substitutes)
        graph_scores = self.compute_graph_scores(anchor_idx)

        # Always mask self
        dense_scores[anchor_idx] = -np.inf
        sem_scores[anchor_idx] = -np.inf
        graph_scores[anchor_idx] = -np.inf

        # Mask external excluded items
        if excluded_items:
            for item in excluded_items:
                try:
                    e_idx = self._resolve_item_index(item)
                    dense_scores[e_idx] = -np.inf
                    sem_scores[e_idx] = -np.inf
                    graph_scores[e_idx] = -np.inf
                except (KeyError, ValueError):
                    pass

        # 2. Multi-Modal Fusion
        if fusion_method.lower() == "rrf":
            # Reciprocal Rank Fusion (k=60)
            k_rrf = 60
            dense_ranks = np.argsort(np.argsort(-dense_scores))
            sem_ranks = np.argsort(np.argsort(-sem_scores))
            graph_ranks = np.argsort(np.argsort(-graph_scores))

            final_scores = (
                weights["dense"] / (k_rrf + dense_ranks) +
                weights["semantic_id"] / (k_rrf + sem_ranks) +
                weights["graph"] / (k_rrf + graph_ranks)
            )
        else:
            # Weighted Score Combination with min-max normalization
            def norm_score(s):
                fin = s[np.isfinite(s)]
                if len(fin) == 0:
                    return np.zeros_like(s)
                s_min, s_max = fin.min(), fin.max()
                if s_max > s_min:
                    return np.where(np.isfinite(s), (s - s_min) / (s_max - s_min), -1e5)
                return np.zeros_like(s)

            final_scores = (
                weights["dense"] * norm_score(dense_scores) +
                weights["semantic_id"] * norm_score(sem_scores) +
                weights["graph"] * norm_score(graph_scores)
            )

        # 3. Suppress Substitutes & Non-Complements in Final Scores
        if filter_substitutes:
            anchor_base = self.idx_to_semantic_id[anchor_idx]
            sub_mask = (self.base_sem_array == anchor_base)
            final_scores[sub_mask] = -np.inf
        final_scores[anchor_idx] = -np.inf

        # 4. Rank Candidates & Apply Category Diversity Throttling
        sorted_indices = np.argsort(-final_scores)

        complements: List[Dict[str, Any]] = []
        cat_counts: Dict[str, int] = defaultdict(int)

        for c_idx in sorted_indices:
            if not np.isfinite(final_scores[c_idx]) or final_scores[c_idx] <= -1e4:
                break

            cand_meta = self.idx_to_meta[c_idx]
            cand_cat_l2 = cand_meta.get("category_l2", "Unknown")

            # Category throttling for diversity
            if max_per_category > 0 and cat_counts[cand_cat_l2] >= max_per_category:
                continue

            cat_counts[cand_cat_l2] += 1
            cand_id = cand_meta["item_id"]
            tier = self.graph_tier.get(anchor_id, {}).get(cand_id, "None")

            complements.append({
                "rank": len(complements) + 1,
                "item_id": cand_id,
                "item_name": cand_meta["item_name"],
                "category_l1": cand_meta["category_l1"],
                "category_l2": cand_cat_l2,
                "brand": cand_meta["brand"],
                "price_inr": cand_meta["price_inr"],
                "semantic_id": cand_meta["semantic_id"],
                "semantic_id_disambiguated": cand_meta["semantic_id_disambiguated"],
                "hybrid_score": round(float(final_scores[c_idx]), 5),
                "dense_similarity": round(float(dense_scores[c_idx]), 4) if np.isfinite(dense_scores[c_idx]) else 0.0,
                "semantic_id_affinity": round(float(sem_scores[c_idx]), 4) if np.isfinite(sem_scores[c_idx]) else 0.0,
                "graph_tier": tier,
                "graph_base_affinity": round(float(graph_scores[c_idx]), 2),
                "is_cross_category": cand_meta["category_l1"] != anchor_meta["category_l1"]
            })

            if len(complements) >= top_k:
                break

        return {
            "anchor": {
                "item_id": anchor_id,
                "item_name": anchor_meta["item_name"],
                "category_l1": anchor_meta["category_l1"],
                "category_l2": anchor_meta["category_l2"],
                "semantic_id": anchor_meta["semantic_id"],
                "semantic_id_disambiguated": anchor_meta["semantic_id_disambiguated"]
            },
            "fusion_method": fusion_method,
            "weights": weights,
            "filter_substitutes": filter_substitutes,
            "total_candidates_evaluated": self.catalog_size,
            "complements": complements
        }

    def get_cart_complements(
        self,
        cart_items: List[Union[str, int]],
        top_k: int = 10,
        fusion_method: str = "rrf",
        weights: Optional[Dict[str, float]] = None,
        max_per_category: int = 2
    ) -> Dict[str, Any]:
        """
        Retrieves top-K complementary items for an entire multi-item shopping cart session.
        Aggregates complementary signals across all items currently in cart.
        """
        if not self.is_loaded:
            self.load()

        cart_indices = [self._resolve_item_index(item) for item in cart_items]
        cart_names = [self.item_names[i] for i in cart_indices]

        # Pool recommendations across all cart items
        combined_scores = np.zeros(self.catalog_size, dtype=np.float32)

        for a_idx in cart_indices:
            res = self.get_complements(
                query_item=a_idx,
                top_k=top_k * 3,
                fusion_method=fusion_method,
                weights=weights,
                filter_substitutes=True,
                max_per_category=0,  # Uncapped in intermediate pooling
                excluded_items=cart_names
            )
            for c in res["complements"]:
                c_idx = self.item_id_to_idx[c["item_id"]]
                combined_scores[c_idx] += c["hybrid_score"]

        # Mask all cart items
        for ci in cart_indices:
            combined_scores[ci] = -np.inf

        sorted_indices = np.argsort(-combined_scores)
        cart_complements = []
        cat_counts = defaultdict(int)

        for c_idx in sorted_indices:
            if combined_scores[c_idx] <= 0 or not np.isfinite(combined_scores[c_idx]):
                break

            cand_meta = self.idx_to_meta[c_idx]
            cand_cat_l2 = cand_meta.get("category_l2", "Unknown")

            if max_per_category > 0 and cat_counts[cand_cat_l2] >= max_per_category:
                continue

            cat_counts[cand_cat_l2] += 1
            cart_complements.append({
                "rank": len(cart_complements) + 1,
                "item_id": cand_meta["item_id"],
                "item_name": cand_meta["item_name"],
                "category_l1": cand_meta["category_l1"],
                "category_l2": cand_cat_l2,
                "brand": cand_meta["brand"],
                "price_inr": cand_meta["price_inr"],
                "semantic_id_disambiguated": cand_meta["semantic_id_disambiguated"],
                "cart_affinity_score": round(float(combined_scores[c_idx]), 5)
            })

            if len(cart_complements) >= top_k:
                break

        return {
            "cart_items": [self.idx_to_meta[i] for i in cart_indices],
            "top_k": top_k,
            "complements": cart_complements
        }
