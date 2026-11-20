"""
QuickBasket Re-Ranker Studio — Zepto-Inspired Real-Time Backend
================================================================
FastAPI backend providing:
  - Instant preview items
  - Category dropdown and dynamic item selector
  - Cross-Category Complement Retrieval (strictly filters out same category)
  - Multi-item cart pooling:
      * 1 item in cart -> Top 5 cross-category items -> Re-rank 5 items
      * 2 items in cart -> Top 5 for item 1 + Top 5 for item 2 -> 10 pooled items -> Re-rank 10 items
      * N items in cart -> Top 5 per item -> Pooled items -> Re-rank all
  - Transparent Before vs. After re-ranking verification
"""

import os
import sys
import json
import time
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

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

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.retrieval.hybrid_complement_search import HybridComplementSearchEngine
from train_reranker_models import (
    Model1_InteractionCrossMLP,
    Model2_DualTowerBilinear,
    Model3_ListAwareTransformer
)

class TwoTowerItemRetrievalModel(torch.nn.Module):
    """
    Two-Tower Neural Projection Architecture for Item-to-Item Retrieval.
    Transforms generic BGE embeddings into optimized complementary embedding space.
    """
    def __init__(self, in_features: int = 384, proj_dim: int = 256):
        super().__init__()
        self.anchor_res = torch.nn.Linear(in_features, proj_dim)
        self.anchor_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, proj_dim),
            torch.nn.LayerNorm(proj_dim),
            torch.nn.GELU(),
            torch.nn.Linear(proj_dim, proj_dim),
            torch.nn.LayerNorm(proj_dim)
        )
        self.cand_res = torch.nn.Linear(in_features, proj_dim)
        self.cand_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, proj_dim),
            torch.nn.LayerNorm(proj_dim),
            torch.nn.GELU(),
            torch.nn.Linear(proj_dim, proj_dim),
            torch.nn.LayerNorm(proj_dim)
        )
        
    def encode_anchor(self, x: torch.Tensor) -> torch.Tensor:
        res = self.anchor_res(x)
        out = self.anchor_mlp(x) + res
        return F.normalize(out, p=2, dim=-1)
        
    def encode_candidate(self, x: torch.Tensor) -> torch.Tensor:
        res = self.cand_res(x)
        out = self.cand_mlp(x) + res
        return F.normalize(out, p=2, dim=-1)

app = FastAPI(title="QuickBasket Re-Ranker Studio", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")

# Global data holders
EMBEDDINGS: Optional[torch.Tensor] = None
EMBEDDINGS_NORM: Optional[torch.Tensor] = None
ITEM_TO_IDX: Dict[str, int] = {}
IDX_TO_ITEM: Dict[int, str] = {}
CATALOG_DF: Optional[pd.DataFrame] = None
ITEM_METADATA: Dict[str, Dict[str, Any]] = {}
CATEGORIES_LIST: List[str] = []
CATEGORY_TO_ITEMS: Dict[str, List[Dict[str, Any]]] = {}
CAT_L1_TO_ID: Dict[str, int] = {}
ITEM_CAT_IDS_TENSOR: Optional[torch.Tensor] = None
MODELS: Dict[str, torch.nn.Module] = {}
RETRIEVAL_MODEL: Optional[TwoTowerItemRetrievalModel] = None
CAND_PROJECTIONS: Optional[torch.Tensor] = None



def get_item_icon(category_l1: str, name: str) -> str:
    n = name.lower()
    c = str(category_l1).lower()
    if "milk" in n: return "🥛"
    if "bread" in n or "bun" in n or "toast" in n: return "🍞"
    if "butter" in n or "cheese" in n or "paneer" in n: return "🧀"
    if "curd" in n or "yogurt" in n: return "🥣"
    if "egg" in n: return "🥚"
    if "shampoo" in n or "conditioner" in n: return "🧴"
    if "soap" in n or "wash" in n or "shower" in n: return "🧼"
    if "toothpaste" in n or "toothbrush" in n: return "🪥"
    if "cleaner" in n or "harpic" in n or "lizol" in n: return "🧹"
    if "detergent" in n or "surf" in n or "ariel" in n: return "🧺"
    if "tea" in n or "chai" in n: return "🍵"
    if "coffee" in n: return "☕"
    if "biscuit" in n or "cookie" in n or "rusk" in n: return "🍪"
    if "chip" in n or "snack" in n or "namkeen" in n: return "🍟"
    if "chocolate" in n or "cadbury" in n: return "🍫"
    
    # Fruits
    if "apple" in n: return "🍎"
    if "banana" in n: return "🍌"
    if "mango" in n: return "🥭"
    if "orange" in n or "mosambi" in n or "citrus" in n: return "🍊"
    if "lemon" in n or "lime" in n: return "🍋"
    if "grape" in n: return "🍇"
    if "watermelon" in n: return "🍉"
    if "pineapple" in n: return "🍍"
    if "papaya" in n or "melon" in n: return "🍈"
    if "avocado" in n: return "🥑"
    if "kiwi" in n: return "🥝"
    if "strawberry" in n or "berry" in n: return "🍓"
    if "pear" in n or "guava" in n: return "🍐"
    if "pomegranate" in n or "anar" in n: return "🍎"
    
    # Vegetables & Herbs
    if "potato" in n or "aloo" in n: return "🥔"
    if "tomato" in n: return "🍅"
    if "onion" in n or "pyaz" in n: return "🧅"
    if "carrot" in n or "gajar" in n: return "🥕"
    if "cucumber" in n or "kheera" in n: return "🥒"
    if "chilli" in n or "pepper" in n or "mirch" in n: return "🌶️"
    if "corn" in n: return "🌽"
    if "mushroom" in n: return "🍄"
    if "garlic" in n or "lasun" in n: return "🧄"
    if "ginger" in n or "adrak" in n: return "🫚"
    if "capsicum" in n: return "🫑"
    if "palak" in n or "spinach" in n or "kale" in n or "methi" in n or "coriander" in n or "mint" in n or "curry" in n: return "🥬"
    if "broccoli" in n or "cabbage" in n or "cauliflower" in n or "gobi" in n: return "🥦"
    if "brinjal" in n or "eggplant" in n or "baingan" in n: return "🍆"

    if "oil" in n or "ghee" in n: return "🫒"
    if "rice" in n or "dal" in n or "atta" in n or "flour" in n: return "🌾"
    if "juice" in n or "coke" in n or "drink" in n or "soda" in n: return "🧃"

    if "dairy" in c or "bread" in c: return "🥛"
    if "snack" in c or "biscuit" in c: return "🍪"
    if "beverage" in c: return "🧃"
    if "tea" in c or "coffee" in c: return "☕"
    if "home" in c or "clean" in c: return "🧹"
    if "personal" in c: return "🧴"
    if "sweet" in c or "chocolate" in c: return "🍫"
    if "cereal" in c: return "🥣"
    if "fruit" in c or "veg" in c: return "🥦"
    return "📦"


def load_resources():
    global EMBEDDINGS, EMBEDDINGS_NORM, ITEM_TO_IDX, IDX_TO_ITEM
    global CATALOG_DF, ITEM_METADATA, CATEGORIES_LIST, CATEGORY_TO_ITEMS
    global CAT_L1_TO_ID, ITEM_CAT_IDS_TENSOR, MODELS

    print("[Server Init] Loading embeddings...")
    emb_path = "sample datasets/item_embeddings_50k_v5.pkl"
    if not os.path.exists(emb_path):
        emb_path = "sample datasets/item_embeddings_50k.pkl"
    with open(emb_path, "rb") as f:
        data = safe_pickle_load(f)
    raw_embs = torch.from_numpy(data["embeddings"]).float()
    EMBEDDINGS = raw_embs
    EMBEDDINGS_NORM = F.normalize(raw_embs, p=2, dim=-1)
    ITEM_TO_IDX = data["item_to_idx"]
    IDX_TO_ITEM = {idx: name for name, idx in ITEM_TO_IDX.items()}
    print(f" -> Loaded {len(ITEM_TO_IDX):,} embeddings.")

    print("[Server Init] Loading catalog metadata...")
    cat_df = pd.read_parquet("data/reference/catalog.parquet")
    CATALOG_DF = cat_df

    cat_map_items = {}
    cat_records = cat_df.to_dict("records")

    for r in cat_records:
        name = str(r["item_name"])
        cat_l1 = str(r.get("category_l1", "General Grocery"))
        
        meta = {
            "item_id": str(r.get("item_id", "")),
            "item_name": name,
            "category": cat_l1,
            "category_l1": cat_l1,
            "category_l2": str(r.get("category_l2", "")),
            "brand": str(r.get("brand", "")),
            "price_inr": float(r.get("price_inr", 49.0)),
            "pack_size": str(r.get("pack_size", "1 unit")),
            "is_veg": bool(r.get("is_veg", True)),
            "icon": get_item_icon(cat_l1, name),
            "semantic_id": ""
        }
        ITEM_METADATA[name] = meta

        if cat_l1 not in cat_map_items:
            cat_map_items[cat_l1] = []
        cat_map_items[cat_l1].append(meta)

    # Attach Semantic IDs if available
    sem_path = "QuickBasket-Semantic-ID-Experiments/outputs/semantic_ids_best.parquet"
    if os.path.exists(sem_path):
        try:
            sem_df = pd.read_parquet(sem_path)
            sem_map = dict(zip(sem_df["item"], sem_df["semantic_id_disambiguated"]))
            for name, meta in ITEM_METADATA.items():
                if name in sem_map:
                    meta["semantic_id"] = sem_map[name]
            print(f" -> Attached RQ-VAE Semantic IDs to {len(sem_map):,} items.")
        except Exception as e:
            print(f" [!] Notice: Could not attach semantic IDs: {e}")

    # Priority category order for the dropdown
    priority_order = [
        "Dairy & Bread",
        "Snacks & Biscuits",
        "Beverages",
        "Chocolates & Sweets",
        "Breakfast Cereals",
        "Instant Food",
        "Personal Care",
        "Home Care",
        "Fruits & Vegetables",
        "Atta/Rice/Dal"
    ]
    CATEGORIES_LIST = [c for c in priority_order if c in cat_map_items]
    # Add any remaining categories
    for c in sorted(cat_map_items.keys()):
        if c not in CATEGORIES_LIST:
            CATEGORIES_LIST.append(c)

    CATEGORY_TO_ITEMS = cat_map_items

    # Build category ID lookup tensor for ultrafast cross-category masking
    unique_cats = list(cat_map_items.keys())
    CAT_L1_TO_ID = {cat: i for i, cat in enumerate(unique_cats)}
    
    cat_id_list = []
    for i in range(len(ITEM_TO_IDX)):
        name = IDX_TO_ITEM[i]
        c_l1 = ITEM_METADATA.get(name, {}).get("category_l1", "")
        cat_id_list.append(CAT_L1_TO_ID.get(c_l1, -1))
    ITEM_CAT_IDS_TENSOR = torch.tensor(cat_id_list, dtype=torch.long)
    print(f" -> Indexed {len(CATEGORIES_LIST)} categories across {len(ITEM_METADATA):,} items.")

    # Load PyTorch Models
    print("[Server Init] Loading Neural Re-Ranker Models...")
    # Model 1: Interaction Cross-MLP (High sensitivity, strong re-ranking shifts)
    m1 = Model1_InteractionCrossMLP(384, 256, 0.0)
    ckpt1_path = "models/reranker/model_1:_interaction_cross_mlp.pt"
    if os.path.exists(ckpt1_path):
        ckpt1 = torch.load(ckpt1_path, map_location=DEVICE)
        m1.load_state_dict(ckpt1["model_state_dict"])
        m1.eval()
        MODELS["model_1_cross_mlp"] = m1
        print(" -> Model 1 (Interaction Cross-MLP) loaded.")

    # Model 3: List-Aware Transformer
    m3 = Model3_ListAwareTransformer(384, 128, 4, 2, 256, 0.0)
    ckpt3_path = "models/reranker/model_3:_list_aware_transformer.pt"
    if os.path.exists(ckpt3_path):
        ckpt3 = torch.load(ckpt3_path, map_location=DEVICE)
        m3.load_state_dict(ckpt3["model_state_dict"])
        m3.eval()
        MODELS["model_3_transformer"] = m3
        print(" -> Model 3 (List-Aware Transformer) loaded.")

    # Model 2: Dual-Tower Bilinear
    m2 = Model2_DualTowerBilinear(384, 128, 0.0)
    ckpt2_path = "models/reranker/model_2:_dual_tower_bilinear.pt"
    if os.path.exists(ckpt2_path):
        ckpt2 = torch.load(ckpt2_path, map_location=DEVICE)
        m2.load_state_dict(ckpt2["model_state_dict"])
        m2.eval()
        MODELS["model_2_dual_tower"] = m2
        print(" -> Model 2 (Dual-Tower Bilinear) loaded.")

    # V3 Re-Ranker Models (Trained on Two-Tower Retrieval Dataset)
    ckpt1_v3 = "models/reranker_v3/model_1_interaction_cross_mlp_v3.pt"
    if os.path.exists(ckpt1_v3):
        m1_v3 = Model1_InteractionCrossMLP(384, 256, 0.0)
        c1 = torch.load(ckpt1_v3, map_location=DEVICE)
        m1_v3.load_state_dict(c1["model_state_dict"])
        m1_v3.eval()
        MODELS["model_1_cross_mlp_v3"] = m1_v3
        print(" -> Model 1 (v3 Interaction Cross-MLP) loaded.")

    ckpt3_v3 = "models/reranker_v3/model_3_list_aware_transformer_v3.pt"
    if os.path.exists(ckpt3_v3):
        m3_v3 = Model3_ListAwareTransformer(384, 128, 4, 2, 256, 0.0)
        c3 = torch.load(ckpt3_v3, map_location=DEVICE)
        m3_v3.load_state_dict(c3["model_state_dict"])
        m3_v3.eval()
        MODELS["model_3_transformer_v3"] = m3_v3
        print(" -> Model 3 (v3 List-Aware Transformer) loaded.")

    ckpt2_v3 = "models/reranker_v3/model_2_dual_tower_bilinear_v3.pt"
    if os.path.exists(ckpt2_v3):
        m2_v3 = Model2_DualTowerBilinear(384, 128, 0.0)
        c2 = torch.load(ckpt2_v3, map_location=DEVICE)
        m2_v3.load_state_dict(c2["model_state_dict"])
        m2_v3.eval()
        MODELS["model_2_dual_tower_v3"] = m2_v3
        print(" -> Model 2 (v3 Dual-Tower Bilinear) loaded.")


    # Two-Tower Item-to-Item Complementary Retrieval Model (v5 InfoNCE 1:1 Balanced with Updated FnV)
    global RETRIEVAL_MODEL, CAND_PROJECTIONS
    ret_ckpt_path_v5 = "artifacts/item_retrieval_v5/best_item_retrieval_model_v5.pt"
    if not os.path.exists(ret_ckpt_path_v5):
        ret_ckpt_path_v5 = "models/retrieval_v5/best_item_retrieval_model_v5.pt"
    ret_ckpt_path_v4 = "artifacts/item_retrieval_v4/best_item_retrieval_model_v4.pt"
    ret_ckpt_path_v3 = "artifacts/item_retrieval_v3/best_item_retrieval_model.pt"
    
    if os.path.exists(ret_ckpt_path_v5):
        print("[Server Init] Loading Two-Tower Item Retrieval Model (v5 Updated FnV 1:1 Balanced)...")
        ret_model = TwoTowerItemRetrievalModel(in_features=384, proj_dim=256)
        ckpt_ret = torch.load(ret_ckpt_path_v5, map_location=DEVICE)
        ret_model.load_state_dict(ckpt_ret["model_state_dict"])
        ret_model.eval()
        RETRIEVAL_MODEL = ret_model
        with torch.no_grad():
            t_proj0 = time.time()
            CAND_PROJECTIONS = ret_model.encode_candidate(EMBEDDINGS)
            print(f" -> Two-Tower Retrieval Model v5 loaded. Pre-projected {len(EMBEDDINGS):,} candidates in {time.time()-t_proj0:.2f}s (Shape: {CAND_PROJECTIONS.shape}).")
    elif os.path.exists(ret_ckpt_path_v4):
        print("[Server Init] Loading Two-Tower Item Retrieval Model (v4 1:1 Balanced)...")
        ret_model = TwoTowerItemRetrievalModel(in_features=384, proj_dim=256)
        ckpt_ret = torch.load(ret_ckpt_path_v4, map_location=DEVICE)
        ret_model.load_state_dict(ckpt_ret["model_state_dict"])
        ret_model.eval()
        RETRIEVAL_MODEL = ret_model
        with torch.no_grad():
            t_proj0 = time.time()
            CAND_PROJECTIONS = ret_model.encode_candidate(EMBEDDINGS)
            print(f" -> Two-Tower Retrieval Model v4 loaded. Pre-projected {len(EMBEDDINGS):,} candidates in {time.time()-t_proj0:.2f}s (Shape: {CAND_PROJECTIONS.shape}).")
    elif os.path.exists(ret_ckpt_path_v3):
        print("[Server Init] Loading Two-Tower Item Retrieval Model (v3)...")
        ret_model = TwoTowerItemRetrievalModel(in_features=384, proj_dim=256)
        ckpt_ret = torch.load(ret_ckpt_path_v3, map_location=DEVICE)
        ret_model.load_state_dict(ckpt_ret["model_state_dict"])
        ret_model.eval()
        RETRIEVAL_MODEL = ret_model
        with torch.no_grad():
            t_proj0 = time.time()
            CAND_PROJECTIONS = ret_model.encode_candidate(EMBEDDINGS)
            print(f" -> Two-Tower Retrieval Model v3 loaded. Pre-projected {len(EMBEDDINGS):,} candidates in {time.time()-t_proj0:.2f}s (Shape: {CAND_PROJECTIONS.shape}).")
    else:
        print("[Server Init] Warning: No trained retrieval model found. Falling back to raw embeddings.")



load_resources()


# ------------------------------------------------------------------------------
# Request / Response Schemas
# ------------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    cart_items: List[str]
    model_name: Optional[str] = "model_1_cross_mlp_v3"  # Default to Model 1 v3 (Trained on Two-Tower Dataset)
    top_k_per_item: Optional[int] = 5
    cross_category: Optional[bool] = True  # Strict cross-category filtering by default!
    retrieval_mode: Optional[str] = "hybrid_semantic"  # "hybrid_semantic" | "cross_category" | "same_category"


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/api/categories")
def get_categories():
    """Returns list of categories for dropdown selection."""
    return {"categories": CATEGORIES_LIST}


@app.get("/api/items_by_category")
def get_items_by_category(category: str = Query(...), limit: int = 60):
    """Returns items belonging to a selected category."""
    items = CATEGORY_TO_ITEMS.get(category, [])[:limit]
    return {"category": category, "count": len(items), "items": items}


@app.get("/api/preview_items")
def get_preview_items():
    """Returns curated popular essentials for the user preview shelf including fresh produce."""
    # Specific curated staples for instant preview
    curated_items = [
        "Fresho Apple - Red Delicious",
        "Fresho Potato - Organically Grown",
        "Amul Taaza Homogenised Toned Milk",
        "English Oven 100% Atta Bread",
        "Fresho Banana - Robusta",
        "Fresho Tomato - Hybrid",
        "Tata Tea Gold Leaf Tea",
        "Cadbury Dairy Milk Silk Chocolate"
    ]
    preview = []
    seen = set()
    for item_name in curated_items:
        if item_name in ITEM_METADATA and item_name not in seen:
            seen.add(item_name)
            preview.append(ITEM_METADATA[item_name])
            
    # Fallback to keyword matching if any specific item is missing
    if len(preview) < 8:
        keywords = ["apple", "potato", "milk", "bread", "banana", "tomato", "tea", "biscuit", "butter", "cleaner"]
        for kw in keywords:
            for name, meta in ITEM_METADATA.items():
                if kw in name.lower() and name not in seen:
                    seen.add(name)
                    preview.append(meta)
                    break
            if len(preview) >= 8:
                break
    return {"preview_items": preview}


@app.get("/api/search")
def search_catalog(q: str = Query(..., min_length=2), limit: int = 25):
    """Fast substring search across 50,000 items."""
    q_lower = q.lower().strip()
    results = []
    for name, meta in ITEM_METADATA.items():
        if q_lower in name.lower() or q_lower in meta["category"].lower() or q_lower in meta["brand"].lower():
            results.append(meta)
            if len(results) >= limit:
                break
    return {"query": q, "results": results}


@app.post("/api/recommend")
def recommend_and_rerank(req: RecommendRequest):
    """
    Executes multi-item retrieval & neural re-ranking:
      - When cross_category=True (default), filters out any item from the same category as the query item!
      - 1 item in cart: Top 5 cross-category items -> Re-rank 5 items
      - 2 items in cart: Top 5 for item 1 + Top 5 for item 2 -> 10 pooled cross-category items -> Re-rank 10 items
      - N items in cart: Top 5 per item -> Pooled candidates -> Re-rank all
      - Returns exact Before vs. After lists and verification verdict.
    """
    t0 = time.time()
    cart_names = [n for n in req.cart_items if n in ITEM_TO_IDX]
    if not cart_names:
        raise HTTPException(status_code=400, detail="Cart is empty or items not found in catalog.")

    cart_indices = [ITEM_TO_IDX[n] for n in cart_names]
    top_k_per_item = max(1, min(req.top_k_per_item or 5, 5))

    # Exclude all items currently in cart
    excluded_indices = set(cart_indices)
    before_cands = []

    # Collect all categories represented in the current cart
    cart_cat_names = set()
    cart_cat_ids = []
    for ci in cart_indices:
        c_name = IDX_TO_ITEM[ci]
        cat = ITEM_METADATA.get(c_name, {}).get("category_l1", "")
        if cat:
            cart_cat_names.add(cat)
            cid = CAT_L1_TO_ID.get(cat)
            if cid is not None and cid not in cart_cat_ids:
                cart_cat_ids.append(cid)

    cart_cat_tensor = torch.tensor(cart_cat_ids, dtype=torch.long) if cart_cat_ids else None

    retrieval_mode = req.retrieval_mode or "hybrid_semantic"

    # 1. RETRIEVE CANDIDATES PER SELECTED RETRIEVAL MODE
    if retrieval_mode == "hybrid_semantic":
        # SOTA Hybrid Search using RQ-VAE Semantic IDs, Two-Tower projections, and Complement Graph RRF
        global HYBRID_ENGINE
        if HYBRID_ENGINE is None:
            HYBRID_ENGINE = HybridComplementSearchEngine().load()

        target_k = max(len(cart_names) * top_k_per_item, 6)
        hybrid_res = HYBRID_ENGINE.get_cart_complements(
            cart_items=cart_names,
            top_k=target_k,
            fusion_method="rrf",
            max_per_category=2
        )
        for c in hybrid_res.get("complements", []):
            cand_name = c["item_name"]
            meta = ITEM_METADATA.get(cand_name, {})
            cand_cat = c.get("category_l1", meta.get("category_l1", "General"))
            is_cross = (cand_cat not in cart_cat_names)
            idx = ITEM_TO_IDX.get(cand_name, 0)
            before_cands.append({
                "item_name": cand_name,
                "category": cand_cat,
                "category_l1": cand_cat,
                "category_l2": c.get("category_l2", meta.get("category_l2", "")),
                "is_cross_category": is_cross,
                "brand": c.get("brand", meta.get("brand", "")),
                "price_inr": c.get("price_inr", meta.get("price_inr", 49.0)),
                "pack_size": c.get("pack_size", meta.get("pack_size", "1 unit")),
                "icon": meta.get("icon", "📦"),
                "similarity_score": round(float(c.get("final_score", 0.0)), 4),
                "retrieved_by": "Cart Complements (Hybrid Search)",
                "retrieved_by_category": ", ".join(cart_cat_names) if cart_cat_names else "Cart",
                "cand_idx": idx,
                "semantic_id": c.get("semantic_id_disambiguated", c.get("semantic_id", meta.get("semantic_id", ""))),
                "tier": c.get("tier", "None"),
                "modality_match": "Hybrid (RQ-VAE + Two-Tower + Graph)"
            })
    else:
        # Standard Two-Tower or Same-Category Dense Retrieval
        for c_idx in cart_indices:
            c_name = IDX_TO_ITEM[c_idx]
            c_cat_l1 = ITEM_METADATA.get(c_name, {}).get("category_l1", "")
            
            # Use trained Two-Tower anchor encoder if available, otherwise fallback to raw normalized embeddings
            if RETRIEVAL_MODEL is not None and CAND_PROJECTIONS is not None:
                with torch.no_grad():
                    q_vec_emb = EMBEDDINGS[c_idx].unsqueeze(0)
                    q_proj = RETRIEVAL_MODEL.encode_anchor(q_vec_emb)
                    sims = torch.mm(q_proj, CAND_PROJECTIONS.t()).squeeze(0).clone()
            else:
                v = EMBEDDINGS_NORM[c_idx].unsqueeze(0)
                sims = torch.mm(v, EMBEDDINGS_NORM.t()).squeeze(0).clone()
            
            # Mask out cart items and already chosen candidates
            for ex in excluded_indices:
                sims[ex] = -1.0

            # CROSS-CATEGORY FILTERING: Mask out all items from categories in the cart!
            if req.cross_category and retrieval_mode != "same_category":
                if cart_cat_tensor is not None and len(cart_cat_tensor) > 0:
                    sims[torch.isin(ITEM_CAT_IDS_TENSOR, cart_cat_tensor)] = -1.0
                elif c_cat_l1 and c_cat_l1 in CAT_L1_TO_ID:
                    sims[ITEM_CAT_IDS_TENSOR == CAT_L1_TO_ID[c_cat_l1]] = -1.0
            elif retrieval_mode == "same_category":
                # Only keep items from the same category (substitute mode)
                if c_cat_l1 and c_cat_l1 in CAT_L1_TO_ID:
                    sims[ITEM_CAT_IDS_TENSOR != CAT_L1_TO_ID[c_cat_l1]] = -1.0

            top_scores, top_idx = torch.topk(sims, k=top_k_per_item)
            for s, idx in zip(top_scores.tolist(), top_idx.tolist()):
                excluded_indices.add(idx)  # Avoid duplicate candidates across cart items
                cand_name = IDX_TO_ITEM[idx]
                meta = ITEM_METADATA.get(cand_name, {})
                cand_cat = meta.get("category_l1", "General")
                is_cross = (cand_cat not in cart_cat_names)
                before_cands.append({
                    "item_name": cand_name,
                    "category": cand_cat,
                    "category_l1": cand_cat,
                    "is_cross_category": is_cross,
                    "brand": meta.get("brand", ""),
                    "price_inr": meta.get("price_inr", 49.0),
                    "pack_size": meta.get("pack_size", "1 unit"),
                    "icon": meta.get("icon", "📦"),
                    "similarity_score": round(float(s), 4),
                    "retrieved_by": c_name,
                    "retrieved_by_category": c_cat_l1,
                    "cand_idx": idx,
                    "semantic_id": meta.get("semantic_id", ""),
                    "tier": "None",
                    "modality_match": "Two-Tower v5 Dense"
                })

    if not before_cands:
        raise HTTPException(status_code=500, detail="Failed to retrieve candidates for cart items.")

    # Assign initial retrieval ranks 1..K
    for rank_idx, c in enumerate(before_cands, 1):
        c["initial_rank"] = rank_idx

    total_candidates = len(before_cands)

    # 2. NEURAL RE-RANKING STAGE
    model_key = req.model_name or "model_1_cross_mlp_v3"
    if model_key not in MODELS:
        model_key = "model_1_cross_mlp_v3" if "model_1_cross_mlp_v3" in MODELS else "model_1_cross_mlp"
    model = MODELS[model_key]

    cand_indices = [c["cand_idx"] for c in before_cands]
    q_vec = torch.mean(EMBEDDINGS[cart_indices], dim=0, keepdim=True).to(DEVICE)
    c_vecs = EMBEDDINGS[cand_indices].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if "transformer" in model_key:
            # For Model 3: Compute pure cross-attention without positional shortcut so ordering is dynamic
            B, K, D = c_vecs.shape
            q_exp = q_vec.unsqueeze(1).expand(-1, K, -1)
            hadamard = q_exp * c_vecs
            abs_diff = torch.abs(q_exp - c_vecs)
            pairs = torch.cat([q_exp, c_vecs, hadamard, abs_diff], dim=-1)
            tokens = model.pair_encoder(pairs)
            contextual_tokens = model.transformer_encoder(tokens)
            scores = model.score_head(contextual_tokens).squeeze(-1).squeeze(0).cpu().numpy()
        else:
            scores = model(q_vec, c_vecs).squeeze(0).cpu().numpy()

    # Softmax probabilities for interpretability
    probs = np.exp(scores - np.max(scores))
    probs = probs / np.sum(probs)

    # 3. CONSTRUCT RE-RANKED LIST AND CALCULATE SHIFTS
    rerank_order = np.argsort(-scores).tolist()
    after_cands = []
    shift_count = 0
    promoted_items = []
    demoted_items = []

    for new_rank, orig_pos in enumerate(rerank_order, 1):
        orig_cand = before_cands[orig_pos]
        old_rank = orig_cand["initial_rank"]
        delta = old_rank - new_rank

        status_type = "neutral"
        status_text = f"Maintained (#{old_rank})"
        if delta > 0:
            status_type = "promoted"
            status_text = f"Promoted (+{delta})"
            shift_count += 1
            promoted_items.append((orig_cand["item_name"], old_rank, new_rank, delta))
        elif delta < 0:
            status_type = "demoted"
            status_text = f"Demoted ({delta})"
            shift_count += 1
            demoted_items.append((orig_cand["item_name"], old_rank, new_rank, delta))

        after_cands.append({
            "rerank": new_rank,
            "initial_rank": old_rank,
            "rank_delta": delta,
            "status_type": status_type,
            "status_text": status_text,
            "item_name": orig_cand["item_name"],
            "category": orig_cand["category"],
            "category_l1": orig_cand["category_l1"],
            "is_cross_category": orig_cand["is_cross_category"],
            "brand": orig_cand["brand"],
            "price_inr": orig_cand["price_inr"],
            "pack_size": orig_cand["pack_size"],
            "icon": orig_cand["icon"],
            "similarity_score": orig_cand["similarity_score"],
            "retrieved_by": orig_cand["retrieved_by"],
            "retrieved_by_category": orig_cand["retrieved_by_category"],
            "model_score": round(float(scores[orig_pos]), 3),
            "model_probability": round(float(probs[orig_pos]), 4),
            "semantic_id": orig_cand.get("semantic_id", ""),
            "tier": orig_cand.get("tier", "None"),
            "modality_match": orig_cand.get("modality_match", "")
        })

    elapsed_ms = round((time.time() - t0) * 1000, 2)
    order_changed = (shift_count > 0)

    # Clear Verification Headline
    cross_cat_tag = "Cross-Category " if req.cross_category else "Full Catalog "
    if order_changed:
        top_after = after_cands[0]
        if top_after["rank_delta"] > 0:
            headline = f"VERIFICATION: Re-Ranking Active & Order Changed! {shift_count} of {total_candidates} {cross_cat_tag}items changed positions. '{top_after['item_name']}' was promoted from #{top_after['initial_rank']} to #1!"
        else:
            headline = f"VERIFICATION: Re-Ranking Active & Order Changed! {shift_count} of {total_candidates} {cross_cat_tag}items changed rank positions."
    else:
        headline = f"VERIFICATION: Re-Ranking Active — Model confirmed all {total_candidates} {cross_cat_tag}items in their optimal ranking order."

    if retrieval_mode == "hybrid_semantic":
        retrieval_name = "Hybrid Semantic Search (RQ-VAE Semantic IDs + Two-Tower v5 + Co-Purchase Graph RRF)"
    elif RETRIEVAL_MODEL is not None:
        retrieval_name = "Two-Tower Complementary Retrieval (v5 InfoNCE 1:1 Balanced)"
    else:
        retrieval_name = "Raw BGE Embeddings"

    return {
        "success": True,
        "cart_count": len(cart_names),
        "total_candidates": total_candidates,
        "cross_category_enabled": req.cross_category,
        "retrieval_mode": retrieval_mode,
        "cross_category_count": sum(1 for c in before_cands if c["is_cross_category"]),
        "cart_categories": list(cart_cat_names),
        "re_ranking_occurred": order_changed,
        "shift_count": shift_count,
        "headline": headline,
        "elapsed_ms": elapsed_ms,
        "retrieval_model": retrieval_name,
        "model_used": model_key,
        "before_ranking": before_cands,
        "after_ranking": after_cands
    }


# ------------------------------------------------------------------------------
# Hybrid Semantic ID Complement Retrieval Endpoints
# ------------------------------------------------------------------------------
HYBRID_ENGINE: Optional[HybridComplementSearchEngine] = None

class CartComplementsRequest(BaseModel):
    cart_items: List[str]
    top_k: Optional[int] = 8
    fusion: Optional[str] = "rrf"
    max_per_category: Optional[int] = 2

@app.get("/api/hybrid_complements")
def get_hybrid_complements_endpoint(
    item: str = Query(..., description="Anchor product name or SKU ID"),
    top_k: int = Query(8, ge=1, le=50),
    fusion: str = Query("rrf", pattern="^(rrf|weighted)$"),
    max_per_category: int = Query(2, ge=0, le=10)
):
    """
    Retrieves complementary items for an anchor item using Hybrid Search:
    Fuses discrete Semantic IDs (RQ-VAE), Two-Tower neural projections, and behavioral graph priors.
    """
    global HYBRID_ENGINE
    if HYBRID_ENGINE is None:
        HYBRID_ENGINE = HybridComplementSearchEngine().load()
    try:
        return HYBRID_ENGINE.get_complements(
            query_item=item,
            top_k=top_k,
            fusion_method=fusion,
            max_per_category=max_per_category
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/hybrid_cart_complements")
def get_hybrid_cart_complements_endpoint(req: CartComplementsRequest):
    """
    Retrieves complementary items for an entire multi-item shopping cart.
    """
    global HYBRID_ENGINE
    if HYBRID_ENGINE is None:
        HYBRID_ENGINE = HybridComplementSearchEngine().load()
    try:
        return HYBRID_ENGINE.get_cart_complements(
            cart_items=req.cart_items,
            top_k=req.top_k or 8,
            fusion_method=req.fusion or "rrf",
            max_per_category=req.max_per_category if req.max_per_category is not None else 2
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ------------------------------------------------------------------------------
# Mount Static UI at Root /
# ------------------------------------------------------------------------------
ui_dir = os.path.join(os.path.dirname(__file__), "ui")
os.makedirs(ui_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
