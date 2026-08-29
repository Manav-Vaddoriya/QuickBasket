import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Set, Optional

def load_cached_embeddings(pkl_path: str = os.path.join("sample datasets", "item_embeddings_50k.pkl")) -> Optional[Dict]:
    """
    Loads precomputed BGE item embeddings pickle file if it exists.
    """
    if os.path.exists(pkl_path):
        print(f"Loading precomputed item embeddings from '{pkl_path}'...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded embeddings for {len(data.get('item_to_idx', {}))} items.")
        return data
    return None

def load_and_prepare_data(
    data_path: str = os.path.join("sample datasets", "triplet_dataset_v3.csv"),
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    include_same_category: bool = True,
    positive_sources: Optional[List[str]] = None,
    negative_sources: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Loads triplet_dataset_v3.csv, applies filters, and performs CATEGORY-STRATIFIED anchor splitting.
    Guarantees every category is represented in Train, Val, and Test splits.
    
    Returns:
        (train_df, val_df, test_df, catalog_df, item_to_category_dict)
    """
    df = pd.read_csv(data_path)
    
    if not include_same_category and 'same_category' in df.columns:
        df = df[df['same_category'] == False]
        
    if positive_sources is not None:
        pos_mask = (df['label'] == 1) & (df['sample_type'].isin(positive_sources))
        neg_mask = (df['label'] == 0)
        df = df[pos_mask | neg_mask]
        
    if negative_sources is not None:
        pos_mask = (df['label'] == 1)
        neg_mask = (df['label'] == 0) & (df['sample_type'].isin(negative_sources))
        df = df[pos_mask | neg_mask]
        
    df = df.reset_index(drop=True)
    
    # Extract Full Item Catalog
    anchors = df[['anchor_item', 'anchor_category']].rename(columns={'anchor_item': 'item_name', 'anchor_category': 'category'})
    candidates = df[['candidate_item', 'candidate_category']].rename(columns={'candidate_item': 'item_name', 'candidate_category': 'category'})
    catalog_df = pd.concat([anchors, candidates]).drop_duplicates(subset=['item_name']).sort_values('item_name').reset_index(drop=True)
    item_to_category = dict(zip(catalog_df['item_name'], catalog_df['category']))
    
    # CATEGORY-STRATIFIED Anchor Splitting
    rng = random.Random(seed)
    train_anchors, val_anchors, test_anchors = set(), set(), set()

    # Group anchors by category
    anchor_cat_df = df[['anchor_item', 'anchor_category']].drop_duplicates()
    for cat, group in anchor_cat_df.groupby('anchor_category'):
        cat_anchors = list(group['anchor_item'])
        rng.shuffle(cat_anchors)
        
        n_cat = len(cat_anchors)
        if n_cat == 1:
            train_anchors.add(cat_anchors[0])
        elif n_cat == 2:
            train_anchors.add(cat_anchors[0])
            val_anchors.add(cat_anchors[1])
        else:
            n_tr = max(1, int(round(n_cat * train_ratio)))
            n_va = max(1, int(round(n_cat * val_ratio)))
            
            # Adjust if split exceeds total
            if n_tr + n_va >= n_cat:
                n_tr = max(1, n_cat - 2)
                n_va = 1
                
            train_anchors.update(cat_anchors[:n_tr])
            val_anchors.update(cat_anchors[n_tr:n_tr + n_va])
            test_anchors.update(cat_anchors[n_tr + n_va:])

    train_df = df[df['anchor_item'].isin(train_anchors)].reset_index(drop=True)
    val_df = df[df['anchor_item'].isin(val_anchors)].reset_index(drop=True)
    test_df = df[df['anchor_item'].isin(test_anchors)].reset_index(drop=True)
    
    # Verify category coverage
    train_cats = train_df['anchor_category'].nunique()
    val_cats = val_df['anchor_category'].nunique()
    test_cats = test_df['anchor_category'].nunique()
    
    print(f"Category-Stratified Data Prepared from '{data_path}':")
    print(f"  - Total Dataset Rows: {len(df)}")
    print(f"  - Total Catalog Items: {len(catalog_df)}")
    print(f"  - Train Set: {len(train_df)} rows ({len(train_anchors)} anchors across {train_cats} categories)")
    print(f"  - Val Set  : {len(val_df)} rows ({len(val_anchors)} anchors across {val_cats} categories)")
    print(f"  - Test Set : {len(test_df)} rows ({len(test_anchors)} anchors across {test_cats} categories)")
    
    return train_df, val_df, test_df, catalog_df, item_to_category


class PairDataset(Dataset):
    """
    Dataset for Positive Pairs (anchor, positive).
    """
    def __init__(self, df: pd.DataFrame):
        pos_df = df[df['label'] == 1].reset_index(drop=True)
        self.records = pos_df[['anchor_item', 'candidate_item']].to_dict(orient='records')
        
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        row = self.records[idx]
        return {
            'anchor_item': row['anchor_item'],
            'positive_item': row['candidate_item']
        }


class TripletDataset(Dataset):
    """
    Dataset for Triplets (anchor, positive, negative).
    """
    def __init__(self, df: pd.DataFrame, seed: int = 42):
        self.rng = random.Random(seed)
        pos_df = df[df['label'] == 1]
        neg_df = df[df['label'] == 0]
        
        anchor_to_negs = {}
        for anchor, group in neg_df.groupby('anchor_item'):
            anchor_to_negs[anchor] = list(group['candidate_item'].unique())
            
        all_candidate_items = list(df['candidate_item'].unique())
        self.triplets = []
        
        for idx, row in pos_df.iterrows():
            anchor = row['anchor_item']
            pos_item = row['candidate_item']
            neg_candidates = anchor_to_negs.get(anchor, [])
            
            if neg_candidates:
                neg_item = self.rng.choice(neg_candidates)
            else:
                neg_item = self.rng.choice(all_candidate_items)
                while neg_item == anchor or neg_item == pos_item:
                    neg_item = self.rng.choice(all_candidate_items)
                    
            self.triplets.append({
                'anchor_item': anchor,
                'positive_item': pos_item,
                'negative_item': neg_item
            })
            
    def __len__(self):
        return len(self.triplets)
        
    def __getitem__(self, idx):
        return self.triplets[idx]
