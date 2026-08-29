import os
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple

class ItemRecommender:
    """
    Two-Tower & Siamese Model Product Recommendation Predictor.
    Given an input product name, uses the model to predict top-K complementary item recommendations
    as well as bottom-K least probable recommendations.
    """
    def __init__(self, model: torch.nn.Module, catalog_df: pd.DataFrame, device: Optional[torch.device] = None, pkl_path: str = os.path.join("sample datasets", "item_embeddings_50k.pkl")):
        self.model = model
        self.catalog_df = catalog_df.reset_index(drop=True)
        self.catalog_items = list(self.catalog_df['item_name'])
        self.item_to_category = dict(zip(self.catalog_df['item_name'], self.catalog_df['category']))
        self.item_to_idx = {name: idx for idx, name in enumerate(self.catalog_items)}
        self.pkl_path = pkl_path
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
        self._initialize_catalog_vectors()
        
    def _initialize_catalog_vectors(self):
        if os.path.exists(self.pkl_path):
            try:
                with open(self.pkl_path, 'rb') as f:
                    cached_data = pickle.load(f)
                cached_matrix = cached_data.get('embeddings')
                cached_item_to_idx = cached_data.get('item_to_idx')
                
                catalog_matrix = np.zeros((len(self.catalog_items), cached_matrix.shape[1]), dtype='float32')
                for i, name in enumerate(self.catalog_items):
                    idx = cached_item_to_idx.get(name)
                    if idx is not None:
                        catalog_matrix[i] = cached_matrix[idx]
                        
                self.catalog_matrix = catalog_matrix
                return
            except Exception as e:
                pass

        embeddings = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(self.catalog_items), batch_size):
                batch_texts = self.catalog_items[i : i + batch_size]
                if hasattr(self.model, 'encode_candidate'):
                    emb = self.model.encode_candidate(batch_texts, device=self.device)
                else:
                    emb = self.model.encode(batch_texts, device=self.device)
                embeddings.append(emb.cpu().numpy())
                
        self.catalog_matrix = np.vstack(embeddings).astype('float32')

    def predict(self, product_name: str, top_k: int = 10, filter_same_category: bool = False) -> pd.DataFrame:
        """
        Input: product_name (str) e.g., 'Amul Milk'
        Params:
            - top_k: number of recommended items to return
            - filter_same_category: if True, strictly excludes items from the same category as input product
        Output: DataFrame of Top-K model predicted complementary recommendations.
        """
        if product_name not in self.item_to_idx:
            raise ValueError(f"Product '{product_name}' not found in item catalog!")
            
        product_cat = self.item_to_category[product_name]
        
        with torch.no_grad():
            if hasattr(self.model, 'encode_anchor'):
                product_vec = self.model.encode_anchor([product_name], device=self.device).cpu().numpy().astype('float32')
            else:
                product_vec = self.model.encode([product_name], device=self.device).cpu().numpy().astype('float32')
                
        scores = np.dot(self.catalog_matrix, product_vec[0]).copy()
        
        # Always exclude self-product
        if product_name in self.item_to_idx:
            scores[self.item_to_idx[product_name]] = -np.inf
            
        # Optional: strictly exclude all items in the same category
        if filter_same_category:
            for idx, item in enumerate(self.catalog_items):
                if self.item_to_category[item] == product_cat:
                    scores[idx] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        top_scores = scores[top_indices]
        
        results = []
        rank = 1
        for idx, score in zip(top_indices, top_scores):
            if np.isneginf(score):
                continue
            rec_item = self.catalog_items[idx]
            if rec_item == product_name:
                continue
            if rank > top_k:
                break
                
            rec_cat = self.item_to_category[rec_item]
            results.append({
                'rank': rank,
                'recommended_item': rec_item,
                'category': rec_cat,
                'confidence_score': round(float(score), 4),
                'same_category': (rec_cat == product_cat)
            })
            rank += 1
            
        return pd.DataFrame(results)

    def predict_least_probable(self, product_name: str, bottom_k: int = 10, filter_same_category: bool = False) -> pd.DataFrame:
        """
        Input: product_name (str)
        Output: DataFrame of bottom-K items with lowest similarity score (least probable recommendations).
        """
        if product_name not in self.item_to_idx:
            raise ValueError(f"Product '{product_name}' not found in item catalog!")
            
        product_cat = self.item_to_category[product_name]
        
        with torch.no_grad():
            if hasattr(self.model, 'encode_anchor'):
                product_vec = self.model.encode_anchor([product_name], device=self.device).cpu().numpy().astype('float32')
            else:
                product_vec = self.model.encode([product_name], device=self.device).cpu().numpy().astype('float32')
                
        scores = np.dot(self.catalog_matrix, product_vec[0]).copy()
        
        # Exclude self-product
        if product_name in self.item_to_idx:
            scores[self.item_to_idx[product_name]] = np.inf
            
        if filter_same_category:
            for idx, item in enumerate(self.catalog_items):
                if self.item_to_category[item] == product_cat:
                    scores[idx] = np.inf

        bottom_indices = np.argsort(scores)[:bottom_k * 2]
        bottom_scores = scores[bottom_indices]
        
        results = []
        rank = 1
        for idx, score in zip(bottom_indices, bottom_scores):
            if np.isposinf(score):
                continue
            rec_item = self.catalog_items[idx]
            if rec_item == product_name:
                continue
            if rank > bottom_k:
                break
                
            rec_cat = self.item_to_category[rec_item]
            results.append({
                'rank': rank,
                'least_probable_item': rec_item,
                'category': rec_cat,
                'confidence_score': round(float(score), 4),
                'same_category': (rec_cat == product_cat)
            })
            rank += 1
            
        return pd.DataFrame(results)

    def recommend(self, product_name: str, top_k: int = 10, filter_same_category: bool = False) -> pd.DataFrame:
        return self.predict(product_name, top_k=top_k, filter_same_category=filter_same_category)

# Alias for backward compatibility
RecommendationServing = ItemRecommender
