import os
import ast
import json
import random
import time
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
K_SIMILAR = 1               # Top-1 similar category for cross-category positives
TOP_K_DISSIMILAR = 5        # Top-5 dissimilar categories for hard negatives
N_CROSS_POSITIVE = 3        # Number of cross-category positives from nearest sub-cluster pool
NEG_TO_POS_RATIO = 2.0      # Target 2 negatives per 1 positive sample (1:2 ratio)
SIMILARITY_SAFETY_FLOOR = 0.55 # Reject hard negatives with similarity > 0.55
ITEMS_PER_CLUSTER = 15      # Target item count per sub-cluster in Stage 2
MIN_CLUSTERS = 2            # Minimum sub-clusters per category (if items >= MIN_ITEMS_FOR_CLUSTERING)
MAX_CLUSTERS = 10           # Maximum sub-clusters per category
MIN_ITEMS_FOR_CLUSTERING = 4 # Fall back to 1 cluster if category has fewer items
RANDOM_SEED = 42

INPUT_FILE = "final_pair_data(2).csv"
OUTPUT_FILE_V3 = os.path.join("sample datasets", "triplet_dataset_v3.csv")
CACHE_EMBEDDINGS_FILE = os.path.join("sample datasets", "item_embeddings_50k.pkl")

def set_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

def parse_list(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        try:
            return json.loads(val)
        except Exception:
            return []

class PoolCycler:
    """
    Utility class for no-duplicate sampling from candidate pools.
    Hand out items sequentially after shuffling; reshuffle when exhausted.
    """
    def __init__(self, items, seed=RANDOM_SEED):
        self.original_items = list(items)
        self.rng = random.Random(seed)
        self.pool = []
        self._reset()

    def _reset(self):
        self.pool = list(self.original_items)
        self.rng.shuffle(self.pool)

    def next(self):
        if not self.original_items:
            return None
        if not self.pool:
            self._reset()
        return self.pool.pop()

def main():
    set_seeds()
    print("=== Stage 1: Loading & Parsing Data from final_pair_data(2).csv ===")
    t0 = time.time()
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from '{INPUT_FILE}'")

    source_item_col = 'item' if 'item' in df.columns else 'source_item'
    source_cat_col = 'category' if 'category' in df.columns else 'source_category'

    df['gold_parsed'] = df['gold_items'].apply(parse_list)
    df['silver_parsed'] = df['silver_items'].apply(parse_list)
    df['bronze_parsed'] = df['bronze_items'].apply(parse_list)

    print(f"Data parsed in {round(time.time() - t0, 2)}s")

    print("\n=== Stage 2: Category Identity & Embeddings (BGE) ===")
    cat_to_items = df.groupby(source_cat_col)[source_item_col].unique().to_dict()
    all_unique_items = sorted(list(df[source_item_col].unique()))
    item_to_cat = dict(zip(df[source_item_col], df[source_cat_col]))

    print(f"Total unique source items: {len(all_unique_items)}")
    print(f"Total unique categories: {len(cat_to_items)}")

    item_embeddings = None
    item_to_idx = {}

    if os.path.exists(CACHE_EMBEDDINGS_FILE):
        print(f"Loading cached BGE embeddings from {CACHE_EMBEDDINGS_FILE}...")
        with open(CACHE_EMBEDDINGS_FILE, 'rb') as f:
            cache = pickle.load(f)
        cached_embeddings = cache['embeddings']
        cached_item_to_idx = cache['item_to_idx']

        # Check if all items are in cache
        missing_items = [it for it in all_unique_items if it not in cached_item_to_idx]
        if not missing_items:
            print(f"All {len(all_unique_items)} items found in embedding cache.")
            item_embeddings = cached_embeddings
            item_to_idx = cached_item_to_idx
        else:
            print(f"Found {len(missing_items)} missing items. Computing embeddings...")
            model = SentenceTransformer(BGE_MODEL_NAME)
            missing_embeddings = model.encode(missing_items, normalize_embeddings=True, show_progress_bar=True, batch_size=256)
            
            combined_items = list(cached_item_to_idx.keys()) + missing_items
            combined_embeddings = np.vstack([cached_embeddings, missing_embeddings])
            item_to_idx = {item: idx for idx, item in enumerate(combined_items)}
            item_embeddings = combined_embeddings

            with open(CACHE_EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump({'embeddings': item_embeddings, 'item_to_idx': item_to_idx}, f)
            print(f"Updated and saved embeddings to {CACHE_EMBEDDINGS_FILE}")
    else:
        print(f"Generating embeddings for {len(all_unique_items)} items using {BGE_MODEL_NAME}...")
        model = SentenceTransformer(BGE_MODEL_NAME)
        item_embeddings = model.encode(all_unique_items, normalize_embeddings=True, show_progress_bar=True, batch_size=256)
        item_to_idx = {item: idx for idx, item in enumerate(all_unique_items)}
        with open(CACHE_EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump({'embeddings': item_embeddings, 'item_to_idx': item_to_idx}, f)
        print(f"Saved embeddings to {CACHE_EMBEDDINGS_FILE}")

    # Compute Category Mean Vectors (normalized)
    cat_names = sorted(list(cat_to_items.keys()))
    cat_embeddings = {}
    for cat in cat_names:
        items = cat_to_items[cat]
        indices = [item_to_idx[it] for it in items]
        mean_vec = item_embeddings[indices].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        cat_embeddings[cat] = mean_vec

    cat_matrix = np.array([cat_embeddings[cat] for cat in cat_names])

    print("\n=== Stage 3: Stage 1 Clustering & Distance-Weighted Dissimilarity ===")
    dist_matrix = squareform(pdist(cat_matrix, metric='cosine'))
    cat_to_idx = {cat: idx for idx, cat in enumerate(cat_names)}

    n_clusters_stage1 = max(2, len(cat_names) // 3)
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters_stage1, metric='cosine', linkage='average')
    labels = agg_clustering.fit_predict(cat_matrix)

    print("Exploratory Category Agglomerative Clusters:")
    for cat, label in zip(cat_names, labels):
        print(f"  - [{label}] {cat}")

    def most_similar(category, k=K_SIMILAR):
        idx = cat_to_idx[category]
        distances = dist_matrix[idx].copy()
        distances[idx] = np.inf
        top_k_indices = np.argsort(distances)[:k]
        return [cat_names[i] for i in top_k_indices]

    def top_k_dissimilar_with_weights(category, k=TOP_K_DISSIMILAR):
        idx = cat_to_idx[category]
        distances = dist_matrix[idx].copy()
        distances[idx] = -np.inf
        top_k_indices = np.argsort(distances)[::-1][:k]
        top_cats = [cat_names[i] for i in top_k_indices]
        top_dists = np.array([distances[i] for i in top_k_indices])
        weights = top_dists / top_dists.sum()
        return top_cats, weights

    top_similar_map = {cat: most_similar(cat, k=K_SIMILAR)[0] for cat in cat_names}
    top_dissimilar_weighted_map = {cat: top_k_dissimilar_with_weights(cat, k=TOP_K_DISSIMILAR) for cat in cat_names}

    print("\nCategory Top-1 Similar & Top-5 Dissimilar (Weighted) Mapping:")
    for cat in cat_names:
        top5_c, top5_w = top_dissimilar_weighted_map[cat]
        w_str = ", ".join([f"{c}({round(w,2)})" for c, w in zip(top5_c, top5_w)])
        print(f"  '{cat}' -> Similar: '{top_similar_map[cat]}' | Dissimilar Top-5: [{w_str}]")

    print("\n=== Stage 4: Stage 2 Clustering (Item-wise within each category) ===")
    cat_subclusters = {}

    for cat in cat_names:
        items = list(cat_to_items[cat])
        n_items = len(items)
        if n_items < MIN_ITEMS_FOR_CLUSTERING:
            n_c = 1
        else:
            n_c = max(MIN_CLUSTERS, min(MAX_CLUSTERS, int(round(n_items / ITEMS_PER_CLUSTER))))
            n_c = min(n_c, n_items)

        item_vecs = np.array([item_embeddings[item_to_idx[it]] for it in items])
        
        if n_c > 1:
            km = KMeans(n_clusters=n_c, random_state=RANDOM_SEED, n_init=10)
            cluster_labels = km.fit_predict(item_vecs)
        else:
            cluster_labels = np.zeros(n_items, dtype=int)

        clusters_dict = {}
        for c_id in range(n_c):
            c_mask = (cluster_labels == c_id)
            c_items = [items[i] for i in range(n_items) if c_mask[i]]
            c_vecs = item_vecs[c_mask]
            centroid = c_vecs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            clusters_dict[c_id] = {
                "items": c_items,
                "centroid": centroid
            }
        cat_subclusters[cat] = clusters_dict

    def nearest_cluster(anchor_embedding, target_category):
        clusters_dict = cat_subclusters[target_category]
        best_cid = None
        best_sim = -np.inf
        for c_id, c_info in clusters_dict.items():
            sim = float(np.dot(anchor_embedding, c_info["centroid"]))
            if sim > best_sim:
                best_sim = sim
                best_cid = c_id
        return best_cid

    print(f"Generated sub-clusters for all {len(cat_names)} categories.")

    print("\n=== Stage 5: Persistent PoolCycler Initialization ===")
    pool_cyclers = {}
    for cat in cat_names:
        for c_id, c_info in cat_subclusters[cat].items():
            pool_key = (cat, c_id)
            pool_seed = RANDOM_SEED + hash(pool_key) % 100000
            pool_cyclers[pool_key] = PoolCycler(c_info["items"], seed=pool_seed)

    print(f"Initialized {len(pool_cyclers)} persistent PoolCycler pools across sub-clusters.")

    print("\n=== Stage 6: Candidate Dataset Generation ===")
    records = []
    rng = random.Random(RANDOM_SEED)

    hard_neg_cat_counts = {cat: 0 for cat in cat_names}

    for idx, row in df.iterrows():
        anchor = row[source_item_col]
        anchor_cat = row[source_cat_col]
        anchor_emb = item_embeddings[item_to_idx[anchor]]

        target_sim_cat = top_similar_map[anchor_cat]
        sim_cid = nearest_cluster(anchor_emb, target_sim_cat)
        sim_pool_key = (target_sim_cat, sim_cid)

        top5_dissim_cats, top5_weights = top_dissimilar_weighted_map[anchor_cat]

        positives_for_anchor = []
        pos_seen = set([anchor])

        # 1. Gold Positives (All gold items)
        for g_item in row['gold_parsed']:
            if g_item and g_item not in pos_seen:
                positives_for_anchor.append((g_item, 'positive_gold'))
                pos_seen.add(g_item)

        # 2. Silver Positives (All silver items not in gold)
        for s_item in row['silver_parsed']:
            if s_item and s_item not in pos_seen:
                positives_for_anchor.append((s_item, 'positive_silver'))
                pos_seen.add(s_item)

        # 3. Cross-Category Positives (N_CROSS_POSITIVE items)
        cross_count = 0
        for _ in range(len(pool_cyclers[sim_pool_key].original_items) + 10):
            if cross_count >= N_CROSS_POSITIVE:
                break
            cand = pool_cyclers[sim_pool_key].next()
            if cand and cand not in pos_seen:
                positives_for_anchor.append((cand, 'positive_cross_category'))
                pos_seen.add(cand)
                cross_count += 1

        # Emit Positive Rows (label = 1)
        for cand_item, stype in positives_for_anchor:
            cand_idx = item_to_idx.get(cand_item)
            if cand_idx is not None:
                sim_score = float(np.dot(anchor_emb, item_embeddings[cand_idx]))
                cand_cat = item_to_cat.get(cand_item, target_sim_cat)
            else:
                sim_score = 0.0
                cand_cat = target_sim_cat

            records.append({
                'anchor_item': anchor,
                'anchor_category': anchor_cat,
                'candidate_item': cand_item,
                'candidate_category': cand_cat,
                'label': 1,
                'sample_type': stype,
                'similarity_score': round(sim_score, 4),
                'same_category': (cand_cat == anchor_cat)
            })

        # ------------------- NEGATIVE SAMPLES (label = 0) -------------------
        total_positives = len(positives_for_anchor)
        target_total_negatives = int(round(total_positives * NEG_TO_POS_RATIO))

        negatives_for_anchor = []
        neg_seen = set(pos_seen)

        # 1. Soft Negatives (All bronze items)
        for b_item in row['bronze_parsed']:
            if b_item and b_item not in neg_seen:
                negatives_for_anchor.append((b_item, 'soft_negative'))
                neg_seen.add(b_item)

        # 2. Hard Negatives (Distance-weighted Top-5 Dissimilar Category + Nearest Sub-Cluster + Safety Floor)
        needed_hard = max(0, target_total_negatives - len(negatives_for_anchor))
        hard_count = 0

        for _ in range(needed_hard * 10 + 50):
            if hard_count >= needed_hard:
                break
            
            chosen_dissim_cat = rng.choices(top5_dissim_cats, weights=top5_weights, k=1)[0]
            dissim_cid = nearest_cluster(anchor_emb, chosen_dissim_cat)
            dissim_pool_key = (chosen_dissim_cat, dissim_cid)

            cand_item = None
            for attempt in range(5):
                candidate_drawn = pool_cyclers[dissim_pool_key].next()
                if candidate_drawn and candidate_drawn not in neg_seen:
                    c_idx = item_to_idx.get(candidate_drawn)
                    if c_idx is not None:
                        sim_val = float(np.dot(anchor_emb, item_embeddings[c_idx]))
                        if sim_val <= SIMILARITY_SAFETY_FLOOR or attempt == 4:
                            cand_item = candidate_drawn
                            break
                    else:
                        cand_item = candidate_drawn
                        break

            if cand_item and cand_item not in neg_seen:
                negatives_for_anchor.append((cand_item, 'hard_negative'))
                neg_seen.add(cand_item)
                hard_neg_cat_counts[chosen_dissim_cat] += 1
                hard_count += 1

        # Emit Negative Rows (label = 0)
        for cand_item, stype in negatives_for_anchor:
            cand_idx = item_to_idx.get(cand_item)
            if cand_idx is not None:
                sim_score = float(np.dot(anchor_emb, item_embeddings[cand_idx]))
                cand_cat = item_to_cat.get(cand_item, 'Unknown')
            else:
                sim_score = 0.0
                cand_cat = 'Unknown'

            records.append({
                'anchor_item': anchor,
                'anchor_category': anchor_cat,
                'candidate_item': cand_item,
                'candidate_category': cand_cat,
                'label': 0,
                'sample_type': stype,
                'similarity_score': round(sim_score, 4),
                'same_category': (cand_cat == anchor_cat)
            })

    dataset_df = pd.DataFrame(records)
    print(f"Generated {len(dataset_df)} total candidate rows.")

    print("\n=== Stage 7: Conflict Resolution & Quality Verification ===")
    pos_pairs = set(dataset_df[dataset_df['label'] == 1][['anchor_item', 'candidate_item']].itertuples(index=False, name=None))
    
    conflicting_mask = (dataset_df['label'] == 0) & dataset_df[['anchor_item', 'candidate_item']].apply(tuple, axis=1).isin(pos_pairs)
    conflict_count = conflicting_mask.sum()
    if conflict_count > 0:
        print(f"Conflict Resolution: Removed {conflict_count} label=0 rows that conflicted with label=1 positive samples.")
        dataset_df = dataset_df[~conflicting_mask].reset_index(drop=True)

    initial_len = len(dataset_df)
    dataset_df = dataset_df.drop_duplicates(subset=['anchor_item', 'candidate_item'], keep='first').reset_index(drop=True)
    dedup_count = initial_len - len(dataset_df)
    print(f"Deduplicated {dedup_count} repeated (anchor_item, candidate_item) rows.")

    self_loops = (dataset_df['anchor_item'] == dataset_df['candidate_item']).sum()
    null_cand_cats = dataset_df['candidate_category'].isna().sum()

    pos_df = dataset_df[dataset_df['label'] == 1]
    neg_df = dataset_df[dataset_df['label'] == 0]

    print(f"\n1. Total positive rows (label=1) : {len(pos_df)}")
    print(f"2. Total negative rows (label=0) : {len(neg_df)}")
    print(f"3. Overall Neg:Pos sampling ratio: {round(len(neg_df) / max(1, len(pos_df)), 2)} (Target: ~2.0)")
    print(f"4. Anchor == Candidate count     : {self_loops} (Target: 0)")
    print(f"5. Null candidate categories     : {null_cand_cats} (Target: 0)")
    
    print("\n6. Unique candidate items and categories per sample_type:")
    sample_type_stats = dataset_df.groupby('sample_type').agg(
        total_rows=('candidate_item', 'count'),
        unique_candidate_items=('candidate_item', 'nunique'),
        unique_candidate_categories=('candidate_category', 'nunique')
    )
    print(sample_type_stats)

    print("\n7. Breakdown of `same_category` flag per sample_type:")
    same_cat_breakdown = dataset_df.groupby(['sample_type', 'same_category']).size().unstack(fill_value=0)
    print(same_cat_breakdown)

    assert self_loops == 0, "Sanity check failed: anchor_item == candidate_item found!"
    assert null_cand_cats == 0, "Sanity check failed: null candidate_category found!"
    print("\nALL SANITY CHECKS PASSED SUCCESSFULLY!")

    print("\n=== Stage 8: Saving Dataset ===")
    out_cols = ['anchor_item', 'anchor_category', 'candidate_item', 'candidate_category', 'label', 'sample_type', 'similarity_score', 'same_category']
    dataset_df[out_cols].to_csv(OUTPUT_FILE_V3, index=False)
    print(f"Successfully saved candidate dataset to '{OUTPUT_FILE_V3}'!")

if __name__ == '__main__':
    main()
