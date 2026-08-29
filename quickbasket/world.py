import random
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from quickbasket import config
from quickbasket.config import (
    DEFAULT_NUM_USERS, WORLD_SEED, MIN_SUPPORT, MIN_LIFT
)
from quickbasket.ulid import new_ulid

# Persona configuration
PERSONA_ARCHETYPES = {
    "bachelor_quick": {
        "share": 0.18,
        "mean_basket_size": 4.0,
        "price_sensitivity": 0.3,
        "brand_loyalty": 0.4,
        "habit_strength": 0.3,
        "session_frequency": 3.5, # sessions per week
        "preferred_categories": ["Instant Food", "Beverages", "Snacks & Biscuits", "Chocolates & Ice Creams"],
        "suppressed_categories": ["Atta/Rice/Dal", "Baby Care", "Home Care"]
    },
    "family_weekly": {
        "share": 0.26,
        "mean_basket_size": 14.0,
        "price_sensitivity": 0.7,
        "brand_loyalty": 0.7,
        "habit_strength": 0.8,
        "session_frequency": 2.0,
        "preferred_categories": ["Atta/Rice/Dal", "Dairy & Bread", "Fruits & Vegetables", "Home Care"],
        "suppressed_categories": ["Paan Corner", "Stationery & Office"]
    },
    "young_couple": {
        "share": 0.16,
        "mean_basket_size": 8.0,
        "price_sensitivity": 0.4,
        "brand_loyalty": 0.6,
        "habit_strength": 0.5,
        "session_frequency": 3.0,
        "preferred_categories": ["Dairy & Bread", "Meat & Seafood", "Bakery & Cakes", "Breakfast Cereals", "Fruits & Vegetables"],
        "suppressed_categories": ["Baby Care", "Pet Care"]
    },
    "fitness_focused": {
        "share": 0.09,
        "mean_basket_size": 6.0,
        "price_sensitivity": 0.5,
        "brand_loyalty": 0.5,
        "habit_strength": 0.6,
        "session_frequency": 4.0,
        "preferred_categories": ["Breakfast Cereals", "Fruits & Vegetables", "Pharma & Wellness", "Meat & Seafood", "Dairy & Bread"],
        "suppressed_categories": ["Chocolates & Ice Creams", "Snacks & Biscuits", "Paan Corner", "Instant Food"]
    },
    "party_host": {
        "share": 0.06,
        "mean_basket_size": 10.0,
        "price_sensitivity": 0.2,
        "brand_loyalty": 0.4,
        "habit_strength": 0.2,
        "session_frequency": 1.5,
        "preferred_categories": ["Snacks & Biscuits", "Beverages", "Chocolates & Ice Creams", "Party & Festivities", "Paan Corner"],
        "suppressed_categories": ["Atta/Rice/Dal", "Baby Care", "Pharma & Wellness"]
    },
    "parent_infant": {
        "share": 0.08,
        "mean_basket_size": 9.0,
        "price_sensitivity": 0.5,
        "brand_loyalty": 0.9,
        "habit_strength": 0.9,
        "session_frequency": 2.5,
        "preferred_categories": ["Baby Care", "Dairy & Bread", "Pharma & Wellness", "Home Care"],
        "suppressed_categories": ["Paan Corner", "Meat & Seafood"]
    },
    "senior_household": {
        "share": 0.07,
        "mean_basket_size": 5.0,
        "price_sensitivity": 0.8,
        "brand_loyalty": 0.8,
        "habit_strength": 0.8,
        "session_frequency": 3.0,
        "preferred_categories": ["Atta/Rice/Dal", "Dairy & Bread", "Pharma & Wellness", "Fruits & Vegetables"],
        "suppressed_categories": ["Pet Care", "Chocolates & Ice Creams", "Paan Corner"]
    },
    "office_pantry": {
        "share": 0.10,
        "mean_basket_size": 16.0,
        "price_sensitivity": 0.1,
        "brand_loyalty": 0.5,
        "habit_strength": 0.4,
        "session_frequency": 2.0,
        "preferred_categories": ["Beverages", "Snacks & Biscuits", "Chocolates & Ice Creams", "Stationery & Office"],
        "suppressed_categories": ["Baby Care", "Meat & Seafood", "Fruits & Vegetables"]
    }
}

# Category-to-category priors for Layer 2 complementarity (anchor_category_l1 -> complement_category_l1)
CATEGORY_PRIORS = {
    ("Beverages", "Snacks & Biscuits"): 0.15,
    ("Atta/Rice/Dal", "Dairy & Bread"): 0.10,  # Atta -> Ghee/Butter
    ("Fruits & Vegetables", "Dairy & Bread"): 0.08,
    ("Snacks & Biscuits", "Beverages"): 0.12,
    ("Baby Care", "Baby Care"): 0.20,
    ("Instant Food", "Beverages"): 0.15,
    ("Bakery & Cakes", "Dairy & Bread"): 0.10,
    ("Breakfast Cereals", "Dairy & Bread"): 0.12,
    ("Chocolates & Ice Creams", "Snacks & Biscuits"): 0.10,
    ("Paan Corner", "Beverages"): 0.08,
    ("Meat & Seafood", "Fruits & Vegetables"): 0.10,
    ("Party & Festivities", "Beverages"): 0.15,
}

def generate_users(num_users=DEFAULT_NUM_USERS, seed=WORLD_SEED):
    """
    Generates user profiles based on persona archetypes.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Setup shares
    archetypes = list(PERSONA_ARCHETYPES.keys())
    shares = [PERSONA_ARCHETYPES[arch]["share"] for arch in archetypes]
    # Add a "mixed" archetype for the remaining 10%
    archetypes.append("mixed")
    shares.append(1.0 - sum(shares))
    
    users = []
    
    # We will assign user_id and sample parameters
    for i in range(num_users):
        arch = np.random.choice(archetypes, p=shares)
        
        # Jitter parameters
        if arch == "mixed":
            # Random combination
            mean_basket = float(np.random.uniform(4.0, 12.0))
            price_sens = float(np.random.uniform(0.1, 0.9))
            brand_loy = float(np.random.uniform(0.3, 0.9))
            habit_str = float(np.random.uniform(0.2, 0.8))
            sess_freq = float(np.random.normal(3.0, 0.5))
            pref_cats = random.sample(list(CATALOG_SPEC.keys()), 3)
            supp_cats = []
        else:
            spec = PERSONA_ARCHETYPES[arch]
            mean_basket = float(max(2.0, np.random.normal(spec["mean_basket_size"], 1.5)))
            price_sens = float(np.clip(np.random.normal(spec["price_sensitivity"], 0.1), 0.0, 1.0))
            brand_loy = float(np.clip(np.random.normal(spec["brand_loyalty"], 0.1), 0.0, 1.0))
            habit_str = float(np.clip(np.random.normal(spec["habit_strength"], 0.1), 0.0, 1.0))
            sess_freq = float(max(0.5, np.random.normal(spec["session_frequency"], 0.5)))
            pref_cats = spec["preferred_categories"]
            supp_cats = spec["suppressed_categories"]
            
        user_id = f"USR_{new_ulid(timestamp_ms=1767225600000 + i)}"
        signup_offset_days = random.randint(10, 100)
        # signup is before the start of simulation (e.g. before week 1)
        base_now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        signup_date = base_now - timedelta(days=100 + signup_offset_days)
        
        city = random.choice(["Mumbai", "Bangalore", "Delhi", "Mumbai", "Bangalore"])
        device = random.choice(["iOS", "Android", "Android", "Web"])
        
        users.append({
            "user_id": user_id,
            "signup_ts": signup_date.strftime("%Y-%m-%d %H:%M:%S"),
            "city": city,
            "persona_id": arch,
            "device_type": device,
            "mean_basket_size": mean_basket,
            "price_sensitivity": price_sens,
            "brand_loyalty": brand_loy,
            "habit_strength": habit_str,
            "session_frequency": sess_freq,
            "preferred_categories": pref_cats,
            "suppressed_categories": supp_cats
        })
        
    return pd.DataFrame(users)

def build_complementarity_matrix(catalog_df, seed=WORLD_SEED):
    """
    Build the complementarity matrix C: a dict mapping anchor_item_id -> {candidate_item_id: weight}.
    We separate Layer 1 (hand-seeded) and Layer 2 (category priors).
    We also return the held-out 10% edges.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    C_layer1 = {}
    C_layer2 = {}
    
    # Layer 1: Hand-seeded anchors (~150 pairs)
    # Define subcategory anchor mappings and find matching items
    # bread→butter 0.55 / butter→bread 0.25
    # coffee→biscuits 0.45
    # potato chips→soft drink 0.40
    # oats (Breakfast Cereals)→milk (Dairy & Bread) 0.35
    # diapers→baby wipes 0.60
    
    breads = catalog_df[catalog_df["category_l2"] == "Bread"].to_dict("records")
    butters = catalog_df[catalog_df["category_l2"] == "Butter"].to_dict("records")
    coffees = catalog_df[catalog_df["category_l2"] == "Coffee"].to_dict("records")
    biscuits = catalog_df[catalog_df["category_l2"] == "Biscuits"].to_dict("records")
    chips = catalog_df[catalog_df["category_l2"] == "Potato Chips"].to_dict("records")
    sodas = catalog_df[catalog_df["category_l2"] == "Soft Drinks"].to_dict("records")
    oats = catalog_df[catalog_df["category_l2"] == "Oats"].to_dict("records")
    milks = catalog_df[catalog_df["category_l2"] == "Milk"].to_dict("records")
    diapers = catalog_df[catalog_df["category_l2"] == "Diapers"].to_dict("records")
    wipes = catalog_df[catalog_df["category_l2"] == "Baby Wipes"].to_dict("records")
    
    def add_layer1_edges(source_list, target_list, weight_val, max_edges=30):
        # Sample edges to not blow up counts but cover exactly a set of obvious pairs
        edges_added = 0
        random.shuffle(source_list)
        random.shuffle(target_list)
        for s in source_list:
            for t in target_list:
                if edges_added >= max_edges:
                    break
                s_id, t_id = s["item_id"], t["item_id"]
                if s_id == t_id:
                    continue
                # Brand consistency bonus: if same brand, add a small extra boost or keep it tight
                final_w = weight_val
                if s["brand"] == t["brand"]:
                    final_w = min(1.0, weight_val * 1.1)
                if s_id not in C_layer1:
                    C_layer1[s_id] = {}
                C_layer1[s_id][t_id] = final_w
                edges_added += 1
                
    # Add Layer 1 edges
    add_layer1_edges(breads, butters, 0.55, max_edges=40)
    add_layer1_edges(butters, breads, 0.25, max_edges=40)
    add_layer1_edges(coffees, biscuits, 0.45, max_edges=35)
    add_layer1_edges(chips, sodas, 0.40, max_edges=35)
    add_layer1_edges(oats, milks, 0.35, max_edges=30)
    add_layer1_edges(diapers, wipes, 0.60, max_edges=30)
    
    # Layer 2: Category priors
    # Sample edges between related level-1 categories
    items_by_cat = {}
    for cat in catalog_df["category_l1"].unique():
        items_by_cat[cat] = catalog_df[catalog_df["category_l1"] == cat].to_dict("records")
        
    for (anchor_cat, comp_cat), base_weight in CATEGORY_PRIORS.items():
        if anchor_cat not in items_by_cat or comp_cat not in items_by_cat:
            continue
        anchors = items_by_cat[anchor_cat]
        candidates = items_by_cat[comp_cat]
        
        # Sample a subset of pairs (e.g. ~5% of possible pairs or up to 20 per anchor)
        for anchor in random.sample(anchors, min(len(anchors), 15)):
            a_id = anchor["item_id"]
            if a_id not in C_layer2:
                C_layer2[a_id] = {}
                
            sampled_cands = random.sample(candidates, min(len(candidates), 5))
            for cand in sampled_cands:
                c_id = cand["item_id"]
                if a_id == c_id:
                    continue
                # Skip if already in Layer 1
                if a_id in C_layer1 and c_id in C_layer1[a_id]:
                    continue
                # Apply brand-consistency bonus (1.2x)
                w = base_weight
                if anchor["brand"] == cand["brand"]:
                    w *= 1.2
                w = min(0.95, w)
                C_layer2[a_id][c_id] = w
                
    # Combine Layer 1 & 2 to create the complete ground truth complementarity graph
    # Let's also split 10% of edges to the held-out evaluation set
    all_edges = []
    for a_id, targets in C_layer1.items():
        for t_id, w in targets.items():
            all_edges.append((a_id, t_id, w, "layer1"))
    for a_id, targets in C_layer2.items():
        for t_id, w in targets.items():
            all_edges.append((a_id, t_id, w, "layer2"))
            
    # Randomly shuffle and hold out 10%
    random.shuffle(all_edges)
    num_held_out = int(len(all_edges) * 0.10)
    held_out_edges = all_edges[:num_held_out]
    public_edges = all_edges[num_held_out:]
    
    # C_full will contain all edges (used by the simulator to generate user actions)
    # The downstream pipelines/evaluators will only have access to catalog/users/events,
    # and they will evaluate using held_out_edges to see if the baseline lift model recovers them.
    C_full = {}
    for a_id, t_id, w, layer in all_edges:
        if a_id not in C_full:
            C_full[a_id] = {}
        C_full[a_id][t_id] = w
        
    # Write held_out and public edges to the evaluation store
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    
    held_out_df = pd.DataFrame(held_out_edges, columns=["anchor_item", "candidate_item", "weight", "layer"])
    held_out_df.to_parquet(os.path.join("held_out_edges.parquet"), index=False)
    
    public_df = pd.DataFrame(public_edges, columns=["anchor_item", "candidate_item", "weight", "layer"])
    public_df.to_parquet(os.path.join("public_edges.parquet"), index=False)
    
    return C_full, held_out_df

def build_substitute_sets(catalog_df):
    """
    Build substitute sets: within each (category_l2, pack_size_band) cell, SKUs form a substitute clique.
    Returns:
      cliques_dict: map from SKU item_id to list of substitute SKU item_ids.
      substitute_df: DataFrame containing the substitute cliques for logging/eval.
    """
    # Group by category_l2 and pack_size_band
    groups = catalog_df.groupby(["category_l2", "pack_size_band"])["item_id"].apply(list).to_dict()
    
    cliques_dict = {}
    for item_id in catalog_df["item_id"]:
        # Find category_l2 and pack_size_band for this item
        row = catalog_df[catalog_df["item_id"] == item_id].iloc[0]
        key = (row["category_l2"], row["pack_size_band"])
        clique = groups.get(key, [item_id])
        # A substitute set excludes the item itself in list, but let's store it as the clique
        # that includes the item itself so we can sample substitutes.
        cliques_dict[item_id] = clique
        
    # Create substitute sets dataframe for the evaluation store
    substitute_records = []
    for key, clique in groups.items():
        cat_l2, pack_band = key
        for item_id in clique:
            substitute_records.append({
                "item_id": item_id,
                "category_l2": cat_l2,
                "pack_size_band": pack_band,
                "substitute_ids": [x for x in clique if x != item_id]
            })
            
    substitute_df = pd.DataFrame(substitute_records)
    substitute_df.to_parquet(os.path.join("substitute_sets.parquet"), index=False)
    
    return cliques_dict
