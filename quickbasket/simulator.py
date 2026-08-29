import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from quickbasket import config
from quickbasket.config import (
    WORLD_SEED, SCALE_FACTOR, SIMULATION_DAYS,
    DUPLICATE_EVENT_PROB, LATE_ARRIVAL_PROB, DROPPED_EVENT_PROB,
    MALFORMED_METADATA_PROB, BOT_USER_SHARE, SKU_STOCKOUT_PROB
)
from quickbasket.ulid import new_ulid
from quickbasket.catalog import CATALOG_SPEC

def sample_hour_of_day(persona_id):
    """
    Sample an hour of day (0-23) based on the user's persona.
    - peaks 8-10 am and 6-9 pm IST.
    - bachelors shop late at night.
    - seniors shop early morning.
    """
    hours = list(range(24))
    weights = [0.01] * 24
    
    # Base peaks
    for h in [8, 9, 10]:
        weights[h] = 0.15
    for h in [18, 19, 20, 21]:
        weights[h] = 0.12
    for h in [11, 12, 13, 14, 15, 16, 17]:
        weights[h] = 0.05
    for h in [22, 23, 7]:
        weights[h] = 0.03
        
    if persona_id == "bachelor_quick":
        # Late night shift
        weights[22] = 0.15
        weights[23] = 0.18
        weights[0] = 0.10
        weights[1] = 0.08
    elif persona_id == "senior_household":
        # Early morning shift
        weights[6] = 0.12
        weights[7] = 0.15
        weights[8] = 0.20
        weights[9] = 0.15
        weights[21] = 0.02
        weights[22] = 0.01
        
    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]
    return np.random.choice(hours, p=weights)

def get_day_multipliers(day_idx, persona_id):
    """
    Calculate the rate multiplier for a given day in the simulation.
    - Day 0 is Friday, April 10, 2026.
    - weekends (Sat=Day 1, Sun=Day 2, etc.) get uplift for family personas.
    - paydays (1st-5th of month) get 1.3x.
    - festival (Diwali week: week 11, days 77-83) gets 1.8x.
    """
    mult = 1.0
    
    # 1. Day of week
    day_of_week = (day_idx + 4) % 7 # 0=Monday, ..., 5=Saturday, 6=Sunday. Day 0 (April 10) was Friday (4).
    if day_of_week in [5, 6]: # Saturday, Sunday
        if persona_id in ["family_weekly", "parent_infant"]:
            mult *= 1.5
        else:
            mult *= 1.1
            
    # 2. Payday (1st to 5th of the month)
    # Start date: April 10.
    # We can map day_idx to a calendar day
    sim_start_date = datetime(2026, 4, 10)
    current_date = sim_start_date + timedelta(days=day_idx)
    if 1 <= current_date.day <= 5:
        mult *= 1.3
        
    # 3. Festival (Diwali week: Day 75 to 81, June 24 to June 30, 2026)
    if 75 <= day_idx <= 81:
        mult *= 1.8
        
    return mult

def run_simulation(users_df, catalog_df, C_full, cliques_dict, scale_factor=SCALE_FACTOR, seed=WORLD_SEED):
    """
    Executes the world simulation over 14 weeks.
    Writes the raw tables directly to the Bronze layer in Parquet format.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    sim_start_date = datetime(2026, 4, 10)
    
    # 1. Assign Zipf base popularity to catalog items
    catalog_df = catalog_df.copy()
    # Sort items randomly or by category to distribute zipf rank, then assign rank
    catalog_df["zipf_rank"] = range(1, len(catalog_df) + 1)
    # Zipf probability distribution: s = 1.05
    s_param = 1.05
    ranks = catalog_df["zipf_rank"].values
    zipf_probs = 1.0 / (ranks ** s_param)
    catalog_df["base_popularity"] = zipf_probs / sum(zipf_probs)
    
    # Save Catalog to Bronze
    catalog_df.to_parquet(os.path.join(config.BRONZE_DIR, "catalog_raw.parquet"), index=False)
    
    # 2. Mark users as bots (0.5% share)
    users_df = users_df.copy()
    users_df["is_bot"] = False
    bot_indices = users_df.sample(frac=BOT_USER_SHARE, random_state=seed).index
    users_df.loc[bot_indices, "is_bot"] = True
    
    # Save Users to Bronze
    users_df.to_parquet(os.path.join(config.BRONZE_DIR, "users_raw.parquet"), index=False)
    
    # 3. Setup stockouts
    # We will simulate daily stockouts for 2% of SKUs
    # Map day -> list of out of stock SKU ids
    daily_stockouts = {}
    item_ids = catalog_df["item_id"].tolist()
    for d in range(SIMULATION_DAYS):
        num_stockouts = int(len(item_ids) * SKU_STOCKOUT_PROB)
        daily_stockouts[d] = set(random.sample(item_ids, num_stockouts))
        
    # --- OPTIMIZATION: PRE-PROCESS CATALOG TO DICTS & PRE-PARSE DATES ---
    catalog_dict = {}
    catalog_by_l1 = {}
    
    for item in catalog_df.to_dict("records"):
        item["launch_dt"] = datetime.strptime(item["launch_date"], "%Y-%m-%d %H:%M:%S")
        catalog_dict[item["item_id"]] = item
        
        cat_l1 = item["category_l1"]
        if cat_l1 not in catalog_by_l1:
            catalog_by_l1[cat_l1] = []
        catalog_by_l1[cat_l1].append(item)
        
    impulse_candidates = [
        item for item in catalog_dict.values()
        if item["category_l1"] in ["Chocolates & Ice Creams", "Snacks & Biscuits", "Beverages"]
    ]
    # --- END OPTIMIZATION ---
    
    # 4. Generate user sessions
    # Track historical purchases for brand loyalty and habit completion
    # Map user_id -> category_l2 -> last_purchased_brand
    user_brand_history = {}
    # Map user_id -> anchor_id -> complement_id -> count of co-purchases
    user_habit_pairs = {}
    # Map user_id -> category_l2 -> last_purchase_day
    user_last_purchase_day = {}
    
    events_raw = []
    orders_raw = []
    
    # Let's iterate through users
    for user_row in users_df.to_dict("records"):
        user_id = user_row["user_id"]
        persona = user_row["persona_id"]
        is_bot = user_row["is_bot"]
        sess_freq = user_row["session_frequency"]
        mean_basket = user_row["mean_basket_size"]
        brand_loyalty = user_row["brand_loyalty"]
        habit_strength = user_row["habit_strength"]
        
        user_brand_history[user_id] = {}
        user_habit_pairs[user_id] = {}
        user_last_purchase_day[user_id] = {}
        
        # Determine user base session rate per day
        base_daily_rate = (sess_freq / 7.0)
        
        if is_bot:
            # Bots have high frequency and view-only behavior
            base_daily_rate *= 3.0
            
        # Simulate day-by-day
        for day in range(SIMULATION_DAYS):
            # Inhomogeneous Poisson process approximation
            day_mult = get_day_multipliers(day, persona)
            daily_rate = base_daily_rate * day_mult
            
            # Number of sessions on this day
            num_sessions = np.random.poisson(daily_rate)
            for _ in range(num_sessions):
                hour = int(sample_hour_of_day(persona))
                session_time = sim_start_date + timedelta(days=day, hours=hour, minutes=random.randint(0, 59))
                session_id = f"SES_{new_ulid(timestamp_ms=int(session_time.timestamp() * 1000))}"
                
                # Check for bot crawl session
                if is_bot:
                    # Bot crawl session: emit search and views, but no adds/purchases
                    events_raw.append({
                        "event_id": f"EVT_{new_ulid(session_time.timestamp()*1000)}",
                        "user_id": user_id,
                        "session_id": session_id,
                        "ts": session_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "event_type": "session_start",
                        "item_id": None,
                        "surface": "app_open",
                        "position": None,
                        "search_query": None,
                        "cart_snapshot_ids": None,
                        "metadata": json.dumps({"device": user_row["device_type"], "entry_surface": "feed"})
                    })
                    # Bot crawl browsing
                    num_views = random.randint(10, 40)
                    for step in range(num_views):
                        view_time = session_time + timedelta(seconds=step * 10)
                        item_id = random.choice(item_ids)
                        events_raw.append({
                            "event_id": f"EVT_{new_ulid(view_time.timestamp()*1000)}",
                            "user_id": user_id,
                            "session_id": session_id,
                            "ts": view_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event_type": "view_item",
                            "item_id": item_id,
                            "surface": "search_results",
                            "position": random.randint(1, 20),
                            "search_query": "crawl_search",
                            "cart_snapshot_ids": None,
                            "metadata": json.dumps({"dwell_ms": random.randint(500, 3000)})
                        })
                    events_raw.append({
                        "event_id": f"EVT_{new_ulid((session_time + timedelta(seconds=num_views*10)).timestamp()*1000)}",
                        "user_id": user_id,
                        "session_id": session_id,
                        "ts": (session_time + timedelta(seconds=num_views*10)).strftime("%Y-%m-%d %H:%M:%S"),
                        "event_type": "session_end",
                        "item_id": None,
                        "surface": "app_close",
                        "position": None,
                        "search_query": None,
                        "cart_snapshot_ids": None,
                        "metadata": json.dumps({"outcome": "browsed_only"})
                    })
                    continue
                
                # Normal user session
                events_raw.append({
                    "event_id": f"EVT_{new_ulid(session_time.timestamp()*1000)}",
                    "user_id": user_id,
                    "session_id": session_id,
                    "ts": session_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": "session_start",
                    "item_id": None,
                    "surface": "app_open",
                    "position": None,
                    "search_query": None,
                    "cart_snapshot_ids": "",
                    "metadata": json.dumps({"device": user_row["device_type"], "entry_surface": random.choice(["feed", "search", "deeplink"])})
                })
                
                # Sample need-set size
                basket_size = max(0, np.random.poisson(mean_basket))
                
                # Pantry depletion check
                pantry_needs = []
                # Milk check: every 2 days
                if user_last_purchase_day[user_id].get("Milk", -99) <= day - 2:
                    pantry_needs.append("Milk")
                # Atta check: every 20 days
                if user_last_purchase_day[user_id].get("Atta (Flour)", -99) <= day - 20:
                    pantry_needs.append("Atta (Flour)")
                
                need_set_cats = pantry_needs.copy()
                
                # Sample remaining categories from preferred categories
                pref_cats = user_row["preferred_categories"]
                supp_cats = user_row["suppressed_categories"]
                all_cats = [c for c in CATALOG_SPEC.keys() if c not in supp_cats]
                
                # Festival shift mix (Diwali week)
                if 75 <= day <= 81:
                    pref_cats = list(set(pref_cats + ["Party & Festivities"]))
                    
                if len(all_cats) > 0:
                    cat_weights = []
                    for c in all_cats:
                        if c in pref_cats:
                            cat_weights.append(5.0)
                        else:
                            cat_weights.append(1.0)
                    total_w = sum(cat_weights)
                    cat_weights = [w / total_w for w in cat_weights]
                    
                    fill_size = max(0, basket_size - len(need_set_cats))
                    sampled_cats = np.random.choice(all_cats, size=min(fill_size, len(all_cats)), replace=True, p=cat_weights)
                    need_set_cats.extend(sampled_cats)
                
                # Keep duplicate categories to allow multiple items from same L1 category
                # need_set_cats = list(set(need_set_cats))
                
                cart = []
                cart_items_added = [] # Chronological items added
                time_offset = 60 # seconds
                
                # Process each need
                for cat_l1 in need_set_cats:
                    # Filter items in level-1 category
                    cat_items = catalog_by_l1.get(cat_l1, [])
                    if not cat_items:
                        continue
                        
                    # Filter out items launched in the future (cold-start)
                    current_ts = session_time + timedelta(seconds=time_offset)
                    launched_items = [item for item in cat_items if item["launch_dt"] <= current_ts]
                            
                    if not launched_items:
                        continue
                        
                    # Pick level-2 subcategory
                    subcat_items = {}
                    for item in launched_items:
                        l2 = item["category_l2"]
                        if l2 not in subcat_items:
                            subcat_items[l2] = []
                        subcat_items[l2].append(item)
                    l2_choice = random.choice(list(subcat_items.keys()))
                    candidates = subcat_items[l2_choice]
                    
                    # Sort candidates by popularity
                    candidates = sorted(candidates, key=lambda x: x["base_popularity"], reverse=True)
                    
                    # Choose entry surface (search vs feed)
                    entry_surface = random.choice(["search", "feed"])
                    search_query = l2_choice if entry_surface == "search" else None
                    
                    # Emit impressions for top 10 items
                    imp_items = candidates[:10]
                    for idx, item in enumerate(imp_items):
                        imp_time = session_time + timedelta(seconds=time_offset)
                        events_raw.append({
                            "event_id": f"EVT_{new_ulid(imp_time.timestamp()*1000)}",
                            "user_id": user_id,
                            "session_id": session_id,
                            "ts": imp_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event_type": "impression",
                            "item_id": item["item_id"],
                            "surface": "search_results" if entry_surface == "search" else "feed_shelf",
                            "position": idx + 1,
                            "search_query": search_query,
                            "cart_snapshot_ids": ",".join(cart),
                            "metadata": json.dumps({})
                        })
                    time_offset += 5
                    
                    # User selects 1 to 3 items to view based on position bias decay
                    viewed_items = []
                    for idx, item in enumerate(imp_items):
                        pos = idx + 1
                        click_prob = 1.0 / np.log2(1 + pos)
                        if random.random() < click_prob * 0.4: # scaled down
                            viewed_items.append(item)
                            if len(viewed_items) >= random.randint(1, 3):
                                break
                                
                    if not viewed_items:
                        # Fallback: view the most popular
                        viewed_items = [imp_items[0]]
                        
                    for item in viewed_items:
                        view_time = session_time + timedelta(seconds=time_offset)
                        events_raw.append({
                            "event_id": f"EVT_{new_ulid(view_time.timestamp()*1000)}",
                            "user_id": user_id,
                            "session_id": session_id,
                            "ts": view_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event_type": "view_item",
                            "item_id": item["item_id"],
                            "surface": "pdp",
                            "position": None,
                            "search_query": None,
                            "cart_snapshot_ids": ",".join(cart),
                            "metadata": json.dumps({"dwell_ms": random.randint(2000, 15000)})
                        })
                        time_offset += 10
                        
                    # Choose which one to add to cart (using brand loyalty)
                    # Get previously bought brand in this subcategory
                    last_brand = user_brand_history[user_id].get(l2_choice)
                    chosen_item = None
                    if last_brand and random.random() < brand_loyalty:
                        # Find item of the same brand in viewed items, or in candidates
                        brand_cands = [x for x in viewed_items if x["brand"] == last_brand]
                        if brand_cands:
                            chosen_item = brand_cands[0]
                        else:
                            brand_cands = [x for x in candidates if x["brand"] == last_brand]
                            if brand_cands:
                                chosen_item = brand_cands[0]
                                
                    if not chosen_item:
                        # Pick from viewed items based on popularity
                        view_probs = [x["base_popularity"] for x in viewed_items]
                        total_p = sum(view_probs)
                        view_probs = [p / total_p for p in view_probs]
                        chosen_item = np.random.choice(viewed_items, p=view_probs)
                        
                    # Stockout check: out-of-stock daily
                    if chosen_item["item_id"] in daily_stockouts[day]:
                        # Switch to substitute in the clique
                        clique = cliques_dict.get(chosen_item["item_id"], [chosen_item["item_id"]])
                        substitutes = [x for x in clique if x != chosen_item["item_id"] and x not in daily_stockouts[day]]
                        if substitutes:
                            # Switch to the substitute
                            sub_id = random.choice(substitutes)
                            chosen_item = catalog_dict[sub_id]
                        else:
                            # If no substitute is in stock, skip adding this need
                            continue
                            
                    # Add to cart
                    cart.append(chosen_item["item_id"])
                    cart_items_added.append(chosen_item["item_id"])
                    add_time = session_time + timedelta(seconds=time_offset)
                    events_raw.append({
                        "event_id": f"EVT_{new_ulid(add_time.timestamp()*1000)}",
                        "user_id": user_id,
                        "session_id": session_id,
                        "ts": add_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "event_type": "add_to_cart",
                        "item_id": chosen_item["item_id"],
                        "surface": "pdp",
                        "position": None,
                        "search_query": None,
                        "cart_snapshot_ids": ",".join(cart[:-1]),
                        "metadata": json.dumps({"qty": 1, "source_surface": "pdp"})
                    })
                    time_offset += 5
                    
                    # --- PLANT COMPLEMENT SUGGESTION ---
                    # Check if adding chosen_item triggers a complement suggestion
                    a_id = chosen_item["item_id"]
                    if a_id in C_full:
                        # Filter to complements in the catalog (i.e. launched already)
                        complements = C_full[a_id]
                        launched_complements = {}
                        for c_id, weight in complements.items():
                            c_item = catalog_dict[c_id]
                            if c_item["launch_dt"] <= (session_time + timedelta(seconds=time_offset)):
                                launched_complements[c_id] = weight
                                
                        if launched_complements:
                            # Persona modulation
                            persona_modulated = {}
                            for c_id, weight in launched_complements.items():
                                c_item = catalog_dict[c_id]
                                c_cat = c_item["category_l1"]
                                # Apply multiplier based on user's preferred/suppressed categories
                                mult = 1.0
                                if c_cat in pref_cats:
                                    mult = 2.0
                                elif c_cat in supp_cats:
                                    mult = 0.0
                                    
                                # Habit multiplier
                                habit_pair_count = user_habit_pairs[user_id].get(a_id, {}).get(c_id, 0)
                                if habit_pair_count > 0:
                                    # Elevate probability heavily
                                    mult *= (1.0 + habit_strength * habit_pair_count)
                                    
                                persona_modulated[c_id] = weight * mult
                                
                            # Filter out zero weight complements
                            persona_modulated = {k: v for k, v in persona_modulated.items() if v > 0}
                            
                            if persona_modulated:
                                # Choose the top complement candidate to show in the widget
                                c_id_choice = max(persona_modulated, key=persona_modulated.get)
                                w_choice = persona_modulated[c_id_choice]
                                
                                # Widget impression
                                widget_time = session_time + timedelta(seconds=time_offset)
                                events_raw.append({
                                    "event_id": f"EVT_{new_ulid(widget_time.timestamp()*1000)}",
                                    "user_id": user_id,
                                    "session_id": session_id,
                                    "ts": widget_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "event_type": "impression",
                                    "item_id": c_id_choice,
                                    "surface": "pair_widget",
                                    "position": 1,
                                    "search_query": None,
                                    "cart_snapshot_ids": ",".join(cart),
                                    "metadata": json.dumps({})
                                })
                                time_offset += 5
                                
                                # Probability of accepting the suggestion is proportional to weight
                                acceptance_prob = w_choice * 0.40
                                if random.random() < acceptance_prob:
                                    # Accept complement!
                                    cart.append(c_id_choice)
                                    cart_items_added.append(c_id_choice)
                                    add_comp_time = session_time + timedelta(seconds=time_offset)
                                    events_raw.append({
                                        "event_id": f"EVT_{new_ulid(add_comp_time.timestamp()*1000)}",
                                        "user_id": user_id,
                                        "session_id": session_id,
                                        "ts": add_comp_time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "event_type": "add_to_cart",
                                        "item_id": c_id_choice,
                                        "surface": "pair_widget",
                                        "position": 1,
                                        "search_query": None,
                                        "cart_snapshot_ids": ",".join(cart[:-1]),
                                        "metadata": json.dumps({"qty": 1, "source_surface": "pair_widget"})
                                    })
                                    time_offset += 5
                                    
                # --- IMPULSE ITEM ---
                # 15% probability of adding an impulse item
                impulse_added = None
                if random.random() < 0.15:
                    if impulse_candidates:
                        impulse_item = random.choice(impulse_candidates)
                        cart.append(impulse_item["item_id"])
                        cart_items_added.append(impulse_item["item_id"])
                        impulse_added = impulse_item["item_id"]
                        
                        imp_add_time = session_time + timedelta(seconds=time_offset)
                        events_raw.append({
                            "event_id": f"EVT_{new_ulid(imp_add_time.timestamp()*1000)}",
                            "user_id": user_id,
                            "session_id": session_id,
                            "ts": imp_add_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "event_type": "add_to_cart",
                            "item_id": impulse_item["item_id"],
                            "surface": "feed_shelf",
                            "position": 3,
                            "search_query": None,
                            "cart_snapshot_ids": ",".join(cart[:-1]),
                            "metadata": json.dumps({"qty": 1, "source_surface": "feed_shelf"})
                        })
                        time_offset += random.randint(15, 60)
                        
                # 50% probability of removing the impulse item before checkout
                if impulse_added and random.random() < 0.5:
                    cart.remove(impulse_added)
                    rem_time = session_time + timedelta(seconds=time_offset)
                    events_raw.append({
                        "event_id": f"EVT_{new_ulid(rem_time.timestamp()*1000)}",
                        "user_id": user_id,
                        "session_id": session_id,
                        "ts": rem_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "event_type": "remove_from_cart",
                        "item_id": impulse_added,
                        "surface": "cart_page",
                        "position": None,
                        "search_query": None,
                        "cart_snapshot_ids": ",".join(cart + [impulse_added]), # snapshot before removal
                        "metadata": json.dumps({"seconds_in_cart": int(random.randint(15, 60))})
                    })
                    time_offset += 5
                    
                # --- TERMINATE SESSION ---
                outcome_prob = random.random()
                if len(cart) == 0:
                    outcome = "browsed_only"
                elif outcome_prob < 0.68:
                    outcome = "purchased"
                elif outcome_prob < 0.90:
                    outcome = "abandoned_cart"
                else:
                    outcome = "browsed_only"
                    cart = [] # emptied
                    
                if outcome == "purchased":
                    # Checkout start
                    chk_time = session_time + timedelta(seconds=time_offset)
                    order_val = sum([catalog_dict[cid]["price_inr"] for cid in cart])
                    events_raw.append({
                        "event_id": f"EVT_{new_ulid(chk_time.timestamp()*1000)}",
                        "user_id": user_id,
                        "session_id": session_id,
                        "ts": chk_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "event_type": "checkout_start",
                        "item_id": None,
                        "surface": "cart_page",
                        "position": None,
                        "search_query": None,
                        "cart_snapshot_ids": ",".join(cart),
                        "metadata": json.dumps({"cart_value": order_val})
                    })
                    time_offset += 15
                    
                    # Purchase success
                    pur_time = session_time + timedelta(seconds=time_offset)
                    order_id = f"ORD_{new_ulid(pur_time.timestamp()*1000)}"
                    
                    events_raw.append({
                        "event_id": f"EVT_{new_ulid(pur_time.timestamp()*1000)}",
                        "user_id": user_id,
                        "session_id": session_id,
                        "ts": pur_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "event_type": "purchase",
                        "item_id": None,
                        "surface": "checkout_screen",
                        "position": None,
                        "search_query": None,
                        "cart_snapshot_ids": ",".join(cart),
                        "metadata": json.dumps({"order_id": order_id, "order_value": order_val})
                    })
                    
                    # Record order
                    orders_raw.append({
                        "order_id": order_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "ts": pur_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "item_ids": cart.copy(),
                        "quantities": [1] * len(cart),
                        "order_value_inr": order_val,
                        "delivery_slot": random.choice(["10-min immediate", "15-min immediate"])
                    })
                    
                    # Update histories for brand-loyalty and pantry depletion
                    for cid in cart:
                        item_row = catalog_dict[cid]
                        user_brand_history[user_id][item_row["category_l2"]] = item_row["brand"]
                        user_last_purchase_day[user_id][item_row["category_l2"]] = day
                        
                    # Update habit co-purchases count
                    for a_idx, a_id in enumerate(cart):
                        for b_idx, b_id in enumerate(cart):
                            if a_idx != b_idx:
                                if a_id not in user_habit_pairs[user_id]:
                                    user_habit_pairs[user_id][a_id] = {}
                                user_habit_pairs[user_id][a_id][b_id] = user_habit_pairs[user_id][a_id].get(b_id, 0) + 1
                                
                    time_offset += 5
                    
                # Session end event
                end_time = session_time + timedelta(seconds=time_offset)
                events_raw.append({
                    "event_id": f"EVT_{new_ulid(end_time.timestamp()*1000)}",
                    "user_id": user_id,
                    "session_id": session_id,
                    "ts": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": "session_end",
                    "item_id": None,
                    "surface": "app_close",
                    "position": None,
                    "search_query": None,
                    "cart_snapshot_ids": ",".join(cart) if outcome == "abandoned_cart" else "",
                    "metadata": json.dumps({"outcome": outcome})
                })
                
    events_raw_df = pd.DataFrame(events_raw)
    orders_raw_df = pd.DataFrame(orders_raw)
    
    # 5. Apply Noise and Realism Layer (Bronze corruption)
    if len(events_raw_df) > 0:
        corrupted_events = events_raw_df.copy()
        
        # (a) 0.5% Duplicate event_ids
        num_dups = int(len(corrupted_events) * DUPLICATE_EVENT_PROB)
        if num_dups > 0:
            dup_indices = np.random.choice(corrupted_events.index, size=num_dups, replace=False)
            dups = corrupted_events.loc[dup_indices].copy()
            corrupted_events = pd.concat([corrupted_events, dups], ignore_index=True)
            
        # (b) 1% Late arrivals
        num_late = int(len(corrupted_events) * LATE_ARRIVAL_PROB)
        if num_late > 0:
            late_indices = np.random.choice(corrupted_events.index, size=num_late, replace=False)
            for idx in late_indices:
                orig_ts = datetime.strptime(corrupted_events.loc[idx, "ts"], "%Y-%m-%d %H:%M:%S")
                shifted_ts = orig_ts + timedelta(seconds=random.randint(3600, 21600))
                corrupted_events.loc[idx, "ts"] = shifted_ts.strftime("%Y-%m-%d %H:%M:%S")
                
        # (c) 0.3% Dropped events
        num_drop = int(len(corrupted_events) * DROPPED_EVENT_PROB)
        if num_drop > 0:
            drop_indices = np.random.choice(corrupted_events.index, size=num_drop, replace=False)
            corrupted_events = corrupted_events.drop(drop_indices).reset_index(drop=True)
            
        # (d) 0.2% Malformed JSON metadata
        num_malformed = int(len(corrupted_events) * MALFORMED_METADATA_PROB)
        if num_malformed > 0:
            has_meta = corrupted_events[corrupted_events["metadata"].notna()].index
            if len(has_meta) > 0:
                malformed_indices = np.random.choice(has_meta, size=min(num_malformed, len(has_meta)), replace=False)
                corrupted_events.loc[malformed_indices, "metadata"] = "{malformed_json: true, missing_quotes"
                
        events_raw_df = corrupted_events
        
    events_raw_df = events_raw_df.sort_values(by="ts").reset_index(drop=True)
    
    events_raw_df.to_parquet(os.path.join(config.BRONZE_DIR, "events_raw.parquet"), index=False)
    orders_raw_df.to_parquet(os.path.join(config.BRONZE_DIR, "orders_raw.parquet"), index=False)
    
    return events_raw_df, orders_raw_df
