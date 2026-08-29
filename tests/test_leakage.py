import os
import shutil
import tempfile
import pandas as pd
from datetime import datetime, timedelta
from quickbasket import config
from quickbasket.pipeline import silver_to_gold

def test_temporal_leakage():
    """
    Verifies temporal hygiene: no feature or training example built
    as of a watermark timestamp contains data from after that boundary (Acceptance Criterion A4).
    """
    print("Testing temporal leakage prevention...")
    
    # Save original paths
    original_silver = config.SILVER_DIR
    original_gold = config.GOLD_DIR
    
    try:
        temp_dir = tempfile.mkdtemp()
        config.SILVER_DIR = os.path.join(temp_dir, "silver")
        config.GOLD_DIR = os.path.join(temp_dir, "gold")
        os.makedirs(config.SILVER_DIR, exist_ok=True)
        os.makedirs(config.GOLD_DIR, exist_ok=True)
        
        # Setup mock Silver events & catalog
        # Simulation starts at 2026-04-10
        sim_start_date = datetime(2026, 4, 10)
        
        # We will create events before and after the week 10 boundary (Day 70, 2026-06-19)
        boundary_dt = sim_start_date + timedelta(days=70)
        boundary_ts = boundary_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        before_dt = boundary_dt - timedelta(days=1)
        after_dt = boundary_dt + timedelta(days=1)
        
        # Catalog with 3 items
        catalog_mock = pd.DataFrame([
            {"item_id": "ITEM_A", "name": "Item A", "category_l1": "Beverages", "category_l2": "Coffee", "brand": "Nescafe", "price_inr": 100.0, "pack_size": "100g", "pack_size_band": "small", "is_veg": True, "shelf_life_days": 365, "launch_date": "2026-01-01 00:00:00"},
            {"item_id": "ITEM_B", "name": "Item B", "category_l1": "Snacks & Biscuits", "category_l2": "Biscuits", "brand": "Britannia", "price_inr": 20.0, "pack_size": "150g", "pack_size_band": "small", "is_veg": True, "shelf_life_days": 180, "launch_date": "2026-01-01 00:00:00"},
            {"item_id": "ITEM_C", "name": "Item C", "category_l1": "Beverages", "category_l2": "Soft Drinks", "brand": "Pepsi", "price_inr": 40.0, "pack_size": "750ml", "pack_size_band": "medium", "is_veg": True, "shelf_life_days": 180, "launch_date": after_dt.strftime("%Y-%m-%d %H:%M:%S")}
        ])
        catalog_mock.to_parquet(os.path.join(config.SILVER_DIR, "catalog.parquet"), index=False)
        
        # Users
        users_mock = pd.DataFrame([
            {"user_id": "USR_1", "signup_ts": "2026-01-01 00:00:00", "city": "Mumbai", "persona_id": "bachelor_quick", "device_type": "iOS", "is_bot": False}
        ])
        users_mock.to_parquet(os.path.join(config.SILVER_DIR, "users.parquet"), index=False)
        
        # Events
        # One session before boundary, one session after boundary
        events_mock = pd.DataFrame([
            # Session 1 (Before)
            {"event_id": "EVT_1", "user_id": "USR_1", "session_id": "SESS_BEFORE", "ts": before_dt.strftime("%Y-%m-%d %H:%M:%S"), "event_type": "session_start", "item_id": None, "surface": "app_open", "position": None, "search_query": None, "cart_snapshot_ids": "", "metadata": "{}"},
            {"event_id": "EVT_2", "user_id": "USR_1", "session_id": "SESS_BEFORE", "ts": (before_dt + timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S"), "event_type": "add_to_cart", "item_id": "ITEM_A", "surface": "pdp", "position": None, "search_query": None, "cart_snapshot_ids": "", "metadata": "{}"},
            {"event_id": "EVT_3", "user_id": "USR_1", "session_id": "SESS_BEFORE", "ts": (before_dt + timedelta(seconds=20)).strftime("%Y-%m-%d %H:%M:%S"), "event_type": "add_to_cart", "item_id": "ITEM_B", "surface": "pdp", "position": None, "search_query": None, "cart_snapshot_ids": "ITEM_A", "metadata": '{"source_surface": "pdp"}'},
            {"event_id": "EVT_4", "user_id": "USR_1", "session_id": "SESS_BEFORE", "ts": (before_dt + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S"), "event_type": "purchase", "item_id": None, "surface": "checkout", "position": None, "search_query": None, "cart_snapshot_ids": "ITEM_A,ITEM_B", "metadata": "{}"},
            {"event_id": "EVT_5", "user_id": "USR_1", "session_id": "SESS_BEFORE", "ts": (before_dt + timedelta(seconds=40)).strftime("%Y-%m-%d %H:%M:%S"), "event_type": "session_end", "item_id": None, "surface": "app_close", "position": None, "search_query": None, "cart_snapshot_ids": "", "metadata": "{}"},
            
            # Session 2 (After)
            {"event_id": "EVT_6", "user_id": "USR_1", "session_id": "SESS_AFTER", "ts": after_dt.strftime("%Y-%m-%d %H:%M:%S"), "event_type": "session_start", "item_id": None, "surface": "app_open", "position": None, "search_query": None, "cart_snapshot_ids": "", "metadata": "{}"},
            {"event_id": "EVT_7", "user_id": "USR_1", "session_id": "SESS_AFTER", "ts": (after_dt + timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S"), "event_type": "add_to_cart", "item_id": "ITEM_C", "surface": "pdp", "position": None, "search_query": None, "cart_snapshot_ids": "", "metadata": "{}"},
            {"event_id": "EVT_8", "user_id": "USR_1", "session_id": "SESS_AFTER", "ts": (after_dt + timedelta(seconds=20)).strftime("%Y-%m-%d %H:%M:%S"), "event_type": "purchase", "item_id": None, "surface": "checkout", "position": None, "search_query": None, "cart_snapshot_ids": "ITEM_C", "metadata": "{}"},
            {"event_id": "EVT_9", "user_id": "USR_1", "session_id": "SESS_AFTER", "ts": (after_dt + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S"), "event_type": "session_end", "item_id": None, "surface": "app_close", "position": None, "search_query": None, "cart_snapshot_ids": "", "metadata": "{}"}
        ])
        events_mock.to_parquet(os.path.join(config.SILVER_DIR, "events.parquet"), index=False)
        
        # Sessions
        sessions_mock = pd.DataFrame([
            {"session_id": "SESS_BEFORE", "user_id": "USR_1", "start_ts": before_dt.strftime("%Y-%m-%d %H:%M:%S"), "end_ts": (before_dt + timedelta(seconds=40)).strftime("%Y-%m-%d %H:%M:%S"), "n_events": 5, "entry_surface": "feed", "outcome": "purchased"},
            {"session_id": "SESS_AFTER", "user_id": "USR_1", "start_ts": after_dt.strftime("%Y-%m-%d %H:%M:%S"), "end_ts": (after_dt + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S"), "n_events": 4, "entry_surface": "feed", "outcome": "purchased"}
        ])
        sessions_mock.to_parquet(os.path.join(config.SILVER_DIR, "sessions.parquet"), index=False)
        
        # Baskets
        baskets_mock = pd.DataFrame([
            {"order_id": "ORD_BEFORE", "user_id": "USR_1", "session_id": "SESS_BEFORE", "ts": (before_dt + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S"), "item_ids": ["ITEM_A", "ITEM_B"], "quantities": [1, 1], "order_value_inr": 120.0, "removed_item_ids": []},
            {"order_id": "ORD_AFTER", "user_id": "USR_1", "session_id": "SESS_AFTER", "ts": (after_dt + timedelta(seconds=20)).strftime("%Y-%m-%d %H:%M:%S"), "item_ids": ["ITEM_C"], "quantities": [1], "order_value_inr": 40.0, "removed_item_ids": []}
        ])
        baskets_mock.to_parquet(os.path.join(config.SILVER_DIR, "baskets.parquet"), index=False)
        
        # Mock Substitute Sets in Evaluation Store
        os.makedirs(config.EVAL_DIR, exist_ok=True)
        substitute_mock = pd.DataFrame([
            {"item_id": "ITEM_A", "category_l2": "Coffee", "pack_size_band": "small", "substitute_ids": []},
            {"item_id": "ITEM_B", "category_l2": "Biscuits", "pack_size_band": "small", "substitute_ids": []},
            {"item_id": "ITEM_C", "category_l2": "Soft Drinks", "pack_size_band": "medium", "substitute_ids": []}
        ])
        substitute_mock.to_parquet(os.path.join(config.EVAL_DIR, "substitute_sets.parquet"), index=False)
        
        # Run Silver -> Gold up to boundary_ts
        silver_to_gold(as_of_ts=boundary_ts)
        
        # Load output Gold tables
        examples_df = pd.read_parquet(os.path.join(config.GOLD_DIR, "pair_training_examples.parquet"))
        sequences_df = pd.read_parquet(os.path.join(config.GOLD_DIR, "user_sequences.parquet"))
        
        # --- VERIFY LEAKAGE ---
        # 1. Check training examples
        # None of the training examples should be associated with SESS_AFTER or ORD_AFTER,
        # and no anchor/candidate should contain ITEM_C (which was only purchased/added after boundary).
        after_sessions = set(examples_df["session_id"].dropna())
        assert "SESS_AFTER" not in after_sessions, "Leakage Error: Session from after boundary included in training examples!"
        
        all_candidates = set(examples_df["candidate_item"])
        all_anchors = set(examples_df["anchor_item"])
        assert "ITEM_C" not in all_candidates, "Leakage Error: Candidate item added after boundary included!"
        assert "ITEM_C" not in all_anchors, "Leakage Error: Anchor item added after boundary included!"
        
        # 2. Check sequences
        # The user sequence for USR_1 must not contain any events from SESS_AFTER (view:ITEM_C or add:ITEM_C)
        usr_seq = sequences_df[sequences_df["user_id"] == "USR_1"]["sequence"].iloc[0].tolist()
        
        # Sequence should end with SESS_BEFORE events
        # ITEM_C should not be referenced in any tokens
        for token in usr_seq:
            assert "ITEM_C" not in token, f"Leakage Error: Found future item ITEM_C in sequence: {token}"
            assert "SESS_AFTER" not in token, f"Leakage Error: Found future session in sequence: {token}"
            
        print("PASS: Temporal leakage prevention verified. No future data leaked past the watermark.")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        config.SILVER_DIR = original_silver
        config.GOLD_DIR = original_gold
