import os
import shutil
import tempfile
import pandas as pd
from quickbasket import config
from quickbasket.pipeline import silver_to_gold

def test_sequence_readiness():
    """
    Validates that user_sequences represents a loss-free, chronologically
    ordered reconstruction of the user's conformed event stream (Acceptance Criterion A6).
    """
    print("Testing sequence readiness and chronological order...")
    
    # Save original paths
    original_silver = config.SILVER_DIR
    original_gold = config.GOLD_DIR
    
    try:
        temp_dir = tempfile.mkdtemp()
        config.SILVER_DIR = os.path.join(temp_dir, "silver")
        config.GOLD_DIR = os.path.join(temp_dir, "gold")
        os.makedirs(config.SILVER_DIR, exist_ok=True)
        os.makedirs(config.GOLD_DIR, exist_ok=True)
        
        # Mock conformed catalog & users
        catalog_mock = pd.DataFrame([
            {"item_id": "ITEM_1", "name": "Item 1", "category_l1": "Beverages", "category_l2": "Coffee", "brand": "Nescafe", "price_inr": 100.0, "pack_size": "100g", "pack_size_band": "small", "is_veg": True, "shelf_life_days": 365, "launch_date": "2026-01-01 00:00:00"}
        ])
        catalog_mock.to_parquet(os.path.join(config.SILVER_DIR, "catalog.parquet"), index=False)
        
        users_mock = pd.DataFrame([
            {"user_id": "USR_999", "signup_ts": "2026-01-01 00:00:00", "city": "Delhi", "persona_id": "young_couple", "device_type": "Android", "is_bot": False}
        ])
        users_mock.to_parquet(os.path.join(config.SILVER_DIR, "users.parquet"), index=False)
        
        # Mock conformed events
        events_mock = pd.DataFrame([
            {"event_id": "E1", "user_id": "USR_999", "session_id": "S1", "ts": "2026-05-01 10:00:00", "event_type": "session_start", "item_id": None, "surface": "app_open", "position": None, "search_query": None, "cart_snapshot_ids": "", "metadata": '{"entry_surface": "search"}'},
            {"event_id": "E2", "user_id": "USR_999", "session_id": "S1", "ts": "2026-05-01 10:00:05", "event_type": "search", "item_id": None, "surface": "search_bar", "position": None, "search_query": "coffee", "cart_snapshot_ids": "", "metadata": "{}"},
            {"event_id": "E3", "user_id": "USR_999", "session_id": "S1", "ts": "2026-05-01 10:00:10", "event_type": "view_item", "item_id": "ITEM_1", "surface": "pdp", "position": None, "search_query": None, "cart_snapshot_ids": "", "metadata": "{}"},
            {"event_id": "E4", "user_id": "USR_999", "session_id": "S1", "ts": "2026-05-01 10:00:15", "event_type": "add_to_cart", "item_id": "ITEM_1", "surface": "pdp", "position": None, "search_query": None, "cart_snapshot_ids": "", "metadata": '{"source_surface": "pdp"}'},
            {"event_id": "E5", "user_id": "USR_999", "session_id": "S1", "ts": "2026-05-01 10:00:20", "event_type": "remove_from_cart", "item_id": "ITEM_1", "surface": "cart_page", "position": None, "search_query": None, "cart_snapshot_ids": "ITEM_1", "metadata": "{}"},
            {"event_id": "E6", "user_id": "USR_999", "session_id": "S1", "ts": "2026-05-01 10:00:25", "event_type": "session_end", "item_id": None, "surface": "app_close", "position": None, "search_query": None, "cart_snapshot_ids": "", "metadata": '{"outcome": "browsed_only"}'}
        ])
        events_mock.to_parquet(os.path.join(config.SILVER_DIR, "events.parquet"), index=False)
        
        # Conformed sessions & baskets
        sessions_mock = pd.DataFrame([
            {"session_id": "S1", "user_id": "USR_999", "start_ts": "2026-05-01 10:00:00", "end_ts": "2026-05-01 10:00:25", "n_events": 6, "entry_surface": "search", "outcome": "browsed_only"}
        ])
        sessions_mock.to_parquet(os.path.join(config.SILVER_DIR, "sessions.parquet"), index=False)
        
        baskets_mock = pd.DataFrame(columns=["order_id", "user_id", "session_id", "ts", "item_ids", "quantities", "order_value_inr", "removed_item_ids"])
        baskets_mock.to_parquet(os.path.join(config.SILVER_DIR, "baskets.parquet"), index=False)
        
        # Mock Substitute Sets in Evaluation Store
        os.makedirs(config.EVAL_DIR, exist_ok=True)
        substitute_mock = pd.DataFrame([
            {"item_id": "ITEM_1", "category_l2": "Coffee", "pack_size_band": "small", "substitute_ids": []}
        ])
        substitute_mock.to_parquet(os.path.join(config.EVAL_DIR, "substitute_sets.parquet"), index=False)
        
        # Run Silver -> Gold
        silver_to_gold()
        
        # Load user sequences
        sequences_df = pd.read_parquet(os.path.join(config.GOLD_DIR, "user_sequences.parquet"))
        
        # Verify sequence exists for USR_999
        assert not sequences_df.empty
        assert "USR_999" in sequences_df["user_id"].values
        
        seq = sequences_df[sequences_df["user_id"] == "USR_999"]["sequence"].iloc[0].tolist()
        
        expected_seq = [
            "[session_start@search]",
            "[search:coffee]",
            "[view:ITEM_1]",
            "[add:ITEM_1]",
            "[remove:ITEM_1]",
            "[session_end@browsed_only]"
        ]
        
        assert len(seq) == len(expected_seq), f"Sequence length mismatch! Expected {len(expected_seq)}, got {len(seq)}: {seq}"
        for s_token, e_token in zip(seq, expected_seq):
            assert s_token == e_token, f"Token mismatch! Expected {e_token}, got {s_token}"
            
        print("PASS: Reconstructed sequence matches conformed event log exactly (loss-free and ordered).")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        config.SILVER_DIR = original_silver
        config.GOLD_DIR = original_gold
