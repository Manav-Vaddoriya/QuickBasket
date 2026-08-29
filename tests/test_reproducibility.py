import os
import shutil
import tempfile
import pandas as pd
from quickbasket.catalog import generate_catalog
from quickbasket.world import generate_users, build_complementarity_matrix, build_substitute_sets
from quickbasket.simulator import run_simulation
from quickbasket import config

def test_simulation_reproducibility():
    """
    Asserts that re-running the generator with the same world_seed
    produces identical data (Acceptance Criterion A5).
    """
    print("Testing simulation reproducibility...")
    
    # Save original paths
    original_bronze = config.BRONZE_DIR
    original_eval = config.EVAL_DIR
    
    try:
        # Create temp directories for runs
        temp_dir1 = tempfile.mkdtemp()
        temp_dir2 = tempfile.mkdtemp()
        
        # --- Run 1 ---
        config.BRONZE_DIR = os.path.join(temp_dir1, "bronze")
        config.EVAL_DIR = os.path.join(temp_dir1, "evaluation")
        os.makedirs(config.BRONZE_DIR, exist_ok=True)
        os.makedirs(config.EVAL_DIR, exist_ok=True)
        
        # Scale down for speed in test (100 users, 200 SKUs)
        seed = 42
        cat1 = generate_catalog(num_skus=200, seed=seed)
        users1 = generate_users(num_users=100, seed=seed)
        C1, _ = build_complementarity_matrix(cat1, seed=seed)
        subs1 = build_substitute_sets(cat1)
        evts1, ords1 = run_simulation(users1, cat1, C1, subs1, scale_factor=0.001, seed=seed)
        
        # Load written parquets from Run 1
        cat_file1 = pd.read_parquet(os.path.join(config.BRONZE_DIR, "catalog_raw.parquet"))
        users_file1 = pd.read_parquet(os.path.join(config.BRONZE_DIR, "users_raw.parquet"))
        evts_file1 = pd.read_parquet(os.path.join(config.BRONZE_DIR, "events_raw.parquet"))
        ords_file1 = pd.read_parquet(os.path.join(config.BRONZE_DIR, "orders_raw.parquet"))
        
        # --- Run 2 ---
        config.BRONZE_DIR = os.path.join(temp_dir2, "bronze")
        config.EVAL_DIR = os.path.join(temp_dir2, "evaluation")
        os.makedirs(config.BRONZE_DIR, exist_ok=True)
        os.makedirs(config.EVAL_DIR, exist_ok=True)
        
        cat2 = generate_catalog(num_skus=200, seed=seed)
        users2 = generate_users(num_users=100, seed=seed)
        C2, _ = build_complementarity_matrix(cat2, seed=seed)
        subs2 = build_substitute_sets(cat2)
        evts2, ords2 = run_simulation(users2, cat2, C2, subs2, scale_factor=0.001, seed=seed)
        
        # Load written parquets from Run 2
        cat_file2 = pd.read_parquet(os.path.join(config.BRONZE_DIR, "catalog_raw.parquet"))
        users_file2 = pd.read_parquet(os.path.join(config.BRONZE_DIR, "users_raw.parquet"))
        evts_file2 = pd.read_parquet(os.path.join(config.BRONZE_DIR, "events_raw.parquet"))
        ords_file2 = pd.read_parquet(os.path.join(config.BRONZE_DIR, "orders_raw.parquet"))
        
        # Verify DataFrame Equality
        pd.testing.assert_frame_equal(cat_file1, cat_file2)
        pd.testing.assert_frame_equal(users_file1, users_file2)
        pd.testing.assert_frame_equal(evts_file1, evts_file2)
        pd.testing.assert_frame_equal(ords_file1, ords_file2)
        
        print("PASS: Reproducibility check successful. Both runs produced identical tables.")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)
        config.BRONZE_DIR = original_bronze
        config.EVAL_DIR = original_eval
