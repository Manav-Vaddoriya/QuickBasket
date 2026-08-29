import os
import sys
import argparse
import pytest
from quickbasket import config
from quickbasket.catalog import generate_catalog
from quickbasket.world import generate_users, build_complementarity_matrix, build_substitute_sets
from quickbasket.simulator import run_simulation
from quickbasket.pipeline import bronze_to_silver, silver_to_gold
from quickbasket.validation import run_validation_suite

def run_all_steps(scale_factor, seed):
    print(f"==========================================")
    print(f"Running QuickBasket Medallion Data Pipeline")
    print(f"Scale Factor: {scale_factor}")
    print(f"Seed: {seed}")
    print(f"==========================================")
    
    # 1. Generate static world objects
    print("\n[Step 1/5] Generating Static World (Catalog, Users, Complementarity, Substitutes)...")
    num_skus = int(config.DEFAULT_NUM_SKUS * min(1.0, scale_factor * 10))
    if num_skus < 200:
        num_skus = 200 # minimum floor
        
    num_users = int(config.DEFAULT_NUM_USERS * scale_factor)
    if num_users < 100:
        num_users = 100 # minimum floor
        
    catalog_df = generate_catalog(num_skus=num_skus, seed=seed)
    users_df = generate_users(num_users=num_users, seed=seed)
    C_full, held_out_df = build_complementarity_matrix(catalog_df, seed=seed)
    cliques_dict = build_substitute_sets(catalog_df)
    
    print(f"Generated Catalog: {len(catalog_df)} SKUs")
    print(f"Generated Users: {len(users_df)} profiles")
    print(f"Ground Truth Edges: {len(held_out_df) * 10} total edges ({len(held_out_df)} held-out)")
    
    # 2. Run simulation (Bronze Raw creation)
    print("\n[Step 2/5] Running Session Simulator to build Bronze (raw)...")
    evts, ords = run_simulation(users_df, catalog_df, C_full, cliques_dict, scale_factor=scale_factor, seed=seed)
    print(f"Bronze Raw Event Log: {len(evts)} rows")
    print(f"Bronze Raw Orders Log: {len(ords)} rows")
    
    # 3. Clean and conform (Silver)
    print("\n[Step 3/5] Running Bronze -> Silver flow...")
    bronze_to_silver()
    
    # 4. Feature and training pairs generation (Gold)
    print("\n[Step 4/5] Running Silver -> Gold flow (as-of-date watermarking)...")
    # Gold up to end of week 14 (full simulation period)
    silver_to_gold()
    
    # 5. Run validation suite (Acceptance checks & Evaluation metrics)
    print("\n[Step 5/5] Running Validation Suite...")
    success = run_validation_suite()
    if success:
        print("\n[SUCCESS] VALIDATION: All QuickBasket data checks passed.")
    else:
        print("\n[FAILURE] VALIDATION: Some assertions on data quality failed.")
    return success

def run_tests():
    print("\n==========================================")
    print("Running QuickBasket Test Suite...")
    print("==========================================")
    
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    exit_code = pytest.main(["-v", test_dir])
    if exit_code == 0:
        print("\n[SUCCESS] TEST SUITE: All unit tests passed.")
    else:
        print("\n[FAILURE] TEST SUITE: Some unit tests failed.")
    return exit_code == 0

def main():
    parser = argparse.ArgumentParser(description="QuickBasket Medallion Pipeline CLI")
    
    parser.add_argument("--scale-factor", type=float, default=0.02,
                        help="Scale factor for simulation users (default 0.02, 2,000 users)")
    parser.add_argument("--seed", type=int, default=config.WORLD_SEED,
                        help=f"World generation seed (default {config.WORLD_SEED})")
    
    # Command targets
    parser.add_argument("--run-simulation", action="store_true", help="Generate Catalog, Users, and run simulation (Bronze)")
    parser.add_argument("--run-silver", action="store_true", help="Transform Bronze -> Silver")
    parser.add_argument("--run-gold", action="store_true", help="Transform Silver -> Gold")
    parser.add_argument("--run-validation", action="store_true", help="Run Validation Suite")
    parser.add_argument("--run-tests", action="store_true", help="Run unit test suite")
    parser.add_argument("--run-all", action="store_true", help="Run full pipeline and unit test suite")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
        
    config.SCALE_FACTOR = args.scale_factor
    config.WORLD_SEED = args.seed
    
    success = True
    
    if args.run_all:
        pipe_success = run_all_steps(args.scale_factor, args.seed)
        test_success = run_tests()
        if not (pipe_success and test_success):
            sys.exit(1)
        sys.exit(0)
        
    if args.run_simulation:
        num_skus = int(config.DEFAULT_NUM_SKUS * min(1.0, args.scale_factor * 10))
        num_skus = max(200, num_skus)
        num_users = max(100, int(config.DEFAULT_NUM_USERS * args.scale_factor))
        
        catalog_df = generate_catalog(num_skus=num_skus, seed=args.seed)
        users_df = generate_users(num_users=num_users, seed=args.seed)
        C_full, _ = build_complementarity_matrix(catalog_df, seed=args.seed)
        cliques_dict = build_substitute_sets(catalog_df)
        
        run_simulation(users_df, catalog_df, C_full, cliques_dict, scale_factor=args.scale_factor, seed=args.seed)
        
    if args.run_silver:
        bronze_to_silver()
        
    if args.run_gold:
        silver_to_gold()
        
    if args.run_validation:
        val_success = run_validation_suite()
        if not val_success:
            success = False
            
    if args.run_tests:
        test_success = run_tests()
        if not test_success:
            success = False
            
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
