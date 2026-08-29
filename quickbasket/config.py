import os

# Project root directory and data subdirectory paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BRONZE_DIR = os.path.join(DATA_DIR, "bronze")
SILVER_DIR = os.path.join(DATA_DIR, "silver")
GOLD_DIR = os.path.join(DATA_DIR, "gold")
EVAL_DIR = os.path.join(DATA_DIR, "evaluation")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

# Make sure all directories exist
for d in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, EVAL_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Simulation scale parameters (can be overridden dynamically)
SCALE_FACTOR = 1.0
WORLD_SEED = 42

# Time bounds (14 weeks total)
# We will define weeks in days: 14 * 7 = 98 days
SIMULATION_DAYS = 98
TRAIN_DAYS = 70       # Weeks 1-10 (Days 0 to 69)
VAL_DAYS = 14         # Weeks 11-12 (Days 70 to 83)
TEST_DAYS = 14        # Weeks 13-14 (Days 84 to 97)

# Catalog settings
DEFAULT_NUM_SKUS = 5000

# Base parameters for simulations
DEFAULT_NUM_USERS = 100000

# Noise injection probabilities
DUPLICATE_EVENT_PROB = 0.005 # 0.5% duplicate events
LATE_ARRIVAL_PROB = 0.01     # 1% late arrivals
DROPPED_EVENT_PROB = 0.003   # 0.3% dropped events
MALFORMED_METADATA_PROB = 0.002 # 0.2% malformed metadata JSON
BOT_USER_SHARE = 0.005       # 0.5% of users are crawlers
SKU_STOCKOUT_PROB = 0.02     # 2% stockout probability daily

# Gold layer filters
MIN_SUPPORT = 20             # Minimum co-purchased baskets for P1
MIN_LIFT = 1.5               # Minimum lift score for P1

# Negative sample configuration
N1_RANDOM_RATIO = 4          # 4 in-batch random negatives per positive
N2_POPULARITY_RATIO = 2      # 2 popularity-matched negatives per positive
N3_SUBSTITUTE_RATIO = 1      # 1 substitute negative per positive
