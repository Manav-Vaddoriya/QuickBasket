# QuickBasket — Synthetic Data Foundation

A fully deterministic **quick-commerce world simulator** and **medallion data pipeline** that generates 14 weeks of realistic Indian quick-delivery shopping behaviour for training, testing, and evaluating personalization models.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [How the Data Is Made](#how-the-data-is-made)
   - [Step 1 — Product Catalog Generation (Stage 1a)](#step-1--product-catalog-generation-stage-1a)
   - [Step 2 — World Objects and Ground-Truth Setup (Stage 1b & 1c)](#step-2--world-objects-and-ground-truth-setup-stage-1b--1c)
   - [Step 3 — Session Simulation (Stage 2a & 2b) (Bronze)](#step-3--session-simulation-stage-2a--2b-bronze)
   - [Step 4 — Bronze → Silver Cleaning Pipeline](#step-4--bronze--silver-cleaning-pipeline)
   - [Step 5 — Silver → Gold Feature & Sequence Pipeline](#step-5--silver--gold-feature--sequence-pipeline)
4. [File-by-File Reference](#file-by-file-reference)
5. [Data Schema Reference](#data-schema-reference)
6. [Running the Pipeline](#running-the-pipeline)
7. [Running the Tests](#running-the-tests)
8. [Acceptance Criteria Checklist](#acceptance-criteria-checklist)

---

## Overview

```
World Generator ──► Bronze (raw) ──► Silver (clean) ──► Gold (features)
  catalog.py           simulator.py       pipeline.py        pipeline.py
  world.py                                bronze_to_silver   silver_to_gold
```

The pipeline runs in five deterministic steps driven by a single integer **seed** (`WORLD_SEED = 42`). Re-running with the same seed always produces byte-identical Parquet files.

---

## Project Structure

```
qb2/
│
├── quickbasket/                  # Core library package
│   ├── __init__.py
│   ├── config.py                 # All paths, constants, and knobs
│   ├── ulid.py                   # Pure-Python ULID generator
│   ├── catalog.py                # Product catalog factory (~5,000 SKUs)
│   ├── world.py                  # Complementarity matrix C, substitute sets S, user personas
│   ├── simulator.py              # 14-week session simulator → Bronze raw events
│   ├── pipeline.py               # Medallion transformations (Bronze→Silver, Silver→Gold)
│   └── validation.py             # Statistical checks + baseline lift evaluation
│
├── tests/                        # Pytest unit tests
│   ├── __init__.py
│   ├── test_reproducibility.py   # A5: same seed → identical Bronze
│   ├── test_leakage.py           # A4: no future data crosses the watermark
│   └── test_sequence_readiness.py# A6: sequences are loss-free & chronological
│
├── data/                         # Auto-created at runtime
│   ├── bronze/                   # Raw, corrupted, unvalidated Parquet files
│   ├── silver/                   # Cleaned, conformed Parquet files
│   ├── gold/                     # ML-ready features, training examples, sequences
│   ├── evaluation/               # Held-out edges and substitute cliques
│   └── reports/                  # Validation report + plots
│
├── run_pipeline.py               # CLI entry-point
└── README.md
```

---

## How the Data Is Made

This section provides a deep technical breakdown of the 5 sequential steps that build the QuickBasket synthetic dataset, as defined in the Product Requirements Document (PRD).

---

### Step 1 — Product Catalog Generation (Stage 1a)
**File:** [`quickbasket/catalog.py`](quickbasket/catalog.py)

The catalog is generated from a hand-curated specification (`CATALOG_SPEC`) covering **18 Level-1 categories** (Dairy & Bread, Beverages, Snacks & Biscuits, Fruits & Vegetables, Atta/Rice/Dal, Instant Food, Personal Care, Home Care, Baby Care, Pet Care, Paan Corner, etc.) and **47 Level-2 sub-categories**.

For each SKU, the generator assigns:
1. **`item_id`**: A sortable, timestamp-prefixed, collision-free **ULID** identifier.
2. **`price_inr`**: Sampled from a **log-normal distribution** tailored to each subcategory. The price is clipped within standard quick-commerce boundaries specified in the PRD (e.g. Biscuits range from 10 INR to 120 INR, Coffee ranges from 90 INR to 650 INR).
3. **`brand`**: Chosen from a realistic brand list per sub-category.
4. **`pack_size` & `pack_size_band`**: Extracted from the catalog specification. The band is mapped as `small`, `medium`, or `large` and defines SKU substitution sets.
5. **`launch_date`**: Used to model cold-start evaluations. **90% of SKUs** receive a launch date before the simulation starts, while **10% of SKUs** receive launch dates distributed *inside* the active 14-week simulation window (cold-start items).
6. **`is_veg` & `shelf_life_days`**: Subcategory attributes defining product properties.

---

### Step 2 — World Objects and Ground-Truth Setup (Stage 1b & 1c)
**File:** [`quickbasket/world.py`](quickbasket/world.py)

Before behavior is simulated, the ground-truth relationship matrices and user profiles are constructed:

#### 1. Complementarity Matrix $C$ (Stage 1b)
$C$ is a directed, weighted, sparse graph where $C[a][b] \in [0, 1]$ represents the probability boost that adding item $a$ (anchor) to the cart gives to wanting item $b$ (candidate complement). It is built in three layers:
- **Layer 1 (Hand-seeded anchors)**: obivous pairings based on domain expertise (~150 pairs total). Weights are asymmetric to model real grocery directionality (e.g., `bread → butter` has a boost of `0.55`, but `butter → bread` has a boost of `0.25`; other anchors include `coffee → biscuits` (0.45), `potato chips → soft drink` (0.40), `oats → milk` (0.35), and `diapers → baby wipes` (0.60)). Items matching the same brand receive a 1.1x brand-match bonus.
- **Layer 2 (Category priors)**: A category-to-category affinity table defining prior associations (e.g., Beverages ↔ Snacks & Biscuits has a prior weight of 0.15; Atta/Rice/Dal → Dairy & Bread has a prior of 0.10). Item pairs within related categories are sampled and receive a brand-consistency bonus of 1.2x.
- **Layer 3 (Persona modulation)**: Each user persona carries a mask that amplifies or suppresses regions of $C$ (e.g. the fitness persona amplifies `oats → protein` and suppresses `chips → cola`).

#### 2. The Ground-Truth Firewall (Held-Out Edges)
Before simulating behavior, **10% of Layer 1 & 2 complementarity edges are randomly held out** and written to `evaluation/held_out_edges.parquet`. These edges still generate shopping behavior in the simulator, but they are hidden from the features and training examples in the Gold layer. Downstream recommender models are evaluated on their ability to recover these hidden edges.

#### 3. Substitute Sets $S$ (Stage 1c)
Within each `(category_l2, pack_size_band)` cell, all SKUs form a substitute clique (e.g. all small packets of biscuits, or all 1L milk brands). A user samples exactly one SKU from a substitute clique to fulfill a category need, sticking to a brand loyalty of ~0.7. The substitutes are later mined as **hard negatives (N3)** to train the model to distinguish complements from substitutes.

#### 4. Persona Library & User Generation (Stage 1d)
100,000 users are generated by sampling from **8 persona archetypes** (`family_weekly`, `bachelor_quick`, `young_couple`, `fitness_focused`, `party_host`, `parent_infant`, `senior_household`, `office_pantry`).
Each persona has unique parameters:
- Mean basket size
- Price sensitivity
- Brand loyalty
- Habit strength ($\eta \in [0, 1]$: higher habit strength means that once a user pairs two items, they are highly likely to keep repeat-pairing them)
- Session frequency ($\lambda$)
To avoid rigid groups, each user's parameters are Gaussian-jittered around the persona mean.

---

### Step 3 — Session Simulation (Stage 2a & 2b) (Bronze)
**File:** [`quickbasket/simulator.py`](quickbasket/simulator.py)

#### 1. Temporal Intensity Model (Stage 2a)
App opens are simulated using an **inhomogeneous Poisson process**:
- **Daily Rates**: Modeled around hour-of-day curves with peaks at 8–10 AM and 6–9 PM IST.
- **Weekend Uplifts**: 1.5x session frequency boost on Sat/Sun for family and parent personas, and 1.1x for other personas.
- **Payday Surges**: 1.3x rate boost during the 1st–5th of every month.
- **Festival Weeks**: Diwali (Days 75–81) applies a 1.8x session rate multiplier and shifts the category mix toward sweets, dry fruits, and diyas, introducing seasonal complement associations (e.g. `gujiya → khoya`).

#### 2. Session Simulator State Machine (Stage 2b)
Each session runs through a state machine:
1. **Need-Set Generation**: Category needs are sampled based on the persona category propensities and a **pantry depletion clock** (e.g., milk is needed every 2 days, atta every 21 days).
2. **Product Browsing**: For each need category, the user views a shelf.
3. **Impressions & Views**: The simulator emits up to 10 impressions. Clicking/viewing a candidate is position-biased based on a decay curve ($\frac{1}{\log_2(1 + \text{position})}$).
4. **Item Selection & Stockouts**: The user selects an item based on brand loyalty first, falling back to Zipf popularity. If a store-level stockout (occurring daily on a random 2% of SKUs) occurs, the user switches to a substitute candidate from the substitute clique, creating realistic brand-switching noise.
5. **Complement Suggestions**: Once item $A$ is added to cart, the simulator checks $C[A]$ and suggests complement item $B$ with probability proportional to the attachment weight.
6. **Impulse Actions**: With a 15% probability, an impulse item is added to the cart. With a 50% probability, it is removed before checkout (generating the add-after-remove signals).
7. **Checkout & Purchase**: The session terminates with one of three outcomes: *purchased* (68%), *abandoned* (22%), or *browsed_only* (10%).

#### 3. Noise & Realism Layer (Stage 2c)
After raw logs are generated, they are run through a corruption layer to mimic production data issues:
- **0.5% duplicate events** (identical event IDs)
- **1.0% late-arriving events** (timestamps shifted forward by 1–6 hours)
- **0.3% dropped events** (broken foreign key relationships)
- **0.2% malformed JSON metadata**
- **0.5% bot-like traffic** (scripted crawlers doing rapid view-only actions)

Raw logs are saved as Parquet files in `data/bronze/`.

---

### Step 4 — Bronze → Silver Cleaning Pipeline
**File:** [`quickbasket/pipeline.py`](quickbasket/pipeline.py) → `bronze_to_silver()`

This nightly batch process runs conformed validations and cleans the Bronze events log:
1. **Deduplication**: Removes duplicate events by keeping only the first occurrence of an `event_id`.
2. **Timestamp Normalization**: Adjusts all timestamps to Indian Standard Time (IST, UTC+5:30) and sorts them chronologically.
3. **Metadata Repair**: Drops rows containing malformed JSON strings in the metadata field.
4. **Bot Filtering**: Identifies bot accounts (user accounts with view-only behavior and zero checkout/purchase actions) and drops their events.
5. **Foreign Key Integrity Validation**: Drops any events that refer to nonexistent `user_id` or `item_id` values.
6. **Session Stitching**: Reconstructs sessions using the **30-minute inactivity gap rule**. If two consecutive events of the same user are separated by more than 30 minutes, they are split into separate sessions.

Cleaned data is written to conformed Parquet tables in `data/silver/`.

---

### Step 5 — Silver → Gold Feature & Sequence Pipeline
**File:** [`quickbasket/pipeline.py`](quickbasket/pipeline.py) → `silver_to_gold(as_of_ts)`

This step converts conformed tables into model-ready features. It strictly enforces a **temporal watermark** (`as_of_ts`). No event occurring after `as_of_ts` can influence the feature calculations or training pairs.

#### 1. Global Pair Statistics (`item_pair_stats.parquet`)
Computed over the training window (Weeks 1–10, up to Day 70 watermark):
- `co_purchase_count` / `support`: Number of baskets containing both items.
- `directional_attach_rate`: $P(B \in \text{basket} \mid A \in \text{basket})$ and $P(A \in \text{basket} \mid B \in \text{basket})$.
- `lift`: $\frac{P(A \text{ and } B)}{P(A) \cdot P(B)}$ (measures association strength independent of popularity).
- `pmi`: $\log_2(\text{lift})$.

#### 2. Label Construction & Negative Mining (`pair_training_examples.parquet`)
Constructs positive and negative candidate pairs:
- **Positives**:
  - `P1` (Co-purchase, weight 1.0): Co-purchased in the same paid basket (filtered by Lift > 1.5 and Support >= 20).
  - `P2` (Add-after-add, weight 0.6): Item $B$ added within 5 events of item $A$ in the same session (recovers signal from abandoned sessions).
  - `P3` (Cross-session habit, weight 1.2): Item $A$ and $B$ co-purchased by the same user in 2 or more distinct calendar weeks.
- **Negatives** (mined at a ratio of 4:2:1 per positive):
  - `N1` (Random negatives, weight 1.0): 4 random catalog items not matching the positive.
  - `N2` (Popularity negatives, weight 1.0): 2 items sampled proportional to their global popularity (prevents the model from always recommending bestsellers).
  - `N3` (Substitute negatives, weight 1.0): 1 item selected from the anchor item's substitute clique (teaches the model to distinguish complements from substitutes).

#### 3. Behavioral Sequence Generation (`user_sequences.parquet`)
Aggregates conformed events into chronological user sequences of action tokens (e.g. `[session_start] [search:milk] [view:SKU_1] [add:SKU_1] [checkout] [purchase] [session_end]`) for behavioral foundation model pre-training.

---

## File-by-File Reference

| File | Role | Key Functions / Classes |
|---|---|---|
| [`config.py`](quickbasket/config.py) | Single source of truth for all paths, rates, seeds | `BRONZE_DIR`, `SILVER_DIR`, `GOLD_DIR`, `SCALE_FACTOR`, `WORLD_SEED` |
| [`ulid.py`](quickbasket/ulid.py) | Pure-Python ULID generator (no dependencies) | `new_ulid(timestamp_ms=None)` |
| [`catalog.py`](quickbasket/catalog.py) | Generates the product catalog DataFrame | `generate_catalog(num_skus, sim_start_date, seed)` |
| [`world.py`](quickbasket/world.py) | Builds C, S, user profiles; writes evaluation store | `generate_users()`, `build_complementarity_matrix()`, `build_substitute_sets()` |
| [`simulator.py`](quickbasket/simulator.py) | Runs 14-week simulation; writes Bronze Parquet | `run_simulation(users_df, catalog_df, C_full, cliques_dict, ...)` |
| [`pipeline.py`](quickbasket/pipeline.py) | Medallion cleaning and feature construction | `bronze_to_silver()`, `silver_to_gold(as_of_ts)` |
| [`validation.py`](quickbasket/validation.py) | Statistical assertions + held-out edge evaluation | `run_validation_suite()` |
| [`run_pipeline.py`](run_pipeline.py) | CLI wrapper for running any/all pipeline steps | `--run-all`, `--scale-factor`, `--seed` |
| [`tests/test_reproducibility.py`](tests/test_reproducibility.py) | Asserts same seed → identical Bronze (A5) | `test_simulation_reproducibility()` |
| [`tests/test_leakage.py`](tests/test_leakage.py) | Asserts no data crosses the temporal watermark (A4) | `test_temporal_leakage()` |
| [`tests/test_sequence_readiness.py`](tests/test_sequence_readiness.py) | Asserts token sequences match conformed event log (A6) | `test_sequence_readiness()` |

---

## Data Schema Reference

### `events.parquet` (Silver)

| Column | Type | Description |
|---|---|---|
| `event_id` | string | ULID-based unique event identifier |
| `user_id` | string | Foreign key → `users.user_id` |
| `session_id` | string | Stitched session identifier |
| `ts` | string | IST timestamp `YYYY-MM-DD HH:MM:SS` |
| `event_type` | string | `session_start`, `impression`, `view_item`, `add_to_cart`, `remove_from_cart`, `checkout_start`, `purchase`, `session_end` |
| `item_id` | string \| null | FK → `catalog.item_id` (null for session/checkout events) |
| `surface` | string | `app_open`, `feed_shelf`, `search_results`, `pdp`, `pair_widget`, `cart_page`, `checkout_screen`, `app_close` |
| `position` | int \| null | Rank position on surface (for impression events) |
| `search_query` | string \| null | Raw search text (for search events) |
| `cart_snapshot_ids` | string | Comma-separated item IDs in cart at event time |
| `metadata` | string | JSON payload (dwell time, order value, etc.) |

### `baskets.parquet` (Silver)

| Column | Type | Description |
|---|---|---|
| `order_id` | string | ULID-based unique order identifier |
| `user_id` | string | FK → `users.user_id` |
| `session_id` | string | FK → `sessions.session_id` |
| `ts` | string | Purchase timestamp |
| `item_ids` | list[string] | Ordered item IDs in the basket |
| `quantities` | list[int] | Quantities (all 1 in v1) |
| `order_value_inr` | float | Total basket value in INR |
| `removed_item_ids` | list[string] | Items added then removed in this session |

### `pair_training_examples.parquet` (Gold)

| Column | Type | Description |
|---|---|---|
| `anchor_item` | string | Item A |
| `candidate_item` | string | Item B |
| `label` | float | 1.0 = positive, 0.0 = negative |
| `weight` | float | Sample weight for loss function |
| `class` | string | `P1`, `P2`, `P3`, `N1`, `N2`, `N3` |
| `user_id` | string | Context user |
| `session_id` | string | Context session |
| `hour_bucket` | int | Hour of day (0–23) |
| `day_of_week` | int | 0=Monday, 6=Sunday |
| `cart_snapshot` | string | Cart contents at time of anchor add |

---

## Running the Pipeline

```powershell
# Full pipeline (2,000 users, ~2-5 minutes)
python run_pipeline.py --scale-factor 0.02 --run-all

# Full scale (100,000 users, needs ~30-60 minutes)
python run_pipeline.py --scale-factor 1.0 --run-all

# Run individual steps
python run_pipeline.py --run-simulation --scale-factor 0.02
python run_pipeline.py --run-silver
python run_pipeline.py --run-gold
python run_pipeline.py --run-validation
```

### Scale Factor Guide

| `--scale-factor` | Users | Approx. Time | Use Case |
|---|---|---|---|
| `0.001` | 100 | < 10 seconds | Unit tests, CI |
| `0.02` | 2,000 | 2–5 minutes | Development / smoke test |
| `0.10` | 10,000 | 15–20 minutes | Staging evaluation |
| `1.00` | 100,000 | 30–60 minutes | Full production dataset |

---

## Running the Tests

```powershell
# Run all unit tests
python run_pipeline.py --run-tests

# Or use pytest directly
pytest tests/ -v
```

| Test | What It Checks |
|---|---|
| `test_reproducibility.py` | Bronze files are byte-identical when re-run with the same seed |
| `test_leakage.py` | Gold features built as-of Week 10 contain zero events from Week 11+ |
| `test_sequence_readiness.py` | User token sequences exactly match the conformed Silver event log |

---

## Acceptance Criteria Checklist

| ID | Criterion | Verified By |
|---|---|---|
| A1 | Catalog has 18 L1 categories, log-normal prices, cold-start items | `catalog.py` spec + `validation.py` |
| A2 | Basket size right-skewed, 6–12 mean; session lengths 2–6 min | `validation.py` assertions |
| A3 | Baseline lift model recovers hand-seeded Layer 1 edges; Spearman ρ > 0.8 on public edges; substitute contamination < 1% | `validation.py` eval section |
| A4 | No temporal leakage across the as-of watermark | `tests/test_leakage.py` |
| A5 | Same seed → identical Bronze Parquet files | `tests/test_reproducibility.py` |
| A6 | `user_sequences` is a lossless, ordered reconstruction of the event stream | `tests/test_sequence_readiness.py` |
