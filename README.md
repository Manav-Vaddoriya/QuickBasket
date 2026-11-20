# QuickBasket: Complementary Product Retrieval with Semantic IDs

## 1. Project Overview
QuickBasket is an e-commerce retrieval engine designed for 10-minute grocery delivery. Its core objective is to recommend high-affinity **complementary items** (e.g., suggesting *Bread, Eggs, or Tea* when a customer adds *Milk* to their cart) across a 50,000+ SKU catalog while strictly eliminating the **"Substitute Trap"** (recommending other brands of milk).

---

## 2. Retrieval Approach Before Semantic IDs

Before introducing Semantic IDs, complementary retrieval was built through an iterative series of continuous embedding and Two-Tower contrastive models (**v3, v4, and v5**):

```
[Catalog & Co-Purchase Graph] ──► [Triplet Datasets v3/v4/v5] ──► [MiniLM-384d] ──► [Two-Tower Models v3/v4/v5] ──► [Dense Vector Search]
```

### End-to-End Pipeline Steps:
1. **Data Preparation**:
   * **Catalog Specification**: 50,029 SKUs across 18 Level-1 categories (e.g., *Dairy & Bread, Beverages*) and 47 Level-2 subcategories.
   * **Behavioral Co-Purchase Graph**: 968,658 historical basket co-occurrence edges categorized into Gold, Silver, and Bronze complementary tiers.
   * **Triplet Dataset Generation**: Mined `(Anchor A, Positive Complement P, Negative Non-Complement N)` across iterations:
     * **v3 (`triplet_dataset_v3.csv`)**: Initial random in-batch negatives.
     * **v4 (`triplet_dataset_v4.csv`)**: Evaluated item-disjoint cross-domain splits to control leakage.
     * **v5 (`triplet_dataset_v5.csv`)**: Refined hard negatives from non-associated categories and distinct brands.
2. **Dense Representation**:
   * Product text formatted as: `Product: <item_cleaned>. Category: <category>.`
   * Encoded using `sentence-transformers/all-MiniLM-L6-v2` into 384-dimensional L2-normalized dense vectors.
3. **Model Architecture (Two-Tower InfoNCE)**:
   * **Anchor Tower**: `Linear(384, 256) + Residual MLP(384 → 256 → 256)` with LayerNorm and GELU.
   * **Candidate Tower**: Symmetric `Linear(384, 256) + Residual MLP(384 → 256 → 256)`.
   * **Loss**: Symmetric InfoNCE contrastive loss with temperature $\tau = 0.05 - 0.07$.
   * **Optimizer**: AdamW (`lr=0.001`, `weight_decay=0.01`) with cosine decay over 5 epochs.
4. **Why Pre-Semantic Retrieval Models Plateaued**:
   * **The "Substitute Trap"**: Pure vector dot products prioritize syntactic and category closeness. When querying *Amul Milk*, other brands of milk (*Mother Dairy, Nandini*) share almost identical embedding space and push out true complements (*Bread, Eggs*).
   * **Performance Ceiling**: Across v3, v4, and v5, full-catalog Hit@10 plateaued at **~1%** (and **6.0%** on held-out gold anchors).

---

## 3. Introducing Semantic IDs

**Semantic IDs** replace opaque continuous vectors with compact, discrete, hierarchical tokens. 

Generated via **Residual-Quantized Variational Autoencoders (RQ-VAE)**, Semantic IDs map each product into a hierarchical sequence of discrete codebook indices $(c_0, c_1, c_2)$ that progressively capture:
* **$c_0$ (Level 0)**: Broad functional domain (e.g., Breakfast Staples).
* **$c_1$ (Level 1)**: Subcategory aisle (e.g., Fresh Dairy & Liquid Milks).
* **$c_2$ (Level 2)**: Fine-grained product cluster (e.g., Toned Cow Milk, 1L packaging).

---

## 4. Semantic ID Generation & Zero-Collision Architecture

### Model Architecture & Training
* **Encoder**: `384 → 512 → 256 → 64` (Latent dimension: 64).
* **Quantizer**: 3 residual levels, codebook size $K = 128$ per level (Total capacity: $128^3 = 2,097,152$).
* **Decoder**: `64 → 256 → 512 → 384` (Reconstructing 384d MiniLM embeddings).
* **Training Optimizations**:
  * **Warm K-Means Codebook Initialization**: Initialized codebooks sequentially on residual vectors, preventing early dead codes.
  * **Straight-Through Estimator (STE)**: Allows gradient backpropagation through non-differentiable $\arg\min$ codebook selection.
  * **Commitment Loss**: Weighted at $\beta = 0.25$ to stabilize encoder representations.

### Achieving 0% Collision (Deterministic Disambiguation)
Quantizing 50,015 products into discrete codes naturally causes near-identical items (e.g., 500ml vs 1L variants, or different brands of identical products) to share the same base code $(c_0, c_1, c_2)$:
* **Base Unique IDs**: 20,795 unique tuples for 50,015 products.
* **Deterministic Suffix Disambiguation**: For every collision cluster sharing $(c_0, c_1, c_2)$, products are sorted deterministically and assigned an incremental suffix index $i \ge 0$:
  `Semantic ID = <code_0>_<code_1>_<code_2>_<suffix>` (e.g. `122_75_38_0`)
* **Result**: **0.00% collisions (100% unique IDs across all 50,015 items)** while strictly preserving cluster identity $(c_0, c_1, c_2)$ for instant substitute masking.

### Concrete Examples

| Product Name | Category | Base Code $(c_0, c_1, c_2)$ | Disambiguated Semantic ID | Relationship to Amul Milk |
|---|---|:---:|:---:|---|
| **Amul Taaza Toned Milk 1L** | Dairy & Bread > Milk | `122_75_38` | `122_75_38_0` | **Anchor** |
| **Mother Dairy Toned Milk 1L** | Dairy & Bread > Milk | `122_75_38` | `122_75_38_1` | **Substitute (Identical Base Code: BLOCKED)** |
| **Nandini Fresh Toned Milk 1L** | Dairy & Bread > Milk | `122_75_38` | `122_75_38_2` | **Substitute (Identical Base Code: BLOCKED)** |
| **English Oven White Bread** | Dairy & Bread > Bread | `83_44_12` | `83_44_12_0` | **Complement (Different Base Code: ALLOWED)** |
| **Farm Fresh Omega 3 Eggs** | Meat & Eggs > Eggs | `119_83_36` | `119_83_36_1` | **Complement (Different Base Code: ALLOWED)** |

---

## 5. Metrics: Comparison Across Retrieval Models

### Pre-Semantic Retrieval Models vs. Semantic Hybrid Search (Full 50k Catalog)
Evaluated on gold/silver complementary test sets across the entire 50,029-item catalog:

| Retrieval Model | Description / Training Configuration | Hit@5 (%) | Hit@10 (%) | MRR | NDCG@10 | Substitute Leakage (%) |
|---|---|---:|---:|---:|---:|---:|
| **Raw Embeddings Baseline** | Zero-shot `MiniLM-L6-v2` (384d Cosine Similarity) | 4.0% | 6.0% | 0.0284 | 0.0120 | 0.07% |
| **Two-Tower Model v3** | Initial InfoNCE model trained on `triplet_dataset_v3` | 3.1% | 6.1% | 0.0196 | 0.0157 | 0.00% |
| **Two-Tower Model v4** | Cross-domain InfoNCE model on `triplet_dataset_v4` | 2.2% | 4.8% | 0.0143 | 0.0117 | 0.00% |
| **Two-Tower Model v5** | Hard-negative InfoNCE model on `triplet_dataset_v5` | 4.0% | 6.0% | 0.0284 | 0.0120 | 0.00% |
| **Semantic Hybrid Search** | **Two-Tower v5 + RQ-VAE Semantic IDs + Graph RRF** | **88.7%** | **93.3%** | **0.7821** | **0.3638** | **0.00%** |

### How Semantic Hybrid Search Solved the Plateau:
Candidates are scored and fused using **Reciprocal Rank Fusion (RRF, $k=60$)**:
$$\text{Score}(c) = \frac{\alpha_{\text{sem}}}{60 + \text{rank}_{\text{sem}}(c)} + \frac{\alpha_{\text{dense}}}{60 + \text{rank}_{\text{dense}}(c)} + \frac{\alpha_{\text{graph}}}{60 + \text{rank}_{\text{graph}}(c)}$$

* $\alpha_{\text{sem}} = 0.40$: Discrete $(c_0, c_1)$ cluster transitions + strict base-ID substitute filter.
* $\alpha_{\text{dense}} = 0.35$: Two-Tower v5 continuous neural projections for fine-grained ranking.
* $\alpha_{\text{graph}} = 0.25$: Empirical co-purchase graph priors (Gold/Silver/Bronze tiers).

---

## 6. End-to-End Retrieval Example

**Query Item**: `Amul Taaza Toned Milk 1L` (Semantic ID: `122_75_38_0`)

```
Input: "Amul Taaza Toned Milk 1L" [122_75_38_0]
   ├── [Anti-Substitute Filter]: Masks all candidates with base code `122_75_38` -> -inf
   ├── [Semantic ID Transitions]: Cluster (122, 75) activates Bread (83), Eggs (119), Coffee/Tea (74)
   ├── [Two-Tower Projections]: Ranks high-affinity breakfast items
   └── [Category Throttling]: Max 2 items per category_l2
```

### Retrieved Top-5 Basket Recommendations:

| Rank | Recommended Product | Category | Semantic ID | Modality Match |
|:---:|---|---|:---:|---|
| **1** | **Farm Fresh Omega 3 Rich Eggs** | Meat & Seafood > Fresh Meat & Eggs | `119_83_36_1` | Graph Gold Tier + Semantic Transition |
| **2** | **Ayur Egg Immune** | Meat & Seafood > Fresh Meat & Eggs | `119_123_65_1` | Graph Gold Tier + Semantic Transition |
| **3** | **Nescafe Super Value Filter Coffee** | Beverages > Coffee | `74_112_94_0` | High Two-Tower Dense Match + Semantic Transition |
| **4** | **English Oven Sandwich White Bread** | Bakery & Dairy > Breads | `83_44_12_0` | High Semantic Transition + Dense Match |
| **5** | **Tata Tea Gold Leaf Tea** | Beverages > Tea | `74_89_15_0` | High Semantic Transition + Graph Tier |

* **Substitute Suppression in Action**: *Mother Dairy Milk* (`122_75_38_1`) and *Nandini Milk* (`122_75_38_2`) were strictly blocked ($-\infty$) by the base Semantic ID filter.
* **Diversity in Action**: Enforcing `max_per_category=2` delivered a well-rounded breakfast basket (*Eggs + Coffee + Bread + Tea*) rather than a redundant list of five breads.
