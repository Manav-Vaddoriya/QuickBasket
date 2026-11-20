"""
QuickBasket Catalogue Preprocessing & Structured Text Construction Pipeline.

Covers:
- Stage 1: Data profiling, quality auditing, text normalization, and deliverable exports.
- Stage 2: Structured product-text construction for representation learning.
"""

import os
import re
import json
import unicodedata
import yaml
import pandas as pd
import numpy as np


def load_config(config_path: str = "QuickBasket-Semantic-IDs/configs/config.yaml") -> dict:
    """Loads configuration file with fallback defaults."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def normalize_product_name(text: str) -> tuple[str, bool]:
    """
    Cleans and standardizes product names safely without destroying catalogue semantics.
    """
    if not isinstance(text, str):
        return "", True
    
    original = text
    cleaned = text
    
    # 1. Normalize mojibake artifacts (e.g. 'Â ' and 'Â\xa0' from UTF-8 / Windows-1252 mismatch)
    cleaned = re.sub(r'Â[\xa0\s]*', ' ', cleaned)
    
    # 2. Normalize smart quotes and typographic dashes to standard ASCII
    cleaned = cleaned.replace('–', '-').replace('—', '-')
    cleaned = cleaned.replace('’', "'").replace('‘', "'")
    cleaned = cleaned.replace('“', '"').replace('”', '"')
    
    # 3. Unicode NFKC normalization
    cleaned = unicodedata.normalize('NFKC', cleaned)
    
    # 4. Collapse repeated whitespace and strip ends
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    was_modified = (cleaned != original)
    return cleaned, was_modified


def construct_product_text(item: str, category: str) -> str:
    """
    Constructs a deterministic, structured textual representation from item and category.
    
    Format:
        "Product: {item_clean}. Category: {category_clean}."
    
    Guarantees:
    1. Preserves meaningful product-name information (brands, units, flavors, variants).
    2. Injects authoritative category context.
    3. Avoids hallucinated or synthetic attributes.
    4. Safe handling of empty or malformed strings.
    """
    item_str = str(item).strip() if pd.notnull(item) else ""
    cat_str = str(category).strip() if pd.notnull(category) else ""
    
    # Clean any trailing periods from values to prevent double periods
    if item_str.endswith('.'):
        item_str = item_str[:-1].strip()
    if cat_str.endswith('.'):
        cat_str = cat_str[:-1].strip()
        
    return f"Product: {item_str}. Category: {cat_str}."


def audit_suspect_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies potential cross-category misclassifications via keyword signatures."""
    category_signatures = {
        "Dairy & Bread": [r"\bmilk\b", r"\bpaneer\b", r"\bcurd\b", r"\byogurt\b", r"\bbutter\b", r"\bcheese\b", r"\bbread\b", r"\bpav\b", r"\bbun\b", r"\bghee\b"],
        "Snacks & Biscuits": [r"\bbiscuit\b", r"\bcookie\b", r"\brusk\b", r"\bnamkeen\b", r"\bchips\b", r"\bwafer\b", r"\bbhujia\b", r"\bcracker\b"],
        "Beverages": [r"\btea\b", r"\bcoffee\b", r"\bjuice\b", r"\bsoda\b", r"\bsyrup\b", r"\benergy drink\b", r"\bmineral water\b"],
        "Atta/Rice/Dal": [r"\batta\b", r"\brice\b", r"\bdal\b", r"\bflour\b", r"\bpulses\b", r"\bchana\b", r"\brajma\b", r"\bbesan\b", r"\bsooji\b"],
        "Pharma & Wellness": [r"\btablet\b", r"\bcapsule\b", r"\bbalm\b", r"\bantiseptic\b", r"\bvitamin\b", r"\bcalcium\b", r"\bcrocin\b", r"\bvicks\b", r"\beno\b"],
        "Personal Care": [r"\bshampoo\b", r"\bconditioner\b", r"\bsoap\b", r"\bbody wash\b", r"\bdeodorant\b", r"\bface wash\b", r"\btoothpaste\b", r"\btoothbrush\b", r"\blotion\b"],
        "Home Care": [r"\bdetergent\b", r"\bfloor cleaner\b", r"\bdishwash\b", r"\bmop\b", r"\bbroom\b", r"\bmosquito repellent\b", r"\btrash bag\b", r"\baluminium foil\b"],
        "Baby Care": [r"\bdiaper\b", r"\bbaby wipes\b", r"\bbaby oil\b", r"\bbaby lotion\b", r"\bfeeding bottle\b", r"\bcerelac\b"],
        "Pet Care": [r"\bdog food\b", r"\bcat food\b", r"\bpuppy\b", r"\bkitten\b", r"\bpedigree\b", r"\bwhiskas\b", r"\bdrools\b"],
        "Stationery & Office": [r"\bball pen\b", r"\bgel pen\b", r"\bpencil\b", r"\bnotebook\b", r"\bmarker\b", r"\bglue stick\b", r"\bstapler\b"],
        "Instant Food": [r"\bnoodles\b", r"\bpasta\b", r"\binstant soup\b", r"\bready to eat\b", r"\bpoha mix\b", r"\bupma mix\b"],
        "Chocolates & Sweets": [r"\bchocolate\b", r"\bcandy\b", r"\btoffee\b", r"\bladdoo\b", r"\bbarfi\b", r"\bgulab jamun\b"],
        "Ice Creams & Frozen Desserts": [r"\bice cream\b", r"\bkulfi\b", r"\bsundae\b", r"\bchocobar\b", r"\bfrozen dessert\b"],
        "Fruits & Vegetables": [r"\bfresh fruit\b", r"\bfresh vegetable\b", r"\bapple\b", r"\bbanana\b", r"\bpotato\b", r"\bonion\b"],
        "Meat & Seafood": [r"\bchicken\b", r"\bmutton\b", r"\bfish\b", r"\bprawn\b", r"\bmeat\b", r"\begg\b"],
        "Breakfast Cereals": [r"\bcorn flakes\b", r"\bmuesli\b", r"\bgranola\b", r"\bchocos\b", r"\boats\b"],
        "Paan Corner": [r"\bpaan\b", r"\bsupari\b", r"\belaichi\b", r"\bmouth freshener\b"]
    }
    
    suspects = []
    for idx, row in df.iterrows():
        item_lower = row["item"].lower()
        curr_cat = row["category"]
        for cat, patterns in category_signatures.items():
            if cat == curr_cat:
                continue
            for pat in patterns:
                if re.search(pat, item_lower):
                    keyword = pat.replace(r"\b", "")
                    if keyword == "wheat" and "bread" in item_lower and curr_cat == "Dairy & Bread":
                        continue
                    if keyword == "milk" and curr_cat in ["Personal Care", "Chocolates & Sweets"]:
                        continue
                    suspects.append({
                        "index": idx,
                        "product_id": row.get("product_id", f"QB_{idx:05d}"),
                        "item": row["item"],
                        "assigned_category": curr_cat,
                        "suggested_category": cat,
                        "matching_keyword": keyword
                    })
                    break
    return pd.DataFrame(suspects)


def run_pipeline(
    input_csv: str = "QuickBasket-Semantic-IDs/data/catalog_items.csv",
    output_dir: str = "QuickBasket-Semantic-IDs",
) -> dict:
    print("=" * 70)
    print("STAGE 1 & 2: QUICKBASKET CATALOGUE PREPROCESSING & TEXT CONSTRUCTION")
    print("=" * 70)

    # 1. Ingestion
    assert os.path.exists(input_csv), f"Input file not found: {input_csv}"
    raw_df = pd.read_csv(input_csv)
    total_rows, total_cols = raw_df.shape
    assert total_rows == 50015, f"Expected 50,015 products, but found {total_rows}."

    # 2. Stable IDs and Cleaning
    df = raw_df.copy()
    df["product_id"] = [f"QB_{i:05d}" for i in range(len(df))]
    
    cleaned_results = [normalize_product_name(x) for x in df["item"]]
    df["item_cleaned"] = [res[0] for res in cleaned_results]
    df["was_cleaned"] = [res[1] for res in cleaned_results]
    num_cleaned = int(df["was_cleaned"].sum())

    df["name_length"] = df["item_cleaned"].str.len()
    df["word_count"] = df["item_cleaned"].apply(lambda x: len(x.split()))

    # 3. Stage 2: Product Text Construction
    print("\n[Stage 2] Constructing Structured Product Texts...")
    df["product_text"] = [
        construct_product_text(item, cat)
        for item, cat in zip(df["item_cleaned"], df["category"])
    ]
    
    # Assertions for Stage 2
    assert not df["product_text"].isnull().any(), "Found null values in product_text!"
    assert (df["product_text"].str.strip() != "").all(), "Found empty product_text strings!"
    assert df["product_text"].str.startswith("Product: ").all(), "Text does not start with 'Product: '!"
    assert df["product_text"].str.contains(r"\. Category: ").all(), "Text does not contain '. Category: '!"
    print(f"    - All {len(df):,} product texts constructed and verified successfully.")
    print(f"    - Sample: '{df['product_text'].iloc[0]}'")

    # 4. Profiling and Reports
    cat_counts = df["category"].value_counts()
    cat_report_rows = []
    for cat, count in cat_counts.items():
        sub = df[df["category"] == cat]
        cat_report_rows.append({
            "category": cat,
            "product_count": count,
            "avg_name_length": round(float(sub["name_length"].mean()), 1),
            "min_name_length": int(sub["name_length"].min()),
            "max_name_length": int(sub["name_length"].max()),
            "unique_first_words": sub["item_cleaned"].apply(lambda x: x.split()[0] if x.split() else "").nunique(),
            "pct_of_total": round(float(count / total_rows * 100), 2)
        })
    cat_dist_df = pd.DataFrame(cat_report_rows)
    suspects_df = audit_suspect_categories(df)

    # 5. Export Deliverables
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)
    
    # Stage 1 Deliverables
    df[["product_id", "item", "category", "item_cleaned", "name_length", "word_count", "was_cleaned"]].to_csv(
        os.path.join(output_dir, "data", "catalogue_profile.csv"), index=False
    )
    df[["product_id", "item", "category", "item_cleaned"]].to_csv(
        os.path.join(output_dir, "data", "cleaned_catalogue.csv"), index=False
    )
    cat_dist_df.to_csv(os.path.join(output_dir, "outputs", "category_distribution_report.csv"), index=False)
    suspects_df.to_csv(os.path.join(output_dir, "outputs", "suspect_category_assignments.csv"), index=False)
    
    # Stage 2 Deliverable: product_texts.csv
    texts_df = df[["product_id", "item", "category", "item_cleaned", "product_text"]]
    texts_path = os.path.join(output_dir, "data", "product_texts.csv")
    texts_df.to_csv(texts_path, index=False)
    print(f"[+] Saved Stage 2 Deliverable: {texts_path} ({os.path.getsize(texts_path):,} bytes)")

    summary_dict = {
        "total_products": total_rows,
        "total_categories": len(cat_counts),
        "items_cleaned": num_cleaned,
        "suspect_category_assignments": len(suspects_df),
        "product_texts_generated": len(texts_df),
        "text_min_chars": int(df["product_text"].str.len().min()),
        "text_max_chars": int(df["product_text"].str.len().max()),
        "text_mean_chars": round(float(df["product_text"].str.len().mean()), 2),
        "text_mean_words": round(float(df["product_text"].apply(lambda x: len(x.split())).mean()), 2),
    }
    
    with open(os.path.join(output_dir, "outputs", "data_quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print("\n" + "=" * 70)
    print("STAGE 1 & 2 PIPELINE EXECUTION COMPLETE AND VALIDATED!")
    print("=" * 70)
    return summary_dict


if __name__ == "__main__":
    run_pipeline()
