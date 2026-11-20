"""
QuickBasket Hybrid Complement Search – Comprehensive Evaluation & Demo Script
Tests retrieval across single items and multi-item shopping carts using generated Semantic IDs.
"""

import os
import sys
import time
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.retrieval.hybrid_complement_search import HybridComplementSearchEngine


def main():
    print("=" * 80)
    print("QUICKBASKET HYBRID COMPLEMENT SEARCH DEMONSTRATION")
    print("=" * 80)

    engine = HybridComplementSearchEngine()
    engine.load()

    test_queries = [
        "Amul Milk",
        "Tata Tea Gold Leaf Tea 1 kg + Leaf Tea 250 g",
        "English Oven Bread",
        "Lizol Disinfectant Floor Cleaner Liquid - Pine, Kills 99.9% Germs",
        "Maggi Chicken Noodles"
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY ANCHOR ITEM: '{q}'")
        print("=" * 80)
        
        # Test query
        res = engine.get_complements(
            query_item=q,
            top_k=8,
            fusion_method="rrf",
            filter_substitutes=True,
            max_per_category=2
        )
        anchor = res["anchor"]
        print(f"Anchor Product: {anchor['item_name']} (SKU: {anchor['item_id']})")
        print(f"Category: {anchor['category_l1']} > {anchor['category_l2']}")
        print(f"Semantic ID: `{anchor['semantic_id_disambiguated']}` (Base tuple: `{anchor['semantic_id']}`)")
        print("\nTop Retrieved Complementary Products (Hybrid Search):")
        print("-" * 105)
        print(f"{'#':<3} | {'Product Name':<35} | {'Category':<22} | {'Semantic ID':<15} | {'Tier':<6} | {'Score':<7}")
        print("-" * 105)
        
        for c in res["complements"]:
            name = c["item_name"][:32] + ("..." if len(c["item_name"]) > 32 else "")
            cat = f"{c['category_l1']} > {c['category_l2']}"
            if len(cat) > 20:
                cat = cat[:19] + ".."
            print(f"{c['rank']:<3} | {name:<35} | {cat:<22} | {c['semantic_id_disambiguated']:<15} | {c['graph_tier']:<6} | {c['hybrid_score']:<7.4f}")
        print("-" * 105)

    # Multi-item cart test
    print("\n" + "=" * 80)
    print("MULTI-ITEM SHOPPING CART COMPLEMENT RETRIEVAL TEST")
    print("Cart: ['Amul Milk', 'Tata Tea Gold Leaf Tea 1 kg + Leaf Tea 250 g']")
    print("=" * 80)
    cart_res = engine.get_cart_complements(
        cart_items=["Amul Milk", "Tata Tea Gold Leaf Tea 1 kg + Leaf Tea 250 g"],
        top_k=8,
        max_per_category=2
    )
    print("\nTop Cart Complements (Pooled across both items):")
    print("-" * 100)
    print(f"{'#':<3} | {'Product Name':<35} | {'Category':<25} | {'Semantic ID':<15} | {'Cart Score':<10}")
    print("-" * 100)
    for c in cart_res["complements"]:
        name = c["item_name"][:32] + ("..." if len(c["item_name"]) > 32 else "")
        cat = f"{c['category_l1']} > {c['category_l2']}"
        if len(cat) > 23:
            cat = cat[:22] + ".."
        print(f"{c['rank']:<3} | {name:<35} | {cat:<25} | {c['semantic_id_disambiguated']:<15} | {c['cart_affinity_score']:<10.4f}")
    print("-" * 100)

    print("\n[+] Demonstration successfully completed!")


if __name__ == "__main__":
    main()
