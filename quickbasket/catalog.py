import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from quickbasket.config import DEFAULT_NUM_SKUS, WORLD_SEED
from quickbasket.ulid import new_ulid

# Level-1 and Level-2 structure with realistic brands, pack sizes, veg tags, and shelf life.
# The prices are log-normally sampled, fitted to a price range [price_min, price_max].
CATALOG_SPEC = {
    "Dairy & Bread": {
        "Milk": {"brands": ["Amul", "Mother Dairy", "Nandini"], "pack_sizes": ["500ml", "1L"], "is_veg": True, "shelf_life_days": 2, "price_min": 25, "price_max": 90, "variants": ["Toned", "Full Cream", "Cow", "Taaza", "Gold", "Organic", "Slim"]},
        "Bread": {"brands": ["Britannia", "Harvest Gold", "English Oven"], "pack_sizes": ["400g", "700g"], "is_veg": True, "shelf_life_days": 5, "price_min": 30, "price_max": 75, "variants": ["White", "Brown", "Whole Wheat", "Multigrain", "Milk", "Sandwich"]},
        "Butter": {"brands": ["Amul", "Mother Dairy", "Nutan"], "pack_sizes": ["100g", "500g"], "is_veg": True, "shelf_life_days": 30, "price_min": 50, "price_max": 275, "variants": ["Pasteurized", "Unsalted", "Garlic & Herb", "Classic"]},
        "Cheese": {"brands": ["Amul", "Britannia", "Go Cheese"], "pack_sizes": ["200g", "400g"], "is_veg": True, "shelf_life_days": 90, "price_min": 90, "price_max": 350, "variants": ["Processed", "Cheddar", "Mozzarella", "Slices", "Cubes"]},
        "Paneer": {"brands": ["Amul", "Mother Dairy", "Nandini", "Milky Mist"], "pack_sizes": ["200g", "400g", "500g"], "is_veg": True, "shelf_life_days": 7, "price_min": 80, "price_max": 220, "variants": ["Fresh", "Malai", "Low Fat", "Organic"]},
        "Curd": {"brands": ["Amul", "Mother Dairy", "Nandini", "Milky Mist"], "pack_sizes": ["200g", "400g", "1kg"], "is_veg": True, "shelf_life_days": 7, "price_min": 35, "price_max": 120, "variants": ["Plain", "Thick", "Set", "Low Fat", "Organic"]},
        "Ghee": {"brands": ["Amul", "Mother Dairy", "Gowardhan", "Patanjali"], "pack_sizes": ["200ml", "500ml", "1L"], "is_veg": True, "shelf_life_days": 270, "price_min": 120, "price_max": 750, "variants": ["Pure Cow", "Desi", "A2 Cow", "Organic", "Danedar"]},
    },
    "Beverages": {
        "Coffee": {"brands": ["Nescafe", "Bru", "Continental", "Tata"], "pack_sizes": ["50g", "100g", "200g"], "is_veg": True, "shelf_life_days": 365, "price_min": 90, "price_max": 650, "variants": ["Instant", "Gold", "Filter", "Classic", "Dark Roast"]},
        "Tea": {"brands": ["Tata Tea", "Red Label", "Taj Mahal", "Society"], "pack_sizes": ["100g", "250g", "500g"], "is_veg": True, "shelf_life_days": 365, "price_min": 40, "price_max": 400, "variants": ["Masala", "Gold", "Green", "Ginger", "Lemon", "Dust"]},
        "Soft Drinks": {"brands": ["Pepsi", "Coca-Cola", "Sprite", "Thums Up"], "pack_sizes": ["250ml", "750ml", "1.25L", "2L"], "is_veg": True, "shelf_life_days": 180, "price_min": 20, "price_max": 100, "variants": ["Diet", "Zero Sugar", "Classic", "Lime", "Original"]},
        "Juices": {"brands": ["Real", "Tropicana", "Paper Boat"], "pack_sizes": ["200ml", "1L"], "is_veg": True, "shelf_life_days": 180, "price_min": 20, "price_max": 120, "variants": ["Orange", "Mixed Fruit", "Apple", "Guava", "Mango"]},
        "Flavored Milk": {"brands": ["Cavins", "Amul Kool", "Paper Boat"], "pack_sizes": ["180ml", "200ml"], "is_veg": True, "shelf_life_days": 120, "price_min": 30, "price_max": 60, "variants": ["Chocolate", "Kesar Badam", "Strawberry", "Vanilla"]},
        "Cold Coffee": {"brands": ["Nescafe Cold", "Amul Kool", "Paper Boat"], "pack_sizes": ["180ml", "200ml"], "is_veg": True, "shelf_life_days": 120, "price_min": 35, "price_max": 90, "variants": ["Classic", "Hazelnut", "Intense", "Sweet Latte"]},
        "Sparkling Water": {"brands": ["Perrier", "Bisleri", "Schweppes"], "pack_sizes": ["500ml", "1L"], "is_veg": True, "shelf_life_days": 365, "price_min": 30, "price_max": 150, "variants": ["Original", "Lime & Lemon", "Peach", "Unflavored"]},
        "Flavored Water": {"brands": ["Vedica", "Himalayan", "Bisleri"], "pack_sizes": ["500ml", "1L"], "is_veg": True, "shelf_life_days": 365, "price_min": 20, "price_max": 80, "variants": ["Peach", "Lemon", "Berry", "Green Apple"]},
    },
    "Snacks & Biscuits": {
        "Biscuits": {"brands": ["Britannia", "Parle", "Sunfeast", "Oreo"], "pack_sizes": ["50g", "100g", "150g", "300g"], "is_veg": True, "shelf_life_days": 180, "price_min": 10, "price_max": 120, "variants": ["Marie", "Bourbon", "Digestive", "Glucose", "NutriChoice"]},
        "Potato Chips": {"brands": ["Lay's", "Kurkure", "Pringles", "Bingo"], "pack_sizes": ["30g", "50g", "90g", "110g"], "is_veg": True, "shelf_life_days": 120, "price_min": 10, "price_max": 150, "variants": ["Classic Salted", "Cream & Onion", "Magic Masala", "Spanish Tomato", "Sizzling Hot"]},
        "Namkeen": {"brands": ["Haldiram's", "Bikaji", "Bikanervala"], "pack_sizes": ["150g", "400g"], "is_veg": True, "shelf_life_days": 180, "price_min": 20, "price_max": 150, "variants": ["Bhujia Sev", "Aloo Bhujia", "Khatta Meetha", "Navratan", "Moong Dal"]},
    },
    "Fruits & Vegetables": {
        "Fresh Vegetables": {"brands": ["Fresho", "Local Farmers"], "pack_sizes": ["250g", "500g", "1kg"], "is_veg": True, "shelf_life_days": 3, "price_min": 10, "price_max": 100, "variants": ["Organic", "Hydroponic", "Local", "Cleaned"]},
        "Fresh Fruits": {"brands": ["Fresho", "Local Farmers"], "pack_sizes": ["250g", "500g", "1kg"], "is_veg": True, "shelf_life_days": 4, "price_min": 30, "price_max": 250, "variants": ["Imported", "Organic", "Sweet", "Fresh"]},
        "Herbs": {"brands": ["Fresho"], "pack_sizes": ["50g", "100g"], "is_veg": True, "shelf_life_days": 2, "price_min": 5, "price_max": 40, "variants": ["Fresh", "Organic", "Bunch"]},
        "Cut Fruits": {"brands": ["Fresho", "Farmley"], "pack_sizes": ["200g", "400g"], "is_veg": True, "shelf_life_days": 2, "price_min": 40, "price_max": 150, "variants": ["Fresh Cut", "Mix Bowl", "Diced"]},
        "Exotic Vegetables": {"brands": ["Fresho", "Local Farmers"], "pack_sizes": ["200g", "250g"], "is_veg": True, "shelf_life_days": 4, "price_min": 60, "price_max": 250, "variants": ["Imported", "Organic", "Select"]},
        "Mushrooms": {"brands": ["Fresho", "Local Farmers"], "pack_sizes": ["200g"], "is_veg": True, "shelf_life_days": 4, "price_min": 50, "price_max": 120, "variants": ["Button", "Organic", "Sliced"]},
    },
    "Atta/Rice/Dal": {
        "Atta": {"brands": ["Aashirvaad", "Pillsbury", "Fortune"], "pack_sizes": ["1kg", "5kg", "10kg"], "is_veg": True, "shelf_life_days": 90, "price_min": 50, "price_max": 450, "variants": ["Chakki Fresh", "Whole Wheat", "Multigrain", "Select", "Fortified"]},
        "Rice": {"brands": ["India Gate", "Daawat", "Fortune"], "pack_sizes": ["1kg", "5kg"], "is_veg": True, "shelf_life_days": 365, "price_min": 60, "price_max": 650, "variants": ["Basmati", "Rozana", "Dubar", "Tibbar", "Kolam", "Sona Masoori"]},
        "Dal": {"brands": ["Tata Sampann", "Fortune", "Organic"], "pack_sizes": ["500g", "1kg"], "is_veg": True, "shelf_life_days": 180, "price_min": 60, "price_max": 180, "variants": ["Toor", "Moong Yellow", "Chana Dal", "Urad Dal", "Masoor Dal"]},
        "Poha": {"brands": ["Fortune", "Tata Sampann", "MTR"], "pack_sizes": ["500g", "1kg"], "is_veg": True, "shelf_life_days": 180, "price_min": 35, "price_max": 100, "variants": ["Thick", "Thin", "Organic"]},
        "Sabudana": {"brands": ["Fortune", "Tata Sampann", "MTR"], "pack_sizes": ["500g", "1kg"], "is_veg": True, "shelf_life_days": 180, "price_min": 45, "price_max": 120, "variants": ["Medium Grain", "Organic", "Big Pearl"]},
        "Besan": {"brands": ["Fortune", "Rajdhani", "Tata Sampann"], "pack_sizes": ["500g", "1kg"], "is_veg": True, "shelf_life_days": 180, "price_min": 45, "price_max": 130, "variants": ["Fine", "Coarse", "Chana", "Organic"]},
        "Sooji": {"brands": ["Fortune", "Rajdhani", "Tata Sampann"], "pack_sizes": ["500g", "1kg"], "is_veg": True, "shelf_life_days": 180, "price_min": 40, "price_max": 110, "variants": ["Fine", "Rava", "Organic"]},
    },
    "Instant Food": {
        "Noodles": {"brands": ["Maggi", "Yippee", "Chings", "Nissin Top Ramen"], "pack_sizes": ["70g", "280g"], "is_veg": True, "shelf_life_days": 180, "price_min": 14, "price_max": 96, "variants": ["Masala", "Atta", "Schezwan", "Chicken", "Soupy"]},
        "Pasta": {"brands": ["Bambino", "Maggi", "Del Monte", "Disano"], "pack_sizes": ["200g", "500g"], "is_veg": True, "shelf_life_days": 365, "price_min": 25, "price_max": 150, "variants": ["Penne", "Fusilli", "Macaroni", "Spaghetti", "Whole Wheat"]},
        "Ready to Eat": {"brands": ["MTR", "Kitchens of India", "Haldiram's", "Tata Q"], "pack_sizes": ["250g", "300g", "400g"], "is_veg": True, "shelf_life_days": 180, "price_min": 60, "price_max": 250, "variants": ["Paneer Butter Masala", "Dal Makhani", "Rajma Chawal", "Pav Bhaji"]},
        "Soups": {"brands": ["Knorr", "Chings", "Maggi"], "pack_sizes": ["40g", "60g", "packs_of_4"], "is_veg": True, "shelf_life_days": 365, "price_min": 15, "price_max": 120, "variants": ["Tomato", "Manchow", "Hot & Sour", "Sweet Corn", "Chicken Clear"]},
        "Frozen Snacks": {"brands": ["McCain", "Godrej Yummiez", "ITC Master Chef"], "pack_sizes": ["320g", "500g", "1kg"], "is_veg": True, "shelf_life_days": 365, "price_min": 90, "price_max": 450, "variants": ["French Fries", "Alu Tikki", "Veg Nuggets", "Cheese Balls"]},
    },
    "Baby Care": {
        "Diapers": {"brands": ["Pampers", "Huggies", "MamyPoko"], "pack_sizes": ["packs_of_10", "packs_of_30", "packs_of_50"], "is_veg": True, "shelf_life_days": 1095, "price_min": 150, "price_max": 1200, "variants": ["Pants Style", "Tape Style", "Extra Care", "Ultra Soft"]},
        "Baby Wipes": {"brands": ["Himalaya", "Johnson's", "Pampers"], "pack_sizes": ["packs_of_80"], "is_veg": True, "shelf_life_days": 730, "price_min": 80, "price_max": 250, "variants": ["Gentle Clean", "Fragrance Free", "Aloe Vera", "Pure Water"]},
        "Baby Lotion": {"brands": ["Himalaya", "Johnson's"], "pack_sizes": ["100ml", "200ml"], "is_veg": True, "shelf_life_days": 730, "price_min": 90, "price_max": 350, "variants": ["Daily Moisture", "Nourishing", "Calming", "Hypoallergenic"]},
    },
    "Personal Care": {
        "Shampoos": {"brands": ["Dove", "Head & Shoulders", "L'Oreal", "Clinic Plus"], "pack_sizes": ["80ml", "180ml", "340ml"], "is_veg": True, "shelf_life_days": 730, "price_min": 60, "price_max": 450, "variants": ["Anti-Dandruff", "Hair Fall Rescue", "Daily Shine", "Intense Repair"]},
        "Soaps": {"brands": ["Dettol", "Lifebuoy", "Dove", "Pears"], "pack_sizes": ["75g", "125g"], "is_veg": True, "shelf_life_days": 730, "price_min": 25, "price_max": 180, "variants": ["Original Antiseptic", "Cool Refreshing", "Deep Moisture", "Pure & Gentle"]},
        "Toothpaste": {"brands": ["Colgate", "Sensodyne", "Pepsodent", "Dabur"], "pack_sizes": ["100g", "200g"], "is_veg": True, "shelf_life_days": 730, "price_min": 40, "price_max": 250, "variants": ["Total Care", "Advanced White", "Herbal Protection", "Sensitivity Relief"]},
    },
    "Home Care": {
        "Detergents": {"brands": ["Surf Excel", "Ariel", "Tide"], "pack_sizes": ["500g", "1kg", "3kg"], "is_veg": True, "shelf_life_days": 730, "price_min": 60, "price_max": 500, "variants": ["Matic Front Load", "Matic Top Load", "Ultra Wash", "Liquid"]},
        "Dishwashers": {"brands": ["Vim", "Pril"], "pack_sizes": ["250ml", "500ml", "750ml"], "is_veg": True, "shelf_life_days": 730, "price_min": 20, "price_max": 150, "variants": ["Lemon Gel", "Neem Liquid", "Tub Bar", "Concentrated"]},
        "Garbage Bags": {"brands": ["Shalimar", "CleanUp"], "pack_sizes": ["packs_of_15", "packs_of_30"], "is_veg": True, "shelf_life_days": 1095, "price_min": 40, "price_max": 120, "variants": ["Medium Size", "Large Size", "Oxo-Biodegradable", "Heavy Duty"]},
    },
    "Pet Care": {
        "Dog Food": {"brands": ["Pedigree", "Royal Canin"], "pack_sizes": ["1kg", "3kg", "10kg"], "is_veg": False, "shelf_life_days": 540, "price_min": 200, "price_max": 3500, "variants": ["Adult Chicken", "Puppy Small Breed", "Meat & Rice"]},
        "Cat Food": {"brands": ["Whiskas", "Drools"], "pack_sizes": ["1kg", "3kg"], "is_veg": False, "shelf_life_days": 540, "price_min": 180, "price_max": 1200, "variants": ["Ocean Fish Adult", "Kitten Chicken", "Tuna in Jelly"]},
        "Pet Treats": {"brands": ["Pedigree", "Meatables", "Purepet"], "pack_sizes": ["100g", "250g"], "is_veg": False, "shelf_life_days": 365, "price_min": 80, "price_max": 450, "variants": ["Chicken Strips", "Crunchy Biscuits", "Munchy Sticks"]},
        "Dog Chews": {"brands": ["Dentastix", "Pedigree"], "pack_sizes": ["100g", "250g"], "is_veg": False, "shelf_life_days": 365, "price_min": 120, "price_max": 650, "variants": ["Dental Sticks", "Rawhide Bone", "Calcium Chews"]},
        "Pet Shampoo": {"brands": ["Himalaya Pet", "Captain Zack", "Wahl"], "pack_sizes": ["200ml", "500ml"], "is_veg": True, "shelf_life_days": 730, "price_min": 150, "price_max": 850, "variants": ["Anti-Tick", "Coat Conditioning", "Gentle Care"]},
        "Pet Wipes": {"brands": ["Himalaya Pet", "Captain Zack"], "pack_sizes": ["packs_of_30", "packs_of_50"], "is_veg": True, "shelf_life_days": 730, "price_min": 120, "price_max": 350, "variants": ["Unscented", "Deodorizing", "Eye & Ear Clean"]},
    },
    "Paan Corner": {
        "Mouth Fresheners": {"brands": ["Pass Pass", "Rajnigandha", "Pulse"], "pack_sizes": ["packs_of_10"], "is_veg": True, "shelf_life_days": 180, "price_min": 10, "price_max": 50, "variants": ["Sweet Saunf", "Mint", "Pan Pasand", "Royal Mix"]},
        "Soda": {"brands": ["Bisleri", "Schweppes"], "pack_sizes": ["300ml", "750ml"], "is_veg": True, "shelf_life_days": 180, "price_min": 15, "price_max": 60, "variants": ["Club Soda", "Salted Soda", "Zero Sugar Soda", "Carbonated"]},
        "Paan Sweets": {"brands": ["Pass Pass", "Chandan"], "pack_sizes": ["packs_of_10", "100g"], "is_veg": True, "shelf_life_days": 365, "price_min": 15, "price_max": 120, "variants": ["Banarasi Paan", "Meetha Paan", "Rose Paan", "Dry Paan"]},
        "Churna": {"brands": ["Hajmola", "Swad"], "pack_sizes": ["100g", "200g"], "is_veg": True, "shelf_life_days": 365, "price_min": 10, "price_max": 80, "variants": ["Chatpata", "Regular", "Anardana", "Imli"]},
        "Energy Drinks": {"brands": ["Red Bull", "Monster", "Sting", "Charged"], "pack_sizes": ["250ml", "350ml", "500ml"], "is_veg": True, "shelf_life_days": 180, "price_min": 20, "price_max": 160, "variants": ["Original Energy", "Sugarfree", "Wild Berry", "Citrus Charge"]},
    },
    "Meat & Seafood": {
        "Chicken": {"brands": ["Licious", "Zorabian"], "pack_sizes": ["250g", "500g", "1kg"], "is_veg": False, "shelf_life_days": 3, "price_min": 100, "price_max": 450, "variants": ["Curry Cut", "Boneless Breast", "Drumsticks", "Whole Chicken"]},
        "Mutton": {"brands": ["Licious"], "pack_sizes": ["500g"], "is_veg": False, "shelf_life_days": 3, "price_min": 400, "price_max": 900, "variants": ["Curry Cut", "Boneless", "Chops", "Keema"]},
        "Fish": {"brands": ["Licious"], "pack_sizes": ["250g", "500g"], "is_veg": False, "shelf_life_days": 2, "price_min": 150, "price_max": 600, "variants": ["Rohu Curry Cut", "Catla Cut", "Surmai Steaks", "Basa Fillet"]},
        "Eggs": {"brands": ["Eggoz", "Fresho", "Farm Fresh"], "pack_sizes": ["packs_of_6", "packs_of_10", "packs_of_30"], "is_veg": False, "shelf_life_days": 14, "price_min": 45, "price_max": 280, "variants": ["Farm Fresh White", "Brown Organic", "Omega 3 Rich"]},
        "Prawns": {"brands": ["Licious", "Captain Fresh", "Cambay Tiger"], "pack_sizes": ["250g", "500g"], "is_veg": False, "shelf_life_days": 2, "price_min": 220, "price_max": 850, "variants": ["Medium Peeled", "Large Tiger Prawns", "Small Cleaned"]},
        "Cold Cuts": {"brands": ["Licious", "Zorabian"], "pack_sizes": ["200g", "350g"], "is_veg": False, "shelf_life_days": 10, "price_min": 180, "price_max": 450, "variants": ["Chicken Salami", "Smoked Turkey", "Chicken Ham"]},
        "Sausages": {"brands": ["Prasuma", "Zorabian"], "pack_sizes": ["200g", "350g"], "is_veg": False, "shelf_life_days": 10, "price_min": 200, "price_max": 550, "variants": ["Chicken Cocktail", "Spicy Pepper", "Cheese & Herb"]},
    },
    "Bakery & Cakes": {
        "Cakes": {"brands": ["Elite", "Winkies", "Local Bakery"], "pack_sizes": ["150g", "400g"], "is_veg": True, "shelf_life_days": 10, "price_min": 40, "price_max": 250, "variants": ["Chocolate Fudge", "Plum", "Vanilla Sponge", "Fruit"]},
        "Buns": {"brands": ["Britannia", "English Oven"], "pack_sizes": ["packs_of_4"], "is_veg": True, "shelf_life_days": 4, "price_min": 20, "price_max": 50, "variants": ["Burger", "Pav", "Fruit", "Hot Dog"]},
        "Cookies": {"brands": ["Unibic", "MOM Magic", "Baking Co"], "pack_sizes": ["150g", "250g"], "is_veg": True, "shelf_life_days": 90, "price_min": 35, "price_max": 180, "variants": ["Butter", "Chocolate Chip", "Cashew Badam"]},
        "Rusks": {"brands": ["Britannia", "Karachi Bakery"], "pack_sizes": ["250g", "400g"], "is_veg": True, "shelf_life_days": 90, "price_min": 40, "price_max": 120, "variants": ["Milk", "Elaichi", "Cake", "Wheat"]},
        "Croissants": {"brands": ["Bauli", "Britannia"], "pack_sizes": ["packs_of_2"], "is_veg": True, "shelf_life_days": 7, "price_min": 40, "price_max": 120, "variants": ["Butter", "Chocolate Filled", "Classic"]},
        "Garlic Bread": {"brands": ["English Oven", "Harvest Gold"], "pack_sizes": ["200g", "300g"], "is_veg": True, "shelf_life_days": 5, "price_min": 30, "price_max": 90, "variants": ["Garlic Toast", "Herb", "Cheese Toast"]},
    },
    "Breakfast Cereals": {
        "Oats": {"brands": ["Quaker", "Saffola", "Kellogg's"], "pack_sizes": ["400g", "1kg"], "is_veg": True, "shelf_life_days": 270, "price_min": 60, "price_max": 250, "variants": ["Rolled", "Instant", "Steel Cut", "Veggie Masala"]},
        "Muesli": {"brands": ["Kellogg's", "Bagrry's"], "pack_sizes": ["500g"], "is_veg": True, "shelf_life_days": 270, "price_min": 180, "price_max": 450, "variants": ["Fruit & Nut", "Crunchy Nuts", "Dark Chocolate"]},
        "Cornflakes": {"brands": ["Kellogg's", "Bagrry's"], "pack_sizes": ["250g", "500g"], "is_veg": True, "shelf_life_days": 270, "price_min": 70, "price_max": 200, "variants": ["Real Honey", "Original Crisp", "Almond & Honey"]},
    },
    "Chocolates & Ice Creams": {
        "Chocolates": {"brands": ["Cadbury", "Nestle", "Amul", "Ferrero Rocher"], "pack_sizes": ["50g", "100g", "150g"], "is_veg": True, "shelf_life_days": 365, "price_min": 10, "price_max": 500, "variants": ["Milk", "Dark 70%", "Fruit & Nut", "Caramel Crisp"]},
        "Ice Creams": {"brands": ["Amul", "Kwality Walls", "Vadilal", "Mother Dairy"], "pack_sizes": ["100ml", "700ml", "1L"], "is_veg": True, "shelf_life_days": 180, "price_min": 20, "price_max": 350, "variants": ["Vanilla Magic", "Chocolate Fudge", "Butterscotch", "Almond Carnival"]},
        "Wafers": {"brands": ["Munch", "KitKat", "Perk"], "pack_sizes": ["20g", "50g"], "is_veg": True, "shelf_life_days": 180, "price_min": 10, "price_max": 60, "variants": ["Chocolate Cream", "Vanilla Roll", "Crispy Chocolate Bar"]},
        "Candies": {"brands": ["Kinder Joy", "Cadbury"], "pack_sizes": ["packs_of_5"], "is_veg": True, "shelf_life_days": 180, "price_min": 35, "price_max": 120, "variants": ["Surprise Egg", "Milk Cocoa Bar", "Fruit Toffee"]},
        "Frozen Desserts": {"brands": ["Havmor", "Kwality Walls", "Baskin Robbins"], "pack_sizes": ["500ml", "1L"], "is_veg": True, "shelf_life_days": 180, "price_min": 90, "price_max": 450, "variants": ["Cassatta", "Choco Bar", "Trio Fudge"]},
        "Kulfi": {"brands": ["Amul", "Vadilal", "Mother Dairy"], "pack_sizes": ["100ml"], "is_veg": True, "shelf_life_days": 180, "price_min": 30, "price_max": 90, "variants": ["Malai Stick", "Kesar Pista", "Matka Kulfi"]},
    },
    "Stationery & Office": {
        "Pens": {"brands": ["Reynolds", "Cello", "Classmate"], "pack_sizes": ["packs_of_5"], "is_veg": True, "shelf_life_days": 3650, "price_min": 25, "price_max": 100, "variants": ["Blue Ballpoint", "Black Gel", "Smooth Rollerball"]},
        "Notebooks": {"brands": ["Classmate", "Navneet"], "pack_sizes": ["single", "packs_of_6"], "is_veg": True, "shelf_life_days": 3650, "price_min": 30, "price_max": 300, "variants": ["Single Line Long Book", "Unruled Journal", "Spiral Bound"]},
        "Glue": {"brands": ["Fevicol", "Fevikwik", "Pidilite"], "pack_sizes": ["20g", "50g"], "is_veg": True, "shelf_life_days": 730, "price_min": 10, "price_max": 60, "variants": ["All Purpose Synthetic", "Instant Adhesive", "Stick"]},
        "Tape": {"brands": ["Scotch", "Pidilite"], "pack_sizes": ["single"], "is_veg": True, "shelf_life_days": 730, "price_min": 15, "price_max": 90, "variants": ["Transparent Cello", "Brown Packing", "Double Sided"]},
        "Calculators": {"brands": ["Casio", "Camlin"], "pack_sizes": ["single"], "is_veg": True, "shelf_life_days": 3650, "price_min": 150, "price_max": 850, "variants": ["Basic Desktop", "Scientific FX", "Financial Desk"]},
        "Staplers": {"brands": ["Classmate", "Camlin"], "pack_sizes": ["single"], "is_veg": True, "shelf_life_days": 3650, "price_min": 50, "price_max": 250, "variants": ["Compact Mini", "Standard Desk", "Heavy Duty"]},
        "Files": {"brands": ["Solo", "Classmate"], "pack_sizes": ["single"], "is_veg": True, "shelf_life_days": 3650, "price_min": 25, "price_max": 150, "variants": ["Cobra Ring File", "Clear Display Folder", "Box File"]},
        "Folders": {"brands": ["Solo", "Classmate"], "pack_sizes": ["packs_of_5"], "is_veg": True, "shelf_life_days": 3650, "price_min": 50, "price_max": 250, "variants": ["Button Document Bag", "Expanding Accordion", "L-Folder"]},
    },
    "Pharma & Wellness": {
        "OTC Medicines": {"brands": ["Crocin", "Dolo", "Vicks", "Eno"], "pack_sizes": ["packs_of_10"], "is_veg": True, "shelf_life_days": 730, "price_min": 10, "price_max": 150, "variants": ["Paracetamol 650mg", "Pain Relief Gel", "Cough Syrup"]},
        "Wellness Supplements": {"brands": ["Revital", "Protinex"], "pack_sizes": ["200g", "500g"], "is_veg": True, "shelf_life_days": 540, "price_min": 150, "price_max": 650, "variants": ["Daily Multivitamin", "Whey Protein", "Calcium & Vitamin D3"]},
        "Bandages": {"brands": ["Band-Aid", "Hansaplast"], "pack_sizes": ["packs_of_10", "packs_of_20"], "is_veg": True, "shelf_life_days": 1095, "price_min": 20, "price_max": 90, "variants": ["Waterproof Strips", "Fabric Bandages", "Antiseptic Strips"]},
        "Antiseptic": {"brands": ["Dettol", "Savlon"], "pack_sizes": ["100ml", "500ml"], "is_veg": True, "shelf_life_days": 1095, "price_min": 35, "price_max": 180, "variants": ["First Aid Liquid", "Germ Protection Wipes", "Spray"]},
        "Pain Relief Balm": {"brands": ["Moov", "Iodex", "Zandu Balm"], "pack_sizes": ["30g", "50g"], "is_veg": True, "shelf_life_days": 730, "price_min": 45, "price_max": 150, "variants": ["Fast Relief", "Ayurvedic Headache", "Herbal Joint"]},
        "Pain Relief Spray": {"brands": ["Volini", "Moov"], "pack_sizes": ["100g"], "is_veg": True, "shelf_life_days": 730, "price_min": 120, "price_max": 260, "variants": ["Instant Relief", "Muscle Relaxant"]},
        "Health Drink": {"brands": ["Bournvita", "Complan", "Horlicks"], "pack_sizes": ["200g", "500g", "1kg"], "is_veg": True, "shelf_life_days": 540, "price_min": 120, "price_max": 550, "variants": ["Chocolate", "Malt Refill", "Vanilla"]},
        "Protein Powder": {"brands": ["Ensure", "Protinex"], "pack_sizes": ["200g", "500g"], "is_veg": True, "shelf_life_days": 540, "price_min": 350, "price_max": 950, "variants": ["Whey Isolate", "Plant Chocolate", "High Protein"]},
    },
    "Party & Festivities": {
        "Sweets": {"brands": ["Haldiram's", "Bikanervala", "Local Sweet Shop"], "pack_sizes": ["250g", "500g", "1kg"], "is_veg": True, "shelf_life_days": 7, "price_min": 100, "price_max": 600, "variants": ["Kaju Katli", "Gulab Jamun", "Rasgulla", "Soan Papdi"]},
        "Dry Fruits": {"brands": ["Happilo", "Nutraj"], "pack_sizes": ["200g", "500g"], "is_veg": True, "shelf_life_days": 270, "price_min": 150, "price_max": 1200, "variants": ["California Almonds", "Whole Cashews", "Raisins", "Pistachios"]},
        "Festival Items": {"brands": ["Local Artisans"], "pack_sizes": ["packs_of_10"], "is_veg": True, "shelf_life_days": 3650, "price_min": 10, "price_max": 150, "variants": ["Brass Diya", "Pooja Thali Set", "Incense Agarbatti", "Rangoli Colors"]},
    }
}

VARIANTS = ["Classic", "Premium", "Regular", "Gold", "Diet", "Salted", "Unsalted", "Natural", "Spicy", "Sweet", "Masala", "Original"]

def sample_lognormal_price(p_min, p_max):
    """
    Sample a price from a log-normal distribution fitted to [p_min, p_max].
    We assume p_min is the 2.5th percentile and p_max is the 97.5th percentile.
    """
    log_min = np.log(p_min)
    log_max = np.log(p_max)
    mu = (log_min + log_max) / 2
    sigma = (log_max - log_min) / 4
    price = np.random.lognormal(mu, sigma)
    # Clip to be safe
    return np.clip(price, p_min, p_max)

def parse_pack_size_val(pack_size):
    """
    Parse a pack size string (e.g. "1L", "500ml", "400g", "1kg", "packs_of_10")
    into a numeric value (in grams or ml) and unit.
    """
    import re
    m = re.match(r"(\d+(?:\.\d+)?)\s*(ml|l|g|kg|packs_of_)?", pack_size.lower())
    if not m:
        return 1.0, "unit"
    val_str, unit_str = m.groups()
    val = float(val_str)
    if unit_str == "l":
        val *= 1000
        unit_str = "ml"
    elif unit_str == "kg":
        val *= 1000
        unit_str = "g"
    elif not unit_str:
        unit_str = "unit"
    return val, unit_str

def generate_catalog(num_skus=DEFAULT_NUM_SKUS, sim_start_date=None, seed=WORLD_SEED):
    """
    Generates a product catalog with a deterministic seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if sim_start_date is None:
        # Default start date is 14 weeks before now (truncated to midnight for reproducibility)
        sim_start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=98)
    
    # List all possible combinations across specifications to sample from
    spec_combinations = []
    for cat_l1, l2_dict in CATALOG_SPEC.items():
        for cat_l2, detail in l2_dict.items():
            for brand in detail["brands"]:
                for pack_size in detail["pack_sizes"]:
                    spec_combinations.append((cat_l1, cat_l2, brand, pack_size, detail))
                    
    # Generate SKUs
    skus = []
    # To generate exactly num_skus, we can iterate and sample
    # If num_skus is larger than available combinations, we add variants.
    # Otherwise, we sample combinations.
    # To guarantee diversity, we shuffle the combos and loop through them.
    # We will append variants as needed.
    
    # Track used names to ensure name uniqueness if needed, but not strictly required
    # since item_id is unique. But nice for realism.
    used_names = set()
    
    for i in range(num_skus):
        combo = spec_combinations[i % len(spec_combinations)]
        cat_l1, cat_l2, brand, pack_size, detail = combo
        
        # Add flavor/variant if we have wrapped around or randomly
        variant = ""
        variants_list = detail.get("variants", VARIANTS)
        if i >= len(spec_combinations) or random.random() < 0.4:
            variant = random.choice(variants_list) + " "
            
        name = f"{brand} {variant}{cat_l2} ({pack_size})"
        
        # Avoid duplicate names by modifying slightly if necessary
        attempts = 0
        while name in used_names and attempts < 10:
            variant = random.choice(variants_list) + " "
            name = f"{brand} {variant}{cat_l2} ({pack_size})"
            attempts += 1
        used_names.add(name)
        
        item_id = new_ulid(timestamp_ms=int(sim_start_date.timestamp() * 1000) + i)
        price = round(sample_lognormal_price(detail["price_min"], detail["price_max"]), 2)
        
        # Calculate pack size band for substitute grouping (small/medium/large)
        pack_size_val, pack_unit = parse_pack_size_val(pack_size)
        if pack_unit in ["g", "ml"]:
            if pack_size_val < 200:
                pack_band = "small"
            elif pack_size_val < 1000:
                pack_band = "medium"
            else:
                pack_band = "large"
        else:
            if pack_size_val <= 5:
                pack_band = "small"
            elif pack_size_val <= 15:
                pack_band = "medium"
            else:
                pack_band = "large"
                
        # Launch date: 10% receive launch_date inside simulation window (cold start).
        # Simulation window is [sim_start_date, sim_start_date + 98 days].
        # 90% receive launch_date before simulation (e.g. 1 year before).
        if random.random() < 0.1:
            days_offset = random.randint(0, 97)
            launch_date = sim_start_date + timedelta(days=days_offset)
        else:
            days_offset = random.randint(100, 365)
            launch_date = sim_start_date - timedelta(days=days_offset)
            
        skus.append({
            "item_id": item_id,
            "name": name,
            "category_l1": cat_l1,
            "category_l2": cat_l2,
            "brand": brand,
            "price_inr": price,
            "pack_size": pack_size,
            "pack_size_band": pack_band, # used to form substitutes
            "is_veg": detail["is_veg"],
            "shelf_life_days": detail["shelf_life_days"],
            "launch_date": launch_date.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    df = pd.DataFrame(skus)
    return df
