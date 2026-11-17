"""Generate the demonstration datasets in ``data/``.

This is **synthetic sample data** for demonstrating the dashboard. A fixed
random seed makes the output reproducible; randomness is used only here, to
create the demo data — never in any operational calculation.

Outputs
-------
* ``data/sample_inventory.csv``          – 160 SKUs across 8 categories,
  including optional ``Unit_Cost`` and ``Lead_Time_Std_Days`` columns
* ``data/sample_sales_history.csv``      – 90 days of daily unit sales per SKU
* ``data/sample_inventory_with_errors.csv`` – small file with deliberate data
  quality problems to demonstrate validation

``Units_Sold`` in the inventory file equals the last 30 days of the history,
so the default 30-day assumed sales window is consistent with the history.

Usage::

    python scripts/generate_sample_data.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261115
#: Separate stream for the optional cost / lead-time-variability columns, so
#: adding them did not change any previously generated value.
EXTRA_SEED = 20261116
END_DATE = date(2026, 11, 15)
HISTORY_DAYS = 90
WINDOW_DAYS = 30
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

CATEGORIES: dict[str, dict] = {
    "Electronics": {
        "prefix": "ELE", "price": (19, 499), "demand": 2.0, "lead": (10, 30),
        "items": ["Wireless Earbuds", "USB-C Charger", "Bluetooth Speaker", "Smart Plug",
                  "Webcam", "Power Bank", "HDMI Cable", "Mechanical Keyboard", "Wireless Mouse",
                  "Fitness Tracker", "Portable SSD", "Phone Stand", "LED Desk Lamp",
                  "Noise-Cancelling Headphones", "Smartwatch Band", "Action Camera",
                  "Wi-Fi Range Extender", "Laptop Sleeve", "Microphone", "Gaming Controller"],
    },
    "Home & Kitchen": {
        "prefix": "HMK", "price": (8, 180), "demand": 2.5, "lead": (7, 21),
        "items": ["Chef Knife", "Non-Stick Pan", "Glass Food Containers", "French Press",
                  "Cutting Board", "Spice Rack", "Electric Kettle", "Bath Towel Set",
                  "Air Fryer", "Storage Baskets", "Cast Iron Skillet", "Water Bottle",
                  "Throw Blanket", "Measuring Cups", "Blender", "Dish Rack", "Coffee Grinder",
                  "Pillow Covers", "Mixing Bowls", "Vacuum Sealer"],
    },
    "Apparel": {
        "prefix": "APP", "price": (9, 120), "demand": 3.0, "lead": (14, 45),
        "items": ["Cotton T-Shirt", "Running Shorts", "Hooded Sweatshirt", "Denim Jacket",
                  "Wool Socks", "Rain Jacket", "Yoga Leggings", "Baseball Cap", "Chino Pants",
                  "Flannel Shirt", "Puffer Vest", "Knit Beanie", "Leather Belt", "Polo Shirt",
                  "Sports Bra", "Pajama Set", "Windbreaker", "Crew Sweater", "Cargo Shorts",
                  "Thermal Base Layer"],
    },
    "Beauty & Personal Care": {
        "prefix": "BTY", "price": (5, 75), "demand": 4.0, "lead": (5, 20),
        "items": ["Facial Cleanser", "Moisturizer SPF 30", "Shampoo", "Conditioner",
                  "Lip Balm", "Electric Toothbrush", "Hair Dryer", "Body Lotion", "Face Serum",
                  "Sunscreen", "Beard Trimmer", "Nail Kit", "Dry Shampoo", "Hand Cream",
                  "Makeup Brushes", "Sheet Masks", "Deodorant", "Eye Cream", "Hair Oil",
                  "Bath Bombs"],
    },
    "Sports & Outdoors": {
        "prefix": "SPT", "price": (10, 250), "demand": 1.5, "lead": (10, 35),
        "items": ["Yoga Mat", "Resistance Bands", "Camping Lantern", "Hiking Backpack",
                  "Dumbbell Set", "Jump Rope", "Insulated Tumbler", "Trekking Poles",
                  "Foam Roller", "Cycling Gloves", "Sleeping Bag", "Tent Footprint",
                  "Swim Goggles", "Kettlebell", "Headlamp", "Soccer Ball", "Bike Light Set",
                  "Hydration Pack", "Pull-Up Bar", "Camping Stove"],
    },
    "Toys & Games": {
        "prefix": "TOY", "price": (7, 90), "demand": 1.8, "lead": (14, 40),
        "items": ["Building Blocks Set", "Puzzle 1000pc", "Board Game Classic", "Plush Bear",
                  "RC Car", "Card Game", "Art Supply Kit", "Science Kit", "Wooden Train Set",
                  "Doll House", "Bubble Machine", "Magnetic Tiles", "Kite", "Play Dough Set",
                  "Toy Drone", "Marble Run", "Dinosaur Figures", "Water Blaster",
                  "Chess Set", "Rubik Cube"],
    },
    "Books": {
        "prefix": "BKS", "price": (6, 45), "demand": 1.2, "lead": (5, 15),
        "items": ["Python Programming Guide", "Data Science Handbook", "Mystery Novel",
                  "Cookbook Everyday", "Children's Picture Book", "Leadership Essentials",
                  "Travel Atlas", "Sci-Fi Anthology", "Self-Help Journal", "History of Trade",
                  "Poetry Collection", "Gardening Basics", "Graphic Novel", "Language Workbook",
                  "Biography Series", "Mindfulness Guide", "Startup Playbook",
                  "Photography Manual", "Fantasy Saga Vol 1", "Personal Finance 101"],
    },
    "Grocery": {
        "prefix": "GRC", "price": (2, 35), "demand": 6.0, "lead": (2, 10),
        "items": ["Organic Coffee Beans", "Green Tea Bags", "Almond Butter", "Rolled Oats",
                  "Olive Oil", "Dark Chocolate", "Protein Bars", "Sparkling Water 12pk",
                  "Basmati Rice", "Pasta Variety", "Honey", "Trail Mix", "Maple Syrup",
                  "Granola", "Coconut Water", "Hot Sauce", "Peanut Snacks", "Herbal Tea",
                  "Dried Mango", "Instant Noodles"],
    },
}

# Inventory scenarios: (name, probability, days-of-cover range)
SCENARIOS = [
    ("healthy", 0.50, (30, 90)),
    ("low", 0.16, (4, 14)),
    ("critical", 0.07, (0.5, 3)),
    ("out", 0.05, (0, 0)),
    ("excess", 0.15, (150, 400)),
    ("dead", 0.07, None),
]


def generate(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the inventory table and the matching daily sales history."""
    dates = pd.date_range(END_DATE - timedelta(days=HISTORY_DAYS - 1), END_DATE, freq="D")
    t = np.arange(HISTORY_DAYS)
    inventory_rows = []
    history_frames = []
    names = [s[0] for s in SCENARIOS]
    probabilities = [s[1] for s in SCENARIOS]

    for category, spec in CATEGORIES.items():
        for index, item in enumerate(spec["items"]):
            sku = f"{spec['prefix']}-{1001 + index}"
            scenario = rng.choice(names, p=probabilities)
            base = spec["demand"] * rng.lognormal(mean=0.0, sigma=0.7)
            trend = rng.choice([-0.006, 0.0, 0.0, 0.008])  # per-day relative drift
            weekly = 1 + 0.15 * np.sin(2 * np.pi * t / 7)
            expected = base * np.clip(1 + trend * (t - HISTORY_DAYS / 2), 0.2, None) * weekly
            if scenario == "dead":
                expected = np.where(t < HISTORY_DAYS - WINDOW_DAYS - 10, expected * 0.3, 0.0)
            daily_units = rng.poisson(expected)
            history_frames.append(
                pd.DataFrame({"Date": dates, "SKU_ID": sku, "Units_Sold": daily_units})
            )

            units_sold = int(daily_units[-WINDOW_DAYS:].sum())
            velocity = units_sold / WINDOW_DAYS
            low_price, high_price = spec["price"]
            price = round(float(np.exp(rng.uniform(np.log(low_price), np.log(high_price)))), 2) - 0.01
            lead_time = int(rng.integers(spec["lead"][0], spec["lead"][1] + 1))
            reorder_level = int(round(max(velocity, 0.3) * lead_time * rng.uniform(0.9, 1.4)))

            cover = dict((s[0], s[2]) for s in SCENARIOS)[scenario]
            if scenario == "dead":
                stock = int(rng.integers(15, 160))
            elif scenario == "out":
                stock = 0
            else:
                stock = int(round(velocity * rng.uniform(*cover)))
                if scenario in {"critical", "low"}:
                    stock = max(stock, 1)
            last_restock = END_DATE - timedelta(days=int(rng.integers(3, 120)))

            inventory_rows.append(
                {
                    "SKU_ID": sku,
                    "Product_Name": item,
                    "Category": category,
                    "Current_Stock": stock,
                    "Units_Sold": units_sold,
                    "Unit_Price": round(price, 2),
                    "Reorder_Level": reorder_level,
                    "Supplier_Lead_Time": lead_time,
                    "Last_Restock_Date": last_restock.isoformat(),
                }
            )

    history = pd.concat(history_frames, ignore_index=True)
    history = history[history["Units_Sold"] > 0]  # transactional exports omit zero days
    history["Date"] = history["Date"].dt.strftime("%Y-%m-%d")
    return pd.DataFrame(inventory_rows), history.reset_index(drop=True)


def add_optional_columns(inventory: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Purchase cost (35–75 % of price) and lead-time standard deviation (10–35 %)."""
    out = inventory.copy()
    out["Unit_Cost"] = (out["Unit_Price"] * rng.uniform(0.35, 0.75, len(out))).round(2)
    out["Lead_Time_Std_Days"] = (out["Supplier_Lead_Time"] * rng.uniform(0.10, 0.35, len(out))).round(1)
    return out


def error_sample() -> pd.DataFrame:
    """A small file containing deliberate data-quality problems."""
    return pd.DataFrame(
        [
            ["DEMO-001", "Desk Organizer", "Office", "45", "60", "14.99", "20", "7", "2026-10-01"],
            ["DEMO-002", "Gel Pens 10pk", "Office", "abc", "120", "6.50", "40", "5", "2026-10-12"],
            ["DEMO-003", "Stapler", "Office", "30", "-4", "9.99", "10", "6", "2026-09-20"],
            ["", "Notebook A5", "Office", "80", "50", "4.25", "25", "4", "2026-10-30"],
            ["DEMO-004", "Paper Clips", "", "500", "90", "$1.99", "100", "3", "not a date"],
            ["DEMO-005", "Monitor Riser", "Office", "12", "15", "39.00", "", "14", "2026-08-15"],
            ["DEMO-001", "Desk Organizer", "Office", "45", "60", "14.99", "20", "7", "2026-10-01"],
            ["DEMO-005", "Monitor Riser v2", "Office", "20", "15", "41.00", "10", "14", "2026-08-15"],
            ["DEMO-006", "Whiteboard", "Office", "0", "0", "55.00", "5", "", "2027-03-01"],
        ],
        columns=[
            "SKU_ID", "Product_Name", "Category", "Current_Stock", "Units_Sold", "Unit_Price",
            "Reorder_Level", "Supplier_Lead_Time", "Last_Restock_Date",
        ],
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    inventory, history = generate(np.random.default_rng(SEED))
    inventory = add_optional_columns(inventory, np.random.default_rng(EXTRA_SEED))
    inventory.to_csv(DATA_DIR / "sample_inventory.csv", index=False)
    history.to_csv(DATA_DIR / "sample_sales_history.csv", index=False)
    error_sample().to_csv(DATA_DIR / "sample_inventory_with_errors.csv", index=False)
    print(f"Wrote {len(inventory)} SKUs and {len(history)} history rows to {DATA_DIR}")


if __name__ == "__main__":
    main()
