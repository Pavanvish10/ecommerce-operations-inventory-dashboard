"""Performance benchmark for the analytics pipeline.

Generates synthetic inventory (and optional daily history) datasets of the
requested sizes in memory, then times each stage the dashboard runs:

    load_csv → validate_inventory → build_inventory_metrics → apply_filters
    → sort → prepare_report + to_csv_bytes

Each stage is timed as the best of ``--repeat`` runs (warm, single process).
Randomness is used only to generate the benchmark data.

Usage::

    python scripts/benchmark.py                     # 1k, 10k, 50k products
    python scripts/benchmark.py --sizes 1000 --history-days 90 --repeat 3
"""

from __future__ import annotations

import argparse
import io
import os
import platform
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AnalyticsConfig  # noqa: E402
import src.data_loader as data_loader  # noqa: E402
from src.data_loader import load_csv  # noqa: E402

# Measure raw throughput beyond the app's upload cap (the cap is a product
# decision, not a technical limit of the pipeline).
data_loader.MAX_FILE_BYTES = 10**10
from src.data_validator import validate_inventory, validate_sales_history  # noqa: E402
from src.export_manager import prepare_report, to_csv_bytes  # noqa: E402
from src.filters import FilterCriteria, apply_filters, sort_products  # noqa: E402
from src.inventory_engine import build_inventory_metrics  # noqa: E402

AS_OF = date(2026, 11, 15)
CATEGORIES = [f"Category {i:02d}" for i in range(20)]


def make_inventory_csv(n: int, seed: int = 7) -> bytes:
    rng = np.random.default_rng(seed)
    daily = rng.lognormal(0.5, 1.0, n)
    sold = rng.poisson(daily * 30)
    lead = rng.integers(2, 45, n)
    stock = np.round(daily * rng.uniform(0, 200, n)).astype(int)
    frame = pd.DataFrame(
        {
            "SKU_ID": [f"SKU-{i:07d}" for i in range(n)],
            "Product_Name": [f"Product {i}" for i in range(n)],
            "Category": rng.choice(CATEGORIES, n),
            "Current_Stock": stock,
            "Units_Sold": sold,
            "Unit_Price": np.round(rng.uniform(2, 400, n), 2),
            "Unit_Cost": np.round(rng.uniform(1, 200, n), 2),
            "Reorder_Level": np.round(daily * lead).astype(int),
            "Supplier_Lead_Time": lead,
            "Last_Restock_Date": [
                (AS_OF - timedelta(days=int(x))).isoformat() for x in rng.integers(1, 200, n)
            ],
        }
    )
    return frame.to_csv(index=False).encode()


def make_history_csv(n: int, days: int, seed: int = 11) -> bytes:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(AS_OF - timedelta(days=days - 1), AS_OF).strftime("%Y-%m-%d")
    rates = rng.lognormal(0.5, 1.0, n)
    units = rng.poisson(np.outer(rates, np.ones(days)))
    sku_idx, day_idx = np.nonzero(units)
    frame = pd.DataFrame(
        {
            "Date": dates[day_idx],
            "SKU_ID": np.char.add("SKU-", np.char.zfill(sku_idx.astype(str), 7)),
            "Units_Sold": units[sku_idx, day_idx],
        }
    )
    return frame.to_csv(index=False).encode()


def best_of(fn: Callable, repeat: int):
    best, result = float("inf"), None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def run(n: int, history_days: int, repeat: int) -> dict[str, float]:
    config = AnalyticsConfig(as_of_date=AS_OF)
    raw_bytes = make_inventory_csv(n)
    timings: dict[str, float] = {"products": n, "csv_mb": len(raw_bytes) / 1e6}

    timings["load_csv"], raw = best_of(lambda: load_csv(raw_bytes), repeat)
    timings["validate"], result = best_of(
        lambda: validate_inventory(raw, reference_date=AS_OF), repeat
    )
    history = None
    if history_days:
        hist_bytes = make_history_csv(n, history_days)
        timings["history_mb"] = len(hist_bytes) / 1e6
        timings["history_load+validate"], hist = best_of(
            lambda: validate_sales_history(load_csv(hist_bytes), reference_date=AS_OF), repeat
        )
        history = hist.data
        timings["history_rows"] = len(history)
    timings["metrics"], metrics = best_of(
        lambda: build_inventory_metrics(result.data, config, history), repeat
    )
    criteria = FilterCriteria(
        search="1", categories=tuple(CATEGORIES[:10]), health_classes=("Healthy", "Moderate")
    )
    timings["filter"], filtered = best_of(lambda: apply_filters(metrics, criteria), repeat)
    timings["sort"], _ = best_of(lambda: sort_products(metrics, "Highest risk"), repeat)
    timings["export_full"], _ = best_of(
        lambda: to_csv_bytes(prepare_report(metrics, "full")), repeat
    )
    timings["filtered_rows"] = len(filtered)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 50_000])
    parser.add_argument("--history-days", type=int, default=0,
                        help="Also benchmark a daily history of this many days.")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    print(f"Python {platform.python_version()} · pandas {pd.__version__} · numpy {np.__version__}")
    print(f"{platform.platform()} · {platform.processor() or platform.machine()} · {os.cpu_count()} logical CPUs")
    print(f"best of {args.repeat} runs; times in seconds\n")
    rows = [run(n, args.history_days, args.repeat) for n in args.sizes]
    table = pd.DataFrame(rows).set_index("products")
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 200):
        print(table.to_string())


if __name__ == "__main__":
    main()
