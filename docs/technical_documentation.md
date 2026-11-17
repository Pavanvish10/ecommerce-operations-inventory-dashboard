# Technical Documentation

This document covers the internals: module responsibilities, data contracts, caching, derived-column dictionary, design decisions, and the engineering log. For the user-level overview and the business formulas, see the [README](../README.md).

## 1. Layering

| Layer | Package | Depends on | Rules |
|---|---|---|---|
| Business logic | `src/` | pandas, numpy | No Streamlit imports. Pure functions: DataFrame in, new DataFrame out (inputs never mutated). |
| UI components | `components/` | `src/`, streamlit, plotly | No business calculations. Charts plot pre-computed columns. |
| Pages | `views/` | `components/`, `src/` | Read the per-run `PageContext`; one `render()` per page. |
| Entry point | `app.py` | all | Page config, navigation, sidebar, context build. |

## 2. Run lifecycle (one Streamlit rerun)

1. `init_session()`: creates `AnalyticsConfig()` and loads the sample bytes on first visit.
2. `build_context(render_global_filters)`:
   1. `validate_inventory_bytes(bytes, duplicate_strategy, as_of)` — cached.
   2. `validate_history_bytes(bytes, skus, as_of)` — cached, only if a history is loaded.
   3. `compute_metrics(data, config.to_dict(), history)` — cached. Runs the whole pipeline.
   4. The sidebar renders global filters and returns a `FilterCriteria`.
   5. `apply_filters(metrics, criteria)` produces `filtered`.
   6. A `PageContext` is stored in `st.session_state["_page_context"]`.
3. `navigation.run()` renders the selected page, which calls `get_context()`.

Because the cache keys are the raw file bytes plus the config dict, changing a filter costs only a vectorised boolean mask. Changing a setting recomputes the metrics once and then caches them.

### Widget-state persistence

Streamlit discards widget state for widgets not rendered on the current page. `restore_widget()` and `save_widget()` mirror explorer and sidebar selections into `_persist_*` keys and drop values that are no longer valid options (for example after uploading a dataset with different categories). This prevents the "value not in options" errors that would otherwise appear.

## 3. Data contracts

### 3.1 `load_csv(source) -> DataFrame`
* Accepts a path, bytes, or a binary file-like object (Streamlit `UploadedFile`).
* Every column is returned as text, so the validator can report the original bad value.
* Raises `DataLoadError` for: empty file, > 200 MB, binary content (NUL bytes, e.g. an `.xlsx`), whitespace-only, semicolon-delimited files, rows with **more fields than the header**, and other parser errors.
* Decoding is tried as `utf-8-sig`, then `cp1252`, then `latin-1`.

### 3.2 `validate_inventory(raw, duplicate_strategy, reference_date) -> ValidationResult`

`ValidationResult` fields: `data` (typed, accepted rows), `rejected` (raw rows plus `Rejection_Reason`), `issues` (`ValidationIssue(severity, message, column, rows)`), `input_rows`. Use `is_valid` before reading `data`.

Invariant (tested): `len(data) + len(rejected) == input_rows`.

Added columns: `Source_Row` (1-based CSV line number; the header is line 1) and `Data_Quality_Notes`.

### 3.3 Derived columns (output of `build_inventory_metrics`)

| Column | Module | Meaning |
|---|---|---|
| `Sales_Window_Days`, `Velocity_Basis` | sales_analytics | Window used for velocity and whether it was reported or assumed |
| `Avg_Daily_Sales`, `Avg_Weekly_Sales`, `Avg_Monthly_Sales` | sales_analytics | Average rates |
| `Revenue`, `Inventory_Value` | sales_analytics | `U × price`, `S × price` |
| `Days_Of_Inventory` | sales_analytics | `S / d`. `0`, `inf` or `NaN` per the edge-case rules. |
| `ABC_Class` | sales_analytics | Pareto class on the **full** (unfiltered) dataset |
| `Safety_Stock`, `Lead_Time_Demand`, `Reorder_Point`, `Effective_Reorder_Threshold`, `Target_Stock` | reorder_engine | Replenishment parameters |
| `Needs_Reorder`, `Reorder_Quantity`, `Reorder_Value` | reorder_engine | Recommendation (`NaN` quantity = cannot compute) |
| `Estimated_Stockout_Date` | reorder_engine | as-of date + ⌊DOI⌋ (`NaT` when no demand) |
| `Alert_Level`, `Alert_Priority`, `Alert_Reason`, `Reorder_Note` | reorder_engine | Alert and its explanation |
| `Score_*` (4 columns, 0–100), `Health_Score`, `Health_Class`, `Health_Note` | health_score | Health score and its components |
| `Stock_Status`, `Is_Slow_Moving`, `Is_Excess`, `Excess_Units`, `Excess_Inventory_Value` | risk_analysis | Risk flags and tied-up capital |
| `Risk_Category`, `Risk_Level`, `Risk_Priority`, `Recommended_Action`, `Action_Rationale` | risk_analysis | Classification and action |
| `Days_Since_Restock` | inventory_engine | Only if `Last_Restock_Date` is present |
| `History_*`, `Trend_Pct_Per_Week`, `Demand_Trend` | sales_analytics | Only meaningful with a history file. Otherwise `Demand_Trend = "No History"`. |

## 4. Configuration (`src/config.py`)

`AnalyticsConfig` is a frozen, self-validating dataclass. Invalid combinations raise `ValueError` at construction, for example `overstock_cover_days <= target_cover_days`, negative weights, or an unknown threshold mode. The Settings page builds a new instance and shows the error instead of applying it. `to_dict()` and `from_dict()` exist so the config can be a `st.cache_data` key.

| Parameter | Default | Used by |
|---|---|---|
| `sales_window_days` | 30 | velocity |
| `safety_stock_days` | 7 | reorder, health A and LT |
| `review_period_days` | 14 | target stock |
| `critical_cover_days` | 3 | Critical alert |
| `approaching_threshold_buffer` | 0.20 | Medium alert, health R |
| `threshold_mode` | `max` | effective reorder threshold |
| `target_cover_days` / `overstock_cover_days` | 30 / 120 | health C, excess detection |
| `health_weights` | 35/30/20/15 | health score |
| `slow_moving_velocity` | 0.5 units/day | slow-moving flag |
| `trend_threshold_pct` / `min_history_days` | 5 % / 14 | demand trends |
| `duplicate_strategy` | `first` | validation |
| `as_of_date` | `None` (today) | stockout dates, restock age, future-date check |

## 5. Design decisions

* **Read everything as text, then coerce.** `pd.read_csv` type inference would silently NaN bad values or strip leading zeros from SKUs. Reading as text lets every coercion be reported per cell.
* **Blank, don't guess.** Invalid numbers become NaN and flow through as *Insufficient Data*. Imputing would produce plausible-looking but invented KPIs.
* **Vectorised engines.** Every calculation uses whole-column pandas/numpy operations (`np.select`, `mask`, `np.fmax`). The only per-row Python is joining note strings. Trends use a closed-form least-squares slope on a SKU × day matrix.
* **Deterministic output.** Every sort is stable (`mergesort`) with `SKU_ID` as the final tie-breaker, so tables, charts and exports are reproducible. The test suite asserts pipeline determinism.
* **Target stock = max of two terms.** This keeps the quantity consistent with the threshold that triggered the reorder (see the README).
* **Colour.** A colour-vision-deficiency-validated palette. Status colours (good/warning/serious/critical) are reserved for state, always paired with a text label and icon. Single-series charts use one hue. No dual-axis charts.

## 6. Engineering log: bugs found and fixed during development

| # | Found by | Problem | Root cause | Fix |
|---|---|---|---|---|
| 1 | `test_invalid_files_raise_meaningful_errors` | A CSV row with **more** fields than the header loaded "successfully", with every value shifted into the wrong column | pandas treats surplus leading fields as an implicit index instead of raising | `_check_field_counts()` scans rows with the `csv` module and raises `DataLoadError` naming the offending line numbers. `index_col=False` added as a second safeguard. |
| 2 | Manual run of the error sample | A validation message claimed missing `Reorder_Level` makes rows "Insufficient Data", but the engine actually re-weights the score | Message drafted before the partial-score design was finalised | Message reworded to "skipped or flagged (never estimated)" |
| 3 | Code review | `risk_analysis` rationale for missing data combined a conditional expression and string concatenation with the wrong precedence | Python evaluates `a + b if c else d` as `(a + b) if c else d` | Rewritten as an explicit `if` |
| 4 | Report-page smoke run | pyarrow serialisation warning in the report preview | Exported `Days_Of_Inventory` mixes numbers and the explicit `"No demand"` marker | Preview casts that column to text. The CSV export is unchanged. |
| 5 | Browser screenshots | `/overview` showed "Page not found" | The default `st.Page` is always served at `/`, so a `url_path` on it is not routable | Removed `url_path` from the default page |
| 6 | Browser screenshots | Outside bar labels clipped (`$141,`) and tick labels touching bars | Plotly autorange ignores outside text | `room_for_labels()` pads the value axis. Template sets `ticklabelstandoff`. |
| 7 | Browser screenshots | "Most urgent" reorder chart was mostly empty bars | The most urgent items are out of stock (zero cover by definition) | Chart now ranks **in-stock** items by cover shortfall vs lead time. Out-of-stock items stay at the top of the table. |
| 8 | `test_app_entrypoint_starts` | `AppTest.from_file("app.py")` not found | Relative paths resolve against the test file's folder | Absolute path from `tests/conftest.ROOT` |

## 7. Verification performed

* `pytest`: **212 passed** (logic, boundaries, and headless UI via `streamlit.testing.v1.AppTest`).
* Real server start (`streamlit run app.py`): `/_stcore/health` returns `ok`, `/` returns 200.
* Every page loaded in headless Microsoft Edge (Playwright) with zero `stException` elements. Product drill-down was exercised by searching for and selecting a row. The screenshots in `docs/screenshots/` come from these runs.
