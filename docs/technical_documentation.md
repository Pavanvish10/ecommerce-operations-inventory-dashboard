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
* Raises `DataLoadError` for: empty file, > 100 MB, binary content (NUL bytes, e.g. an `.xlsx`), whitespace-only, semicolon-delimited files, rows with **more fields than the header**, and other parser errors.
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
| `Revenue`, `Inventory_Value` | sales_analytics | `U × price`, `S × price` (retail value) |
| `Unit_Value`, `Valuation_Basis` | sales_analytics | Per-unit value for working capital: cost if known, else price, and which one was used |
| `Inventory_Cost_Value`, `Gross_Profit`, `Gross_Margin_Pct` | sales_analytics | Only when `Unit_Cost` is supplied (`NaN` margin when price = 0) |
| `Days_Of_Inventory` | sales_analytics | `S / d`. `0`, `inf` or `NaN` per the edge-case rules. |
| `ABC_Class` | sales_analytics | Pareto class on the **full** (unfiltered) dataset |
| `Safety_Stock`, `Safety_Stock_Method`, `Lead_Time_Demand`, `Reorder_Point`, `Effective_Reorder_Threshold`, `Target_Stock` | reorder_engine | Replenishment parameters and which safety-stock method each row used |
| `Needs_Reorder`, `Reorder_Quantity`, `Reorder_Value` | reorder_engine | Recommendation (`NaN` quantity = cannot compute) |
| `Estimated_Stockout_Date` | reorder_engine | as-of date + ⌊DOI⌋ (`NaT` when there is no demand or DOI > 3,650 days) |
| `Alert_Level`, `Alert_Priority`, `Alert_Reason`, `Reorder_Note` | reorder_engine | Alert and its explanation |
| `Score_*` (4 columns, 0–100), `Health_Score`, `Health_Class`, `Health_Note` | health_score | Health score and its components |
| `Stock_Status`, `Is_Slow_Moving`, `Is_Excess`, `Excess_Units`, `Excess_Inventory_Value` | risk_analysis | Risk flags and tied-up capital |
| `Risk_Category`, `Risk_Level`, `Risk_Priority`, `Recommended_Action`, `Action_Rationale` | risk_analysis | Classification and action |
| `Days_Since_Restock` | inventory_engine | Only if `Last_Restock_Date` is present |
| `History_*`, `Demand_Std_Daily`, `Trend_Pct_Per_Week`, `Demand_Trend` | sales_analytics | Only meaningful with a history file. Otherwise `Demand_Trend = "No History"`. |

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
| `safety_stock_method` / `service_level` | `days` / 0.95 | safety stock (statistical method uses history σd and `Lead_Time_Std_Days`) |
| `forecast_window_days` / `forecast_holdout_days` | 28 / 14 | forecast fitting window and backtest holdout |

## 5. Design decisions

* **Read everything as text, then coerce.** `pd.read_csv` type inference would silently NaN bad values or strip leading zeros from SKUs. Reading as text lets every coercion be reported per cell.
* **Blank, don't guess.** Invalid numbers become NaN and flow through as *Insufficient Data*. Imputing would produce plausible-looking but invented KPIs.
* **Vectorised engines.** Every calculation uses whole-column pandas/numpy operations (`np.select`, `mask`, `np.fmax`). Note strings are joined with a vectorised `join_nonempty`, and rationale numbers are formatted only for the rows each rule applies to. Trends, demand variability and forecasts all use closed-form operations on a SKU × day matrix.
* **Float-safe integer rounding.** `utils.safe_ceil` and `safe_floor` round to 9 decimals before `ceil`/`floor`, so `3.0000000000000004` gives 3 units, not 4. A property test checks the engine against exact `fractions.Fraction` arithmetic.
* **Measured, not assumed, forecast accuracy.** Forecasts are only shown together with a holdout backtest. If there is not enough history, nothing is shown.
* **Untrusted text.** Uploaded values are HTML-escaped before any raw-HTML rendering, Markdown-escaped in captions and messages, and formula-neutralised in CSV exports.
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

## 7. Production-readiness audit (17 Nov 2026)

The audit re-verified the previous report independently (212/212 tests re-run, all source files reviewed), then probed edge cases with targeted inputs. Every defect below was reproduced before being fixed, and has a regression test in `tests/test_audit_regressions.py`.

### 7.1 Defects found

| # | Severity | Defect | Reproduction | Fix |
|---|---|---|---|---|
| A1 | **Critical** (silent data corruption) | Decimal commas were silently multiplied: `12,50` → **1250**, `1,2,3` → 123 | price column `"12,50"` | A comma is accepted only in the strict thousands pattern `^\d{1,3}(,\d{3})+(\.\d+)?$`. Anything else is rejected with a hint. |
| A2 | **Critical** (crash) | Timezone-aware or mixed-offset dates raised `ValueError: Mixed timezones`, which crashed validation and therefore the app | `2026-01-01T10:00:00Z` next to `2026-02-01` | `to_datetime(..., utc=True)` then drop the timezone |
| A3 | **Critical** (crash) | Very long cover overflowed the stockout-date conversion (`OutOfBoundsTimedelta`) and crashed the pipeline | stock 5×10⁹, 1 unit sold | Dates are not projected beyond 3,650 days, with an explanatory note |
| A4 | High (wrong decision) | Float noise inflated reorder quantities by one unit (`ceil(3.0000000000000004) = 4`) and raised false **High** alerts when cover exactly equalled lead time (`29.999999999999996 < 30`) | U = 3, L = 9 → 4 units; U = S = 23, L = 30 → High | Round to 9 decimals before `ceil`/`floor` and in days-of-inventory. Property-tested against exact rationals over 630 combinations. |
| A5 | High (security) | **XSS:** the uploaded `Product_Name` was interpolated into `st.markdown(..., unsafe_allow_html=True)` | `<img src=x onerror=alert(1)>` | `html.escape`. Verified in a real browser: rendered as text, 0 injected elements, no dialog. |
| A6 | High (security) | **CSV/formula injection:** exported cells beginning with `=`, `+`, `-` or `@` execute in spreadsheets | `=HYPERLINK(...)` product name | Apostrophe prefix (OWASP); plain negative numbers are left untouched |
| A7 | Medium | Implausible magnitudes produced `inf` totals | `1e308` stock | Values above 10¹² are rejected with a warning |
| A8 | Medium | Ambiguous day/month dates were silently read month-first | `03/04/2026` | Still read month-first, now flagged as info with row numbers |
| A9 | Medium | SKUs differing only by case were silently treated as separate products | `abc-1` / `ABC-1` | Warning with row numbers. Rows are kept, because they may legitimately differ. |
| A10 | Medium (UX consistency) | "Low Stock" meant two different things: the risk category (≤ effective threshold) and the stock-status filter (≤ `Reorder_Level`). A product could be risk "Low Stock" but status "In Stock". | demand-based ROP > `Reorder_Level` | Risk category renamed to **Reorder Point** |
| A11 | Medium (correctness) | The explorer drill-down used a positional selection, so after a filter change the detail panel could show a different product | select a row, then change filters | Table key derived from the result set, so the selection resets |
| A12 | Low (security) | File names and CSV header text were rendered as Markdown (links, formatting) | file named `[click](https://…).csv` | `md_escape`. Browser-verified: no link rendered. |
| A13 | Performance | 50k products: metrics took 2.75 s because of row-wise `apply(axis=1)` string joins and 550k per-cell format calls | `scripts/benchmark.py` | Vectorised joins and per-subset formatting: **0.60 s** |
| A14 | Performance | 3.3M-row history: validation took 14.4 s (per-element regex on every cell) | `benchmark.py --history-days 90` | Vectorised `to_numeric` fast path; the regex cleaning runs only on failures. pyarrow strings: **6.0 s** |
| A15 | Resource use | Upload cap of 200 MB with up to 8 + 16 cached results per process: an unbounded-memory risk (an 87 MB history already peaks at 0.9 GB) | measured peak working set | Cap set to 100 MB; caches bounded by `max_entries` and a 1-hour TTL |

Also verified correct, with no change needed: the health-score formulas and weights match the documentation; division by zero is guarded everywhere (zero stock, zero sales, zero price, zero reorder level, zero lead time); negative stock is rejected at validation, not computed on; KPI aggregates ignore NaN; ABC uses the full dataset, as documented; filters are ANDed and exports respect them; no file-system writes occur; there are no secrets in the repo.

### 7.2 Advanced analytics evaluated (Phase 3)

| Capability | Decision | Rationale |
|---|---|---|
| Demand forecasting | **Implemented** (`src/forecasting.py`) | Three baselines with a holdout backtest. Accuracy is reported (WAPE/bias/MAE) and the best method is chosen by measurement. Kept informational so reorder logic stays traceable. |
| Statistical safety stock | **Implemented** (opt-in) | Standard `z·√(L·σd² + d²·σL²)`, with per-product fallback that is labelled in `Safety_Stock_Method` |
| Lead-time variability | **Implemented** | Optional `Lead_Time_Std_Days` column feeds σL |
| Purchase cost & gross margin | **Implemented** | Optional `Unit_Cost`. Cost-based working capital with per-row valuation basis. Margin KPI computed only over costed products. |
| Multi-warehouse | **Not implemented** | Would change the primary key from `SKU_ID` to `(SKU_ID, Location)` across validation, merges, reorder plan, filters and exports. Too invasive to do reliably in this audit; the design path is documented in the README. |

### 7.3 Test-suite changes

* Original 212 tests: all kept. Two were edited:
  * One expected-label string changed from "Low Stock" to "Reorder Point" (A10). The assertion is otherwise unchanged.
  * The settings test now finds its input by label instead of by position (`number_input[5]`), which would have broken silently as inputs were added.
* 51 new tests (263 total).
* **Mutation check:** with the float-noise fix disabled, all three float regression tests fail, confirming they detect A4.

## 8. Verification performed

* `pytest`: **263 passed, 0 failed** (logic, boundaries, audit regressions, property test, and headless UI via `streamlit.testing.v1.AppTest`).
* Hostile-upload end-to-end test in headless Edge: decimal comma, case-variant SKU, ambiguous date, XSS payload, formula payload and a Markdown-link file name. Validation warnings were shown, nothing was injected, the export was neutralised, and no page raised an exception.
* `pip-audit` over the 40 pinned packages: no known vulnerabilities.
* `scripts/benchmark.py` at 1k / 10k / 50k products, with and without a 90-day history (see README "Performance").
* Real server start (`streamlit run app.py`): `/_stcore/health` returns `ok`, `/` returns 200.
* Every page loaded in headless Microsoft Edge (Playwright) with zero `stException` elements. Product drill-down was exercised by searching for and selecting a row. The screenshots in `docs/screenshots/` come from these runs.
