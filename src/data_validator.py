"""Dataset validation and cleaning.

Guiding rule: **nothing is changed or dropped silently.** Every coercion,
fill, and rejected row produces a :class:`ValidationIssue`, rejected rows are
returned in :attr:`ValidationResult.rejected` together with the reason, and
each accepted row carries a ``Data_Quality_Notes`` column describing any cell
that was blanked because it was invalid.

Cleaning policy
---------------
* Column names are matched case-insensitively; spaces and hyphens are treated
  as underscores (``"unit price"`` -> ``Unit_Price``).
* Missing required columns are a blocking error.
* Rows without a ``SKU_ID`` and completely empty rows are rejected.
* Non-numeric, negative, or infinite values in numeric columns are blanked
  (NaN) and reported; downstream engines mark the affected metrics as
  "Insufficient Data" instead of guessing.
* Missing ``Product_Name`` / ``Category`` are filled with explicit placeholders
  and reported.
* Exact duplicate rows are collapsed; conflicting records for the same SKU keep
  the first (or last) occurrence and the others are rejected with a reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import (
    HISTORY_REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
)

SOURCE_ROW = "Source_Row"
QUALITY_NOTES = "Data_Quality_Notes"
REJECTION_REASON = "Rejection_Reason"
UNCATEGORIZED = "Uncategorized"

_CURRENCY_PATTERN = re.compile(r"^\s*[$€£¥₹]\s*")
#: A comma is only accepted as a thousands separator in this exact shape
#: ("1,234" / "12,345.67"). Anything else ("12,50", "1,2,3") is ambiguous —
#: possibly a decimal comma — and is reported instead of being guessed.
_THOUSANDS_PATTERN = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")
_DAY_MONTH_PATTERN = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}")
#: Values above this magnitude are treated as data-entry errors (they would
#: overflow currency totals and date arithmetic).
MAX_NUMERIC_VALUE = 1e12
_MAX_ROWS_IN_MESSAGE = 10


@dataclass
class ValidationIssue:
    """A single finding produced during validation."""

    severity: str  # "error" | "warning" | "info"
    message: str
    column: str | None = None
    rows: list[int] = field(default_factory=list)
    """1-based line numbers in the original CSV (the header is line 1)."""

    def rows_preview(self) -> str:
        """Short human-readable list of affected CSV lines."""
        if not self.rows:
            return ""
        shown = ", ".join(str(r) for r in self.rows[:_MAX_ROWS_IN_MESSAGE])
        extra = len(self.rows) - _MAX_ROWS_IN_MESSAGE
        return shown + (f" (+{extra} more)" if extra > 0 else "")


@dataclass
class ValidationResult:
    """Outcome of validating a dataset."""

    data: pd.DataFrame
    rejected: pd.DataFrame
    issues: list[ValidationIssue]
    input_rows: int

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def infos(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "info"]

    @property
    def is_valid(self) -> bool:
        """True when there are no blocking errors and at least one usable row."""
        return not self.errors and not self.data.empty

    def summary(self) -> dict[str, int]:
        return {
            "input_rows": self.input_rows,
            "accepted_rows": len(self.data),
            "rejected_rows": len(self.rejected),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _canonical_key(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip()).lower()


def normalise_columns(
    frame: pd.DataFrame, known_columns: Iterable[str]
) -> tuple[pd.DataFrame, list[ValidationIssue]]:
    """Rename columns to their canonical spelling.

    Returns the renamed frame and issues describing renames or collisions.
    """
    lookup = {_canonical_key(name): name for name in known_columns}
    rename: dict[str, str] = {}
    issues: list[ValidationIssue] = []
    for original in frame.columns:
        stripped = str(original).strip()
        canonical = lookup.get(_canonical_key(stripped), stripped)
        if canonical != original:
            rename[original] = canonical
    renamed = frame.rename(columns=rename)

    duplicated = renamed.columns[renamed.columns.duplicated()].unique().tolist()
    if duplicated:
        issues.append(
            ValidationIssue(
                "error",
                "Duplicate column names after normalisation: "
                + ", ".join(map(str, duplicated))
                + ". Each column must appear only once.",
            )
        )
    meaningful = {k: v for k, v in rename.items() if str(k).strip() != v}
    if meaningful:
        pairs = ", ".join(f"'{k}' → '{v}'" for k, v in meaningful.items())
        issues.append(ValidationIssue("info", f"Column names standardised: {pairs}."))
    return renamed, issues


def _strip_text(series: pd.Series) -> pd.Series:
    # pyarrow-backed strings: regex / strip run vectorised in C (~20x faster on large files)
    stripped = series.astype("string[pyarrow]").str.strip()
    return stripped.mask(stripped == "", pd.NA)


def _source_rows(frame: pd.DataFrame, mask: pd.Series) -> list[int]:
    return frame.loc[mask, SOURCE_ROW].astype(int).tolist()


def _append_note(notes: pd.Series, mask: pd.Series, text: pd.Series | str) -> pd.Series:
    if not mask.any():
        return notes
    addition = text if isinstance(text, str) else text[mask]
    current = notes[mask]
    notes.loc[mask] = np.where(current == "", addition, current + "; " + addition)
    return notes


def _coerce_numeric(
    frame: pd.DataFrame,
    column: str,
    kind: str,
    notes: pd.Series,
    issues: list[ValidationIssue],
) -> pd.Series:
    raw = _strip_text(frame[column])
    # Fast path: plain numbers (the vast majority) convert in one vectorised call.
    numeric = pd.to_numeric(raw, errors="coerce").astype("float64")
    ambiguous_comma = pd.Series(False, index=raw.index)
    needs_cleaning = raw.notna() & numeric.isna()
    if needs_cleaning.any():
        # Slow path only for values that failed: currency symbols, thousands separators.
        cleaned = raw[needs_cleaning].str.replace(_CURRENCY_PATTERN.pattern, "", regex=True)
        has_comma = cleaned.str.contains(",", regex=False).fillna(False).astype(bool)
        valid = cleaned.str.fullmatch(_THOUSANDS_PATTERN.pattern).fillna(False).astype(bool)
        ambiguous = has_comma & ~valid
        ambiguous_comma.loc[needs_cleaning] = ambiguous.to_numpy()
        cleaned = cleaned.str.replace(",", "", regex=False).mask(ambiguous, pd.NA)
        numeric.loc[needs_cleaning] = pd.to_numeric(cleaned, errors="coerce").astype("float64").to_numpy()
    numeric = numeric.where(np.isfinite(numeric))  # treat inf as invalid

    missing = raw.isna()
    unparseable = ~missing & numeric.isna()
    too_large = numeric.notna() & (numeric.abs() > MAX_NUMERIC_VALUE)
    if kind == "positive_number":
        below_range = numeric.notna() & (numeric <= 0)
        range_text = "must be greater than zero"
    else:
        below_range = numeric.notna() & (numeric < 0)
        range_text = "cannot be negative"
    out_of_range = below_range | too_large

    if unparseable.any():
        hint = (
            " Commas are only accepted as thousands separators (e.g. 1,234.50); "
            "use a dot for decimals."
            if (unparseable & ambiguous_comma).any()
            else ""
        )
        issues.append(
            ValidationIssue(
                "warning",
                f"{int(unparseable.sum())} non-numeric value(s) in {column} were "
                "treated as missing." + hint,
                column,
                _source_rows(frame, unparseable),
            )
        )
        notes = _append_note(notes, unparseable, f"{column} invalid ('" + raw.fillna("") + "')")
    for mask, text in (
        (below_range, range_text),
        (too_large, f"exceed the plausible maximum of {MAX_NUMERIC_VALUE:,.0f}"),
    ):
        if mask.any():
            issues.append(
                ValidationIssue(
                    "warning",
                    f"{int(mask.sum())} value(s) in {column} {text} and were treated as missing.",
                    column,
                    _source_rows(frame, mask),
                )
            )
    if out_of_range.any():
        notes = _append_note(
            notes, out_of_range, f"{column} out of range (" + raw.fillna("") + ")"
        )
    if missing.any():
        issues.append(
            ValidationIssue(
                "warning",
                f"{int(missing.sum())} missing value(s) in {column}. Metrics that depend "
                "on this field are skipped or flagged for those rows (never estimated).",
                column,
                _source_rows(frame, missing),
            )
        )
        notes = _append_note(notes, missing, f"{column} missing")

    return numeric.mask(out_of_range)


def _coerce_date(
    frame: pd.DataFrame,
    column: str,
    notes: pd.Series,
    issues: list[ValidationIssue],
    reference: date,
    *,
    allow_future: bool = False,
) -> pd.Series:
    raw = _strip_text(frame[column])
    # utc=True makes mixed / timezone-aware inputs safe (they are converted to
    # UTC and the timezone dropped); naive dates are unaffected.
    parsed = pd.to_datetime(raw, errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
    unparseable = raw.notna() & parsed.isna()
    non_iso = parsed.notna() & ~raw.str.match(_ISO_DATE_PATTERN).fillna(False).astype(bool)
    ambiguous = pd.Series(False, index=raw.index)
    if non_iso.any():  # extraction only for the non-ISO subset
        parts = raw[non_iso].str.extract(_DAY_MONTH_PATTERN).astype("float64")
        ambiguous.loc[non_iso] = (
            (parts[0] <= 12) & (parts[1] <= 12) & (parts[0] != parts[1])
        ).to_numpy()
    if ambiguous.any():
        issues.append(
            ValidationIssue(
                "info",
                f"{int(ambiguous.sum())} date(s) in {column} are ambiguous between day/month "
                "and month/day order and were read as month/day (e.g. 03/04/2026 = 4 March). "
                "Use YYYY-MM-DD to avoid ambiguity.",
                column,
                _source_rows(frame, ambiguous),
            )
        )
    if unparseable.any():
        issues.append(
            ValidationIssue(
                "warning",
                f"{int(unparseable.sum())} unrecognised date(s) in {column} were treated "
                "as missing. Use the YYYY-MM-DD format.",
                column,
                _source_rows(frame, unparseable),
            )
        )
        _append_note(notes, unparseable, f"{column} invalid ('" + raw.fillna("") + "')")
    if not allow_future:
        future = parsed.notna() & (parsed.dt.normalize() > pd.Timestamp(reference))
        if future.any():
            issues.append(
                ValidationIssue(
                    "warning",
                    f"{int(future.sum())} date(s) in {column} are in the future and were "
                    "treated as missing.",
                    column,
                    _source_rows(frame, future),
                )
            )
            _append_note(notes, future, f"{column} in the future")
            parsed = parsed.mask(future)
    return parsed.dt.normalize()


def _reject(
    frame: pd.DataFrame, mask: pd.Series, reason: str | pd.Series
) -> pd.DataFrame:
    rejected = frame.loc[mask].copy()
    rejected[REJECTION_REASON] = reason if isinstance(reason, str) else reason[mask]
    return rejected


def _empty_result(raw: pd.DataFrame, issues: list[ValidationIssue]) -> ValidationResult:
    return ValidationResult(
        data=pd.DataFrame(columns=list(REQUIRED_COLUMNS)),
        rejected=pd.DataFrame(),
        issues=issues,
        input_rows=len(raw),
    )


# --------------------------------------------------------------------------- #
# Inventory dataset
# --------------------------------------------------------------------------- #


def validate_inventory(
    raw: pd.DataFrame,
    duplicate_strategy: str = "first",
    reference_date: date | None = None,
) -> ValidationResult:
    """Validate and type-convert a raw inventory table.

    Args:
        raw: Output of :func:`src.data_loader.load_csv` (all-text columns), or
            any DataFrame with the expected columns.
        duplicate_strategy: ``"first"`` or ``"last"`` — which record to keep
            when the same SKU appears with conflicting values.
        reference_date: "Today" for future-date checks (defaults to today).

    Returns:
        A :class:`ValidationResult`. Check :attr:`ValidationResult.is_valid`
        before using :attr:`ValidationResult.data`.
    """
    if duplicate_strategy not in {"first", "last"}:
        raise ValueError("duplicate_strategy must be 'first' or 'last'.")
    reference = reference_date or date.today()
    issues: list[ValidationIssue] = []

    if raw is None or raw.shape[1] == 0:
        return _empty_result(
            pd.DataFrame(), [ValidationIssue("error", "The dataset has no columns.")]
        )

    frame, rename_issues = normalise_columns(raw, [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS])
    issues.extend(rename_issues)
    if any(i.severity == "error" for i in issues):
        return _empty_result(raw, issues)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing_columns:
        issues.append(
            ValidationIssue(
                "error",
                "Missing required column(s): "
                + ", ".join(missing_columns)
                + ". Required columns are: "
                + ", ".join(REQUIRED_COLUMNS)
                + ".",
            )
        )
        return _empty_result(raw, issues)

    if frame.empty:
        issues.append(ValidationIssue("error", "The dataset has a header but no data rows."))
        return _empty_result(raw, issues)

    known = [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]
    extra = [c for c in frame.columns if c not in known]
    if extra:
        issues.append(
            ValidationIssue(
                "info",
                "Additional column(s) kept as-is but not used in calculations: "
                + ", ".join(map(str, extra))
                + ".",
            )
        )

    frame = frame.reset_index(drop=True).copy()
    frame.insert(0, SOURCE_ROW, np.arange(2, len(frame) + 2))
    raw_copy = frame.copy()  # untouched values for the rejected-rows report
    rejected_parts: list[pd.DataFrame] = []

    present_known = [c for c in known if c in frame.columns]
    for column in present_known:
        frame[column] = _strip_text(frame[column])

    # 1. Completely empty rows
    empty_rows = frame[present_known].isna().all(axis=1)
    if empty_rows.any():
        issues.append(
            ValidationIssue(
                "info",
                f"{int(empty_rows.sum())} completely empty row(s) were skipped.",
                rows=_source_rows(frame, empty_rows),
            )
        )
        rejected_parts.append(_reject(raw_copy, empty_rows, "Empty row"))

    # 2. Missing SKU
    missing_sku = frame["SKU_ID"].isna() & ~empty_rows
    if missing_sku.any():
        issues.append(
            ValidationIssue(
                "warning",
                f"{int(missing_sku.sum())} row(s) without a SKU_ID were excluded from "
                "analysis (see the rejected rows report).",
                "SKU_ID",
                _source_rows(frame, missing_sku),
            )
        )
        rejected_parts.append(_reject(raw_copy, missing_sku, "Missing SKU_ID"))

    keep = ~(empty_rows | missing_sku)
    frame = frame.loc[keep].reset_index(drop=True)
    raw_copy = raw_copy.loc[keep].reset_index(drop=True)
    notes = pd.Series("", index=frame.index, dtype=object)

    # 3. Type coercion
    for column, (kind, _) in {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}.items():
        if column not in frame.columns:
            continue
        if kind.endswith("number"):
            frame[column] = _coerce_numeric(frame, column, kind, notes, issues)
        elif kind == "date":
            frame[column] = _coerce_date(frame, column, notes, issues, reference)

    # 4. Placeholder text for descriptive fields
    for column, placeholder in (("Product_Name", None), ("Category", UNCATEGORIZED)):
        missing = frame[column].isna()
        if missing.any():
            fill = (
                "Unnamed product (" + frame["SKU_ID"].astype(str) + ")"
                if placeholder is None
                else pd.Series(placeholder, index=frame.index)
            )
            frame[column] = frame[column].fillna(fill)
            issues.append(
                ValidationIssue(
                    "warning",
                    f"{int(missing.sum())} missing {column} value(s) were labelled "
                    f"'{placeholder or 'Unnamed product (SKU)'}'.",
                    column,
                    _source_rows(frame, missing),
                )
            )
            _append_note(notes, missing, f"{column} missing (placeholder used)")
    frame["SKU_ID"] = frame["SKU_ID"].astype(str)
    frame["Product_Name"] = frame["Product_Name"].astype(str)
    frame["Category"] = frame["Category"].astype(str)

    # 5. Duplicates
    compare_cols = [c for c in present_known]
    exact_dup = frame.duplicated(subset=compare_cols, keep="first")
    if exact_dup.any():
        issues.append(
            ValidationIssue(
                "info",
                f"{int(exact_dup.sum())} exact duplicate row(s) were collapsed into one.",
                "SKU_ID",
                _source_rows(frame, exact_dup),
            )
        )
        rejected_parts.append(_reject(raw_copy, exact_dup, "Exact duplicate row"))
        frame = frame.loc[~exact_dup].reset_index(drop=True)
        raw_copy = raw_copy.loc[~exact_dup].reset_index(drop=True)
        notes = notes.loc[~exact_dup].reset_index(drop=True)

    case_key = frame["SKU_ID"].str.casefold()
    case_variants = case_key.duplicated(keep=False) & ~frame["SKU_ID"].duplicated(keep=False)
    if case_variants.any():
        issues.append(
            ValidationIssue(
                "warning",
                f"{int(case_variants.sum())} row(s) have SKU_IDs that differ only by letter "
                "case (e.g. 'abc-1' vs 'ABC-1'). They are kept as separate products — check "
                "whether they are the same item.",
                "SKU_ID",
                _source_rows(frame, case_variants),
            )
        )

    conflict = frame.duplicated(subset=["SKU_ID"], keep=duplicate_strategy)
    if conflict.any():
        kept_rows = (
            frame.loc[~conflict].set_index("SKU_ID")[SOURCE_ROW].astype(int).to_dict()
        )
        reasons = (
            "Conflicting duplicate SKU (kept CSV line "
            + frame["SKU_ID"].map(kept_rows).astype(str)
            + ")"
        )
        issues.append(
            ValidationIssue(
                "warning",
                f"{int(conflict.sum())} row(s) reuse an existing SKU_ID with different "
                f"values; the {duplicate_strategy} occurrence was kept and the others "
                "were excluded (see the rejected rows report).",
                "SKU_ID",
                _source_rows(frame, conflict),
            )
        )
        rejected_parts.append(_reject(raw_copy, conflict, reasons))
        frame = frame.loc[~conflict].reset_index(drop=True)
        notes = notes.loc[~conflict].reset_index(drop=True)

    frame[QUALITY_NOTES] = notes.values
    rejected = (
        pd.concat(rejected_parts, ignore_index=True).sort_values(SOURCE_ROW)
        if rejected_parts
        else pd.DataFrame(columns=[*raw_copy.columns, REJECTION_REASON])
    )

    if frame.empty:
        issues.append(
            ValidationIssue("error", "No valid rows remain after validation.")
        )

    return ValidationResult(
        data=frame, rejected=rejected.reset_index(drop=True), issues=issues, input_rows=len(raw)
    )


# --------------------------------------------------------------------------- #
# Optional daily sales history
# --------------------------------------------------------------------------- #


def validate_sales_history(
    raw: pd.DataFrame,
    known_skus: Iterable[str] | None = None,
    reference_date: date | None = None,
) -> ValidationResult:
    """Validate a daily sales history table (``Date``, ``SKU_ID``, ``Units_Sold``).

    Rows with an invalid date, SKU, or unit count cannot contribute to a trend
    and are rejected (with reasons). Multiple records for the same SKU and day
    are summed, which is the correct treatment for transactional exports.
    """
    reference = reference_date or date.today()
    issues: list[ValidationIssue] = []
    if raw is None or raw.shape[1] == 0:
        return _empty_result(
            pd.DataFrame(), [ValidationIssue("error", "The history file has no columns.")]
        )
    frame, rename_issues = normalise_columns(raw, HISTORY_REQUIRED_COLUMNS)
    issues.extend(rename_issues)
    missing_columns = [c for c in HISTORY_REQUIRED_COLUMNS if c not in frame.columns]
    if missing_columns or any(i.severity == "error" for i in issues):
        if missing_columns:
            issues.append(
                ValidationIssue(
                    "error",
                    "Sales history is missing column(s): " + ", ".join(missing_columns)
                    + ". Expected: Date, SKU_ID, Units_Sold.",
                )
            )
        return _empty_result(raw, issues)
    if frame.empty:
        issues.append(ValidationIssue("error", "The history file has no data rows."))
        return _empty_result(raw, issues)

    frame = frame[list(HISTORY_REQUIRED_COLUMNS)].reset_index(drop=True).copy()
    frame.insert(0, SOURCE_ROW, np.arange(2, len(frame) + 2))
    raw_copy = frame.copy()
    notes = pd.Series("", index=frame.index, dtype=object)
    frame["SKU_ID"] = _strip_text(frame["SKU_ID"])
    frame["Date"] = _coerce_date(frame, "Date", notes, issues, reference)
    frame["Units_Sold"] = _coerce_numeric(
        frame, "Units_Sold", "non_negative_number", notes, issues
    )

    reason = pd.Series("", index=frame.index, dtype=object)
    reason = _append_note(reason, frame["SKU_ID"].isna(), "Missing SKU_ID")
    reason = _append_note(reason, frame["Date"].isna(), "Missing or invalid Date")
    reason = _append_note(reason, frame["Units_Sold"].isna(), "Missing or invalid Units_Sold")
    bad = reason != ""
    rejected = _reject(raw_copy, bad, reason)
    if bad.any():
        issues.append(
            ValidationIssue(
                "warning",
                f"{int(bad.sum())} history row(s) could not be used and were excluded.",
                rows=_source_rows(frame, bad),
            )
        )
    frame = frame.loc[~bad]

    before = len(frame)
    frame = frame.groupby(["SKU_ID", "Date"], as_index=False)["Units_Sold"].sum()
    if len(frame) < before:
        issues.append(
            ValidationIssue(
                "info",
                f"{before - len(frame)} row(s) shared a SKU and date with another row "
                "and were summed into daily totals.",
            )
        )

    if known_skus is not None:
        unknown = sorted(set(frame["SKU_ID"]) - set(map(str, known_skus)))
        if unknown:
            preview = ", ".join(unknown[:_MAX_ROWS_IN_MESSAGE])
            issues.append(
                ValidationIssue(
                    "info",
                    f"{len(unknown)} SKU(s) in the history are not in the inventory file "
                    f"and are ignored in product-level views: {preview}"
                    + (" …" if len(unknown) > _MAX_ROWS_IN_MESSAGE else ""),
                )
            )
    if frame.empty:
        issues.append(ValidationIssue("error", "No valid history rows remain."))
    return ValidationResult(
        data=frame.sort_values(["SKU_ID", "Date"]).reset_index(drop=True),
        rejected=rejected.reset_index(drop=True),
        issues=issues,
        input_rows=len(raw),
    )
