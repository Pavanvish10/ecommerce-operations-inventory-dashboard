"""CSV ingestion and validation: upload, invalid files, missing values, duplicates."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoadError, load_csv
from src.data_validator import (
    QUALITY_NOTES,
    REJECTION_REASON,
    UNCATEGORIZED,
    validate_inventory,
    validate_sales_history,
)
from tests.conftest import AS_OF, ERROR_SAMPLE, SAMPLE_INVENTORY, make_raw

HEADER = "SKU_ID,Product_Name,Category,Current_Stock,Units_Sold,Unit_Price,Reorder_Level,Supplier_Lead_Time\n"


# --------------------------------------------------------------------------- #
# CSV upload / loading
# --------------------------------------------------------------------------- #


class TestLoadCsv:
    def test_loads_from_path(self):
        frame = load_csv(SAMPLE_INVENTORY)
        assert len(frame) == 160
        assert "SKU_ID" in frame.columns

    def test_loads_from_bytes_and_file_like(self):
        content = (HEADER + "A1,Pen,Office,10,5,1.5,2,3\n").encode()
        from_bytes = load_csv(content)
        from_buffer = load_csv(io.BytesIO(content))
        pd.testing.assert_frame_equal(from_bytes, from_buffer)
        assert from_bytes.loc[0, "Current_Stock"] == "10"  # raw text preserved

    def test_reads_values_as_text_to_preserve_raw_input(self):
        frame = load_csv((HEADER + "007,Pen,Office,10,5,1.50,2,3\n").encode())
        assert frame.loc[0, "SKU_ID"] == "007"  # leading zeros not lost

    def test_handles_utf8_bom_and_cp1252(self):
        bom = ("﻿" + HEADER + "A1,Café,Office,1,1,1,1,1\n").encode("utf-8")
        assert load_csv(bom).columns[0] == "SKU_ID"
        legacy = (HEADER + "A1,Café,Office,1,1,1,1,1\n").encode("cp1252")
        assert load_csv(legacy).loc[0, "Product_Name"] == "Café"

    @pytest.mark.parametrize(
        "content, message",
        [
            (b"", "empty"),
            (b"   \n  \n", "no data"),
            (b"PK\x03\x04\x00\x00binary", "binary"),
            (b"a,b\n1,2,3,4\n", "well-formed"),
            (b"SKU_ID;Product_Name;Category\nA;B;C\n", "semicolons"),
        ],
    )
    def test_invalid_files_raise_meaningful_errors(self, content, message):
        with pytest.raises(DataLoadError, match=message):
            load_csv(content)

    def test_extra_fields_report_line_numbers_instead_of_shifting_columns(self):
        content = (HEADER + "A1,Pen,Office,10,5,1.5,2,3\nA2,Pen, blue,Office,10,5,1.5,2,3\n").encode()
        with pytest.raises(DataLoadError, match=r"line\(s\) 3 have more fields"):
            load_csv(content)

    def test_quoted_commas_are_fine(self):
        frame = load_csv((HEADER + 'A1,"Pen, blue",Office,10,5,1.5,2,3\n').encode())
        assert frame.loc[0, "Product_Name"] == "Pen, blue"

    def test_missing_path(self, tmp_path):
        with pytest.raises(DataLoadError, match="not found"):
            load_csv(tmp_path / "missing.csv")


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


class TestSchema:
    def test_sample_dataset_is_clean(self):
        result = validate_inventory(load_csv(SAMPLE_INVENTORY), reference_date=AS_OF)
        assert result.is_valid
        assert result.summary()["accepted_rows"] == 160
        assert not result.warnings and result.rejected.empty

    def test_missing_required_columns_is_blocking_error(self):
        raw = make_raw([{}]).drop(columns=["Unit_Price", "Reorder_Level"])
        result = validate_inventory(raw)
        assert not result.is_valid
        assert "Unit_Price" in result.errors[0].message
        assert "Reorder_Level" in result.errors[0].message

    def test_header_only_is_error(self):
        result = validate_inventory(load_csv(HEADER.encode()))
        assert not result.is_valid
        assert "no data rows" in result.errors[0].message

    def test_column_names_are_normalised(self):
        raw = make_raw([{}]).rename(
            columns={"SKU_ID": " sku id ", "Unit_Price": "unit-price", "Category": "CATEGORY"}
        )
        result = validate_inventory(raw)
        assert result.is_valid
        assert {"SKU_ID", "Unit_Price", "Category"} <= set(result.data.columns)
        assert any("standardised" in i.message for i in result.infos)

    def test_duplicate_column_names_after_normalisation(self):
        raw = make_raw([{}])
        raw["unit price"] = "5"
        result = validate_inventory(raw)
        assert not result.is_valid
        assert "Duplicate column" in result.errors[0].message

    def test_optional_columns_not_required(self):
        raw = make_raw([{}]).drop(columns=["Last_Restock_Date"])
        assert validate_inventory(raw).is_valid

    def test_extra_columns_preserved(self):
        raw = make_raw([{}])
        raw["Warehouse"] = "North"
        result = validate_inventory(raw)
        assert result.data.loc[0, "Warehouse"] == "North"
        assert any("Warehouse" in i.message for i in result.infos)


# --------------------------------------------------------------------------- #
# Values: missing, invalid numbers, dates
# --------------------------------------------------------------------------- #


class TestValues:
    def test_numeric_columns_are_typed(self):
        result = validate_inventory(make_raw([{}]))
        for column in ["Current_Stock", "Units_Sold", "Unit_Price", "Reorder_Level"]:
            assert pd.api.types.is_float_dtype(result.data[column])
        assert pd.api.types.is_datetime64_any_dtype(result.data["Last_Restock_Date"])

    @pytest.mark.parametrize("bad", ["abc", "12..5", "ten", "--3"])
    def test_non_numeric_values_become_missing_and_are_reported(self, bad):
        result = validate_inventory(make_raw([{"Current_Stock": bad}, {}]))
        assert result.is_valid
        assert np.isnan(result.data.loc[0, "Current_Stock"])
        assert result.data.loc[1, "Current_Stock"] == 100
        warning = next(i for i in result.warnings if i.column == "Current_Stock")
        assert warning.rows == [2]  # CSV line 2 (header is line 1)
        assert "invalid" in result.data.loc[0, QUALITY_NOTES]

    @pytest.mark.parametrize("column", ["Current_Stock", "Units_Sold", "Unit_Price", "Reorder_Level",
                                        "Supplier_Lead_Time"])
    def test_negative_values_are_rejected_as_invalid(self, column):
        result = validate_inventory(make_raw([{column: -5}]))
        assert np.isnan(result.data.loc[0, column])
        assert any("cannot be negative" in i.message for i in result.warnings)

    def test_infinite_values_are_invalid(self):
        result = validate_inventory(make_raw([{"Unit_Price": "inf"}]))
        assert np.isnan(result.data.loc[0, "Unit_Price"])

    def test_currency_and_thousands_separators_are_parsed(self):
        result = validate_inventory(make_raw([{"Unit_Price": "$1,200.50", "Current_Stock": "1,000"}]))
        assert result.data.loc[0, "Unit_Price"] == pytest.approx(1200.5)
        assert result.data.loc[0, "Current_Stock"] == 1000
        assert not result.warnings

    def test_zero_values_are_valid(self):
        result = validate_inventory(make_raw([{"Current_Stock": 0, "Units_Sold": 0, "Reorder_Level": 0}]))
        assert not result.warnings
        assert result.data.loc[0, "Current_Stock"] == 0

    def test_missing_numeric_values_reported_not_filled(self):
        result = validate_inventory(make_raw([{"Reorder_Level": None}, {"Reorder_Level": "  "}]))
        assert result.data["Reorder_Level"].isna().all()
        warning = next(i for i in result.warnings if i.column == "Reorder_Level")
        assert warning.rows == [2, 3]

    def test_missing_text_fields_get_explicit_placeholders(self):
        result = validate_inventory(make_raw([{"Category": None, "Product_Name": ""}]))
        row = result.data.iloc[0]
        assert row["Category"] == UNCATEGORIZED
        assert row["Product_Name"] == "Unnamed product (SKU-1)"
        assert len([i for i in result.warnings if i.column in {"Category", "Product_Name"}]) == 2

    def test_invalid_and_future_dates(self):
        result = validate_inventory(
            make_raw([{"Last_Restock_Date": "yesterday"}, {"Last_Restock_Date": "2030-01-01"},
                      {"Last_Restock_Date": "2026-11-15"}]),
            reference_date=AS_OF,
        )
        dates = result.data["Last_Restock_Date"]
        assert pd.isna(dates[0]) and pd.isna(dates[1])
        assert dates[2] == pd.Timestamp("2026-11-15")  # boundary: today is allowed
        assert any("future" in i.message for i in result.warnings)

    def test_sales_period_must_be_positive(self):
        raw = make_raw([{}, {}])
        raw["Sales_Period_Days"] = ["0", "14"]
        result = validate_inventory(raw)
        assert np.isnan(result.data.loc[0, "Sales_Period_Days"])
        assert result.data.loc[1, "Sales_Period_Days"] == 14

    def test_whitespace_is_trimmed(self):
        result = validate_inventory(make_raw([{"SKU_ID": "  A-1 ", "Current_Stock": " 5 "}]))
        assert result.data.loc[0, "SKU_ID"] == "A-1"
        assert result.data.loc[0, "Current_Stock"] == 5


# --------------------------------------------------------------------------- #
# Row-level rules: missing SKU, empty rows, duplicates
# --------------------------------------------------------------------------- #


class TestRows:
    def test_rows_without_sku_are_rejected_with_reason(self):
        result = validate_inventory(make_raw([{"SKU_ID": None}, {}]))
        assert len(result.data) == 1
        assert result.rejected.loc[0, REJECTION_REASON] == "Missing SKU_ID"
        assert result.rejected.loc[0, "Source_Row"] == 2

    def test_completely_empty_rows_are_skipped_and_reported(self):
        raw = make_raw([{}, {}])
        raw.loc[1, :] = None
        result = validate_inventory(raw)
        assert len(result.data) == 1
        assert result.rejected.loc[0, REJECTION_REASON] == "Empty row"

    def test_exact_duplicates_collapsed(self):
        raw = make_raw([{"SKU_ID": "X"}, {"SKU_ID": "X"}])
        result = validate_inventory(raw)
        assert len(result.data) == 1
        assert result.rejected.loc[0, REJECTION_REASON] == "Exact duplicate row"
        assert any("exact duplicate" in i.message for i in result.infos)

    @pytest.mark.parametrize("strategy, kept_stock, kept_line", [("first", 10, 2), ("last", 30, 4)])
    def test_conflicting_duplicate_skus(self, strategy, kept_stock, kept_line):
        raw = make_raw([
            {"SKU_ID": "X", "Current_Stock": 10},
            {"SKU_ID": "Y"},
            {"SKU_ID": "X", "Current_Stock": 30},
        ])
        result = validate_inventory(raw, duplicate_strategy=strategy)
        assert result.data["SKU_ID"].is_unique
        assert result.data.set_index("SKU_ID").loc["X", "Current_Stock"] == kept_stock
        assert len(result.rejected) == 1
        assert f"kept CSV line {kept_line}" in result.rejected.loc[0, REJECTION_REASON]

    def test_invalid_duplicate_strategy(self):
        with pytest.raises(ValueError):
            validate_inventory(make_raw([{}]), duplicate_strategy="merge")

    def test_all_rows_rejected_is_error(self):
        result = validate_inventory(make_raw([{"SKU_ID": None}, {"SKU_ID": ""}]))
        assert not result.is_valid
        assert "No valid rows" in result.errors[-1].message

    def test_nothing_discarded_silently(self):
        """Every input row is either accepted or listed as rejected."""
        result = validate_inventory(load_csv(ERROR_SAMPLE), reference_date=AS_OF)
        assert len(result.data) + len(result.rejected) == result.input_rows
        assert set(result.rejected[REJECTION_REASON]) == {
            "Missing SKU_ID", "Exact duplicate row", "Conflicting duplicate SKU (kept CSV line 7)",
        }


# --------------------------------------------------------------------------- #
# Sales history
# --------------------------------------------------------------------------- #


class TestSalesHistory:
    def _raw(self, rows):
        return pd.DataFrame(rows, columns=["Date", "SKU_ID", "Units_Sold"], dtype=object)

    def test_valid_history_and_same_day_rows_summed(self):
        result = validate_sales_history(self._raw([
            ["2026-11-01", "A", "2"], ["2026-11-01", "A", "3"], ["2026-11-02", "A", "1"],
        ]), reference_date=AS_OF)
        assert result.is_valid
        assert result.data["Units_Sold"].tolist() == [5, 1]
        assert any("summed" in i.message for i in result.infos)

    def test_bad_rows_rejected_with_reasons(self):
        result = validate_sales_history(self._raw([
            ["bad-date", "A", "2"], ["2026-11-01", None, "3"], ["2026-11-02", "A", "-1"],
            ["2026-11-03", "A", "4"],
        ]), reference_date=AS_OF)
        assert len(result.data) == 1
        assert len(result.rejected) == 3
        assert "Missing or invalid Date" in result.rejected[REJECTION_REASON].iloc[0]

    def test_unknown_skus_reported(self):
        result = validate_sales_history(
            self._raw([["2026-11-01", "ZZZ", "1"]]), known_skus=["A"], reference_date=AS_OF
        )
        assert any("not in the inventory" in i.message for i in result.infos)

    def test_missing_columns(self):
        result = validate_sales_history(pd.DataFrame({"Date": ["2026-11-01"]}))
        assert not result.is_valid
