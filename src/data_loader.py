"""CSV ingestion.

Files are read with every column as text so that the validator can see the raw
values and report exactly which cells failed type conversion instead of pandas
silently guessing (or silently turning bad values into NaN).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

CsvSource = Union[str, Path, bytes, BinaryIO]

#: Uploads larger than this are rejected to protect the app's memory.
MAX_FILE_BYTES = 100 * 1024 * 1024  # ~1 GB peak RAM at this size; see README "Performance"
_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


class DataLoadError(Exception):
    """Raised when a file cannot be parsed as a usable CSV table."""


def _read_bytes(source: CsvSource) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"File not found: {path}")
        return path.read_bytes()
    if hasattr(source, "getvalue"):
        return source.getvalue()
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        return source.read()
    raise DataLoadError(f"Unsupported data source type: {type(source).__name__}")


def _decode(raw: bytes) -> str:
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    # latin-1 can decode any byte sequence, so this is unreachable in practice.
    raise DataLoadError("Could not decode the file. Please save it as UTF-8 CSV.")


def _check_field_counts(text: str, max_reported: int = 10) -> None:
    """Reject rows with more fields than the header.

    pandas would otherwise silently turn the surplus leading fields into an
    index, shifting every value into the wrong column.
    """
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header:
        return
    expected = len(header)
    offending: list[int] = []
    try:
        for row in reader:
            if len(row) > expected:
                offending.append(reader.line_num)
    except csv.Error as exc:
        raise DataLoadError(f"The file is not a well-formed CSV: {exc}") from exc
    if offending:
        lines = ", ".join(map(str, offending[:max_reported]))
        more = f" (+{len(offending) - max_reported} more)" if len(offending) > max_reported else ""
        raise DataLoadError(
            f"The file is not a well-formed CSV: line(s) {lines}{more} have more fields than "
            f"the {expected}-column header. Check for unquoted commas inside values."
        )


def load_csv(source: CsvSource) -> pd.DataFrame:
    """Read a CSV file into a DataFrame of strings.

    Args:
        source: A path, raw bytes, or a binary file-like object (such as a
            Streamlit ``UploadedFile``).

    Returns:
        DataFrame where every column has ``object`` dtype and empty cells are NaN.

    Raises:
        DataLoadError: If the file is empty, too large, not text, or malformed.
    """
    raw = _read_bytes(source)
    if len(raw) == 0:
        raise DataLoadError("The file is empty.")
    if len(raw) > MAX_FILE_BYTES:
        raise DataLoadError(
            f"The file is {len(raw) / 1e6:.0f} MB; the maximum supported size is "
            f"{MAX_FILE_BYTES / 1e6:.0f} MB."
        )
    if b"\x00" in raw[:4096]:
        raise DataLoadError(
            "The file appears to be binary (for example an Excel workbook). "
            "Please export it as CSV."
        )

    text = _decode(raw)
    if not text.strip():
        raise DataLoadError("The file contains no data.")
    _check_field_counts(text)

    try:
        frame = pd.read_csv(
            io.StringIO(text),
            dtype=str,
            skipinitialspace=True,
            on_bad_lines="error",
            index_col=False,
        )
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The file contains no columns.") from exc
    except (pd.errors.ParserError, csv.Error) as exc:
        raise DataLoadError(
            f"The file is not a well-formed CSV (inconsistent number of fields): {exc}"
        ) from exc

    if frame.shape[1] == 0:
        raise DataLoadError("The file contains no columns.")
    if frame.shape[1] == 1 and ";" in str(frame.columns[0]):
        raise DataLoadError(
            "Only one column was detected and the header contains ';'. "
            "The file seems to use semicolons as separators; please save it with commas."
        )
    return frame
