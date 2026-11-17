"""Small vectorised helpers shared by the analytics engines."""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Decimal places used to absorb binary floating-point noise before integer
#: rounding (e.g. 3.0000000000000004 → 3.0 so that ceil() gives 3, not 4).
FLOAT_TOLERANCE_DECIMALS = 9


def clean_float(values: pd.Series) -> pd.Series:
    """Round away floating-point noise; NaN and ±inf are preserved."""
    return values.astype("float64").round(FLOAT_TOLERANCE_DECIMALS)


def safe_ceil(values: pd.Series) -> pd.Series:
    """``ceil`` that is robust to floating-point noise."""
    return np.ceil(clean_float(values))


def safe_floor(values: pd.Series) -> pd.Series:
    """``floor`` that is robust to floating-point noise."""
    return np.floor(clean_float(values))


def join_nonempty(parts: list[pd.Series], sep: str = "; ") -> pd.Series:
    """Element-wise join of string Series, skipping empty strings and NaN.

    Vectorised replacement for ``frame.apply(lambda row: sep.join(...), axis=1)``,
    which is orders of magnitude slower on large frames.
    """
    if not parts:
        raise ValueError("join_nonempty needs at least one Series")
    result = parts[0].fillna("").astype(str)
    for part in parts[1:]:
        part = part.fillna("").astype(str)
        both = (result != "") & (part != "")
        result = (result + np.where(both, sep, "") + part)
    return result.astype(object)
