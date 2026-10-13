"""Input normalisation and unit/sign/rate conversions shared by every conformance target.

Every target accepts exactly one array-like shape at its Python boundary (a pandas
``Series``/``DataFrame``, a NumPy array, or a plain sequence) and immediately converts it here to
a contiguous ``float64`` NumPy array before handing it to quantcore or to the oracle. This is the
one place that decides what counts as a valid numeric input, so both sides of a comparison always
see byte-identical data.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

type FloatArray = npt.NDArray[np.float64]
type ArrayLike1D = pd.Series | npt.NDArray[np.generic] | Sequence[float]
type ArrayLike2D = pd.DataFrame | npt.NDArray[np.generic] | Sequence[Sequence[float]]

_REJECTED_KINDS = {
    "b": "bool",
    "O": "object",
    "U": "str",
    "S": "bytes",
    "c": "complex",
    "M": "datetime64",
    "m": "timedelta64",
}


def as_float_array(
    x: ArrayLike1D | ArrayLike2D,
    *,
    name: str,
    ndim: Literal[1, 2] = 1,
    min_len: int = 1,
) -> FloatArray:
    """Convert array-like input to a contiguous ``float64`` array, rejecting anything unsafe.

    Any pandas index or column labels are dropped: only the numeric values are kept. The returned
    array always owns its memory (it is either a fresh allocation or a defensive copy of the
    input), so a callee mutating it in place can never corrupt the caller's original array.

    Args:
        x: Data to convert: a pandas ``Series``/``DataFrame``, a NumPy array, or a sequence of
            floats.
        name: Name of the argument, used in every raised ``ValueError`` so the caller can tell
            which input failed.
        ndim: Required number of dimensions (``1`` for a series, ``2`` for a matrix).
        min_len: Minimum required length along the first axis.

    Returns:
        A C-contiguous ``float64`` array with the same shape as the input, owning its own memory.

    Raises:
        ValueError: ``x`` has the wrong number of dimensions, is shorter than ``min_len`` along
            its first axis, contains a dtype that is not real numeric (``bool``, ``object``,
            ``str``, ``bytes``, ``complex``, a datetime/timedelta type), or contains NaN or ±inf.
    """
    if isinstance(x, np.ma.MaskedArray):
        if np.ma.is_masked(x):
            raise ValueError(f"{name} must not contain masked elements")
        raw: npt.NDArray[np.generic] = np.ma.getdata(x)
    elif isinstance(x, pd.Series | pd.DataFrame):
        raw = x.to_numpy()
    else:
        raw = np.asarray(x)

    kind = raw.dtype.kind
    if kind in _REJECTED_KINDS:
        raise ValueError(f"{name} must be real-valued numeric data, got dtype {raw.dtype!r}")
    if kind not in "iuf":
        raise ValueError(f"{name} must be real-valued numeric data, got dtype {raw.dtype!r}")

    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {raw.ndim}-D")
    if raw.shape[0] < min_len:
        raise ValueError(f"{name} must have at least {min_len} observation(s), got {raw.shape[0]}")

    array: FloatArray = np.array(raw, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must not contain NaN or infinite values")
    return array


def annual_rate_to_per_period(rate_annual: float, periods_per_year: int) -> float:
    """De-annualise a rate geometrically to a per-period rate.

    Uses the convention of ``empyrical``/``quantstats``: ``(1 + rate_annual) ** (1 / p) - 1``, so
    that compounding the per-period rate ``p`` times reproduces ``rate_annual`` exactly.

    Args:
        rate_annual: Annualised rate, e.g. ``0.02`` for 2%.
        periods_per_year: Number of periods per year, e.g. ``252`` for daily data.

    Returns:
        The equivalent per-period rate.
    """
    return float((1.0 + rate_annual) ** (1.0 / periods_per_year) - 1.0)


def bps_to_fraction(x: float) -> float:
    """Convert a value in basis points to a fraction, e.g. ``10`` (bps) -> ``0.001``."""
    return x / 10_000.0


def fraction_to_bps(x: float) -> float:
    """Convert a fraction to basis points, e.g. ``0.001`` -> ``10`` (bps)."""
    return x * 10_000.0
