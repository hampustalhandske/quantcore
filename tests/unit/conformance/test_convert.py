"""Tests for conformance.convert: as_float_array and the rate/unit helpers."""

from decimal import Decimal

import numpy as np
import pytest

# pandas and hypothesis are installed by the `oracle` extra only; skip this module without them.
pytest.importorskip("pandas")
pytest.importorskip("hypothesis")

import pandas as pd
from conformance.convert import (
    annual_rate_to_per_period,
    as_float_array,
    bps_to_fraction,
    fraction_to_bps,
)
from hypothesis import given, settings
from hypothesis import strategies as st

_ACCEPTED_1D = [
    ("list", [1.0, 2.0, 3.0]),
    ("tuple", (1.0, 2.0, 3.0)),
    ("ndarray_float64", np.array([1.0, 2.0, 3.0], dtype=np.float64)),
    ("ndarray_float32", np.array([1.0, 2.0, 3.0], dtype=np.float32)),
    ("ndarray_int64", np.array([1, 2, 3], dtype=np.int64)),
    ("non_contiguous_view", np.array([1.0, 9.0, 2.0, 9.0, 3.0, 9.0])[::2]),
    (
        "series_datetime_index",
        pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3)),
    ),
]


@pytest.mark.parametrize(("kind", "value"), _ACCEPTED_1D, ids=[k for k, _ in _ACCEPTED_1D])
def test_as_float_array_accepts_every_1d_input_kind(kind, value) -> None:
    """Every accepted 1-D input kind converts to a C-contiguous float64 array with equal values."""
    out = as_float_array(value, name="x", ndim=1, min_len=1)
    assert out.dtype == np.float64
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, np.asarray(value, dtype=np.float64))


def test_as_float_array_accepts_fortran_ordered_2d() -> None:
    """A Fortran-ordered 2-D array converts to a C-contiguous float64 array with equal values."""
    src = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert not src.flags["C_CONTIGUOUS"]
    out = as_float_array(src, name="x", ndim=2, min_len=1)
    assert out.dtype == np.float64
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, src)


def test_as_float_array_accepts_dataframe_for_ndim_2() -> None:
    """A pandas DataFrame is accepted for ndim=2 and its index/columns are dropped."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = as_float_array(df, name="x", ndim=2, min_len=1)
    assert out.dtype == np.float64
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, df.to_numpy(dtype=np.float64))


_REJECTED_VALUES = [
    ("nan", [1.0, float("nan"), 3.0], 1),
    ("plus_inf", [1.0, float("inf"), 3.0], 1),
    ("minus_inf", [1.0, float("-inf"), 3.0], 1),
    ("bool_dtype", np.array([True, False, True]), 1),
    ("object_dtype", np.array([1.0, "x", 3.0], dtype=object), 1),
    ("str_dtype", np.array(["1.0", "2.0"]), 1),
    ("complex_dtype", np.array([1.0 + 1j, 2.0 + 2j]), 1),
    ("datetime64_dtype", np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"), 1),
]


@pytest.mark.parametrize(
    ("kind", "value", "ndim"), _REJECTED_VALUES, ids=[k for k, _, _ in _REJECTED_VALUES]
)
def test_as_float_array_rejects_bad_values(kind, value, ndim) -> None:
    """NaN, +/-inf and non-numeric dtypes raise ValueError naming the argument."""
    with pytest.raises(ValueError, match="thing") as exc_info:
        as_float_array(value, name="thing", ndim=ndim, min_len=1)
    assert "thing" in str(exc_info.value)


def test_as_float_array_rejects_wrong_ndim_both_directions() -> None:
    """A 1-D input when ndim=2 is required, and a 2-D input when ndim=1 is required, both raise."""
    with pytest.raises(ValueError, match="thing"):
        as_float_array([1.0, 2.0, 3.0], name="thing", ndim=2, min_len=1)
    with pytest.raises(ValueError, match="thing"):
        as_float_array([[1.0, 2.0], [3.0, 4.0]], name="thing", ndim=1, min_len=1)


def test_as_float_array_rejects_length_below_min_len() -> None:
    """An input shorter than min_len raises ValueError naming the argument."""
    with pytest.raises(ValueError, match="thing"):
        as_float_array([1.0, 2.0], name="thing", ndim=1, min_len=3)


def test_as_float_array_mutating_result_does_not_affect_caller_array() -> None:
    """Mutating the array returned for ndarray input never changes the caller's original array."""
    original = np.array([1.0, 2.0, 3.0])
    out = as_float_array(original, name="x", ndim=1, min_len=1)
    out[0] = 999.0
    assert original[0] == 1.0


def test_as_float_array_mutating_result_does_not_affect_caller_series() -> None:
    """Mutating the array returned for a Series input never changes the caller's original Series."""
    original = pd.Series([1.0, 2.0, 3.0])
    out = as_float_array(original, name="x", ndim=1, min_len=1)
    out[0] = 999.0
    assert original.iloc[0] == 1.0


def test_annual_rate_to_per_period_hand_values_and_zero() -> None:
    """annual_rate_to_per_period matches a geometric hand computation and is 0 at 0% annual rate."""
    assert annual_rate_to_per_period(0.0, 12) == 0.0
    assert annual_rate_to_per_period(0.0, 252) == 0.0
    got = annual_rate_to_per_period(0.05, 12)
    expected = (1.05) ** (1.0 / 12.0) - 1.0
    assert got == pytest.approx(expected, abs=1e-15)


@given(
    rate_annual=st.floats(min_value=-0.99, max_value=10.0, allow_nan=False, allow_infinity=False),
    periods_per_year=st.integers(min_value=1, max_value=366),
)
@settings(max_examples=100, deadline=None)
def test_annual_rate_to_per_period_compounding_round_trips(rate_annual, periods_per_year) -> None:
    """Compounding the per-period rate periods_per_year times recovers the annual rate to 1e-12."""
    per_period = annual_rate_to_per_period(rate_annual, periods_per_year)
    recovered = (1.0 + per_period) ** periods_per_year - 1.0
    assert recovered == pytest.approx(rate_annual, abs=1e-12, rel=1e-12)


def test_as_float_array_rejects_masked_array_with_masked_element() -> None:
    """A MaskedArray with any masked element is rejected, even if its fill value looks finite."""
    garbage = np.ma.MaskedArray([1.0, 2.0, 1e20], mask=[False, False, True])
    with pytest.raises(ValueError, match="thing"):
        as_float_array(garbage, name="thing", ndim=1, min_len=1)


def test_as_float_array_accepts_masked_array_with_no_masked_element() -> None:
    """A MaskedArray with no masked element converts normally."""
    clean = np.ma.MaskedArray([1.0, 2.0, 3.0], mask=[False, False, False])
    out = as_float_array(clean, name="x", ndim=1, min_len=1)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_as_float_array_rejects_nullable_int64_series_with_na() -> None:
    """A pandas Int64 (nullable) Series holding pd.NA is rejected."""
    s = pd.Series(pd.array([1, 2, pd.NA], dtype="Int64"))
    with pytest.raises(ValueError, match="thing"):
        as_float_array(s, name="thing", ndim=1, min_len=1)


def test_as_float_array_converts_nullable_int64_series_without_na() -> None:
    """A pandas Int64 (nullable) Series with no missing values converts to float64."""
    s = pd.Series(pd.array([1, 2, 3], dtype="Int64"))
    out = as_float_array(s, name="x", ndim=1, min_len=1)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_as_float_array_rejects_nullable_float64_series_with_na() -> None:
    """A pandas Float64 (nullable) Series holding pd.NA is rejected."""
    s = pd.Series(pd.array([1.0, 2.0, pd.NA], dtype="Float64"))
    with pytest.raises(ValueError, match="thing"):
        as_float_array(s, name="thing", ndim=1, min_len=1)


def test_as_float_array_converts_nullable_float64_series_without_na() -> None:
    """A pandas Float64 (nullable) Series with no missing values converts to float64."""
    s = pd.Series(pd.array([1.0, 2.0, 3.0], dtype="Float64"))
    out = as_float_array(s, name="x", ndim=1, min_len=1)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_as_float_array_rejects_pandas_string_dtype() -> None:
    """A pandas string-dtype Series is rejected, not silently coerced."""
    s = pd.Series(["1.0", "2.0"], dtype="string")
    with pytest.raises(ValueError, match="thing"):
        as_float_array(s, name="thing", ndim=1, min_len=1)


def test_as_float_array_rejects_decimal_objects() -> None:
    """An array of decimal.Decimal objects is rejected."""
    s = pd.Series([Decimal("1.0"), Decimal("2.0")])
    with pytest.raises(ValueError, match="thing"):
        as_float_array(s, name="thing", ndim=1, min_len=1)


def test_as_float_array_rejects_0d_array_as_wrong_ndim() -> None:
    """A 0-d array is rejected as a wrong-ndim error, not silently reshaped."""
    with pytest.raises(ValueError, match="thing"):
        as_float_array(np.array(1.0), name="thing", ndim=1, min_len=1)


def test_as_float_array_rejects_structured_dtype() -> None:
    """A structured (void) dtype array is rejected as non-numeric."""
    structured = np.zeros(3, dtype=[("x", "f4"), ("y", "f4")])
    with pytest.raises(ValueError, match="thing"):
        as_float_array(structured, name="thing", ndim=1, min_len=1)


def test_as_float_array_rejects_empty_input_below_min_len() -> None:
    """Empty input is rejected the same way as any other input shorter than min_len."""
    with pytest.raises(ValueError, match="thing"):
        as_float_array([], name="thing", ndim=1, min_len=1)


def test_bps_to_fraction_and_back_hand_values() -> None:
    """bps_to_fraction and fraction_to_bps match hand-computed values and invert each other."""
    assert bps_to_fraction(100.0) == pytest.approx(0.01)
    assert bps_to_fraction(1.0) == pytest.approx(0.0001)
    assert fraction_to_bps(0.01) == pytest.approx(100.0)
    assert fraction_to_bps(0.0001) == pytest.approx(1.0)


@given(x=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_bps_to_fraction_round_trips_with_fraction_to_bps(x) -> None:
    """fraction_to_bps(bps_to_fraction(x)) recovers x, and the reverse round trip holds too."""
    assert fraction_to_bps(bps_to_fraction(x)) == pytest.approx(x, abs=1e-9, rel=1e-9)
    assert bps_to_fraction(fraction_to_bps(x)) == pytest.approx(x, abs=1e-9, rel=1e-9)
