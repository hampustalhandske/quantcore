"""Tests for conformance.compare: `ConformanceTarget` registration rules and the comparators."""

import dataclasses
import math

import conformance.compare as compare_module
import numpy as np
import pytest
from conformance.compare import (
    ArrayComparator,
    ConformanceTarget,
    DecisiveFloatArrayComparator,
    ReportOnlyArrayComparator,
    ReportOnlyScalarComparator,
    ScalarComparator,
    Tolerance,
    TupleComparator,
    registry,
)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test its own empty registry, so a throwaway target never reaches the real one."""
    monkeypatch.setattr(compare_module, "_registry", {})


def test_conformance_target_registration_and_registry() -> None:
    """Constructing a ConformanceTarget registers it by name in a read-only registry view."""
    name = "toy_target"
    fn = ConformanceTarget(
        name=name,
        tier=2,
        oracle="test-oracle",
        citation="test-citation",
        compare=ScalarComparator(),
        quantcore=lambda: 1.0,
    )
    assert registry()[name] is fn
    with pytest.raises(TypeError):
        registry()[name] = fn


def test_conformance_target_duplicate_name_raises_value_error() -> None:
    """Registering a second ConformanceTarget under a used name raises ValueError naming it."""
    name = "toy_target"
    ConformanceTarget(
        name=name,
        tier=2,
        oracle="o",
        citation="c",
        compare=ScalarComparator(),
        quantcore=lambda: 1.0,
    )
    with pytest.raises(ValueError, match=name):
        ConformanceTarget(
            name=name,
            tier=2,
            oracle="o",
            citation="c",
            compare=ScalarComparator(),
            quantcore=lambda: 2.0,
        )


def test_conformance_target_requires_at_least_one_callable() -> None:
    """A ConformanceTarget with neither `quantcore` nor `reference` raises and is not registered."""
    with pytest.raises(ValueError, match="toy_target"):
        ConformanceTarget(
            name="toy_target", tier=2, oracle="o", citation="c", compare=ScalarComparator()
        )
    assert "toy_target" not in registry()


def test_conformance_target_accepts_reference_only() -> None:
    """A target with only an oracle (no quantcore callable) is valid and registered."""
    fn = ConformanceTarget(
        name="toy_target",
        tier=1,
        oracle="o",
        citation="c",
        compare=ScalarComparator(),
        reference=lambda: 1.0,
    )
    assert fn.quantcore is None
    assert registry()["toy_target"] is fn


def test_conformance_target_is_frozen_data_not_a_callable() -> None:
    """A target only carries its two callables: it cannot be called itself, nor mutated."""
    fn = ConformanceTarget(
        name="toy_target",
        tier=2,
        oracle="o",
        citation="c",
        compare=ScalarComparator(),
        quantcore=lambda: 1.0,
        reference=lambda: 2.0,
    )
    assert not callable(fn)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fn.tier = 1  # type: ignore[misc]


def test_registry_returns_a_snapshot() -> None:
    """A mapping returned by registry() does not change when another target registers later."""
    before = registry()
    ConformanceTarget(
        name="toy_target",
        tier=2,
        oracle="o",
        citation="c",
        compare=ScalarComparator(),
        quantcore=lambda: 1.0,
    )
    assert "toy_target" not in before
    assert "toy_target" in registry()


_MAX_ERRORS_CASES = [
    ("simple", 3.0, 1.0, 2.0, 2.0),
    ("expected_zero_uses_tiny", 1e-200, 0.0, 1e-200, 1e-200 / 1e-300),
    ("actual_equals_expected", 5.0, 5.0, 0.0, 0.0),
    ("negative_values", -3.0, -1.0, 2.0, 2.0),
]


@pytest.mark.parametrize(
    ("name", "actual", "expected", "abs_err", "rel_err"),
    _MAX_ERRORS_CASES,
    ids=[c[0] for c in _MAX_ERRORS_CASES],
)
def test_scalar_comparator_max_errors_hand_computed(
    name, actual, expected, abs_err, rel_err
) -> None:
    """ScalarComparator.max_errors matches the hand-computed abs/rel error, including expected 0."""
    got_abs, got_rel = ScalarComparator().max_errors(actual, expected)
    assert got_abs == pytest.approx(abs_err)
    assert got_rel == pytest.approx(rel_err, rel=1e-6)


def test_array_comparator_max_errors_hand_computed_and_shape_mismatch() -> None:
    """ArrayComparator.max_errors matches hand computation and raises on mismatched shapes."""
    actual = np.array([1.0, 2.0, 3.0])
    expected = np.array([1.0, 2.5, 2.0])
    abs_err, rel_err = ArrayComparator().max_errors(actual, expected)
    assert abs_err == pytest.approx(1.0)
    assert rel_err == pytest.approx(0.5, rel=1e-12)
    with pytest.raises(ValueError, match="shape"):
        ArrayComparator().max_errors(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_tuple_comparator_element_wise() -> None:
    """TupleComparator applies each comparator element-wise and aggregates the worst error."""
    comparator = TupleComparator(ScalarComparator(), ScalarComparator())
    abs_err, rel_err = comparator.max_errors((1.0, 10.0), (1.0, 8.0))
    assert abs_err == pytest.approx(2.0)
    assert rel_err == pytest.approx(2.0 / 8.0, rel=1e-6)
    assert comparator.is_finite((1.0, 2.0))
    assert not comparator.is_finite((1.0, float("nan")))


def test_assert_close_boundary_exactly_at_tolerance_passes_just_beyond_fails() -> None:
    """assert_close passes exactly at atol + rtol*|expected| and fails just beyond that boundary."""
    expected = 10.0
    tol = Tolerance(rtol=0.01, atol=0.1, justification="test")
    boundary = tol.atol + tol.rtol * abs(expected)
    ScalarComparator().assert_close(expected + boundary, expected, tol)
    with pytest.raises(AssertionError):
        ScalarComparator().assert_close(expected + boundary + 1e-9, expected, tol)


def test_scalar_comparator_is_finite() -> None:
    """ScalarComparator.is_finite is True for finite floats and False for NaN and inf."""
    assert ScalarComparator().is_finite(1.0)
    assert not ScalarComparator().is_finite(float("nan"))
    assert not ScalarComparator().is_finite(float("inf"))
    assert not math.isfinite(float("-inf"))


def test_array_comparator_assert_close_passes_within_and_raises_beyond_tolerance() -> None:
    """ArrayComparator.assert_close is silent within tolerance and raises with values otherwise."""
    tol = Tolerance(rtol=0.0, atol=0.01, justification="test")
    expected = np.array([1.0, 2.0, 3.0])
    close = np.array([1.005, 2.0, 3.0])
    far = np.array([1.005, 2.0, 9.0])
    ArrayComparator().assert_close(close, expected, tol)
    with pytest.raises(AssertionError) as exc_info:
        ArrayComparator().assert_close(far, expected, tol)
    message = str(exc_info.value)
    assert "max_abs_error" in message
    assert "max_rel_error" in message
    with pytest.raises(ValueError, match="shape"):
        ArrayComparator().assert_close(np.array([1.0, 2.0]), expected, tol)


def test_tuple_comparator_assert_close_delegates_to_each_component() -> None:
    """TupleComparator.assert_close is silent when every component is close, else raises."""
    comparator = TupleComparator(ScalarComparator(), ScalarComparator())
    tol = Tolerance(rtol=0.0, atol=0.01, justification="test")
    comparator.assert_close((1.0, 2.0), (1.005, 2.005), tol)
    with pytest.raises(AssertionError):
        comparator.assert_close((1.0, 2.0), (1.0, 5.0), tol)


def test_array_comparator_is_finite() -> None:
    """ArrayComparator.is_finite is False as soon as one element is NaN or infinite."""
    assert ArrayComparator().is_finite(np.array([1.0, 2.0, 3.0]))
    assert not ArrayComparator().is_finite(np.array([1.0, float("nan"), 2.0]))
    assert not ArrayComparator().is_finite(np.array([1.0, float("inf")]))


def test_report_only_comparators_measure_but_never_judge() -> None:
    """Report-only comparators report errors and finiteness, but assert_close never raises."""
    tol = Tolerance(rtol=0.0, atol=0.0, justification="test")

    scalar = ReportOnlyScalarComparator()
    assert scalar.report_only_top_level is True
    assert scalar.max_errors(3.0, 1.0) == ScalarComparator().max_errors(3.0, 1.0)
    assert not scalar.is_finite(float("nan"))
    scalar.assert_close(3.0, 1.0, tol)

    array = ReportOnlyArrayComparator()
    actual = np.array([1.0, 9.0])
    expected = np.array([1.0, 2.0])
    assert array.max_errors(actual, expected) == ArrayComparator().max_errors(actual, expected)
    assert not array.is_finite(np.array([1.0, float("inf")]))
    array.assert_close(actual, expected, tol)
    assert not hasattr(array, "report_only_top_level")


def test_decisive_float_array_comparator_judges_the_decision_component_only() -> None:
    """Errors and closeness key off the scalar; finiteness requires both components."""
    comparator = DecisiveFloatArrayComparator()
    tol = Tolerance(rtol=0.0, atol=0.01, justification="test")
    far_secondary = np.array([100.0, -100.0])
    secondary = np.array([1.0, 2.0])

    assert comparator.max_errors((1.5, far_secondary), (1.0, secondary)) == (0.5, 0.5)
    comparator.assert_close((1.005, far_secondary), (1.0, secondary), tol)
    with pytest.raises(AssertionError):
        comparator.assert_close((2.0, secondary), (1.0, secondary), tol)

    assert comparator.is_finite((1.0, secondary))
    assert not comparator.is_finite((float("nan"), secondary))
    assert not comparator.is_finite((1.0, np.array([1.0, float("nan")])))
