"""Tests for conformance.harness: `check()` and `check_second_oracle()` on toy targets.

Each test builds a throwaway `ConformanceTarget` whose two callables are plain functions, so the
recorded outcome can be stated by hand. The registry and the harness's result log are swapped for
empty per-test ones, and the toy target's tolerance is declared only for the test's duration, so
nothing recorded here reaches a real conformance run.
"""

import math
from collections.abc import Callable
from typing import Any

import conformance.compare as compare_module
import numpy as np
import pytest
from conformance import harness
from conformance.compare import (
    Comparator,
    ConformanceTarget,
    DecisiveFloatArrayComparator,
    ScalarComparator,
    Tolerance,
    TupleComparator,
)
from conformance.tolerances import ILL_CONDITIONED_RTOL_CAP, ConditionAwareTolerance

_NAME = "toy_target"
_TOLERANCE = Tolerance(rtol=0.0, atol=0.01, justification="test")


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use an empty registry and result log per test, and declare the toy target's tolerance."""
    monkeypatch.setattr(compare_module, "_registry", {})
    monkeypatch.setattr(harness, "_results", [])
    monkeypatch.setattr(harness, "_details", [])
    monkeypatch.setitem(harness.TOLERANCES, _NAME, _TOLERANCE)


def _target(
    quantcore: Callable[..., Any] | None,
    reference: Callable[..., Any] | None,
    compare: Comparator[Any] | None = None,
) -> ConformanceTarget[Any, Any]:
    return ConformanceTarget(
        name=_NAME,
        tier=2,
        oracle="hand-computed",
        citation="test",
        compare=compare if compare is not None else ScalarComparator(),
        quantcore=quantcore,
        reference=reference,
    )


def _raise_value_error(*args: Any, **kwargs: Any) -> float:
    raise ValueError("boom")


def test_check_records_an_in_tolerance_case() -> None:
    """A quantcore value within tolerance of the oracle's is recorded as within_tolerance."""
    target = _target(lambda x: 1.005, lambda x: 1.0)
    (case,) = harness.check(target, 3.0, family="normal", length_bucket="short", seed=7)

    assert case.function == _NAME
    assert case.family == "normal"
    assert case.length_bucket == "short"
    assert case.quantity == "value"
    assert case.within_tolerance is True
    assert case.quantcore_raised is None
    assert case.reference_raised is None
    assert case.ill_conditioned is False
    assert case.oracle == ""
    assert case.abs_err == pytest.approx(0.005)
    assert case.rel_err == pytest.approx(0.005)
    assert harness.results() == [case]

    (detail,) = harness.details()
    assert detail.function == _NAME
    assert detail.quantity == "value"
    assert detail.seed == 7
    assert detail.expected_abs == 1.0
    assert detail.quantcore_value == "1.005"
    assert detail.reference_value == "1.0"
    assert detail.inputs_repr == "(3.0,)"
    assert detail.oracle == ""


def test_check_records_an_out_of_tolerance_case_without_raising() -> None:
    """A quantcore value far from the oracle's is recorded as a mismatch; check() does not raise."""
    target = _target(lambda x: 2.0, lambda x: 1.0)
    (case,) = harness.check(target, 3.0, family="normal", length_bucket="short")

    assert case.within_tolerance is False
    assert case.quantcore_raised is None
    assert case.abs_err == 1.0
    assert case.rel_err == 1.0


def test_check_records_a_quantcore_exception_without_raising() -> None:
    """An exception from quantcore is recorded by type and message, with no error to measure."""
    target = _target(_raise_value_error, lambda x: 1.0)
    (case,) = harness.check(target, 3.0, family="normal", length_bucket="short")

    assert case.within_tolerance is False
    assert case.quantcore_raised == "ValueError: boom"
    assert math.isnan(case.abs_err)
    assert math.isnan(case.rel_err)

    (detail,) = harness.details()
    assert detail.abs_err is None
    assert detail.rel_err is None
    assert detail.quantcore_value is None
    assert detail.reference_value == "1.0"


def test_check_records_a_non_finite_quantcore_result_as_a_failure() -> None:
    """A NaN from quantcore is recorded like an exception: never within tolerance."""
    target = _target(lambda x: float("nan"), lambda x: 1.0)
    (case,) = harness.check(target, 3.0, family="normal", length_bucket="short")

    assert case.within_tolerance is False
    assert case.quantcore_raised == "non-finite result"


@pytest.mark.parametrize(
    "reference",
    [_raise_value_error, lambda x: float("inf"), None],
    ids=["raises", "non-finite", "missing"],
)
def test_check_raises_assertion_error_when_the_reference_is_unusable(
    reference: Callable[..., Any] | None,
) -> None:
    """A reference that raises, returns a non-finite value or is absent fails the run outright."""
    target = _target(lambda x: 1.0, reference)
    with pytest.raises(AssertionError, match=_NAME):
        harness.check(target, 3.0, family="normal", length_bucket="short")
    assert harness.results() == []
    assert harness.details() == []


def test_check_passes_the_same_arguments_to_both_callables() -> None:
    """Positional and keyword arguments reach quantcore and the reference unchanged."""
    seen: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def quantcore(*args: Any, **kwargs: Any) -> float:
        seen["quantcore"] = (args, kwargs)
        return 1.0

    def reference(*args: Any, **kwargs: Any) -> float:
        seen["reference"] = (args, kwargs)
        return 1.0

    harness.check(
        _target(quantcore, reference), 3.0, 4, family="normal", length_bucket="short", scale=2.0
    )
    assert seen["quantcore"] == ((3.0, 4), {"scale": 2.0})
    assert seen["reference"] == seen["quantcore"]


def test_check_records_one_case_per_named_quantity() -> None:
    """A tuple result judged by a TupleComparator yields one case per quantity, each on its own."""
    target = _target(
        lambda x: (1.0, 5.0),
        lambda x: (1.0, 2.0),
        TupleComparator(ScalarComparator(), ScalarComparator()),
    )
    statistic, p_value = harness.check(
        target, 3.0, family="normal", length_bucket="short", quantities=("statistic", "p_value")
    )

    assert (statistic.quantity, statistic.within_tolerance) == ("statistic", True)
    assert (p_value.quantity, p_value.within_tolerance) == ("p_value", False)
    assert p_value.abs_err == 3.0
    assert [detail.quantity for detail in harness.details()] == ["statistic", "p_value"]


def test_check_never_fails_the_secondary_quantity_of_a_decisive_comparator() -> None:
    """Only the decision scalar is judged; the secondary array's error is recorded, not failed."""
    target = _target(
        lambda x: (1.005, np.array([100.0, 2.0])),
        lambda x: (1.0, np.array([1.0, 2.0])),
        DecisiveFloatArrayComparator(),
    )
    log_likelihood, params = harness.check(
        target, 3.0, family="normal", length_bucket="short", quantities=("log_likelihood", "params")
    )

    assert log_likelihood.within_tolerance is True
    assert params.within_tolerance is True
    assert params.abs_err == 99.0


def test_check_widens_the_tolerance_and_flags_an_ill_conditioned_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A condition-aware tolerance is evaluated on the case's own arguments."""
    tolerance = ConditionAwareTolerance(
        rtol=1e-10, atol=0.0, justification="test", kappa=lambda x: x, K=1.0, p=1.0
    )
    monkeypatch.setitem(harness.TOLERANCES, _NAME, tolerance)
    target = _target(lambda x: x * (1.0 + 1e-5), lambda x: x)

    (well_conditioned,) = harness.check(target, 1.0, family="normal", length_bucket="short")
    assert well_conditioned.ill_conditioned is False
    assert well_conditioned.within_tolerance is False

    assert tolerance.rtol_eff(1e12) > ILL_CONDITIONED_RTOL_CAP
    (ill_conditioned,) = harness.check(target, 1e12, family="normal", length_bucket="short")
    assert ill_conditioned.ill_conditioned is True
    assert ill_conditioned.within_tolerance is True


def test_reset_clears_the_recorded_results_and_details() -> None:
    """reset() empties both logs."""
    harness.check(
        _target(lambda x: 1.0, lambda x: 1.0), 3.0, family="normal", length_bucket="short"
    )
    assert len(harness.results()) == 1
    assert len(harness.details()) == 1

    harness.reset()
    assert harness.results() == []
    assert harness.details() == []


def test_check_second_oracle_compares_the_oracle_against_the_reference() -> None:
    """The second oracle is judged against the reference, each called with its own keywords."""
    target = _target(_raise_value_error, lambda x, rf: x - rf)
    case = harness.check_second_oracle(
        target,
        3.0,
        family="normal",
        length_bucket="short",
        oracle_name="second",
        oracle_callable=lambda x, rf_annual: x - rf_annual / 2.0,
        quantity="rf_nonzero",
        seed=11,
        reference_kwargs={"rf": 1.0},
        rf_annual=2.0,
    )

    assert case.oracle == "second"
    assert case.quantity == "rf_nonzero"
    assert case.within_tolerance is True
    assert case.quantcore_raised is None
    assert case.abs_err == 0.0
    assert harness.results() == [case]

    (detail,) = harness.details()
    assert detail.oracle == "second"
    assert detail.seed == 11
    assert detail.quantcore_value == "2.0"
    assert detail.reference_value == "2.0"


def test_check_second_oracle_records_a_raising_oracle_without_raising() -> None:
    """A second oracle that raises is the finding itself: recorded, never an error."""
    case = harness.check_second_oracle(
        _target(None, lambda x: 1.0),
        3.0,
        family="normal",
        length_bucket="short",
        oracle_name="second",
        oracle_callable=_raise_value_error,
        quantity="value",
    )

    assert case.within_tolerance is False
    assert case.quantcore_raised == "ValueError: boom"
    assert math.isnan(case.abs_err)


def test_check_second_oracle_raises_assertion_error_when_the_reference_raises() -> None:
    """A reference that raises invalidates the cross-check, exactly as in check()."""
    with pytest.raises(AssertionError, match=_NAME):
        harness.check_second_oracle(
            _target(lambda x: 1.0, _raise_value_error),
            3.0,
            family="normal",
            length_bucket="short",
            oracle_name="second",
            oracle_callable=lambda x: 1.0,
            quantity="value",
        )
    assert harness.results() == []
