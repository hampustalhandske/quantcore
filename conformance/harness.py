"""`check()`: run quantcore and its oracle on one input, compare, and record the outcome.

`check()` calls a `ConformanceTarget`'s `quantcore` callable and its `reference` (oracle) callable
with identical arguments and appends one `CaseResult` per recorded quantity to an in-process log.
A quantcore mismatch, exception or non-finite result never raises here: it is recorded, and
`classify.py` turns the recorded cases into a status afterwards, so one failing function cannot
stop the rest of the suite from being measured. A `reference` callable that raises or returns a
non-finite value does raise `AssertionError`, because a broken oracle invalidates the comparison
itself rather than saying anything about quantcore.

`classify.CaseResult` (what `classify()` consumes) carries only the fields a status decision
needs. The seed and raw values needed to reconstruct a failing case by hand for a findings
write-up are recorded alongside it, in a separate `ReproDetail`, so `CaseResult`'s shape stays
exactly what `conformance/classify.py`'s tests pin down.
"""

import dataclasses
import math
import threading
from typing import Any

import numpy as np

from conformance.classify import CaseResult
from conformance.compare import (
    Comparator,
    ConformanceTarget,
    DecisiveFloatArrayComparator,
    ReportOnlyArrayComparator,
    ScalarComparator,
    Tolerance,
    TupleComparator,
)
from conformance.tolerances import TOLERANCES, ConditionAwareTolerance

_results: list[CaseResult] = []
_details: list["ReproDetail"] = []
_lock = threading.Lock()


@dataclasses.dataclass(frozen=True)
class ReproDetail:
    """Everything needed to reconstruct one recorded case by hand, for a findings write-up.

    One `check()` call producing several named quantities (see `CaseResult.quantity`) yields one
    `ReproDetail` per quantity, matching the corresponding `CaseResult` one-to-one.

    Attributes:
        function: Name of the `ConformanceTarget` under test.
        family: Generator family this case came from.
        quantity: Which named quantity this detail describes; `"value"` for a single-result
            function.
        seed: Seed reproducing this case's input via `numpy.random.default_rng(seed)`, or `None`
            for a hand-picked (non-generated) case.
        abs_err: Maximum absolute error of quantcore against the reference for this quantity, or
            `None` if quantcore raised.
        rel_err: Maximum relative error of quantcore against the reference for this quantity, or
            `None` if quantcore raised.
        expected_abs: Absolute magnitude of the reference (oracle) value for this quantity; used
            to exclude a near-zero denominator from `max_rel_error` aggregation.
        quantcore_value: `repr()` of quantcore's value for this quantity, or `None` if it raised.
            For a second-oracle cross-check record (see `CaseResult.oracle`), this holds the
            second oracle's own value instead — there is no quantcore involved.
        reference_value: `repr()` of the reference (oracle) value for this quantity.
        inputs_repr: `repr()` of the positional arguments passed to both callables, truncated for
            large arrays.
        oracle: Mirrors `CaseResult.oracle`: `""` for the primary comparison, or the name of a
            second oracle (e.g. `"quantstats"`) this record cross-checks instead.
    """

    function: str
    family: str
    quantity: str
    seed: int | None
    abs_err: float | None
    rel_err: float | None
    expected_abs: float
    quantcore_value: str | None
    reference_value: str
    inputs_repr: str
    oracle: str = ""


def _describe_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


_MAX_INPUT_REPR_LEN = 60_000
"""Truncation length for one argument's repr in a recorded case.

Generous rather than tight: a findings write-up copy-pastes this verbatim as its minimal repro.
A `float64` array's `repr()` runs roughly 20 characters per element, so this comfortably holds a
full `n<=1000` regression/cointegration array or an `n<=2000` returns array (the largest a Tier-1
generator draws by default under the `ci` hypothesis profile) without truncating mid-array;
only the `full` profile's rarer `n` close to 5000 can still exceed it.
"""


def _repr_inputs(args: tuple[Any, ...]) -> str:
    """Render ``args`` as a Python tuple literal, so `f"call(*{_repr_inputs(args)})"` is valid.

    Always wraps in `(...,)` (trailing comma included) so a single-argument call reproduces as a
    genuine 1-tuple rather than as the bare argument, which `*args`-unpacking would otherwise
    silently misinterpret (e.g. spreading an array's own elements as separate positional args).
    """
    parts = []
    for arg in args:
        text = repr(arg)
        if len(text) > _MAX_INPUT_REPR_LEN:
            text = text[:_MAX_INPUT_REPR_LEN] + "...<truncated>"
        parts.append(text)
    return "(" + ", ".join(parts) + ",)"


def _component_comparators(
    compare: Comparator[Any], n_quantities: int
) -> tuple[Comparator[Any], ...]:
    """Split a function's `Comparator` into one comparator per recorded quantity.

    A `TupleComparator` already holds one sub-comparator per tuple position (e.g. `statistic` and
    `p_value`); a `DecisiveFloatArrayComparator` describes a `(decision, secondary)` pair (e.g. an
    iterative estimator's attained objective plus its raw, never-compared parameter vector) and
    splits into a `ScalarComparator` for the first quantity and a `ReportOnlyArrayComparator` (its
    `assert_close` never raises) for the second. Any other comparator describes a single quantity
    (a scalar or a whole array).
    """
    if isinstance(compare, TupleComparator):
        return compare.comparators
    if isinstance(compare, DecisiveFloatArrayComparator):
        return (ScalarComparator(), ReportOnlyArrayComparator())
    return (compare,) * n_quantities


def _split_value(value: Any, n_quantities: int) -> tuple[Any, ...]:
    """Split a callable's return value into one component per recorded quantity."""
    if n_quantities == 1:
        return (value,)
    return tuple(value)


def _magnitude(comparator: Comparator[Any], value: Any) -> float:
    """Absolute magnitude of a scalar or array quantity, for the near-zero rel-error floor.

    A single recorded "value" quantity that is itself a tuple (e.g. a Monte Carlo pricer's
    `(price, standard_error)`, judged as one opaque unit by its own custom `Comparator` rather
    than split via `quantities=`) reports the magnitude of its first, primary component. A
    quantity with no numeric magnitude at all (e.g. a `datetime.date`, judged by its own
    `Comparator` such as `_DateComparator`) reports `0.0`: this value only ever feeds the
    near-zero relative-error-floor exclusion in `ReproDetail`, not a pass/fail decision.
    """
    if isinstance(value, np.ndarray):
        return float(np.max(np.abs(value))) if value.size else 0.0
    if isinstance(value, tuple):
        return abs(float(value[0]))
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return 0.0


def check(
    fn: ConformanceTarget[Any, Any],
    *args: Any,
    family: str,
    length_bucket: str,
    seed: int | None = None,
    quantities: tuple[str, ...] = ("value",),
    **kwargs: Any,
) -> list[CaseResult]:
    """Run `fn`'s quantcore and reference callables on the same input and record the comparison.

    A quantcore exception, non-finite result or out-of-tolerance value is recorded in the returned
    `CaseResult`s (`quantcore_raised`, `within_tolerance=False`); it does not raise.

    Args:
        fn: The `ConformanceTarget` under test. Must have a `reference` callable; a quantcore-only
            target has nothing to differentially test here.
        *args: Positional arguments passed identically to both callables.
        family: Generator family this case came from, for grouping in the report.
        length_bucket: Declared coverage bucket (`"short"`, `"medium"`, `"long"`) this input's
            length falls into; see `generators.length_bucket_for`.
        seed: Seed reproducing this case's input, if it was generated (see `generators.py`).
        quantities: Name of each value `fn` returns, in order. A scalar- or array-returning
            function passes the default, single `"value"`; a function returning a fixed-length
            tuple (via `TupleComparator`) names each position, e.g. `("statistic", "p_value")`.
        **kwargs: Keyword arguments passed identically to both callables.

    Returns:
        One recorded `CaseResult` per entry in ``quantities``, already appended to this module's
        result log, in the same order as ``quantities``.

    Raises:
        AssertionError: `fn` has no `reference` callable, or that callable raised or returned a
            non-finite value. A broken reference implementation invalidates the whole differential
            comparison, so this always fails the run; it is never recorded as a mere case result.
    """
    if fn.reference is None:
        raise AssertionError(f"{fn.name!r} has no reference callable to check quantcore against")

    quantcore_raised: str | None = None
    quantcore_value: object | None = None
    if fn.quantcore is not None:
        try:
            quantcore_value = fn.quantcore(*args, **kwargs)
        except Exception as exc:
            quantcore_raised = _describe_exception(exc)

    try:
        reference_value = fn.reference(*args, **kwargs)
    except Exception as exc:
        raise AssertionError(
            f"{fn.name!r}: reference callable raised on a generator-produced input "
            f"(family={family!r}, seed={seed!r}): {_describe_exception(exc)}"
        ) from exc
    if not fn.compare.is_finite(reference_value):
        raise AssertionError(
            f"{fn.name!r}: reference callable returned a non-finite result "
            f"(family={family!r}, seed={seed!r}): {reference_value!r}"
        )

    if (
        quantcore_raised is None
        and quantcore_value is not None
        and not fn.compare.is_finite(quantcore_value)
    ):
        quantcore_raised = "non-finite result"

    comparators = _component_comparators(fn.compare, len(quantities))
    expected_components = _split_value(reference_value, len(quantities))
    actual_components = (
        _split_value(quantcore_value, len(quantities)) if quantcore_raised is None else None
    )
    tolerance = _tolerance_for(fn.name)
    case_tolerance = tolerance
    ill_conditioned = False
    if isinstance(tolerance, ConditionAwareTolerance):
        # The condition number is a property of this case's *inputs*, shared by every quantity it
        # records, so it is computed once, from the exact positional/keyword arguments passed to
        # both callables.
        ill_conditioned = tolerance.is_ill_conditioned(*args, **kwargs)
        case_tolerance = dataclasses.replace(tolerance, rtol=tolerance.rtol_eff(*args, **kwargs))

    results: list[CaseResult] = []
    details: list[ReproDetail] = []
    for index, quantity in enumerate(quantities):
        comparator = comparators[index]
        expected = expected_components[index]
        abs_err: float | None = None
        rel_err: float | None = None
        within_tolerance = False
        actual_repr: str | None = None
        if actual_components is not None:
            actual = actual_components[index]
            abs_err, rel_err = comparator.max_errors(actual, expected)
            try:
                comparator.assert_close(actual, expected, case_tolerance)
                within_tolerance = True
            except AssertionError:
                within_tolerance = False
            actual_repr = repr(actual)

        results.append(
            CaseResult(
                function=fn.name,
                family=family,
                length_bucket=length_bucket,
                abs_err=abs_err if abs_err is not None else float("nan"),
                rel_err=rel_err if rel_err is not None else float("nan"),
                within_tolerance=within_tolerance,
                quantcore_raised=quantcore_raised,
                reference_raised=None,
                quantity=quantity,
                ill_conditioned=ill_conditioned,
            )
        )
        details.append(
            ReproDetail(
                function=fn.name,
                family=family,
                quantity=quantity,
                seed=seed,
                abs_err=abs_err,
                rel_err=rel_err,
                expected_abs=_magnitude(comparator, expected),
                quantcore_value=actual_repr,
                reference_value=repr(expected),
                inputs_repr=_repr_inputs(args),
            )
        )

    with _lock:
        _results.extend(results)
        _details.extend(details)
    return results


def _tolerance_for(function_name: str) -> Tolerance:
    return TOLERANCES[function_name]


def check_second_oracle(
    fn: ConformanceTarget[Any, Any],
    *args: Any,
    family: str,
    length_bucket: str,
    oracle_name: str,
    oracle_callable: Any,
    quantity: str,
    seed: int | None = None,
    reference_kwargs: dict[str, Any] | None = None,
    **oracle_kwargs: Any,
) -> CaseResult:
    """Cross-check a second, independent oracle against `fn`'s reference (its standard definition).

    Unlike `check()`, this never touches quantcore and never influences its status: it exists
    purely to record whether a second oracle (e.g. `quantstats`) agrees with the target's own
    reference implementation of the same standard definition. A run never fails because the
    second oracle disagrees, raises, or returns a non-finite value — that is itself the finding
    this records, not a defect in the harness.

    Args:
        fn: The `ConformanceTarget` under test. Must have a `reference` callable: that callable's
            output is what the second oracle is compared against.
        *args: Positional arguments passed to `fn.reference`.
        family: Generator family this case's input came from.
        length_bucket: Declared coverage bucket the input's length falls into.
        oracle_name: Name of the second oracle, e.g. `"quantstats"`; becomes `CaseResult.oracle`.
        oracle_callable: The second oracle's callable, e.g. `quantstats.stats.sharpe`.
        quantity: Label distinguishing this comparison from any other recorded for the same
            function and oracle (e.g. `"rf_zero"` vs. `"rf_nonzero"`), reported as separate rows.
        seed: Seed reproducing this case's input, if generated.
        reference_kwargs: Keyword arguments passed to `fn.reference` only (e.g. `fn`'s own
            per-period risk-free rate); `None` means no keyword arguments.
        **oracle_kwargs: Keyword arguments passed to ``oracle_callable`` only (e.g. the oracle's
            own annualised risk-free rate and periods-per-year, in the form it expects).

    Returns:
        The recorded `CaseResult`, already appended to this module's result log alongside a
        matching `ReproDetail`.

    Raises:
        AssertionError: `fn` has no `reference` callable, or `fn.reference` (the target's own
            standard-definition computation, not the second oracle) raised or returned a
            non-finite value on this input — that fails the run exactly as in `check()`, since it
            invalidates the comparison entirely.
    """
    if fn.reference is None:
        raise AssertionError(f"{fn.name!r} has no reference callable to cross-check against")

    try:
        reference_value = fn.reference(*args, **(reference_kwargs or {}))
    except Exception as exc:
        raise AssertionError(
            f"{fn.name!r}: reference callable raised while building an oracle cross-check "
            f"(family={family!r}, seed={seed!r}): {_describe_exception(exc)}"
        ) from exc
    if not math.isfinite(reference_value):
        raise AssertionError(
            f"{fn.name!r}: reference callable returned a non-finite result while building an "
            f"oracle cross-check (family={family!r}, seed={seed!r}): {reference_value!r}"
        )

    oracle_raised: str | None = None
    oracle_value: float | None = None
    try:
        raw = oracle_callable(*args, **oracle_kwargs)
        oracle_value = float(raw)
        if not math.isfinite(oracle_value):
            oracle_raised = f"non-finite result: {oracle_value!r}"
            oracle_value = None
    except Exception as exc:
        oracle_raised = _describe_exception(exc)

    abs_err: float | None = None
    rel_err: float | None = None
    within_tolerance = False
    if oracle_value is not None:
        abs_err = abs(oracle_value - reference_value)
        rel_err = abs_err / max(abs(reference_value), 1e-300)
        tolerance = _tolerance_for(fn.name)
        within_tolerance = tolerance.is_close(oracle_value, reference_value)

    case = CaseResult(
        function=fn.name,
        family=family,
        length_bucket=length_bucket,
        abs_err=abs_err if abs_err is not None else float("nan"),
        rel_err=rel_err if rel_err is not None else float("nan"),
        within_tolerance=within_tolerance,
        quantcore_raised=oracle_raised,
        reference_raised=None,
        quantity=quantity,
        oracle=oracle_name,
    )
    detail = ReproDetail(
        function=fn.name,
        family=family,
        quantity=quantity,
        seed=seed,
        abs_err=abs_err,
        rel_err=rel_err,
        expected_abs=abs(reference_value),
        quantcore_value=repr(oracle_value) if oracle_value is not None else None,
        reference_value=repr(reference_value),
        inputs_repr=_repr_inputs(args),
        oracle=oracle_name,
    )
    with _lock:
        _results.append(case)
        _details.append(detail)
    return case


def results() -> list[CaseResult]:
    """Return every `CaseResult` recorded so far in this process, in recording order."""
    with _lock:
        return list(_results)


def details() -> list[ReproDetail]:
    """Return every `ReproDetail` recorded so far in this process, in recording order."""
    with _lock:
        return list(_details)


def reset() -> None:
    """Clear all recorded results and details.

    Used by the pytest plugin between sessions and in the plugin's own tests.
    """
    with _lock:
        _results.clear()
        _details.clear()
