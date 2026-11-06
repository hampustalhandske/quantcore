"""`ConformanceTarget` and the comparators that judge a quantcore result against an oracle's.

Each verified quantity is declared once as a `ConformanceTarget`: a name, a tier, the standard it
is measured against, a `Comparator`, and two callables taking identical arguments -- `quantcore`
(an adapter around the quantcore function under test) and `reference` (an independent oracle for
the same quantity). Constructing a target registers it, so `registry()` lists everything the suite
covers. Nothing here runs a comparison: `conformance.harness.check()` calls both callables on one
input and uses the target's `Comparator` to record how far apart their results are.
"""

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

__all__ = [
    "ArrayComparator",
    "Comparator",
    "ConformanceTarget",
    "DecisiveFloatArrayComparator",
    "ReportOnlyArrayComparator",
    "ReportOnlyScalarComparator",
    "ScalarComparator",
    "Tolerance",
    "TupleComparator",
    "registry",
]

_TINY = 1e-300


@dataclass(frozen=True)
class Tolerance:
    """Closeness threshold for judging a quantcore result equal to its oracle's.

    Two values are close when ``|actual - expected| <= atol + rtol * |expected|``.

    Attributes:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        justification: Why this tolerance is correct for the function it is attached to. Must
            explain any tolerance looser than ``1e-10`` relative (the closed-form default) in
            terms of a concrete numerical reason, e.g. summation order over many terms.
    """

    rtol: float
    atol: float
    justification: str

    def is_close(self, actual: float, expected: float) -> bool:
        """Return whether ``actual`` is within this tolerance of ``expected``."""
        return abs(actual - expected) <= self.atol + self.rtol * abs(expected)


def _rel_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), _TINY)


def _require_same_shape(actual: npt.NDArray[np.float64], expected: npt.NDArray[np.float64]) -> None:
    if actual.shape != expected.shape:
        raise ValueError(
            f"shape mismatch: actual.shape={actual.shape} expected.shape={expected.shape}"
        )


class Comparator[R](Protocol):
    """How a quantcore result and its oracle's are judged finite, close, and how far apart.

    Implemented once per result shape (`ScalarComparator` for `float`, `ArrayComparator` for a
    `FloatArray`, `TupleComparator` for a tuple of sub-results), so `conformance.harness.check()`
    never needs to know the concrete return type of the target it runs.
    """

    def is_finite(self, value: R) -> bool:
        """Return whether every number in ``value`` is finite (no NaN, no ±inf)."""
        ...

    def max_errors(self, actual: R, expected: R) -> tuple[float, float]:
        """Return ``(max_abs_error, max_rel_error)`` of ``actual`` against ``expected``.

        Relative error at each scalar position uses ``|a - e| / max(|e|, 1e-300)``.
        """
        ...

    def assert_close(self, actual: R, expected: R, tol: Tolerance) -> None:
        """Raise `AssertionError` (naming both values and ``tol``) unless ``actual`` is close."""
        ...


@dataclass(frozen=True)
class ScalarComparator:
    """`Comparator` for a function that returns a single `float`."""

    def is_finite(self, value: float) -> bool:
        """Return whether ``value`` is finite."""
        return math.isfinite(value)

    def max_errors(self, actual: float, expected: float) -> tuple[float, float]:
        """Return the absolute and relative error of ``actual`` against ``expected``."""
        return abs(actual - expected), _rel_error(actual, expected)

    def assert_close(self, actual: float, expected: float, tol: Tolerance) -> None:
        """Raise `AssertionError` unless ``actual`` is within ``tol`` of ``expected``."""
        if not tol.is_close(actual, expected):
            raise AssertionError(f"actual={actual!r} expected={expected!r} tol={tol!r}")


@dataclass(frozen=True)
class ArrayComparator:
    """`Comparator` for a function that returns a `FloatArray`."""

    def is_finite(self, value: npt.NDArray[np.float64]) -> bool:
        """Return whether every element of ``value`` is finite."""
        return bool(np.all(np.isfinite(value)))

    def max_errors(
        self, actual: npt.NDArray[np.float64], expected: npt.NDArray[np.float64]
    ) -> tuple[float, float]:
        """Return the largest absolute and relative element-wise error.

        Raises:
            ValueError: ``actual`` and ``expected`` have different shapes.
        """
        _require_same_shape(actual, expected)
        abs_errors = np.abs(actual - expected)
        rel_errors = abs_errors / np.maximum(np.abs(expected), _TINY)
        return float(np.max(abs_errors)), float(np.max(rel_errors))

    def assert_close(
        self,
        actual: npt.NDArray[np.float64],
        expected: npt.NDArray[np.float64],
        tol: Tolerance,
    ) -> None:
        """Raise `AssertionError` unless every element is within ``tol``.

        Raises:
            ValueError: ``actual`` and ``expected`` have different shapes.
        """
        _require_same_shape(actual, expected)
        if not np.allclose(actual, expected, rtol=tol.rtol, atol=tol.atol):
            max_abs, max_rel = self.max_errors(actual, expected)
            raise AssertionError(
                f"actual={actual!r} expected={expected!r} tol={tol!r} "
                f"max_abs_error={max_abs!r} max_rel_error={max_rel!r}"
            )


@dataclass(frozen=True)
class ReportOnlyArrayComparator:
    """`Comparator` for a `FloatArray` quantity that is recorded and reported but never judged.

    `assert_close` never raises, so this quantity can never by itself flip a case's
    `within_tolerance` to `False`; `is_finite`/`max_errors` still behave exactly like
    `ArrayComparator` (a genuinely non-finite result is still detected, and the
    magnitude of any difference is still informative in a report). Used for a secondary quantity
    that is reported alongside a decision quantity but is not itself well-posed to compare, e.g. an
    iterative estimator's raw parameter vector, whose optimum is not unique (see
    `DecisiveFloatArrayComparator`).
    """

    def is_finite(self, value: npt.NDArray[np.float64]) -> bool:
        """Return whether every element of ``value`` is finite."""
        return ArrayComparator().is_finite(value)

    def max_errors(
        self, actual: npt.NDArray[np.float64], expected: npt.NDArray[np.float64]
    ) -> tuple[float, float]:
        """Return the largest absolute and relative element-wise error, for reporting only."""
        return ArrayComparator().max_errors(actual, expected)

    def assert_close(
        self,
        actual: npt.NDArray[np.float64],
        expected: npt.NDArray[np.float64],
        tol: Tolerance,
    ) -> None:
        """No-op: this quantity is never judged close or far, by design."""
        return None


@dataclass(frozen=True)
class ReportOnlyScalarComparator:
    """`Comparator` for a `float` quantity that is recorded and reported but never judged.

    `assert_close` never raises, so this quantity can never by itself flip a case's
    `within_tolerance` to `False`; `is_finite`/`max_errors` still behave exactly like
    `ScalarComparator` (a genuinely non-finite result is still detected, and the
    magnitude of any difference is still informative in a report). Used for a scalar quantity
    whose "correct" value is inherently ambiguous (e.g. an information-criterion-optimal model
    order, where a near-tie between two orders is common and not a defect) -- mirrors
    `ReportOnlyArrayComparator` for a scalar rather than an array.

    A `ConformanceTarget` whose whole `compare` carries `report_only_top_level = True` is to be
    reported as `unverifiable` rather than run through `conformance.classify.classify()`, since
    every one of its cases reads `within_tolerance=True` regardless of how far apart quantcore and
    the oracle actually are.
    """

    report_only_top_level: bool = True
    """Marks this comparator as never producing a real pass/fail signal when used as a function's
    *entire* `compare` (as opposed to only the secondary half of a `DecisiveFloatArrayComparator`
    split, where the decision quantity alone still decides conformance status)."""

    def is_finite(self, value: float) -> bool:
        """Return whether ``value`` is finite."""
        return ScalarComparator().is_finite(value)

    def max_errors(self, actual: float, expected: float) -> tuple[float, float]:
        """Return abs/rel error of ``actual`` against ``expected``, for reporting only."""
        return ScalarComparator().max_errors(actual, expected)

    def assert_close(self, actual: float, expected: float, tol: Tolerance) -> None:
        """No-op: this quantity is never judged close or far, by design."""
        return None


@dataclass(frozen=True)
class DecisiveFloatArrayComparator:
    """`Comparator` for `(decision, secondary)`: a scalar decision quantity plus a reported array.

    A two-quantity `ConformanceTarget` (e.g. an iterative estimator's attained log-likelihood and
    its fitted parameter vector) registers this as its `compare`. Called on the whole
    `(decision, secondary)` pair, `max_errors` and `assert_close` key off the decision component
    alone, so an out-of-tolerance secondary value never fails the comparison; `is_finite` requires
    both components to be finite. `conformance.harness.check()` splits this comparator into its
    two per-quantity halves (`ScalarComparator` for the decision, `ReportOnlyArrayComparator` for
    the secondary) so each is recorded as its own named quantity.
    """

    def is_finite(self, value: tuple[float, npt.NDArray[np.float64]]) -> bool:
        """Return whether both the decision scalar and every element of the secondary are finite."""
        decision, secondary = value
        return ScalarComparator().is_finite(decision) and ArrayComparator().is_finite(secondary)

    def max_errors(
        self,
        actual: tuple[float, npt.NDArray[np.float64]],
        expected: tuple[float, npt.NDArray[np.float64]],
    ) -> tuple[float, float]:
        """Return the decision component's absolute and relative error; the secondary is ignored."""
        return ScalarComparator().max_errors(actual[0], expected[0])

    def assert_close(
        self,
        actual: tuple[float, npt.NDArray[np.float64]],
        expected: tuple[float, npt.NDArray[np.float64]],
        tol: Tolerance,
    ) -> None:
        """Raise `AssertionError` unless the decision component alone is within ``tol``."""
        ScalarComparator().assert_close(actual[0], expected[0], tol)


@dataclass(frozen=True)
class TupleComparator:
    """`Comparator` for a function returning a fixed-length tuple of scalar sub-results.

    Every recorded quantity in the tuple (e.g. a test statistic and its p-value) is judged by its
    own `Comparator[float]`, and the tuple as a whole is finite/close only if every component is.

    Attributes:
        comparators: One `Comparator[float]` per tuple position, in order.
    """

    comparators: tuple[Comparator[float], ...]

    def __init__(self, *comparators: Comparator[float]) -> None:
        """Build a `TupleComparator` from one comparator per tuple position, given positionally."""
        object.__setattr__(self, "comparators", comparators)

    def is_finite(self, value: tuple[float, ...]) -> bool:
        """Return whether every component of ``value`` is finite under its own comparator."""
        return all(c.is_finite(v) for c, v in zip(self.comparators, value, strict=True))

    def max_errors(
        self, actual: tuple[float, ...], expected: tuple[float, ...]
    ) -> tuple[float, float]:
        """Return the component-wise worst absolute and relative error, each taken independently."""
        abs_errors = []
        rel_errors = []
        for comparator, a, e in zip(self.comparators, actual, expected, strict=True):
            abs_err, rel_err = comparator.max_errors(a, e)
            abs_errors.append(abs_err)
            rel_errors.append(rel_err)
        return max(abs_errors), max(rel_errors)

    def assert_close(
        self, actual: tuple[float, ...], expected: tuple[float, ...], tol: Tolerance
    ) -> None:
        """Raise `AssertionError` unless every component is within ``tol`` under its comparator."""
        for comparator, a, e in zip(self.comparators, actual, expected, strict=True):
            comparator.assert_close(a, e, tol)


@dataclass(frozen=True)
class ConformanceTarget[**P, R]:
    """One quantity the suite verifies: a quantcore callable paired with an independent oracle.

    Constructing a `ConformanceTarget` registers it in the module-level registry returned by
    `registry()`; a duplicate `name` raises `ValueError`. At least one of `quantcore` and
    `reference` must be given. Both callables take identical arguments, so one generated input can
    be passed to each and the two results judged with `compare`.

    Attributes:
        name: Unique, snake_case identifier; the key used in the registry, in
            `conformance.tolerances.TOLERANCES` and in every recorded result.
        tier: `1` for a function a downstream pipeline is known to call, `2` for the rest of
            quantcore's surface.
        oracle: Independent library or libraries quantcore's output is checked against, e.g.
            `"empyrical-reloaded; quantstats"`.
        citation: Primary source for the standard definition this quantity implements.
        compare: How quantcore's result and the oracle's result for the same call are judged equal.
        quantcore: Adapter calling the quantcore function under test, or `None` if quantcore has
            no such function.
        reference: The oracle, an independent implementation of the standard definition, or `None`
            if there is nothing independent to check quantcore against.
    """

    name: str
    tier: Literal[1, 2]
    oracle: str
    citation: str
    compare: Comparator[R]
    quantcore: Callable[P, R] | None = None
    reference: Callable[P, R] | None = None

    def __post_init__(self) -> None:
        """Validate that at least one callable was given, then register this target by name."""
        if self.quantcore is None and self.reference is None:
            raise ValueError(f"{self.name!r}: at least one of quantcore, reference must be given")
        _register(self)


_registry: dict[str, ConformanceTarget[..., Any]] = {}
_lock = threading.Lock()


def _register(target: ConformanceTarget[..., Any]) -> None:
    """Add ``target`` to the registry, keyed by its name.

    Raises:
        ValueError: A target with this name is already registered.
    """
    with _lock:
        if target.name in _registry:
            raise ValueError(f"a conformance target named {target.name!r} is already registered")
        _registry[target.name] = target


def registry() -> Mapping[str, ConformanceTarget[..., Any]]:
    """Return a read-only snapshot of every registered `ConformanceTarget`, keyed by name."""
    with _lock:
        return MappingProxyType(dict(_registry))
