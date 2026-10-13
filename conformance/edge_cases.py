"""Recorder for what a raw function does when called with a contract-violating edge-case input.

`observe()` calls a function directly on an input outside its contract (all-NaN, empty, a single
observation, zero variance, ...) and logs whether it returned a value or raised. The outcome is
merely logged here, never asserted on, so a raw quantcore misbehaviour on a contract-violating
input never fails the conformance run. It still matters: it is exactly the kind of fact a
findings write-up needs ("quantcore silently returns 0.0 on all-NaN input" is a citable defect).
"""

import dataclasses
import threading
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass(frozen=True)
class EdgeCaseObservation:
    """What a raw function did when called directly with a contract-violating input.

    Attributes:
        function: Name of the function under test.
        provider: Which implementation was called directly, e.g. `"quantcore"`.
        case: Short label for the edge case, e.g. `"all-NaN"`, `"empty"`, `"zero variance"`.
        outcome: `"returned {value!r}"` or `"raised {ExceptionType}: {message}"`.
    """

    function: str
    provider: str
    case: str
    outcome: str


_observations: list[EdgeCaseObservation] = []
_lock = threading.Lock()


def observe(
    function: str,
    provider: str,
    case: str,
    raw_callable: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> EdgeCaseObservation:
    """Call ``raw_callable`` and record what it did, without raising on the caller's behalf."""
    try:
        result = raw_callable(*args, **kwargs)
        outcome = f"returned {result!r}"
    except Exception as exc:
        outcome = f"raised {type(exc).__name__}: {exc}"
    observation = EdgeCaseObservation(
        function=function, provider=provider, case=case, outcome=outcome
    )
    with _lock:
        _observations.append(observation)
    return observation


def observations() -> list[EdgeCaseObservation]:
    """Return every observation recorded so far in this process."""
    with _lock:
        return list(_observations)


def reset() -> None:
    """Clear all recorded observations."""
    with _lock:
        _observations.clear()
