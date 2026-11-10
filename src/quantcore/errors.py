"""Exceptions shared across quantcore.

`UndefinedResultError` is what a public function that returns a single number raises when that
number is undefined or infinite for valid input: a ratio whose denominator is zero, a statistic
of a constant series, a result that overflows float64. Such a function never returns NaN or
infinity. Invalid input (a wrong shape, a non-finite value, an unknown option) raises a plain
`ValueError` instead. See `docs/UNDEFINED_RESULTS.md` for the cases of every function.
"""

from __future__ import annotations

import math


class UndefinedResultError(ValueError):
    """The requested number is undefined or infinite for this (valid) input.

    A subclass of `ValueError`, so code that catches `ValueError` also catches it. The message
    names the condition, e.g. "returns have zero range; the Sharpe ratio is undefined".
    """


def _finite_result(value: float, quantity: str) -> float:
    """Return `value` unchanged when it is finite.

    Args:
        value: The computed result.
        quantity: Name of the quantity, used in the error message.

    Raises:
        UndefinedResultError: `value` is NaN or infinite.
    """
    if not math.isfinite(value):
        raise UndefinedResultError(f"{quantity} is not finite ({value!r}) for this input")
    return value
