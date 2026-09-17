"""Brinson-Fachler single-period performance attribution.

    Given per-segment (e.g. per-sector or per-asset-class) portfolio and
    benchmark weights and returns for a single period, decompose the
    portfolio's active return into allocation, selection, and interaction
    effects:

      r_b_total = sum(w_b * r_b)

      Allocation effect (segment i):
        A_i = (w_p,i - w_b,i) * (r_b,i - r_b_total)

      Selection effect (segment i):
        S_i = w_b,i * (r_p,i - r_b,i)

      Interaction effect (segment i):
        I_i = (w_p,i - w_b,i) * (r_p,i - r_b,i)

    By construction, A_i + S_i + I_i equals the segment's contribution to
    active return, and summing A_i, S_i, I_i across all segments recovers
    the portfolio's total active return:
      sum(w_p * r_p) - sum(w_b * r_b) = sum(A) + sum(S) + sum(I)

    Note the shape convention: all four inputs are 1-D arrays of length n,
    one entry per SEGMENT for a single period — not a T-length time series
    of portfolio-level returns as used elsewhere in this package (e.g.
    `sharpe_ratio`, `information_ratio`). Multi-period attribution requires
    linking single-period results across time and is out of scope here.

References:
    Brinson, G.P., Hood, L.R., and Beebower, G.L. (1986), "Determinants of
    Portfolio Performance," Financial Analysts Journal, 42(4), 39-44.
    Brinson, G.P. and Fachler, N. (1985), "Measuring Non-U.S. Equity
    Portfolio Performance," Journal of Portfolio Management, 11(3), 73-76.
    See docs/REFERENCES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_WEIGHT_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class BrinsonFachlerAttribution:
    """Per-segment and total Brinson-Fachler attribution effects."""

    allocation_effect: npt.NDArray[np.float64]
    selection_effect: npt.NDArray[np.float64]
    interaction_effect: npt.NDArray[np.float64]
    total_allocation_effect: float
    total_selection_effect: float
    total_interaction_effect: float


def _validate_brinson_fachler_attribution(
    portfolio_weights: npt.NDArray[np.float64],
    benchmark_weights: npt.NDArray[np.float64],
    portfolio_returns: npt.NDArray[np.float64],
    benchmark_returns: npt.NDArray[np.float64],
) -> None:
    arrays = {
        "portfolio_weights": portfolio_weights,
        "benchmark_weights": benchmark_weights,
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
    }
    for name, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array")
        if array.size == 0:
            raise ValueError(f"{name} must be non-empty")

    shape = portfolio_weights.shape
    for array in arrays.values():
        if array.shape != shape:
            raise ValueError(
                "portfolio_weights, benchmark_weights, portfolio_returns, and "
                "benchmark_returns must all have the same shape"
            )

    if abs(float(np.sum(portfolio_weights)) - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError("portfolio_weights must sum to 1.0")
    if abs(float(np.sum(benchmark_weights)) - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError("benchmark_weights must sum to 1.0")


def brinson_fachler_attribution(
    portfolio_weights: npt.NDArray[np.float64],
    benchmark_weights: npt.NDArray[np.float64],
    portfolio_returns: npt.NDArray[np.float64],
    benchmark_returns: npt.NDArray[np.float64],
) -> BrinsonFachlerAttribution:
    """Decompose active return into allocation, selection, and interaction effects.

    Args:
        portfolio_weights: Per-segment portfolio weights, shape (n,), summing to 1.0.
        benchmark_weights: Per-segment benchmark weights, shape (n,), summing to 1.0.
        portfolio_returns: Per-segment portfolio returns for the period, shape (n,).
        benchmark_returns: Per-segment benchmark returns for the period, shape (n,).

    Returns:
        `BrinsonFachlerAttribution` with per-segment effect arrays and their totals.
    """
    _validate_brinson_fachler_attribution(
        portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
    )
    return _brinson_fachler_attribution(
        portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
    )


def _brinson_fachler_attribution(
    portfolio_weights: npt.NDArray[np.float64],
    benchmark_weights: npt.NDArray[np.float64],
    portfolio_returns: npt.NDArray[np.float64],
    benchmark_returns: npt.NDArray[np.float64],
) -> BrinsonFachlerAttribution:
    benchmark_total_return = float(np.sum(benchmark_weights * benchmark_returns))
    weight_diff = portfolio_weights - benchmark_weights
    return_diff = portfolio_returns - benchmark_returns

    allocation_effect = weight_diff * (benchmark_returns - benchmark_total_return)
    selection_effect = benchmark_weights * return_diff
    interaction_effect = weight_diff * return_diff

    return BrinsonFachlerAttribution(
        allocation_effect=allocation_effect,
        selection_effect=selection_effect,
        interaction_effect=interaction_effect,
        total_allocation_effect=float(np.sum(allocation_effect)),
        total_selection_effect=float(np.sum(selection_effect)),
        total_interaction_effect=float(np.sum(interaction_effect)),
    )
