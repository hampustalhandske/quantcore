"""Tests for `conformance.tolerances.ConditionAwareTolerance`.

A condition-aware tolerance widens `rtol` by `K * eps * kappa ** p` for a caller-supplied
condition number `kappa`, so that a well-conditioned case still uses a tight `rtol` while an
ill-conditioned one is not flagged as a mismatch purely from floating-point error.
"""

import numpy as np
import pytest
from conformance.compare import Tolerance
from conformance.tolerances import (
    ILL_CONDITIONED_RTOL_CAP,
    TOLERANCES,
    ConditionAwareTolerance,
    _calmar_kappa,
    _sharpe_kappa,
    _ulcer_kappa,
)


def _tol(*, rtol=1e-10, kappa=None, big_k=1.0, p=2.0) -> ConditionAwareTolerance:
    return ConditionAwareTolerance(
        rtol=rtol, atol=1e-9, justification="test", kappa=kappa, K=big_k, p=p
    )


def test_rtol_eff_equals_plain_rtol_without_kappa() -> None:
    """A `ConditionAwareTolerance` with no `kappa` behaves exactly like a plain `Tolerance`."""
    tol = _tol(kappa=None)
    assert tol.rtol_eff(1.0, 2.0) == tol.rtol
    assert tol.is_ill_conditioned(1.0, 2.0) is False


def test_rtol_eff_widens_with_kappa() -> None:
    """`rtol_eff` grows with `K * eps * kappa ** p`, and is at least the plain `rtol` floor."""
    tol = _tol(rtol=1e-10, kappa=lambda x: 10.0, big_k=1.0, p=2.0)
    eps = float(np.finfo(np.float64).eps)
    assert tol.rtol_eff(0) == max(1e-10, eps * 100.0)

    tiny_kappa_tol = _tol(rtol=1e-10, kappa=lambda x: 1e-6, big_k=1.0, p=2.0)
    assert tiny_kappa_tol.rtol_eff(0) == 1e-10  # the plain rtol floor still applies


def test_is_ill_conditioned_crosses_the_cap() -> None:
    """A large enough kappa pushes `rtol_eff` past the cap; a small one stays under it."""
    well_conditioned = _tol(kappa=lambda x: 2.0, big_k=1.0, p=2.0)
    assert well_conditioned.is_ill_conditioned(0) is False

    ill_conditioned = _tol(kappa=lambda x: 1e6, big_k=1.0, p=2.0)
    assert ill_conditioned.rtol_eff(0) > ILL_CONDITIONED_RTOL_CAP
    assert ill_conditioned.is_ill_conditioned(0) is True


def test_is_close_still_uses_the_declared_rtol_not_rtol_eff() -> None:
    """`is_close` (from `Tolerance`) ignores `kappa`; callers apply `rtol_eff` themselves."""
    tol = _tol(rtol=1e-10, kappa=lambda x: 1e6, big_k=1.0, p=2.0)
    assert isinstance(tol, Tolerance)
    assert tol.is_close(1.0 + 1e-3, 1.0) is False


def test_sharpe_ratio_tolerance_is_condition_aware() -> None:
    """`TOLERANCES["sharpe_ratio"]` is a `ConditionAwareTolerance` with `kappa` and `p` set."""
    tol = TOLERANCES["sharpe_ratio"]
    assert isinstance(tol, ConditionAwareTolerance)
    assert tol.kappa is not None
    assert tol.p == 2.0


def test_sharpe_kappa_is_large_on_a_near_constant_series() -> None:
    """A near-constant series (tiny std, large level) has a large condition number."""
    near_constant = np.full(20, 0.0004) + np.random.default_rng(0).normal(0.0, 1e-9, 20)
    kappa = _sharpe_kappa(near_constant, 0.0, 252)
    assert kappa > 1e4


def test_sharpe_kappa_is_small_on_an_ordinary_series() -> None:
    """An ordinary series (std comparable to the level) has a modest condition number."""
    ordinary = np.random.default_rng(0).normal(0.0005, 0.01, 500)
    kappa = _sharpe_kappa(ordinary, 0.0, 252)
    assert kappa < 20.0


def test_sharpe_kappa_near_constant_series_is_ill_conditioned_under_the_full_tolerance() -> None:
    """The actual `sharpe_ratio` tolerance marks a severely near-constant series ill-conditioned.

    A near-constant draw's realised `std` varies widely (it is itself sampled from a tiny-scale
    noise process), from "merely small" to a near-total ddof=1 cancellation; this uses a
    deliberately severe case (noise at machine-epsilon scale) to exercise the genuinely
    pathological end of a near-constant short series, rather than depending on one arbitrary
    seed's realised severity.
    """
    tol = TOLERANCES["sharpe_ratio"]
    assert isinstance(tol, ConditionAwareTolerance)
    near_constant = np.full(20, 0.0004) + np.random.default_rng(0).normal(0.0, 1e-15, 20)
    assert tol.is_ill_conditioned(near_constant, 0.0, 252) is True


def test_sharpe_kappa_ordinary_series_is_well_conditioned_under_the_full_tolerance() -> None:
    tol = TOLERANCES["sharpe_ratio"]
    assert isinstance(tol, ConditionAwareTolerance)
    ordinary = np.random.default_rng(0).normal(0.0005, 0.01, 500)
    assert tol.is_ill_conditioned(ordinary, 0.0, 252) is False


def test_sharpe_kappa_handles_zero_std() -> None:
    """A perfectly constant series has zero std; kappa is infinity, never a ZeroDivisionError."""
    constant = np.full(10, 0.0004)
    assert _sharpe_kappa(constant, 0.0, 252) == float("inf")


# Returns [0.2, -0.5, 1.5, -0.25, 0.1, 0.3] compound to wealth [1.2, 0.6, 1.5, 1.125, 1.2375,
# 1.60875] with running peak [1.2, 1.2, 1.5, 1.5, 1.5, 1.60875], so the drawdowns are
# [0, 0.5, 0, 0.25, 0.175, 0].
_DRAWDOWN_RETURNS = np.array([0.2, -0.5, 1.5, -0.25, 0.1, 0.3])
_NO_DRAWDOWN_RETURNS = np.array([0.1, 0.2])


def test_calmar_kappa_is_the_reciprocal_of_the_maximum_drawdown() -> None:
    """The maximum drawdown of the hand-worked path is 0.5, so kappa is 1 / 0.5 = 2."""
    assert _calmar_kappa(_DRAWDOWN_RETURNS, 252) == pytest.approx(2.0, rel=1e-12)


def test_calmar_kappa_ignores_periods_per_year() -> None:
    """`periods_per_year` scales only the numerator of the ratio and does not enter kappa."""
    assert _calmar_kappa(_DRAWDOWN_RETURNS, 12) == _calmar_kappa(_DRAWDOWN_RETURNS, 252)


def test_calmar_kappa_is_infinite_without_a_drawdown() -> None:
    """A path that never declines has a maximum drawdown of 0 and an infinite kappa."""
    assert _calmar_kappa(_NO_DRAWDOWN_RETURNS, 252) == float("inf")


def test_ulcer_kappa_is_the_reciprocal_of_the_ulcer_index() -> None:
    """Ulcer Index is sqrt((0.5**2 + 0.25**2 + 0.175**2) / (6 - 1)) = sqrt(0.068625)."""
    ulcer_index = (0.25 + 0.0625 + 0.030625) ** 0.5 / 5**0.5
    assert ulcer_index == pytest.approx(0.261963, rel=1e-5)
    assert _ulcer_kappa(_DRAWDOWN_RETURNS) == pytest.approx(1.0 / ulcer_index, rel=1e-12)


def test_ulcer_kappa_is_infinite_without_a_drawdown() -> None:
    """A path that never declines has an Ulcer Index of 0 and an infinite kappa."""
    assert _ulcer_kappa(_NO_DRAWDOWN_RETURNS) == float("inf")
