"""Tests for how the conformance suite judges a Monte Carlo price against the closed form.

`conformance.targets.pricing_extra` accepts a `(price, std_error)` estimate when
`|price - closed_form| <= atol + k * max(std_error, true_std_error)`, where `true_std_error` is
the standard deviation of the estimator itself and is returned by the oracle next to the
closed-form price. Three things are pinned here:

- the oracle's `true_std_error` is right, checked against numerical quadrature of the payoff
  moments and against the spread of quantcore's own estimates over many seeds;
- the comparator accepts what a correct pricer returns, including the `(0.0, 0.0)` it returns for
  a deep out-of-the-money option when no simulated path finishes in the money;
- the comparator still rejects a price that is more than `k` true standard errors away.
"""

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy import integrate

pytest.importorskip("QuantLib")

import conformance.targets.pricing_extra as pricing_extra
from conformance.targets.pricing import OptionRight
from conformance.tolerances import TOLERANCES

from quantcore.pricing import monte_carlo_call_price, monte_carlo_put_price

type Pricer = Callable[..., tuple[float, float]]
type Reference = Callable[..., tuple[float, float]]

_COMPARATOR = pricing_extra._MonteCarloComparator()
_K = pricing_extra._MONTE_CARLO_K_STD_ERRORS

_PRICERS: dict[OptionRight, Pricer] = {
    "call": monte_carlo_call_price,
    "put": monte_carlo_put_price,
}
_REFERENCES: dict[OptionRight, Reference] = {
    "call": pricing_extra._monte_carlo_call_price_reference,
    "put": pricing_extra._monte_carlo_put_price_reference,
}
_TOLERANCE_NAMES: dict[OptionRight, str] = {
    "call": "monte_carlo_call_price",
    "put": "monte_carlo_put_price",
}

# (s, k, t, r, sigma). Each right is in the money at one point and out of the money at another.
_AT_THE_MONEY = (100.0, 100.0, 1.0, 0.03, 0.2)
_SPOT_ABOVE_STRIKE = (130.0, 100.0, 0.5, 0.02, 0.25)
_SPOT_BELOW_STRIKE = (80.0, 100.0, 1.5, 0.05, 0.3)
_POINTS = (_AT_THE_MONEY, _SPOT_ABOVE_STRIKE, _SPOT_BELOW_STRIKE)

# Deep out of the money: fewer than one path in 20,000 is expected to finish in the money, yet
# the closed-form price is above the comparator's `atol`.
_DEEP_OTM: dict[OptionRight, tuple[float, float, float, float, float]] = {
    "call": (24.0, 100.0, 3.9370300962959544, 0.001, 0.15562650723991206),
    "put": (100.0, 24.0, 4.0, 0.001, 0.16),
}
_NUM_PATHS = 20_000
_NUM_STEPS = 50


def _reference(
    right: OptionRight,
    point: tuple[float, float, float, float, float],
    *,
    num_paths: int = _NUM_PATHS,
    antithetic: bool = True,
) -> tuple[float, float]:
    """The oracle's `(closed_form_price, true_std_error)` for one input."""
    s, k, t, r, sigma = point
    return _REFERENCES[right](
        s,
        k,
        t,
        r=r,
        sigma=sigma,
        num_paths=num_paths,
        num_steps=_NUM_STEPS,
        seed=0,
        antithetic=antithetic,
    )


def _quadrature_std_error(
    point: tuple[float, float, float, float, float],
    *,
    num_paths: int,
    antithetic: bool,
    right: OptionRight,
) -> float:
    """Standard deviation of the price estimator from numerically integrated payoff moments.

    Integrates the payoff, its square, and its product with the mirrored path's payoff against
    the standard normal density, then combines them for `num_paths // 2` mirrored pairs plus one
    unpaired path when `num_paths` is odd (no pairs at all without `antithetic`).
    """
    s, k, t, r, sigma = point
    total_vol = sigma * math.sqrt(t)
    forward = s * math.exp(r * t)

    def payoff(z: float) -> float:
        terminal = forward * math.exp(-0.5 * total_vol * total_vol + total_vol * z)
        return max(terminal - k, 0.0) if right == "call" else max(k - terminal, 0.0)

    def density(z: float) -> float:
        return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    # The payoff has a kink where the terminal price crosses the strike; the mirrored payoff has
    # one at the opposite point.
    kink = (math.log(k / forward) + 0.5 * total_vol * total_vol) / total_vol
    edges = sorted({-40.0, -abs(kink), abs(kink), 40.0})

    def integral(integrand: Callable[[float], float]) -> float:
        return sum(
            integrate.quad(integrand, lo, hi, epsabs=0.0, epsrel=1e-13, limit=400)[0]
            for lo, hi in itertools.pairwise(edges)
        )

    mean = integral(lambda z: payoff(z) * density(z))
    variance = integral(lambda z: payoff(z) ** 2 * density(z)) - mean * mean
    covariance = integral(lambda z: payoff(z) * payoff(-z) * density(z)) - mean * mean
    n_pairs = num_paths // 2 if antithetic else 0
    n_unpaired = num_paths - 2 * n_pairs
    variance_of_sum = 2.0 * n_pairs * (variance + covariance) + n_unpaired * variance
    return math.exp(-r * t) * math.sqrt(variance_of_sum) / num_paths


@pytest.mark.parametrize("right", ["call", "put"])
@pytest.mark.parametrize("antithetic", [True, False])
@pytest.mark.parametrize("num_paths", [1, 2, 3, 20_000, 20_001])
@pytest.mark.parametrize("point", [*_POINTS, *_DEEP_OTM.values()])
def test_true_std_error_matches_quadrature_of_the_payoff_moments(
    right: OptionRight,
    antithetic: bool,
    num_paths: int,
    point: tuple[float, float, float, float, float],
) -> None:
    """The closed-form standard error equals the one built from numerically integrated moments."""
    s, k, t, r, sigma = point
    closed_form = pricing_extra._monte_carlo_true_std_error(
        s, k, t, r=r, sigma=sigma, num_paths=num_paths, antithetic=antithetic, right=right
    )
    quadrature = _quadrature_std_error(
        point, num_paths=num_paths, antithetic=antithetic, right=right
    )
    assert closed_form == pytest.approx(quadrature, rel=1e-8)


@pytest.mark.parametrize("right", ["call", "put"])
@pytest.mark.parametrize("antithetic", [True, False])
@pytest.mark.parametrize(("num_paths", "n_seeds"), [(400, 1500), (3, 6000)])
@pytest.mark.parametrize("point", [_SPOT_ABOVE_STRIKE, _SPOT_BELOW_STRIKE])
def test_true_std_error_matches_the_spread_of_quantcore_estimates(
    right: OptionRight,
    antithetic: bool,
    num_paths: int,
    n_seeds: int,
    point: tuple[float, float, float, float, float],
) -> None:
    """The standard deviation of quantcore's price over many seeds is the true standard error.

    Three paths are one mirrored pair plus one unpaired path, so that count exercises the odd
    case. Where the option is in the money, ignoring the pairing or treating the unpaired path
    as half a pair changes the standard error by 20 percent or more, well outside the 10 percent
    allowed for the sampling noise of a standard deviation taken over this many seeds.
    """
    s, k, t, r, sigma = point
    prices = np.array(
        [
            _PRICERS[right](s, k, r, sigma, t, num_paths, 4, seed, antithetic)[0]
            for seed in range(n_seeds)
        ]
    )
    expected = pricing_extra._monte_carlo_true_std_error(
        s, k, t, r=r, sigma=sigma, num_paths=num_paths, antithetic=antithetic, right=right
    )
    assert float(prices.std(ddof=1)) == pytest.approx(expected, rel=0.10)


@pytest.mark.parametrize("right", ["call", "put"])
def test_true_std_error_is_zero_for_a_deterministic_payoff(right: OptionRight) -> None:
    """With zero volatility or zero time to expiry every path pays the same, so nothing varies."""
    for sigma, t in ((0.0, 1.0), (0.2, 0.0)):
        assert (
            pricing_extra._monte_carlo_true_std_error(
                100.0, 90.0, t, r=0.03, sigma=sigma, num_paths=1000, antithetic=True, right=right
            )
            == 0.0
        )


@pytest.mark.parametrize("right", ["call", "put"])
def test_reference_returns_the_closed_form_price_and_the_true_std_error(
    right: OptionRight,
) -> None:
    """The oracle pairs the Black-Scholes-Merton price with the estimator's standard error."""
    s, k, t, r, sigma = _AT_THE_MONEY
    price, std_error = _reference(right, _AT_THE_MONEY)
    assert price == pricing_extra._bsm_price_closed_form(s, k, t, r, 0.0, sigma, right)
    assert std_error == pricing_extra._monte_carlo_true_std_error(
        s, k, t, r=r, sigma=sigma, num_paths=_NUM_PATHS, antithetic=True, right=right
    )
    assert std_error > 0.0


@pytest.mark.parametrize("right", ["call", "put"])
@pytest.mark.parametrize("point", _POINTS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_comparator_accepts_quantcore_estimates(
    right: OptionRight, point: tuple[float, float, float, float, float], seed: int
) -> None:
    """An unbiased estimate lands inside the band."""
    s, k, t, r, sigma = point
    actual = _PRICERS[right](s, k, r, sigma, t, _NUM_PATHS, _NUM_STEPS, seed, True)
    _COMPARATOR.assert_close(actual, _reference(right, point), TOLERANCES[_TOLERANCE_NAMES[right]])


@pytest.mark.parametrize("right", ["call", "put"])
def test_comparator_accepts_zero_price_and_zero_std_error_deep_out_of_the_money(
    right: OptionRight,
) -> None:
    """`(0.0, 0.0)` is accepted when the closed form is within `k` true standard errors of zero.

    The closed-form price is above `atol`, so the reported standard error of `0.0` and `atol`
    together would not cover the gap: the true standard error does.
    """
    tol = TOLERANCES[_TOLERANCE_NAMES[right]]
    closed_form, true_std_error = _reference(right, _DEEP_OTM[right])
    assert tol.atol < closed_form < _K * true_std_error
    _COMPARATOR.assert_close((0.0, 0.0), (closed_form, true_std_error), tol)


@pytest.mark.parametrize("right", ["call", "put"])
def test_quantcore_returns_zero_price_and_zero_std_error_deep_out_of_the_money(
    right: OptionRight,
) -> None:
    """With no simulated path in the money quantcore returns `(0.0, 0.0)`, which is accepted."""
    s, k, t, r, sigma = _DEEP_OTM[right]
    actual = _PRICERS[right](s, k, r, sigma, t, _NUM_PATHS, _NUM_STEPS, 16751827, True)
    assert actual == (0.0, 0.0)
    _COMPARATOR.assert_close(
        actual, _reference(right, _DEEP_OTM[right]), TOLERANCES[_TOLERANCE_NAMES[right]]
    )


@pytest.mark.parametrize("right", ["call", "put"])
@pytest.mark.parametrize("point", [*_POINTS, *_DEEP_OTM.values()])
@pytest.mark.parametrize("reported_fraction", [0.0, 1.0])
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_comparator_rejects_a_price_six_true_std_errors_away(
    right: OptionRight,
    point: tuple[float, float, float, float, float],
    reported_fraction: float,
    direction: float,
) -> None:
    """A price 6 true standard errors off is rejected, whether the reported one is right or 0."""
    tol = TOLERANCES[_TOLERANCE_NAMES[right]]
    closed_form, true_std_error = _reference(right, point)
    actual = (closed_form + direction * 6.0 * true_std_error, reported_fraction * true_std_error)
    with pytest.raises(AssertionError):
        _COMPARATOR.assert_close(actual, (closed_form, true_std_error), tol)


@pytest.mark.parametrize("right", ["call", "put"])
@pytest.mark.parametrize(
    ("reported_fraction", "band_in_true_std_errors"), [(0.0, 4.0), (0.5, 4.0), (2.0, 8.0)]
)
def test_comparator_band_is_k_times_the_larger_std_error(
    right: OptionRight, reported_fraction: float, band_in_true_std_errors: float
) -> None:
    """The band is `k = 4` times whichever of the reported and true standard errors is larger."""
    tol = TOLERANCES[_TOLERANCE_NAMES[right]]
    closed_form, true_std_error = _reference(right, _AT_THE_MONEY)
    expected = (closed_form, true_std_error)
    reported = reported_fraction * true_std_error
    assert tol.atol < 0.01 * true_std_error

    inside = closed_form + (band_in_true_std_errors - 0.1) * true_std_error
    outside = closed_form + (band_in_true_std_errors + 0.1) * true_std_error
    _COMPARATOR.assert_close((inside, reported), expected, tol)
    with pytest.raises(AssertionError):
        _COMPARATOR.assert_close((outside, reported), expected, tol)


@pytest.mark.parametrize("right", ["call", "put"])
def test_comparator_rejects_a_three_percent_bias_in_quantcore_estimates(
    right: OptionRight,
) -> None:
    """With 200,000 paths a 3 percent bias is more than 10 standard errors and is rejected."""
    s, k, t, r, sigma = _AT_THE_MONEY
    num_paths = 200_000
    tol = TOLERANCES[_TOLERANCE_NAMES[right]]
    expected = _reference(right, _AT_THE_MONEY, num_paths=num_paths)
    price, std_error = _PRICERS[right](s, k, r, sigma, t, num_paths, 1, 7, True)

    _COMPARATOR.assert_close((price, std_error), expected, tol)
    with pytest.raises(AssertionError):
        _COMPARATOR.assert_close((1.03 * price, std_error), expected, tol)
    with pytest.raises(AssertionError):
        _COMPARATOR.assert_close((0.97 * price, std_error), expected, tol)


def test_comparator_reports_errors_of_the_price_only() -> None:
    """`max_errors` measures the price; the two standard errors do not enter it."""
    abs_error, rel_error = _COMPARATOR.max_errors((10.5, 0.2), (10.0, 0.05))
    assert abs_error == pytest.approx(0.5)
    assert rel_error == pytest.approx(0.05)


def test_comparator_requires_both_components_finite() -> None:
    """A non-finite price or standard error makes the pair non-finite."""
    assert _COMPARATOR.is_finite((1.0, 0.0))
    assert not _COMPARATOR.is_finite((math.nan, 0.1))
    assert not _COMPARATOR.is_finite((1.0, math.inf))
