"""Oracle tests for `arima_fit` vs `statsmodels.tsa.arima.model.ARIMA`.

Covers Workstream D: "ARIMA, VAR, IRF, Granger vs statsmodels."

`arima_fit` maximizes the exact Gaussian likelihood by default, the same
objective `statsmodels`' `ARIMA.fit` maximizes with a Kalman filter. Two
local optimizers stop at slightly different points near the same maximum,
so the decisive comparison is on the attained objective, not on the
parameters: `statsmodels`' own `loglike` is evaluated at quantcore's
solution (never re-fit) and compared with the log-likelihood `statsmodels`
reports for its own fit. Parameters are compared too, at a looser bound.

`arima_fit(..., method="css")` is a different estimator (conditional sum of
squares, Box, Jenkins & Reinsel 2015) that agrees with maximum likelihood
only asymptotically, and `statsmodels`' `ARIMA` offers no CSS fit to
compare against directly. What is checked for it is the large-sample
agreement: on a long, well-behaved series the CSS estimate lands close to
`statsmodels`' maximum-likelihood estimate.

`arima_forecast` is compared at fixed coefficients (no fit on either side)
with the forecast of `statsmodels`' state-space ARIMA model filtered at
those coefficients: both are the Gaussian conditional mean of the future
values given the whole sample.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")

from scipy.signal import lfilter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from quantcore.time_series.arima import arima_fit, arima_forecast

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921

# Zero-mean ARMA processes, each fit at its own true (p, q) with d = 0:
# name -> (AR coefficients, MA coefficients, order fit). White noise is fit
# as an AR(1) whose true coefficient is 0.
_FAMILIES: dict[str, tuple[tuple[float, ...], tuple[float, ...], tuple[int, int, int]]] = {
    "white_noise": ((), (), (1, 0, 0)),
    "ar1": ((0.6,), (), (1, 0, 0)),
    "ar2": ((0.5, -0.3), (), (2, 0, 0)),
    "ma1": ((), (0.5,), (0, 0, 1)),
    "ma2": ((), (0.4, -0.2), (0, 0, 2)),
    "arma11": ((0.5,), (0.3,), (1, 0, 1)),
    "near_unit_root": ((0.97,), (), (1, 0, 0)),
}
_LENGTHS = (120, 400, 800)
_BURN_IN = 200


def _simulate_arma(family: str, n: int, seed: int) -> np.ndarray:
    ar, ma, _ = _FAMILIES[family]
    innovations = np.random.default_rng(seed).normal(size=n + _BURN_IN)
    series = lfilter(np.r_[1.0, ma], np.r_[1.0, -np.asarray(ar)], innovations)
    return np.asarray(series[_BURN_IN:], dtype=np.float64)


class TestArimaFitAttainsStatsmodelsLikelihood:
    @pytest.mark.parametrize("n", _LENGTHS)
    @pytest.mark.parametrize("family", list(_FAMILIES))
    def test_exact_loglike_at_default_fit_matches_statsmodels_mle(
        self, family: str, n: int
    ) -> None:
        order = _FAMILIES[family][2]
        seed = RNG_SEED + 1000 * list(_FAMILIES).index(family) + n
        series = _simulate_arma(family, n, seed)

        ar_coefs, ma_coefs, sigma2 = arima_fit(series, *order)

        model = ARIMA(series, order=order, trend="n", concentrate_scale=False)
        quantcore_loglike = float(model.loglike(np.concatenate([ar_coefs, ma_coefs, [sigma2]])))
        sm_result = model.fit()
        sm_loglike = float(sm_result.llf)

        # Lower side, the optimizer-based policy bound of 1e-6 on the
        # objective: quantcore's solution is never a worse fit than
        # statsmodels' own optimum.
        assert quantcore_loglike >= sm_loglike - 1e-6
        # Upper side, 1e-3 nats: statsmodels' own optimizer can stop short
        # of the maximum (it ends some ARMA fits on its iteration limit,
        # with a ConvergenceWarning), by up to several 1e-4 nats on series
        # of this kind, so quantcore's solution may sit that far above it.
        # A larger excess would mean the two fits found different maxima.
        assert quantcore_loglike <= sm_loglike + 1e-3

        # Parameters at 1e-3: both sides maximize the same likelihood, so
        # they differ only by where each optimizer stops, which for
        # statsmodels is up to a few 1e-4 from the maximizer on series of
        # this kind. 1e-3 is still under 3% of the estimates' sampling
        # standard error, which is of order 1/sqrt(n).
        quantcore_params = np.concatenate([ar_coefs, ma_coefs, [sigma2]])
        np.testing.assert_allclose(quantcore_params, sm_result.params, atol=1e-3, rtol=0.0)


def _simulate_ar1(phi: float, n: int, innovation_scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    series = np.empty(n)
    series[0] = rng.normal()
    for t in range(1, n):
        series[t] = phi * series[t - 1] + rng.normal(scale=innovation_scale)
    return series


def _simulate_arma11(phi: float, theta: float, n: int, seed: int) -> np.ndarray:
    eps = np.random.default_rng(seed).normal(scale=0.2, size=n)
    series = np.empty(n)
    series[0] = eps[0]
    for t in range(1, n):
        series[t] = phi * series[t - 1] + eps[t] + theta * eps[t - 1]
    return series


def _simulate_integrated_ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    diffs[0] = 0.0
    for t in range(1, n):
        diffs[t] = phi * diffs[t - 1] + rng.normal(scale=0.2)
    return np.cumsum(diffs) + 100.0


# (absolute bound on a coefficient, relative bound on sigma2) per estimator.
# Maximum likelihood: both sides maximize the same likelihood, so they
# differ only by where each optimizer stops (the 1e-3 argument above). With
# d = 1 statsmodels also gives the integrated state an approximate diffuse
# prior, which moves its optimum by less than that.
# CSS: a different estimator whose distance to the maximum-likelihood
# estimate shrinks like 1/n, so only large-sample closeness is asserted.
_PARAMETER_BOUNDS = {"mle": (1e-3, 1e-3), "css": (0.02, 0.05)}


@pytest.mark.parametrize("method", ["mle", "css"])
class TestArimaFitClosesToStatsmodelsMle:
    def test_ar1_matches_closely(self, method: str) -> None:
        coef_atol, sigma2_rtol = _PARAMETER_BOUNDS[method]
        series = _simulate_ar1(phi=0.6, n=1000, innovation_scale=0.3, seed=RNG_SEED)

        ar_coefs, _, sigma2 = arima_fit(series, p=1, d=0, q=0, method=method)
        sm_result = ARIMA(series, order=(1, 0, 0), trend="n").fit()

        assert ar_coefs[0] == pytest.approx(sm_result.params[0], abs=coef_atol)
        assert sigma2 == pytest.approx(sm_result.params[1], rel=sigma2_rtol)

    def test_arma11_matches_closely(self, method: str) -> None:
        coef_atol, sigma2_rtol = _PARAMETER_BOUNDS[method]
        series = _simulate_arma11(phi=0.5, theta=0.3, n=1500, seed=RNG_SEED)

        ar_coefs, ma_coefs, sigma2 = arima_fit(series, p=1, d=0, q=1, method=method)
        sm_result = ARIMA(series, order=(1, 0, 1), trend="n").fit()

        assert ar_coefs[0] == pytest.approx(sm_result.params[0], abs=coef_atol)
        assert ma_coefs[0] == pytest.approx(sm_result.params[1], abs=coef_atol)
        assert sigma2 == pytest.approx(sm_result.params[2], rel=sigma2_rtol)

    def test_ar1_with_differencing_matches_closely(self, method: str) -> None:
        coef_atol, sigma2_rtol = _PARAMETER_BOUNDS[method]
        series = _simulate_integrated_ar1(phi=0.4, n=1000, seed=RNG_SEED + 1)

        ar_coefs, _, sigma2 = arima_fit(series, p=1, d=1, q=0, method=method)
        sm_result = ARIMA(series, order=(1, 1, 0), trend="n").fit()

        assert ar_coefs[0] == pytest.approx(sm_result.params[0], abs=coef_atol)
        assert sigma2 == pytest.approx(sm_result.params[1], rel=sigma2_rtol)


def _simulate_arima(
    ar: tuple[float, ...], ma: tuple[float, ...], d: int, n: int, level: float, seed: int
) -> np.ndarray:
    innovations = np.random.default_rng(seed).normal(size=n + _BURN_IN)
    series = lfilter(np.r_[1.0, ma], np.r_[1.0, -np.asarray(ar)], innovations)[_BURN_IN:]
    for _ in range(d):
        series = np.cumsum(series)
    return np.asarray(series + level, dtype=np.float64)


def _statsmodels_forecast(
    series: np.ndarray, ar: tuple[float, ...], ma: tuple[float, ...], d: int, h: int
) -> np.ndarray:
    # `SARIMAX` is the state-space model `statsmodels`' `ARIMA` wraps. Two of
    # its options make its forecast the exact conditional mean rather than
    # an approximation to it: `use_exact_diffuse=True` gives the integrated
    # states (d > 0) an exactly diffuse prior instead of a variance of 1e6,
    # and `tolerance=0.0` keeps the filter updating its gain at every step
    # instead of freezing it at an early steady-state estimate (which leaves
    # an error of 1e-10 to 1e-9 in the forecast). The forecast does not
    # depend on the innovation variance, set to 1 here.
    model = SARIMAX(
        series, order=(len(ar), d, len(ma)), trend="n", use_exact_diffuse=True, tolerance=0.0
    )
    return np.asarray(model.filter(np.r_[ar, ma, 1.0]).forecast(h), dtype=np.float64)


# name -> (AR coefficients, MA coefficients, d, series length, series level).
# The MA(2) with coefficients (-1.9, 0.9025) is (1 - 0.95*B)^2, a double
# root at distance 0.05 from the unit circle; an MA coefficient of -0.98 or
# -0.95 puts a root at 0.02 or 0.05 from it. With a root that close the
# filter is still far from its steady state at the end of the sample, as it
# is on the short series.
_FORECAST_CASES: dict[str, tuple[tuple[float, ...], tuple[float, ...], int, int, float]] = {
    "arma11": ((0.5,), (0.3,), 0, 300, 0.0),
    "arma11_short": ((0.5,), (0.3,), 0, 5, 0.0),
    "arma11_ma_root_near_unit_circle": ((0.5,), (-0.98,), 0, 120, 0.0),
    "ma2": ((), (0.4, -0.2), 0, 300, 0.0),
    "ma2_short": ((), (0.4, -0.2), 0, 6, 0.0),
    "ma2_double_root_near_unit_circle": ((), (-1.9, 0.9025), 0, 200, 0.0),
    "arima111": ((0.5,), (0.3,), 1, 300, 100.0),
    "arima111_short": ((0.5,), (0.3,), 1, 15, 100.0),
    "arima111_ma_root_near_unit_circle": ((0.5,), (-0.95,), 1, 40, 100.0),
    "arima022": ((), (0.4, -0.2), 2, 120, 50.0),
    "ar1": ((0.6,), (), 0, 300, 0.0),
    "arima110": ((0.4,), (), 1, 300, 100.0),
}


class TestArimaForecastMatchesStatsmodelsConditionalMean:
    @pytest.mark.parametrize("case", list(_FORECAST_CASES))
    def test_forecast_at_fixed_coefficients_matches_statsmodels(self, case: str) -> None:
        ar, ma, d, n, level = _FORECAST_CASES[case]
        seed = RNG_SEED + list(_FORECAST_CASES).index(case)
        series = _simulate_arima(ar, ma, d, n, level, seed)
        h = 10

        forecast = arima_forecast(series, np.array(ar), np.array(ma), d, h)

        # Both sides evaluate the same linear function of the sample in
        # double precision, so the closed-form bound of 1e-10 relative
        # applies. The absolute term covers forecasts that are zero or
        # decay toward it (an MA(q) forecast is exactly 0 beyond q steps),
        # where a relative error is not defined.
        np.testing.assert_allclose(
            forecast, _statsmodels_forecast(series, ar, ma, d, h), rtol=1e-10, atol=1e-10
        )
