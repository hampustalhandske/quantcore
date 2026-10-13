"""Tests for Ljung-Box autocorrelation testing and ARIMA(p,d,q) fitting/forecasting.

The exact-likelihood code is checked against formulations that share
nothing with it: the dense multivariate-normal density of the sample under
the ARMA autocovariance matrix, closed forms for AR(1) and MA(1), a plain
matrix Kalman filter in NumPy, and `scipy.signal.lfilter`.
"""

from __future__ import annotations

import itertools
import platform
import time

import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.optimize import OptimizeResult, approx_fprime, minimize_scalar
from scipy.signal import lfilter, lfiltic
from scipy.stats import multivariate_normal

import quantcore.risk
import quantcore.time_series.arima as arima_module
from quantcore.time_series.arima import (
    _EXACT_MLE_INFEASIBLE_PENALTY,
    _MAX_ABS_UNCONSTRAINED,
    _PACF_MARGIN,
    _arima_fit_css,
    _arma_kalman_filter_kernel,
    _coefs_to_pacf_kernel,
    _css_objective,
    _css_residuals_numba_kernel,
    _difference,
    _exact_mle_objective,
    _exact_mle_objective_and_gradient_kernel,
    _exact_mle_objective_kernel,
    _fit_exact_mle,
    _is_strictly_stationary,
    _pacf_to_coefs_kernel,
    _run_exact_mle_optimizer,
    _stationary_state_covariance_kernel,
    _unconstrained_to_arma_kernel,
    arima_fit,
    arima_forecast,
    arima_residuals,
    ljung_box_test,
    select_arima_order,
)

RNG_SEED = 7

# Stationary and invertible ARMA(p, q) coefficient sets, (phi, theta), used
# wherever a test needs "several (p, q)". The last two sit close to the edge
# of the stationary and of the invertible region.
ARMA_COEFFICIENTS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "ar1": ((0.6,), ()),
    "ar2": ((0.5, -0.3), ()),
    "ma1": ((), (0.5,)),
    "ma2": ((), (0.4, -0.2)),
    "arma11": ((0.5,), (0.3,)),
    "arma21": ((1.2, -0.5), (-0.4,)),
    "arma12": ((-0.7,), (0.3, 0.2)),
    "arma33": ((0.4, -0.2, 0.1), (0.3, 0.2, -0.1)),
    "ar1_near_unit_root": ((0.97,), ()),
    "ma1_near_unit_root": ((), (-0.95,)),
}


def _simulate_ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, 1.0, size=n)
    series = np.empty(n, dtype=np.float64)
    series[0] = innovations[0]
    for t in range(1, n):
        series[t] = phi * series[t - 1] + innovations[t]
    return series


def _simulate_arma(ar: tuple[float, ...], ma: tuple[float, ...], n: int, seed: int) -> np.ndarray:
    burn_in = 200
    innovations = np.random.default_rng(seed).normal(size=n + burn_in)
    series = lfilter(np.r_[1.0, ma], np.r_[1.0, -np.asarray(ar)], innovations)
    return np.asarray(series[burn_in:], dtype=np.float64)


def _arma_autocovariances(
    phi: tuple[float, ...], theta: tuple[float, ...], n_lags: int
) -> np.ndarray:
    """Autocovariances gamma(0..n_lags-1) of an ARMA(p, q) with unit innovation variance.

    Solves the defining linear equations
    `gamma(k) - sum_i phi_i * gamma(|k - i|) = sum_{j >= k} theta_j * psi_{j-k}`
    for k = 0..max(p, q), with `psi` the MA(infinity) weights and
    `theta_0 = 1`, then extends by the AR recursion. No state space and no
    Kalman filter is involved.
    """
    p, q = len(phi), len(theta)
    theta_full = np.r_[1.0, theta]
    psi = np.zeros(q + 1)
    for j in range(q + 1):
        psi[j] = theta_full[j] + sum(phi[i] * psi[j - 1 - i] for i in range(min(j, p)))
    m = max(p, q) + 1
    lhs = np.zeros((m, m))
    rhs = np.zeros(m)
    for k in range(m):
        lhs[k, k] += 1.0
        for i in range(1, p + 1):
            lhs[k, abs(k - i)] -= phi[i - 1]
        rhs[k] = sum(theta_full[j] * psi[j - k] for j in range(k, q + 1))
    gamma = list(np.linalg.solve(lhs, rhs))
    for k in range(m, n_lags):
        gamma.append(sum(phi[i - 1] * gamma[k - i] for i in range(1, p + 1)))
    return np.array(gamma[:n_lags])


def _dense_loglike(
    w: np.ndarray, phi: tuple[float, ...], theta: tuple[float, ...], sigma2: float
) -> float:
    """Log-density of `w` as one multivariate normal with the ARMA covariance matrix."""
    covariance = sigma2 * toeplitz(_arma_autocovariances(phi, theta, w.size))
    return float(multivariate_normal(mean=np.zeros(w.size), cov=covariance).logpdf(w))


def _kernel_loglike(
    w: np.ndarray, phi: tuple[float, ...], theta: tuple[float, ...], sigma2: float
) -> float:
    """Exact log-likelihood assembled from the Kalman kernel's two sums."""
    sum_sq, sum_log, _, ok = _arma_kalman_filter_kernel(np.array(phi), np.array(theta), w)
    assert ok
    return float(
        -0.5 * w.size * np.log(2.0 * np.pi * sigma2) - 0.5 * sum_log - 0.5 * sum_sq / sigma2
    )


def _concentrated_loglike(w: np.ndarray, phi: np.ndarray, theta: np.ndarray) -> float:
    """Exact log-likelihood at (phi, theta) with sigma2 at its maximizing value."""
    sum_sq, sum_log, _, ok = _arma_kalman_filter_kernel(
        np.array(phi, dtype=np.float64), np.array(theta, dtype=np.float64), w
    )
    assert ok
    n = w.size
    return float(-0.5 * n * (np.log(2.0 * np.pi) + 1.0 + np.log(sum_sq / n)) - 0.5 * sum_log)


def _numpy_kalman_filter(
    phi: tuple[float, ...], theta: tuple[float, ...], w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Textbook Kalman filter of the ARMA state-space model with full matrix products.

    Returns `(v, f, next_state)`: the one-step prediction errors, their
    variances in units of sigma2, and the state predicted for the first
    time point after the sample. Every step updates the full covariance;
    there is no steady-state shortcut.
    """
    p, q = len(phi), len(theta)
    r = max(p, q + 1)
    transition = np.zeros((r, r))
    transition[:p, 0] = phi
    transition[:-1, 1:] += np.eye(r - 1)
    selection = np.zeros(r)
    selection[0] = 1.0
    selection[1 : q + 1] = theta
    state_noise = np.outer(selection, selection)
    covariance = np.linalg.solve(
        np.eye(r * r) - np.kron(transition, transition), state_noise.ravel()
    ).reshape(r, r)
    state = np.zeros(r)
    errors = np.empty(w.size)
    variances = np.empty(w.size)
    for t, value in enumerate(w):
        variances[t] = covariance[0, 0]
        errors[t] = value - state[0]
        gain = covariance[:, 0] / variances[t]
        state = transition @ (state + gain * errors[t])
        covariance = (
            transition @ (covariance - np.outer(gain, covariance[0, :])) @ transition.T
            + state_noise
        )
    return errors, variances, state


def _max_root_modulus(polynomial_tail: np.ndarray) -> float:
    """Largest root modulus of `z^k + c_1*z^(k-1) + ... + c_k`; 0.0 for an empty tail."""
    if polynomial_tail.size == 0:
        return 0.0
    return float(np.max(np.abs(np.roots(np.r_[1.0, polynomial_tail]))))


class TestLjungBoxTest:
    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            ljung_box_test(np.array([]), n_lags=5)

    def test_invalid_n_lags_non_positive(self) -> None:
        residuals = np.arange(20, dtype=np.float64)
        with pytest.raises(ValueError):
            ljung_box_test(residuals, n_lags=0)

    def test_invalid_n_lags_too_large(self) -> None:
        residuals = np.arange(10, dtype=np.float64)
        with pytest.raises(ValueError):
            ljung_box_test(residuals, n_lags=10)

    def test_white_noise_not_rejected(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        residuals = rng.normal(0.0, 1.0, size=500)
        _, p_value = ljung_box_test(residuals, n_lags=10)
        assert p_value > 0.05

    def test_autocorrelated_series_rejected(self) -> None:
        series = _simulate_ar1(phi=0.9, n=500, seed=RNG_SEED)
        _, p_value = ljung_box_test(series, n_lags=10)
        assert p_value < 0.05

    def test_invalid_negative_model_df_raises(self) -> None:
        residuals = np.arange(20, dtype=np.float64)
        with pytest.raises(ValueError):
            ljung_box_test(residuals, n_lags=5, model_df=-1)

    def test_invalid_model_df_at_least_n_lags_raises(self) -> None:
        residuals = np.arange(20, dtype=np.float64)
        with pytest.raises(ValueError):
            ljung_box_test(residuals, n_lags=5, model_df=5)

    def test_default_model_df_is_zero(self) -> None:
        rng = np.random.default_rng(RNG_SEED)
        residuals = rng.normal(0.0, 1.0, size=200)
        stat_default, p_default = ljung_box_test(residuals, n_lags=10)
        stat_explicit, p_explicit = ljung_box_test(residuals, n_lags=10, model_df=0)
        assert stat_default == pytest.approx(stat_explicit)
        assert p_default == pytest.approx(p_explicit)

    def test_model_df_reduces_degrees_of_freedom_and_pvalue(self) -> None:
        # Same Q statistic, fewer df -> the survival function is evaluated
        # further into the chi-squared tail -> a smaller (not larger)
        # p-value for model_df > 0, all else equal.
        rng = np.random.default_rng(RNG_SEED)
        residuals = rng.normal(0.0, 1.0, size=200)
        q_no_adjustment, p_no_adjustment = ljung_box_test(residuals, n_lags=10, model_df=0)
        q_adjusted, p_adjusted = ljung_box_test(residuals, n_lags=10, model_df=3)
        assert q_adjusted == pytest.approx(q_no_adjustment)
        assert p_adjusted < p_no_adjustment


class TestArimaFit:
    def test_invalid_inputs_empty_series(self) -> None:
        with pytest.raises(ValueError):
            arima_fit(np.array([]), p=1, d=0, q=0)

    def test_invalid_negative_orders(self) -> None:
        series = np.arange(50, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_fit(series, p=-1, d=0, q=0)

    def test_invalid_d_out_of_range(self) -> None:
        series = np.arange(50, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_fit(series, p=1, d=3, q=0)

    def test_invalid_p_and_q_both_zero(self) -> None:
        series = np.arange(50, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_fit(series, p=0, d=0, q=0)

    def test_ar1_recovers_known_coefficient(self) -> None:
        series = _simulate_ar1(phi=0.6, n=300, seed=RNG_SEED)
        ar_coefs, ma_coefs, sigma2 = arima_fit(series, p=1, d=0, q=0)
        assert ar_coefs.shape == (1,)
        assert ma_coefs.shape == (0,)
        assert ar_coefs[0] == pytest.approx(0.6, abs=0.15)
        assert sigma2 > 0.0

    @pytest.mark.parametrize("method", ["mle", "css"])
    def test_invalid_series_too_short_for_order(self, method: str) -> None:
        with pytest.raises(ValueError):
            arima_fit(np.arange(3, dtype=np.float64), p=2, d=1, q=0, method=method)

    @pytest.mark.parametrize("method", ["mle", "css"])
    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_invalid_non_finite_series(self, method: str, bad_value: float) -> None:
        series = _simulate_ar1(phi=0.6, n=50, seed=RNG_SEED)
        series[17] = bad_value
        with pytest.raises(ValueError, match="finite"):
            arima_fit(series, p=1, d=0, q=1, method=method)

    @pytest.mark.parametrize("method", ["mle", "css"])
    def test_invalid_series_not_one_dimensional(self, method: str) -> None:
        series = _simulate_ar1(phi=0.6, n=50, seed=RNG_SEED).reshape(25, 2)
        with pytest.raises(ValueError, match="1-D"):
            arima_fit(series, p=1, d=0, q=0, method=method)

    def test_invalid_method_raises(self) -> None:
        series = _simulate_ar1(phi=0.6, n=50, seed=RNG_SEED)
        with pytest.raises(ValueError, match="method"):
            arima_fit(series, p=1, d=0, q=0, method="ols")  # type: ignore[arg-type]

    def test_method_is_keyword_only(self) -> None:
        series = _simulate_ar1(phi=0.6, n=50, seed=RNG_SEED)
        with pytest.raises(TypeError):
            arima_fit(series, 1, 0, 0, "css")  # type: ignore[misc]

    def test_default_method_is_exact_maximum_likelihood(self) -> None:
        series = _simulate_arma((0.5,), (0.3,), 200, RNG_SEED)
        default = arima_fit(series, 1, 0, 1)
        explicit = arima_fit(series, 1, 0, 1, method="mle")
        np.testing.assert_array_equal(default[0], explicit[0])
        np.testing.assert_array_equal(default[1], explicit[1])
        assert default[2] == explicit[2]

    @pytest.mark.parametrize("method", ["mle", "css"])
    @pytest.mark.parametrize("order", [(1, 0, 0), (0, 1, 2), (2, 2, 1)])
    def test_return_types_and_shapes(self, method: str, order: tuple[int, int, int]) -> None:
        p, d, q = order
        series = np.cumsum(_simulate_ar1(phi=0.3, n=120, seed=RNG_SEED))
        ar_coefs, ma_coefs, sigma2 = arima_fit(series, p, d, q, method=method)
        assert isinstance(ar_coefs, np.ndarray) and ar_coefs.dtype == np.float64
        assert isinstance(ma_coefs, np.ndarray) and ma_coefs.dtype == np.float64
        assert ar_coefs.shape == (p,)
        assert ma_coefs.shape == (q,)
        assert type(sigma2) is float
        assert np.isfinite(sigma2) and sigma2 > 0.0


class TestSelectArimaOrder:
    def test_invalid_criterion_raises(self) -> None:
        series = _simulate_ar1(phi=0.6, n=100, seed=RNG_SEED)
        with pytest.raises(ValueError):
            select_arima_order(series, max_p=2, max_d=1, max_q=2, criterion="hqic")

    def test_does_not_crash_on_infeasible_small_orders(self) -> None:
        # A short series makes most (p, d, q) candidates infeasible for
        # arima_fit; select_arima_order must silently skip them rather than
        # propagate the ValueError, as long as at least one candidate fits.
        series = _simulate_ar1(phi=0.5, n=6, seed=RNG_SEED)
        p, d, q = select_arima_order(series, max_p=3, max_d=2, max_q=3, criterion="aic")
        assert p >= 0 and d >= 0 and q >= 0

    def test_ar1_series_selects_low_order_with_no_differencing(self) -> None:
        # An information criterion is noisy on a single finite sample: an
        # extra near-zero coefficient raises the likelihood a little and can
        # occasionally outweigh its penalty, so the recovered order is
        # asserted to be small and undifferenced rather than pinned to the
        # true order.
        series = _simulate_ar1(phi=0.7, n=400, seed=RNG_SEED)
        p, d, q = select_arima_order(series, max_p=3, max_d=1, max_q=2, criterion="bic")
        assert p >= 1
        assert d == 0
        assert q <= 1


class TestArimaResiduals:
    def test_invalid_empty_series(self) -> None:
        with pytest.raises(ValueError):
            arima_residuals(np.array([]), np.array([0.5]), np.array([]), d=0)

    def test_invalid_d_out_of_range(self) -> None:
        series = np.arange(50, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_residuals(series, np.array([0.5]), np.array([]), d=3)

    def test_invalid_ar_coefs_not_1d(self) -> None:
        series = np.arange(50, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_residuals(series, np.array([[0.5]]), np.array([]), d=0)

    def test_invalid_ma_coefs_not_1d(self) -> None:
        series = np.arange(50, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_residuals(series, np.array([0.5]), np.array([[0.1]]), d=0)

    def test_invalid_series_too_short(self) -> None:
        series = np.arange(2, dtype=np.float64)
        with pytest.raises(ValueError):
            arima_residuals(series, np.array([0.5, 0.2, 0.1]), np.array([]), d=0)

    def test_bit_identical_to_arima_fit_internal_residuals(self) -> None:
        # arima_residuals must reproduce the CSS fit's internal residual
        # computation exactly, since both route through the same CSS kernel
        # call with the winning parameters.
        series = _simulate_ar1(phi=0.6, n=300, seed=RNG_SEED)
        p, d, q = 2, 0, 1

        ar_coefs, ma_coefs, sigma2 = arima_fit(series, p=p, d=d, q=q, method="css")

        y = _difference(series, d)
        params = np.concatenate([ar_coefs, ma_coefs])
        internal_sum_sq = _css_objective(params, y, p, q)
        internal_residuals = _css_residuals_numba_kernel(params[:p], params[p : p + q], y)

        residuals = arima_residuals(series, ar_coefs, ma_coefs, d)

        assert np.array_equal(residuals, internal_residuals)
        assert float(np.sum(residuals**2)) == internal_sum_sq
        assert sigma2 == internal_sum_sq / residuals.size

    def test_residuals_feed_ljung_box_no_rejection_on_well_specified_fit(self) -> None:
        series = _simulate_ar1(phi=0.6, n=500, seed=RNG_SEED)
        ar_coefs, ma_coefs, _ = arima_fit(series, p=1, d=0, q=0)
        residuals = arima_residuals(series, ar_coefs, ma_coefs, d=0)
        _, p_value = ljung_box_test(residuals, n_lags=10)
        assert p_value > 0.05

    def test_output_length_matches_css_slicing_convention(self) -> None:
        series = _simulate_ar1(phi=0.5, n=200, seed=RNG_SEED)
        p, d, q = 2, 1, 1
        ar_coefs, ma_coefs, _ = arima_fit(series, p=p, d=d, q=q)
        residuals = arima_residuals(series, ar_coefs, ma_coefs, d)
        differenced = _difference(series, d)
        assert residuals.shape == (differenced.size - max(p, q),)


class TestCssRefactorRegression:
    def test_ar1_fit_matches_expected_coefficient_after_numba_refactor(self) -> None:
        series = _simulate_ar1(phi=0.6, n=300, seed=RNG_SEED)
        ar_coefs, ma_coefs, sigma2 = arima_fit(series, p=1, d=0, q=0, method="css")
        assert ar_coefs[0] == pytest.approx(0.6, abs=0.15)
        assert ma_coefs.shape == (0,)
        assert sigma2 > 0.0

    @pytest.mark.parametrize("method", ["mle", "css"])
    def test_arma_fit_is_self_consistent_across_repeated_calls(self, method: str) -> None:
        series = _simulate_ar1(phi=0.4, n=250, seed=RNG_SEED + 1)
        first = arima_fit(series, p=1, d=0, q=1, method=method)
        second = arima_fit(series, p=1, d=0, q=1, method=method)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        assert first[2] == second[2]

    def test_short_series_grid_still_recovers_a_feasible_order(self) -> None:
        series = _simulate_ar1(phi=0.5, n=60, seed=RNG_SEED)
        p, d, q = select_arima_order(series, max_p=2, max_d=1, max_q=2, criterion="aic")
        assert p >= 0 and d >= 0 and q >= 0


class TestSelectArimaOrderBenchmark:
    def test_select_arima_order_completes_within_a_generous_ceiling(self) -> None:
        # Informational timing guard, not a tight assertion: warms up the
        # Numba JIT kernel first (a fresh process pays that compile cost
        # once) so the measured call only reflects steady-state performance,
        # matching how a long-lived daily refit process would behave after
        # its first asset.
        series = _simulate_ar1(phi=0.6, n=500, seed=RNG_SEED)
        select_arima_order(series[:20], max_p=1, max_d=0, max_q=1, criterion="aic")

        start = time.perf_counter()
        select_arima_order(series, max_p=2, max_d=1, max_q=2, criterion="aic")
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0


class TestArimaForecast:
    def test_ar1_forecast_matches_expected_next_step(self) -> None:
        series = _simulate_ar1(phi=0.6, n=300, seed=RNG_SEED)
        ar_coefs, ma_coefs, _ = arima_fit(series, p=1, d=0, q=0)
        forecast = arima_forecast(series, ar_coefs=ar_coefs, ma_coefs=ma_coefs, d=0, h=5)
        assert forecast.shape == (5,)
        assert np.all(np.isfinite(forecast))
        expected_first_step = ar_coefs[0] * series[-1]
        assert forecast[0] == pytest.approx(expected_first_step, abs=0.5)


class TestPacfTransform:
    def test_ar2_step_up_matches_hand_formula(self) -> None:
        # Durbin-Levinson for two lags: phi_1 = a_1 * (1 - a_2), phi_2 = a_2.
        a1, a2 = 0.4, -0.7
        coefs = _pacf_to_coefs_kernel(np.array([a1, a2]))
        np.testing.assert_allclose(coefs, [a1 * (1.0 - a2), a2], rtol=1e-15)

    @pytest.mark.parametrize("order", [1, 2, 3, 6])
    def test_round_trip_through_step_down(self, order: int) -> None:
        rng = np.random.default_rng(RNG_SEED + order)
        for _ in range(50):
            pacf = rng.uniform(-0.98, 0.98, size=order)
            recovered, stationary = _coefs_to_pacf_kernel(_pacf_to_coefs_kernel(pacf))
            assert stationary
            # Each step-down divides by 1 - a^2 >= 0.04, so rounding grows
            # by at most 25x per lag.
            np.testing.assert_allclose(recovered, pacf, rtol=0.0, atol=1e-9)

    @pytest.mark.parametrize("order", [1, 2, 3, 5])
    def test_step_up_lands_in_stationary_region(self, order: int) -> None:
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            coefs = _pacf_to_coefs_kernel(rng.uniform(-0.98, 0.98, size=order))
            assert _max_root_modulus(-coefs) < 1.0

    @pytest.mark.parametrize("order", [1, 2])
    def test_every_corner_of_the_optimizer_bound_is_stationary_up_to_order_two(
        self, order: int
    ) -> None:
        # Partial autocorrelations of magnitude 1 - 1e-6 leave a margin of
        # 1e-6 (one of them) or 1e-12 (two) to the edge of the stationary
        # region, both wider than the spacing of doubles.
        edge = 1.0 - _PACF_MARGIN
        for signs in itertools.product((1.0, -1.0), repeat=order):
            coefs = _pacf_to_coefs_kernel(edge * np.array(signs))
            assert _is_strictly_stationary(coefs)

    def test_rounding_leaves_the_stationary_region_where_three_bounds_meet(self) -> None:
        # With three partial autocorrelations at 1 - 1e-6 the margin is of
        # order 1e-18, below the spacing of doubles near 1: the rounded
        # coefficients are not a stationary polynomial.
        coefs = _pacf_to_coefs_kernel(np.full(3, 1.0 - _PACF_MARGIN))
        assert not _is_strictly_stationary(coefs)

    @pytest.mark.parametrize(
        "coefs",
        [[1.0], [-1.0], [1.5], [0.5, 0.6], [2.0, -1.0], [0.2, 0.3, 0.6]],
    )
    def test_step_down_flags_non_stationary_coefficients(self, coefs: list[float]) -> None:
        assert _max_root_modulus(-np.array(coefs)) >= 1.0 - 1e-12
        _, stationary = _coefs_to_pacf_kernel(np.array(coefs))
        assert not stationary

    @pytest.mark.parametrize(
        ("coefs", "expected"),
        [
            ([], True),
            ([0.999999], True),
            ([1.0], False),
            ([-1.0], False),
            ([0.5, 0.4], True),
            ([0.5, 0.5], False),
            ([0.5, 0.6], False),
            ([0.0, -1.0], False),
            ([1.2, -0.5], True),
        ],
    )
    def test_exact_stationarity_test_on_hand_cases(
        self, coefs: list[float], expected: bool
    ) -> None:
        # AR(2) is stationary exactly when phi_2 > -1, phi_1 + phi_2 < 1
        # and phi_2 - phi_1 < 1.
        assert _is_strictly_stationary(np.array(coefs, dtype=np.float64)) is expected

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_exact_stationarity_test_agrees_with_polynomial_roots(self, order: int) -> None:
        rng = np.random.default_rng(RNG_SEED + order)
        seen = set()
        for _ in range(200):
            coefs = rng.uniform(-1.5, 1.5, size=order)
            largest = _max_root_modulus(-coefs)
            if abs(largest - 1.0) < 1e-6:
                continue
            assert _is_strictly_stationary(coefs) is (largest < 1.0)
            seen.add(largest < 1.0)
        assert seen == {True, False}

    def test_unconstrained_map_is_tanh_for_first_orders(self) -> None:
        phi, theta = _unconstrained_to_arma_kernel(np.array([0.8, -1.3]), 1, 1)
        assert phi[0] == pytest.approx(np.tanh(0.8), rel=1e-15)
        assert theta[0] == pytest.approx(np.tanh(-1.3), rel=1e-15)

    @pytest.mark.parametrize("order", [(2, 0), (0, 2), (2, 3), (4, 4)])
    def test_unconstrained_map_is_stationary_and_invertible(self, order: tuple[int, int]) -> None:
        p, q = order
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(50):
            u = rng.normal(scale=2.0, size=p + q)
            phi, theta = _unconstrained_to_arma_kernel(u, p, q)
            assert phi.shape == (p,) and theta.shape == (q,)
            assert _max_root_modulus(-phi) < 1.0
            assert _max_root_modulus(theta) < 1.0


class TestStationaryStateCovariance:
    def test_ar1_is_process_variance(self) -> None:
        phi = 0.8
        covariance, ok = _stationary_state_covariance_kernel(np.array([phi]), np.array([1.0]))
        assert ok
        assert covariance[0, 0] == pytest.approx(1.0 / (1.0 - phi**2), rel=1e-14)

    def test_ma1_matches_hand_covariance(self) -> None:
        # State (w_t, theta * e_t) with w_t = e_t + theta * e_{t-1}.
        theta = 0.5
        covariance, ok = _stationary_state_covariance_kernel(
            np.array([0.0, 0.0]), np.array([1.0, theta])
        )
        assert ok
        expected = np.array([[1.0 + theta**2, theta], [theta, theta**2]])
        np.testing.assert_allclose(covariance, expected, rtol=1e-14, atol=1e-15)

    @pytest.mark.parametrize("name", list(ARMA_COEFFICIENTS))
    def test_solves_the_lyapunov_equation(self, name: str) -> None:
        phi, theta = ARMA_COEFFICIENTS[name]
        r = max(len(phi), len(theta) + 1)
        phi_full = np.zeros(r)
        phi_full[: len(phi)] = phi
        selection = np.zeros(r)
        selection[0] = 1.0
        selection[1 : len(theta) + 1] = theta
        transition = np.zeros((r, r))
        transition[:, 0] = phi_full
        transition[:-1, 1:] += np.eye(r - 1)

        covariance, ok = _stationary_state_covariance_kernel(phi_full, selection)

        assert ok
        residual = covariance - (
            transition @ covariance @ transition.T + np.outer(selection, selection)
        )
        assert np.max(np.abs(residual)) <= 1e-12 * np.max(np.abs(covariance))
        np.testing.assert_array_equal(covariance, covariance.T)
        assert covariance[0, 0] == pytest.approx(_arma_autocovariances(phi, theta, 1)[0], rel=1e-12)

    def test_unit_root_has_no_stationary_covariance(self) -> None:
        _, ok = _stationary_state_covariance_kernel(np.array([1.0]), np.array([1.0]))
        assert not ok


class TestArmaKalmanFilterKernel:
    @pytest.mark.parametrize("n", [1, 2, 5, 12, 30])
    @pytest.mark.parametrize("name", list(ARMA_COEFFICIENTS))
    def test_matches_dense_multivariate_normal_log_density(self, name: str, n: int) -> None:
        phi, theta = ARMA_COEFFICIENTS[name]
        w = 1.7 * np.random.default_rng(RNG_SEED + n).normal(size=n)
        sigma2 = 1.3
        assert _kernel_loglike(w, phi, theta, sigma2) == pytest.approx(
            _dense_loglike(w, phi, theta, sigma2), rel=1e-10
        )

    @pytest.mark.parametrize("phi", [0.0, 0.6, -0.9, 0.999])
    def test_ar1_matches_closed_form_likelihood(self, phi: float) -> None:
        w = _simulate_ar1(phi=0.5, n=40, seed=RNG_SEED)
        sigma2 = 0.7
        sum_sq = (1.0 - phi**2) * w[0] ** 2 + float(np.sum((w[1:] - phi * w[:-1]) ** 2))
        closed_form = (
            -0.5 * w.size * np.log(2.0 * np.pi * sigma2)
            + 0.5 * np.log(1.0 - phi**2)
            - 0.5 * sum_sq / sigma2
        )
        assert _kernel_loglike(w, (phi,), (), sigma2) == pytest.approx(closed_form, rel=1e-10)

    def test_ma1_two_observations_match_hand_formula(self) -> None:
        # Covariance sigma2 * [[1 + theta^2, theta], [theta, 1 + theta^2]].
        theta, sigma2 = 0.5, 2.0
        w = np.array([1.0, 2.0])
        variance = 1.0 + theta**2
        determinant = sigma2**2 * (variance**2 - theta**2)
        quadratic_form = (variance * (w[0] ** 2 + w[1] ** 2) - 2.0 * theta * w[0] * w[1]) / (
            sigma2 * (variance**2 - theta**2)
        )
        by_hand = -np.log(2.0 * np.pi) - 0.5 * np.log(determinant) - 0.5 * quadratic_form
        assert _kernel_loglike(w, (), (theta,), sigma2) == pytest.approx(by_hand, rel=1e-13)

    @pytest.mark.parametrize("coefs", [((1.0,), ()), ((1.5,), ()), ((0.5, 0.6), (0.3,))])
    def test_reports_not_ok_outside_the_stationary_region(
        self, coefs: tuple[tuple[float, ...], tuple[float, ...]]
    ) -> None:
        w = np.random.default_rng(RNG_SEED).normal(size=30)
        *_, ok = _arma_kalman_filter_kernel(np.array(coefs[0]), np.array(coefs[1]), w)
        assert not ok

    def test_objective_is_negative_concentrated_loglike_per_observation(self) -> None:
        w = _simulate_arma((0.5,), (0.3,), 80, RNG_SEED)
        u = np.array([0.4, -0.2])
        phi, theta = _unconstrained_to_arma_kernel(u, 1, 1)
        sum_sq, _, _, _ = _arma_kalman_filter_kernel(phi, theta, w)
        sigma2_hat = sum_sq / w.size

        objective = _exact_mle_objective_kernel(u, w, 1, 1)

        dense = _dense_loglike(w, tuple(phi), tuple(theta), sigma2_hat)
        assert objective == pytest.approx(
            -dense / w.size - 0.5 * (np.log(2.0 * np.pi) + 1.0), rel=1e-10
        )

    def test_gradient_matches_independent_finite_differences(self) -> None:
        w = _simulate_arma((0.5, -0.3), (0.4,), 150, RNG_SEED)
        u = np.array([0.3, -0.2, 0.5])

        value, gradient = _exact_mle_objective(u, w, 2, 1)

        assert value == _exact_mle_objective_kernel(u, w, 2, 1)
        forward = approx_fprime(u, lambda x: _exact_mle_objective_kernel(x, w, 2, 1), 1e-7)
        # Forward differences with step 1e-7 on an O(1) function carry an
        # error of order 1e-7 themselves.
        np.testing.assert_allclose(gradient, forward, rtol=0.0, atol=1e-5)

    def test_unevaluable_point_gives_penalty_and_zero_gradient(self) -> None:
        # Squares of values near 1e200 overflow, so the sum of squared
        # prediction errors is infinite at every parameter value.
        w = 1e200 * np.random.default_rng(RNG_SEED).normal(size=30)
        value, gradient = _exact_mle_objective(np.array([0.2]), w, 1, 0)
        assert value == _EXACT_MLE_INFEASIBLE_PENALTY
        np.testing.assert_array_equal(gradient, np.zeros(1))


class TestArmaKalmanNumbaNumpyAgreement:
    """Numba/NumPy agreement for the exact-likelihood Kalman recursion.

    The NumPy side is a textbook filter with full matrix products at every
    step; the Numba kernel exploits the shift structure of the transition
    matrix and switches to a steady-state recursion.
    """

    @pytest.mark.parametrize("n", [1, 7, 50, 400, 2000])
    @pytest.mark.parametrize("name", list(ARMA_COEFFICIENTS))
    def test_agree_across_orders_and_lengths(self, name: str, n: int) -> None:
        phi, theta = ARMA_COEFFICIENTS[name]
        w = np.random.default_rng(RNG_SEED + n).normal(size=n)

        errors, variances, numpy_state = _numpy_kalman_filter(phi, theta, w)
        sum_sq, sum_log, numba_state, ok = _arma_kalman_filter_kernel(
            np.array(phi), np.array(theta), w
        )

        assert ok
        assert sum_sq == pytest.approx(float(np.sum(errors**2 / variances)), rel=1e-10)
        assert sum_log == pytest.approx(float(np.sum(np.log(variances))), rel=1e-10, abs=1e-10)
        assert np.allclose(numba_state, numpy_state, rtol=1e-10, atol=1e-10)

    def test_agree_on_non_invertible_ma(self) -> None:
        # The steady state of a non-invertible MA is not F = 1 with gain R,
        # so the kernel must stay in its general recursion throughout.
        w = np.random.default_rng(RNG_SEED).normal(size=300)
        errors, variances, numpy_state = _numpy_kalman_filter((0.4,), (-1.07,), w)
        sum_sq, sum_log, numba_state, ok = _arma_kalman_filter_kernel(
            np.array([0.4]), np.array([-1.07]), w
        )
        assert ok
        assert sum_sq == pytest.approx(float(np.sum(errors**2 / variances)), rel=1e-10)
        assert sum_log == pytest.approx(float(np.sum(np.log(variances))), rel=1e-10)
        assert np.allclose(numba_state, numpy_state, rtol=1e-10, atol=1e-10)


# name -> (AR coefficients, MA coefficients, (p, d, q) fit); each process is
# fit at its own true order. White noise is fit as an AR(1).
FIT_FAMILIES: dict[str, tuple[tuple[float, ...], tuple[float, ...], tuple[int, int, int]]] = {
    "white_noise": ((), (), (1, 0, 0)),
    "ar1": ((0.6,), (), (1, 0, 0)),
    "ar2": ((0.5, -0.3), (), (2, 0, 0)),
    "ma1": ((), (0.5,), (0, 0, 1)),
    "ma2": ((), (0.4, -0.2), (0, 0, 2)),
    "arma11": ((0.5,), (0.3,), (1, 0, 1)),
    "near_unit_root": ((0.97,), (), (1, 0, 0)),
}


def _degenerate_series() -> dict[str, tuple[np.ndarray, tuple[int, int, int]]]:
    """Series whose likelihood peaks at or near the edge of the admissible region."""
    t = np.arange(40, dtype=np.float64)
    return {
        "linear_trend_differenced_once": (3.0 + 0.5 * t, (1, 1, 0)),
        "linear_trend_in_levels": (3.0 + 0.5 * t, (1, 0, 0)),
        "alternating_signs": ((-1.0) ** t, (1, 0, 0)),
        "quadratic_trend_differenced_once": (t * t, (2, 1, 0)),
        "random_walk_in_levels": (
            np.cumsum(np.random.default_rng(1).normal(size=200)),
            (1, 0, 0),
        ),
        "over_differenced_white_noise": (np.random.default_rng(2).normal(size=200), (0, 1, 1)),
        "white_noise_as_arma22": (np.random.default_rng(3).normal(size=150), (2, 0, 2)),
        "ar1_phi_0999": (_simulate_ar1(phi=0.999, n=100, seed=4), (1, 0, 0)),
        "seven_observations_arma11": (np.random.default_rng(5).normal(size=7), (1, 0, 1)),
    }


class TestArimaFitExactMle:
    @pytest.mark.parametrize("family", list(FIT_FAMILIES))
    def test_sigma2_is_the_maximum_likelihood_variance(self, family: str) -> None:
        ar, ma, order = FIT_FAMILIES[family]
        series = _simulate_arma(ar, ma, 250, RNG_SEED)

        ar_coefs, ma_coefs, sigma2 = arima_fit(series, *order)

        errors, variances, _ = _numpy_kalman_filter(tuple(ar_coefs), tuple(ma_coefs), series)
        assert sigma2 == pytest.approx(
            float(np.sum(errors**2 / variances)) / series.size, rel=1e-10
        )

    @pytest.mark.parametrize(("phi", "n", "seed"), [(0.6, 300, 7), (0.97, 200, 3), (-0.5, 50, 1)])
    def test_ar1_matches_one_dimensional_closed_form_optimum(
        self, phi: float, n: int, seed: int
    ) -> None:
        # For an AR(1) the concentrated exact likelihood is an explicit
        # function of phi alone, maximized here by a bracketing scalar
        # search that uses none of the fitting code.
        w = _simulate_ar1(phi=phi, n=n, seed=seed)

        def sum_of_squares(candidate: float) -> float:
            return float(
                (1.0 - candidate**2) * w[0] ** 2 + np.sum((w[1:] - candidate * w[:-1]) ** 2)
            )

        def negative_loglike(candidate: float) -> float:
            return float(
                0.5 * np.log(sum_of_squares(candidate) / n) - 0.5 * np.log(1.0 - candidate**2) / n
            )

        optimum = minimize_scalar(
            negative_loglike, bounds=(-0.999, 0.999), method="bounded", options={"xatol": 1e-12}
        ).x

        ar_coefs, _, sigma2 = arima_fit(w, p=1, d=0, q=0)

        # The objective is flat to second order at its minimum, so both
        # searches locate it to about the square root of their function
        # tolerance (1e-12 here), i.e. 1e-6.
        assert ar_coefs[0] == pytest.approx(optimum, abs=1e-6)
        assert sigma2 == pytest.approx(sum_of_squares(optimum) / n, rel=1e-6)

    @pytest.mark.parametrize("family", list(FIT_FAMILIES))
    def test_no_nearby_or_rival_point_has_a_higher_exact_likelihood(self, family: str) -> None:
        ar, ma, order = FIT_FAMILIES[family]
        p, _, q = order
        series = _simulate_arma(ar, ma, 250, RNG_SEED)

        ar_coefs, ma_coefs, _ = arima_fit(series, *order)
        attained = _concentrated_loglike(series, ar_coefs, ma_coefs)

        fitted = np.concatenate([ar_coefs, ma_coefs])
        for index in range(p + q):
            for shift in (-1e-3, 1e-3):
                moved = fitted.copy()
                moved[index] += shift
                assert _concentrated_loglike(series, moved[:p], moved[p:]) < attained

        css_ar, css_ma, _ = arima_fit(series, *order, method="css")
        assert _concentrated_loglike(series, css_ar, css_ma) < attained
        if len(ar) == p and len(ma) == q:
            assert _concentrated_loglike(series, np.array(ar), np.array(ma)) < attained

    @pytest.mark.parametrize("order", [(1, 0), (0, 1), (1, 1), (2, 1), (0, 2)])
    def test_optimum_does_not_depend_on_the_starting_point(self, order: tuple[int, int]) -> None:
        p, q = order
        w = _simulate_arma((0.5, -0.3)[:p], (0.4, -0.2)[:q], 300, RNG_SEED)
        reference = _fit_exact_mle(w, p, q)
        assert reference is not None and reference.converged
        reference_objective = -reference.log_likelihood / w.size

        rng = np.random.default_rng(RNG_SEED)
        for _ in range(8):
            run = _run_exact_mle_optimizer(w, p, q, rng.uniform(-1.0, 1.0, size=p + q))
            assert run is not None and run.converged
            phi, theta = _unconstrained_to_arma_kernel(run.u, p, q)
            # L-BFGS-B stops on a projected gradient of 1e-7 or a relative
            # decrease of 1e-12, either of which pins the optimizer down to
            # about 1e-6 in the coefficients.
            np.testing.assert_allclose(phi, reference.ar_coefs, rtol=0.0, atol=1e-5)
            np.testing.assert_allclose(theta, reference.ma_coefs, rtol=0.0, atol=1e-5)
            shifted = run.objective + 0.5 * (np.log(2.0 * np.pi) + 1.0)
            assert shifted == pytest.approx(reference_objective, abs=1e-10)

    def test_public_fit_is_unchanged_by_a_different_starting_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        series = _simulate_arma((0.5,), (0.3,), 300, RNG_SEED)
        baseline = arima_fit(series, 1, 0, 1)

        monkeypatch.setattr(arima_module, "_starting_values", lambda w, p, q: np.full(p + q, -0.9))
        restarted = arima_fit(series, 1, 0, 1)

        np.testing.assert_allclose(restarted[0], baseline[0], rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(restarted[1], baseline[1], rtol=0.0, atol=1e-5)
        assert restarted[2] == pytest.approx(baseline[2], rel=1e-6)

    @pytest.mark.parametrize("name", list(_degenerate_series()))
    def test_coefficients_are_strictly_stationary_and_invertible(self, name: str) -> None:
        series, (p, d, q) = _degenerate_series()[name]

        ar_coefs, ma_coefs, sigma2 = arima_fit(series, p, d, q)

        assert _max_root_modulus(-ar_coefs) < 1.0
        assert _max_root_modulus(ma_coefs) < 1.0
        ar_pacf, stationary = _coefs_to_pacf_kernel(ar_coefs)
        ma_pacf, invertible = _coefs_to_pacf_kernel(-ma_coefs)
        assert stationary and invertible
        largest_allowed = 1.0 - _PACF_MARGIN + 1e-12
        assert np.all(np.abs(ar_pacf) <= largest_allowed)
        assert np.all(np.abs(ma_pacf) <= largest_allowed)
        assert np.isfinite(sigma2) and sigma2 > 0.0

    def test_constant_differenced_series_gives_the_boundary_estimate(self) -> None:
        # The differences of a linear trend are constant, which an AR(1)
        # with phi = 1 reproduces exactly: the likelihood rises all the way
        # to the edge of the stationary region.
        series = 3.0 + 0.5 * np.arange(40, dtype=np.float64)
        ar_coefs, _, sigma2 = arima_fit(series, p=1, d=1, q=0)
        assert ar_coefs[0] == pytest.approx(1.0 - _PACF_MARGIN, abs=1e-12)
        assert 0.0 < sigma2 < 1e-6

    @pytest.mark.parametrize(
        ("series", "d"),
        [(np.zeros(20), 0), (np.full(20, 4.0), 1), (3.0 + 0.5 * np.arange(20.0), 2)],
    )
    def test_identically_zero_differenced_series_returns_zeros(
        self, series: np.ndarray, d: int
    ) -> None:
        ar_coefs, ma_coefs, sigma2 = arima_fit(series, p=1, d=d, q=1)
        np.testing.assert_array_equal(ar_coefs, np.zeros(1))
        np.testing.assert_array_equal(ma_coefs, np.zeros(1))
        assert sigma2 == 0.0

    @pytest.mark.parametrize("scale", [1e-6, 1e6])
    def test_fit_is_scale_equivariant(self, scale: float) -> None:
        series = _simulate_arma((0.5,), (0.3,), 200, RNG_SEED)
        ar_coefs, ma_coefs, sigma2 = arima_fit(series, 1, 0, 1)
        scaled_ar, scaled_ma, scaled_sigma2 = arima_fit(scale * series, 1, 0, 1)
        np.testing.assert_allclose(scaled_ar, ar_coefs, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(scaled_ma, ma_coefs, rtol=0.0, atol=1e-6)
        assert scaled_sigma2 == pytest.approx(scale**2 * sigma2, rel=1e-6)

    @pytest.mark.parametrize("d", [1, 2])
    def test_fit_depends_on_the_series_only_through_its_differences(self, d: int) -> None:
        differenced = _simulate_arma((0.4,), (0.3,), 150, RNG_SEED)
        series = differenced
        for _ in range(d):
            series = np.concatenate([[0.0], np.cumsum(series)])

        integrated_fit = arima_fit(series + 250.0, 1, d, 1)
        direct_fit = arima_fit(np.diff(series, n=d), 1, 0, 1)

        np.testing.assert_allclose(integrated_fit[0], direct_fit[0], rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(integrated_fit[1], direct_fit[1], rtol=0.0, atol=1e-5)
        assert integrated_fit[2] == pytest.approx(direct_fit[2], rel=1e-6)

    def test_any_array_layout_reuses_the_one_compiled_kernel(self) -> None:
        series = _simulate_arma((0.5,), (0.3,), 240, RNG_SEED)
        arima_fit(series, 1, 0, 1)
        compiled = (
            list(_arma_kalman_filter_kernel.signatures),
            list(_exact_mle_objective_and_gradient_kernel.signatures),
        )

        read_only = series.copy()
        read_only.setflags(write=False)
        strided = np.repeat(series, 2)[::2]
        assert not strided.flags["C_CONTIGUOUS"]
        integers = np.round(100.0 * series).astype(np.int64)
        single_precision = series.astype(np.float32)

        np.testing.assert_array_equal(
            arima_fit(read_only, 1, 0, 1)[0], arima_fit(series, 1, 0, 1)[0]
        )
        np.testing.assert_array_equal(arima_fit(strided, 1, 0, 1)[0], arima_fit(series, 1, 0, 1)[0])
        arima_fit(integers, 1, 0, 1)
        arima_fit(single_precision, 1, 0, 1)
        arima_forecast(read_only, np.array([0.5]), np.array([0.3]), d=0, h=3)
        arima_forecast(integers, np.array([0.5], dtype=np.float32), np.array([0.3]), d=1, h=3)

        assert list(_arma_kalman_filter_kernel.signatures) == compiled[0]
        assert list(_exact_mle_objective_and_gradient_kernel.signatures) == compiled[1]


# CSS estimates on fixed inputs: (series, order) -> (AR coefficients, MA
# coefficients, sigma2_hat). The values are the float64 results on macOS
# arm64, where they are compared to the last bit. A Nelder-Mead search turns a
# last-bit difference in a dot product (another BLAS, another CPU) into a
# slightly different end point, so elsewhere they are held to 1e-6 relative,
# which a different estimator misses by orders of magnitude.
_CSS_PINS_ARE_BIT_EXACT_HERE = platform.system() == "Darwin" and platform.machine() == "arm64"
_CSS_PIN_RTOL_ELSEWHERE = 1e-6

CSS_PINNED = [
    ((0.6, 300, 7), (1, 0, 0), [0.6365242614746099], [], 0.8689611975116575),
    (
        (0.6, 300, 7),
        (2, 0, 1),
        [1.3516771615835992, -0.42011017595510924],
        [-0.7581443290783403],
        0.8671102827045445,
    ),
    ((0.4, 250, 8), (1, 0, 1), [0.4806375099555108], [-0.08093142856255497], 1.1265769859167853),
    (
        (0.5, 200, 7),
        (0, 2, 2),
        [],
        [-1.3600562715297642, 0.4076538057877188],
        1.0015197150863469,
    ),
]


class TestArimaFitCss:
    @pytest.mark.parametrize(("simulation", "order", "ar", "ma", "sigma2"), CSS_PINNED)
    def test_reproduces_pinned_estimates(
        self,
        simulation: tuple[float, int, int],
        order: tuple[int, int, int],
        ar: list[float],
        ma: list[float],
        sigma2: float,
    ) -> None:
        phi, n, seed = simulation
        series = _simulate_ar1(phi=phi, n=n, seed=seed)

        ar_coefs, ma_coefs, sigma2_hat = arima_fit(series, *order, method="css")

        expected_ar = np.array(ar, dtype=np.float64)
        expected_ma = np.array(ma, dtype=np.float64)
        if _CSS_PINS_ARE_BIT_EXACT_HERE:
            np.testing.assert_array_equal(ar_coefs, expected_ar)
            np.testing.assert_array_equal(ma_coefs, expected_ma)
            assert sigma2_hat == sigma2
        else:
            np.testing.assert_allclose(ar_coefs, expected_ar, rtol=_CSS_PIN_RTOL_ELSEWHERE)
            np.testing.assert_allclose(ma_coefs, expected_ma, rtol=_CSS_PIN_RTOL_ELSEWHERE)
            assert sigma2_hat == pytest.approx(sigma2, rel=_CSS_PIN_RTOL_ELSEWHERE)

    @pytest.mark.skipif(
        not _CSS_PINS_ARE_BIT_EXACT_HERE,
        reason="the point where Nelder-Mead exhausts its budget is pinned on macOS arm64 only",
    )
    def test_evaluation_cap_raises_with_the_pinned_stopping_point(self) -> None:
        # On this input Nelder-Mead uses all 5,000 evaluations without
        # converging; the point it stops at, a non-invertible MA(1)
        # coefficient included, travels in the exception.
        series = _simulate_ar1(phi=0.5, n=200, seed=7)
        with pytest.raises(quantcore.risk.ConvergenceError) as excinfo:
            arima_fit(series, 2, 1, 1, method="css")
        assert excinfo.value.params == (
            0.5414763610774975,
            0.06958851561399748,
            -1.0739280352460199,
            0.6991458191721374,
        )

    @pytest.mark.parametrize("order", [(1, 0, 0), (1, 1, 1), (0, 2, 2), (3, 0, 2)])
    def test_is_the_conditional_sum_of_squares_routine(self, order: tuple[int, int, int]) -> None:
        series = np.cumsum(_simulate_ar1(phi=0.3, n=150, seed=RNG_SEED))
        public = arima_fit(series, *order, method="css")
        direct = _arima_fit_css(series, *order)
        np.testing.assert_array_equal(public[0], direct[0])
        np.testing.assert_array_equal(public[1], direct[1])
        assert public[2] == direct[2]

    def test_sigma2_divides_by_the_number_of_residuals(self) -> None:
        series = _simulate_ar1(phi=0.5, n=200, seed=RNG_SEED)
        p, d, q = 1, 1, 2
        ar_coefs, ma_coefs, sigma2 = arima_fit(series, p, d, q, method="css")
        residuals = arima_residuals(series, ar_coefs, ma_coefs, d)
        assert residuals.size == series.size - d - max(p, q)
        assert sigma2 == float(np.sum(residuals**2)) / residuals.size

    def test_places_no_constraint_where_the_exact_fit_does(self) -> None:
        # A linear trend in levels: the unconstrained CSS estimate is an
        # explosive AR(1), the exact-likelihood estimate a stationary one.
        series = 3.0 + 0.5 * np.arange(40, dtype=np.float64)
        css_ar, _, _ = arima_fit(series, p=1, d=0, q=0, method="css")
        mle_ar, _, _ = arima_fit(series, p=1, d=0, q=0)
        assert css_ar[0] > 1.0
        assert 0.0 < mle_ar[0] < 1.0


def _never_converging_minimize(fun: object, x0: np.ndarray, **kwargs: object) -> OptimizeResult:
    """Stand-in for `scipy.optimize.minimize` that returns its starting point unconverged."""
    return OptimizeResult(
        x=np.asarray(x0, dtype=np.float64), success=False, message="stopped by the test"
    )


class TestArimaFitConvergence:
    def test_non_convergence_raises_with_the_best_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        series = _simulate_arma((0.5,), (0.3,), 300, RNG_SEED)
        monkeypatch.setattr(arima_module, "minimize", _never_converging_minimize)

        with pytest.raises(quantcore.risk.ConvergenceError) as excinfo:
            arima_fit(series, 1, 0, 1)

        error = excinfo.value
        assert isinstance(error, RuntimeError)
        assert "ARIMA(1,0,1)" in str(error)
        assert error.optimizer_message == "stopped by the test"
        assert error.params is not None and len(error.params) == 3
        assert all(isinstance(value, float) and np.isfinite(value) for value in error.params)
        phi, theta, sigma2 = error.params
        assert abs(phi) < 1.0 and abs(theta) < 1.0 and sigma2 > 0.0
        assert error.log_likelihood == pytest.approx(
            _kernel_loglike(series, (phi,), (theta,), sigma2), rel=1e-12
        )

    def test_iteration_cap_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        series = _simulate_arma((0.5, -0.3), (0.4,), 300, RNG_SEED)
        monkeypatch.setattr(arima_module, "_MAX_ITERATIONS", 1)
        with pytest.raises(quantcore.risk.ConvergenceError) as excinfo:
            arima_fit(series, 2, 0, 1)
        assert excinfo.value.params is not None and len(excinfo.value.params) == 4
        assert excinfo.value.optimizer_message

    @pytest.mark.parametrize("scale", [1e200, 1e-200])
    def test_unrepresentable_squares_raise_without_a_best_attempt(self, scale: float) -> None:
        # The squares of these values overflow to inf or underflow to 0, so
        # the likelihood cannot be evaluated at any parameter value.
        series = scale * _simulate_ar1(phi=0.6, n=100, seed=RNG_SEED)
        with pytest.raises(quantcore.risk.ConvergenceError) as excinfo:
            arima_fit(series, 1, 0, 1)
        assert excinfo.value.params is None
        assert excinfo.value.log_likelihood is None
        assert excinfo.value.optimizer_message is None

    def test_end_point_where_three_bounds_meet_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every run is made to end where three partial autocorrelations
        # meet their bound, a point whose coefficients round out of the
        # stationary region.
        series = _simulate_arma((0.5, -0.3, 0.1), (), 200, RNG_SEED)
        monkeypatch.setattr(
            arima_module,
            "minimize",
            lambda fun, x0, **kwargs: OptimizeResult(
                x=np.full(len(x0), _MAX_ABS_UNCONSTRAINED), success=True, message="stub"
            ),
        )

        assert _run_exact_mle_optimizer(series, 3, 0, np.zeros(3)) is None
        with pytest.raises(quantcore.risk.ConvergenceError) as excinfo:
            arima_fit(series, 3, 0, 0)
        assert excinfo.value.params is None
        assert select_arima_order(series, max_p=3, max_d=0, max_q=0)[0] < 3

    def test_end_point_failing_the_exact_stationarity_test_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        series = _simulate_arma((0.5,), (0.3,), 200, RNG_SEED)
        assert _run_exact_mle_optimizer(series, 1, 1, np.zeros(2)) is not None

        monkeypatch.setattr(arima_module, "_is_strictly_stationary", lambda coefs: False)

        assert _run_exact_mle_optimizer(series, 1, 1, np.zeros(2)) is None
        with pytest.raises(quantcore.risk.ConvergenceError) as excinfo:
            arima_fit(series, 1, 0, 1)
        assert excinfo.value.params is None
        assert excinfo.value.log_likelihood is None

    def test_css_method_raises_when_its_optimizer_reports_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        series = _simulate_arma((0.5,), (0.3,), 100, RNG_SEED)
        monkeypatch.setattr(
            arima_module,
            "minimize",
            lambda fun, x0, **kwargs: OptimizeResult(
                x=np.asarray(x0), fun=fun(x0, *kwargs["args"]), success=False, message="stub"
            ),
        )

        with pytest.raises(quantcore.risk.ConvergenceError) as excinfo:
            arima_fit(series, 1, 0, 1, method="css")

        # The stub stops at the starting point, all coefficients 0, where
        # the residuals are the series from index 1 on.
        error = excinfo.value
        sigma2 = float(np.sum(series[1:] ** 2)) / (series.size - 1)
        assert error.params == (0.0, 0.0, sigma2)
        assert error.optimizer_message == "stub"
        assert error.log_likelihood == pytest.approx(
            -0.5 * (series.size - 1) * (np.log(2.0 * np.pi * sigma2) + 1.0), rel=1e-12
        )

    def test_css_evaluation_cap_raises_with_the_best_attempt(self) -> None:
        # Nelder-Mead exhausts its evaluation budget on this ARMA(4,4).
        series = np.random.default_rng(3).standard_normal(200)
        with pytest.raises(quantcore.risk.ConvergenceError) as excinfo:
            arima_fit(series, 4, 0, 4, method="css")
        error = excinfo.value
        assert "ARIMA(4,0,4)" in str(error)
        assert error.params is not None and len(error.params) == 9
        assert all(np.isfinite(value) for value in error.params)
        assert error.log_likelihood is not None and np.isfinite(error.log_likelihood)
        assert error.optimizer_message


def _simulate_multimodal_arma22(seed: int) -> np.ndarray:
    """ARMA(2,2) sample, n = 300, whose exact likelihood has several local maxima."""
    rng = np.random.default_rng(seed)
    n = 300
    innovations = rng.standard_normal(n + 2)
    w = np.empty(n + 2)
    w[:2] = innovations[:2]
    for t in range(2, n + 2):
        w[t] = (
            0.3 * w[t - 1]
            + 0.4 * w[t - 2]
            + innovations[t]
            + 0.2 * innovations[t - 1]
            - 0.6 * innovations[t - 2]
        )
    return np.array(w[2:])


class TestArimaFitNestedOrders:
    def test_arma22_is_not_below_its_nested_fits_where_two_starts_find_a_local_maximum(
        self,
    ) -> None:
        # From the Hannan-Rissanen and white-noise starts alone L-BFGS-B
        # converges on this series to a local maximum 0.33 log-likelihood
        # units below the ARMA(1,2) fit, which a true maximum of the larger
        # model cannot be.
        series = _simulate_multimodal_arma22(75)
        starts = [arima_module._starting_values(series, 2, 2), np.zeros(4)]
        two_start_best = min(
            run.objective
            for run in (_run_exact_mle_optimizer(series, 2, 2, start) for start in starts)
            if run is not None
        )
        nested = max(
            _concentrated_loglike(series, *arima_fit(series, p, 0, q)[:2])
            for p, q in ((1, 2), (2, 1))
        )
        constant = 0.5 * (np.log(2.0 * np.pi) + 1.0)
        assert -series.size * (two_start_best + constant) < nested - 0.1

        ar_coefs, ma_coefs, _ = arima_fit(series, 2, 0, 2)

        assert _concentrated_loglike(series, ar_coefs, ma_coefs) >= nested - 1e-6

    @pytest.mark.parametrize("order", [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2)])
    @pytest.mark.parametrize("seed", [70, 73, 75, 78])
    def test_fit_is_never_below_the_fits_of_the_orders_nested_in_it(
        self, seed: int, order: tuple[int, int]
    ) -> None:
        p, q = order
        series = _simulate_multimodal_arma22(seed)
        attained = _concentrated_loglike(series, *arima_fit(series, p, 0, q)[:2])
        for nested_p, nested_q in ((p - 1, q), (p, q - 1)):
            if nested_p + nested_q == 0:
                continue
            nested = _concentrated_loglike(series, *arima_fit(series, nested_p, 0, nested_q)[:2])
            assert attained >= nested - 1e-6

    def test_nested_fit_that_is_unusable_is_skipped_as_a_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        series = _simulate_multimodal_arma22(75)
        run_optimizer = arima_module._run_exact_mle_optimizer

        def without_arma12(w: np.ndarray, p: int, q: int, start: np.ndarray) -> object:
            return None if (p, q) == (1, 2) else run_optimizer(w, p, q, start)

        monkeypatch.setattr(arima_module, "_run_exact_mle_optimizer", without_arma12)

        assert _fit_exact_mle(series, 1, 2) is None
        fit = _fit_exact_mle(series, 2, 2)
        assert fit is not None and fit.converged
        arma21 = _fit_exact_mle(series, 2, 1)
        assert arma21 is not None
        assert fit.log_likelihood >= arma21.log_likelihood - 1e-6

    def test_fit_below_a_nested_fit_is_not_reported_as_converged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every ARMA(2,2) run is made to end, converged, at white noise,
        # which the nested fits beat.
        series = _simulate_multimodal_arma22(75)
        run_optimizer = arima_module._run_exact_mle_optimizer

        def white_noise_for_arma22(w: np.ndarray, p: int, q: int, start: np.ndarray) -> object:
            if (p, q) != (2, 2):
                return run_optimizer(w, p, q, start)
            objective, _ = _exact_mle_objective(np.zeros(4), w, 2, 2)
            return arima_module._OptimizerRun(np.zeros(4), objective, True, "stub")

        monkeypatch.setattr(arima_module, "_run_exact_mle_optimizer", white_noise_for_arma22)

        fit = _fit_exact_mle(series, 2, 2)
        assert fit is not None and not fit.converged
        with pytest.raises(quantcore.risk.ConvergenceError):
            arima_fit(series, 2, 0, 2)

    def test_select_arima_order_fits_each_order_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        series = _simulate_multimodal_arma22(75)
        run_optimizer = arima_module._run_exact_mle_optimizer
        runs_per_order: dict[tuple[int, int], int] = {}

        def counting(w: np.ndarray, p: int, q: int, start: np.ndarray) -> object:
            runs_per_order[(p, q)] = runs_per_order.get((p, q), 0) + 1
            return run_optimizer(w, p, q, start)

        monkeypatch.setattr(arima_module, "_run_exact_mle_optimizer", counting)

        select_arima_order(series, max_p=2, max_d=0, max_q=2, criterion="bic")

        # Eight orders, each run from at most four starting points: two of
        # its own and one from each of the two orders nested in it.
        assert set(runs_per_order) == {(p, q) for p in range(3) for q in range(3)} - {(0, 0)}
        assert max(runs_per_order.values()) <= 4


class TestSelectArimaOrderExactLikelihood:
    @pytest.mark.parametrize("criterion", ["aic", "bic"])
    def test_selected_order_minimizes_the_exact_likelihood_criterion(self, criterion: str) -> None:
        series = _simulate_arma((0.5, -0.3), (), 300, RNG_SEED)
        n = series.size
        scores = {}
        for p in range(3):
            for q in range(3):
                if p == 0 and q == 0:
                    continue
                ar_coefs, ma_coefs, sigma2 = arima_fit(series, p, 0, q)
                log_likelihood = _kernel_loglike(series, tuple(ar_coefs), tuple(ma_coefs), sigma2)
                penalty = 2.0 * (p + q + 1) if criterion == "aic" else (p + q + 1) * np.log(n)
                scores[(p, 0, q)] = penalty - 2.0 * log_likelihood

        selected = select_arima_order(series, max_p=2, max_d=0, max_q=2, criterion=criterion)

        assert selected == min(scores, key=lambda order: scores[order])

    @pytest.mark.parametrize(
        ("ar", "ma", "n", "true_order"),
        [
            ((0.5, -0.3), (), 500, (2, 0, 0)),
            ((), (0.5,), 500, (0, 0, 1)),
            ((0.5,), (0.3,), 800, (1, 0, 1)),
        ],
    )
    def test_bic_recovers_the_true_order_of_a_long_series(
        self,
        ar: tuple[float, ...],
        ma: tuple[float, ...],
        n: int,
        true_order: tuple[int, int, int],
    ) -> None:
        series = _simulate_arma(ar, ma, n, RNG_SEED)
        assert select_arima_order(series, max_p=3, max_d=0, max_q=2, criterion="bic") == true_order

    def test_bic_differences_an_integrated_series_once(self) -> None:
        series = 50.0 + np.cumsum(_simulate_arma((0.4,), (), 400, RNG_SEED))
        assert select_arima_order(series, max_p=2, max_d=2, max_q=2, criterion="bic") == (1, 1, 0)

    def test_candidates_that_do_not_converge_are_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        series = _simulate_ar1(phi=0.6, n=100, seed=RNG_SEED)
        monkeypatch.setattr(arima_module, "minimize", _never_converging_minimize)
        with pytest.raises(ValueError, match="no feasible"):
            select_arima_order(series, max_p=1, max_d=1, max_q=1, criterion="aic")

    @pytest.mark.parametrize("seed", range(8))
    def test_selected_order_can_always_be_fit(self, seed: int) -> None:
        # Six observations leave several candidates with about as many
        # coefficients as observations, some of which do not converge.
        series = np.random.default_rng(seed).normal(size=6)
        order = select_arima_order(series, max_p=3, max_d=2, max_q=3, criterion="aic")
        ar_coefs, ma_coefs, _ = arima_fit(series, *order)
        assert ar_coefs.shape == (order[0],) and ma_coefs.shape == (order[2],)

    def test_non_finite_series_raises(self) -> None:
        series = _simulate_ar1(phi=0.6, n=60, seed=RNG_SEED)
        series[10] = np.nan
        with pytest.raises(ValueError, match="finite"):
            select_arima_order(series, max_p=1, max_d=1, max_q=1)

    def test_series_not_one_dimensional_raises(self) -> None:
        series = _simulate_ar1(phi=0.6, n=60, seed=RNG_SEED).reshape(30, 2)
        with pytest.raises(ValueError, match="1-D"):
            select_arima_order(series, max_p=1, max_d=1, max_q=1)

    def test_empty_series_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            select_arima_order(np.array([]), max_p=1, max_d=1, max_q=1)

    def test_constant_series_has_no_comparable_candidate(self) -> None:
        # In levels the series is constant and its first difference is
        # identically zero: neither has a likelihood to compare.
        with pytest.raises(ValueError, match="no variation"):
            select_arima_order(np.full(40, 2.0), max_p=2, max_d=1, max_q=2)

    def test_differences_without_variation_are_not_selected(self) -> None:
        # A linear trend: constant first difference, zero second difference.
        _, d, _ = select_arima_order(np.arange(40.0), max_p=2, max_d=2, max_q=2)
        assert d == 0

    def test_trend_plus_noise_can_still_select_a_difference(self) -> None:
        series = np.arange(200.0) + np.cumsum(np.random.default_rng(RNG_SEED).normal(size=200))
        _, d, _ = select_arima_order(series, max_p=1, max_d=1, max_q=1, criterion="bic")
        assert d == 1


class TestArimaResidualsDefinition:
    @pytest.mark.parametrize("d", [0, 1])
    @pytest.mark.parametrize("name", list(ARMA_COEFFICIENTS))
    def test_matches_scipy_lfilter_from_the_same_initial_conditions(
        self, name: str, d: int
    ) -> None:
        # The CSS residuals are the inverse ARMA filter phi(B) / theta(B)
        # applied to the differenced series, started at index max(p, q)
        # with earlier residuals 0 and earlier observations as they are.
        phi, theta = ARMA_COEFFICIENTS[name]
        series = (
            np.cumsum(_simulate_arma((0.3,), (), 120, RNG_SEED))
            if d
            else _simulate_arma((0.3,), (), 120, RNG_SEED)
        )
        w = np.diff(series, n=d)
        start = max(len(phi), len(theta))
        numerator = np.r_[1.0, -np.asarray(phi)]
        denominator = np.r_[1.0, theta]
        initial_state = lfiltic(
            numerator, denominator, y=np.zeros(len(theta)), x=w[start - len(phi) : start][::-1]
        )
        expected, _ = lfilter(numerator, denominator, w[start:], zi=initial_state)

        residuals = arima_residuals(series, np.array(phi), np.array(theta), d)

        np.testing.assert_allclose(residuals, expected, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("phi", [(0.6,), (0.5, -0.3), (0.4, -0.2, 0.1)])
    def test_pure_ar_residuals_are_the_exact_prediction_errors_after_p(
        self, phi: tuple[float, ...]
    ) -> None:
        series = _simulate_arma(phi, (), 80, RNG_SEED)
        errors, variances, _ = _numpy_kalman_filter(phi, (), series)

        residuals = arima_residuals(series, np.array(phi), np.array([]), d=0)

        np.testing.assert_allclose(residuals, errors[len(phi) :], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(variances[len(phi) :], 1.0, rtol=1e-12)
        assert np.all(variances[: len(phi)] > 1.0)

    def test_ma_residuals_approach_the_exact_prediction_errors_geometrically(self) -> None:
        # With theta = 0.5 the two differ by a term that halves at every
        # step: visible at the start of the sample, gone long before its end.
        theta = 0.5
        series = _simulate_arma((), (theta,), 300, RNG_SEED)
        errors, _, _ = _numpy_kalman_filter((), (theta,), series)

        residuals = arima_residuals(series, np.array([]), np.array([theta]), d=0)

        difference = np.abs(residuals - errors[1:])
        assert difference[0] > 1e-2
        assert np.all(difference[1:40] < difference[:39])
        assert np.max(difference[60:]) < 1e-12


class TestArimaForecastConditionalMean:
    @pytest.mark.parametrize("phi", [(0.6,), (0.5, -0.3), (1.0,), (1.05,), ()])
    @pytest.mark.parametrize("d", [0, 1, 2])
    def test_pure_ar_forecast_is_the_ar_recursion(self, phi: tuple[float, ...], d: int) -> None:
        series = np.cumsum(_simulate_ar1(phi=0.4, n=60, seed=RNG_SEED)) + 20.0
        h = 6
        w = list(np.diff(series, n=d))
        for _ in range(h):
            w.append(float(np.dot(phi, w[: -len(phi) - 1 : -1])) if phi else 0.0)
        expected = np.array(w[-h:])
        for level in range(d - 1, -1, -1):
            expected = np.diff(series, n=level)[-1] + np.cumsum(expected)

        forecast = arima_forecast(series, np.array(phi), np.array([]), d=d, h=h)

        np.testing.assert_allclose(forecast, expected, rtol=1e-13, atol=1e-13)

    def test_all_zero_ma_coefficients_are_the_same_as_none(self) -> None:
        series = np.cumsum(_simulate_ar1(phi=0.4, n=60, seed=RNG_SEED))
        without = arima_forecast(series, np.array([0.5, 0.2]), np.array([]), d=1, h=5)
        with_zeros = arima_forecast(series, np.array([0.5, 0.2]), np.zeros(2), d=1, h=5)
        np.testing.assert_array_equal(with_zeros, without)

    @pytest.mark.parametrize(
        ("name", "n"),
        [
            *((name, n) for name in ("ma1", "ma2", "ma1_near_unit_root") for n in (1, 2)),
            *(
                (name, n)
                for name in ARMA_COEFFICIENTS
                if ARMA_COEFFICIENTS[name][1]
                for n in (4, 7, 25)
            ),
        ],
    )
    def test_matches_dense_conditional_expectation(self, name: str, n: int) -> None:
        # E[w_{n+k} | w] = c_k' * Sigma^{-1} * w, with Sigma the covariance
        # matrix of the sample and c_k its covariances with w_{n+k}.
        phi, theta = ARMA_COEFFICIENTS[name]
        h = 5
        w = np.random.default_rng(RNG_SEED + n).normal(size=n)
        gamma = _arma_autocovariances(phi, theta, n + h)
        weights = np.linalg.solve(toeplitz(gamma[:n]), w)
        expected = np.array([gamma[k : n + k][::-1] @ weights for k in range(1, h + 1)])

        forecast = arima_forecast(w, np.array(phi), np.array(theta), d=0, h=h)

        np.testing.assert_allclose(forecast, expected, rtol=1e-9, atol=1e-11)

    def test_ma1_single_observation_matches_hand_formula(self) -> None:
        # Cov(w_2, w_1) / Var(w_1) = theta / (1 + theta^2), and nothing
        # carries beyond one step.
        theta = 0.5
        forecast = arima_forecast(np.array([2.0]), np.array([]), np.array([theta]), d=0, h=3)
        np.testing.assert_allclose(
            forecast, [2.0 * theta / (1.0 + theta**2), 0.0, 0.0], rtol=1e-15, atol=0.0
        )

    def test_ma_forecast_is_exactly_zero_beyond_q_steps(self) -> None:
        series = _simulate_arma((), (0.4, -0.2), 100, RNG_SEED)
        forecast = arima_forecast(series, np.array([]), np.array([0.4, -0.2]), d=0, h=6)
        assert np.all(forecast[:2] != 0.0)
        np.testing.assert_array_equal(forecast[2:], np.zeros(4))

    def test_last_innovation_carries_into_the_forecast(self) -> None:
        # ARMA(1,1) on a long sample: the one-step forecast is
        # phi * w_T + theta * e_T, with e_T the last innovation.
        phi, theta = 0.5, 0.3
        series = _simulate_arma((phi,), (theta,), 400, RNG_SEED)
        innovations = lfilter([1.0, -phi], [1.0, theta], series)

        forecast = arima_forecast(series, np.array([phi]), np.array([theta]), d=0, h=2)

        assert forecast[0] == pytest.approx(phi * series[-1] + theta * innovations[-1], abs=1e-10)
        assert forecast[1] == pytest.approx(phi * forecast[0], rel=1e-13)
        assert abs(theta * innovations[-1]) > 1e-3

    def test_non_invertible_ma_matches_dense_conditional_expectation(self) -> None:
        phi, theta = (0.4,), (-1.07,)
        n, h = 30, 4
        w = np.random.default_rng(RNG_SEED).normal(size=n)
        gamma = _arma_autocovariances(phi, theta, n + h)
        weights = np.linalg.solve(toeplitz(gamma[:n]), w)
        expected = np.array([gamma[k : n + k][::-1] @ weights for k in range(1, h + 1)])
        forecast = arima_forecast(w, np.array(phi), np.array(theta), d=0, h=h)
        np.testing.assert_allclose(forecast, expected, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize("d", [1, 2])
    def test_integrates_the_forecast_of_the_differences(self, d: int) -> None:
        series = 100.0 + np.cumsum(np.cumsum(_simulate_arma((0.3,), (0.4,), 80, RNG_SEED)))
        coefs = (np.array([0.3]), np.array([0.4]))
        h = 5
        expected = arima_forecast(np.diff(series, n=d), *coefs, d=0, h=h)
        for level in range(d - 1, -1, -1):
            expected = np.diff(series, n=level)[-1] + np.cumsum(expected)

        forecast = arima_forecast(series, *coefs, d=d, h=h)

        np.testing.assert_allclose(forecast, expected, rtol=1e-14)

    @pytest.mark.parametrize("phi", [[1.0], [1.2], [0.5, 0.6]])
    def test_non_stationary_ar_with_an_ma_part_raises(self, phi: list[float]) -> None:
        series = _simulate_ar1(phi=0.5, n=50, seed=RNG_SEED)
        with pytest.raises(ValueError, match="stationary"):
            arima_forecast(series, np.array(phi), np.array([0.3]), d=0, h=3)

    def test_invalid_non_finite_inputs(self) -> None:
        series = _simulate_ar1(phi=0.5, n=50, seed=RNG_SEED)
        broken = series.copy()
        broken[20] = np.nan
        with pytest.raises(ValueError, match="finite"):
            arima_forecast(broken, np.array([0.5]), np.array([0.3]), d=0, h=3)
        with pytest.raises(ValueError, match="finite"):
            arima_forecast(series, np.array([np.nan]), np.array([]), d=0, h=3)
        with pytest.raises(ValueError, match="finite"):
            arima_forecast(series, np.array([0.5]), np.array([np.inf]), d=0, h=3)

    @pytest.mark.parametrize("ma", [[], [0.3]])
    def test_invalid_series_not_one_dimensional(self, ma: list[float]) -> None:
        series = _simulate_ar1(phi=0.5, n=50, seed=RNG_SEED).reshape(25, 2)
        with pytest.raises(ValueError, match="series must be a 1-D"):
            arima_forecast(series, np.array([0.5]), np.array(ma), d=0, h=3)

    def test_invalid_coefficients_not_one_dimensional(self) -> None:
        series = _simulate_ar1(phi=0.5, n=50, seed=RNG_SEED)
        with pytest.raises(ValueError, match="1-D"):
            arima_forecast(series, np.array([[0.5]]), np.array([]), d=0, h=3)
        with pytest.raises(ValueError, match="1-D"):
            arima_forecast(series, np.array([0.5]), np.array([[0.3]]), d=0, h=3)

    @pytest.mark.parametrize(("d", "h"), [(3, 2), (0, 0), (0, -1)])
    def test_invalid_d_or_h(self, d: int, h: int) -> None:
        series = _simulate_ar1(phi=0.5, n=50, seed=RNG_SEED)
        with pytest.raises(ValueError):
            arima_forecast(series, np.array([0.5]), np.array([0.3]), d=d, h=h)

    def test_invalid_series_empty_or_too_short(self) -> None:
        with pytest.raises(ValueError):
            arima_forecast(np.array([]), np.array([0.5]), np.array([0.3]), d=0, h=3)
        with pytest.raises(ValueError):
            arima_forecast(np.array([1.0, 2.0]), np.array([0.5, 0.1]), np.array([0.3]), d=0, h=3)
