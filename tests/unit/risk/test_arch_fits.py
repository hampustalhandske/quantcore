"""Tests for the GARCH-family fitters, which delegate to the `arch` package."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from arch import arch_model
from scipy.optimize import OptimizeResult

from quantcore import UndefinedResultError
from quantcore.risk import (
    ConvergenceError,
    egarch_11_variance,
    egarch_fit,
    fit_garch_11,
    garch_11_variance,
    gjr_garch_11_variance,
    gjr_garch_fit,
    variance_backcast,
)

RNG_SEED = 20240921


def _simulate_garch(n: int, omega: float, alpha: float, beta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = np.empty(n)
    variance = omega / (1.0 - alpha - beta)
    previous = 0.0
    for t in range(n):
        variance = omega + alpha * previous * previous + beta * variance
        previous = np.sqrt(variance) * rng.standard_normal()
        returns[t] = previous
    return returns


def _gaussian_log_likelihood(epsilon: np.ndarray, omega: float, alpha: float, beta: float) -> float:
    variance = garch_11_variance(epsilon, omega=omega, alpha=alpha, beta=beta)
    return float(-0.5 * np.sum(np.log(2.0 * np.pi) + np.log(variance) + epsilon**2 / variance))


class TestFitGarch11:
    RETURNS = _simulate_garch(1500, 0.05, 0.08, 0.9, RNG_SEED) + 0.02

    def test_returns_what_arch_fits_on_the_demeaned_series(self) -> None:
        epsilon = self.RETURNS - self.RETURNS.mean()
        spread = float(np.std(epsilon))
        fitted = arch_model(
            epsilon / spread, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
        ).fit(disp="off")
        omega, alpha, beta = (float(value) for value in np.asarray(fitted.params))

        result = fit_garch_11(self.RETURNS)

        assert result == pytest.approx((omega * spread**2, alpha, beta), rel=1e-12)
        assert all(type(value) is float for value in result)

    def test_result_is_stationary_and_positive(self) -> None:
        omega, alpha, beta = fit_garch_11(self.RETURNS)
        assert omega > 0.0
        assert alpha >= 0.0
        assert beta >= 0.0
        assert alpha + beta < 1.0

    @pytest.mark.parametrize("scale", [1e-6, 1e-2, 1.0, 1e4])
    def test_units_of_the_returns_only_rescale_omega(self, scale: float) -> None:
        # Scaling changes the standardised series in its last bits, and arch's optimiser stops
        # at a slightly different point. On this series a change of one unit in the last place
        # moves omega by up to 1.0e-6 relative and alpha and beta by up to 1.2e-7. The
        # tolerances are about 100 times that; a unit error moves omega by a factor of scale**2.
        omega, alpha, beta = fit_garch_11(self.RETURNS)
        scaled_omega, scaled_alpha, scaled_beta = fit_garch_11(self.RETURNS * scale)
        assert scaled_omega == pytest.approx(omega * scale**2, rel=1e-4)
        assert scaled_alpha == pytest.approx(alpha, abs=1e-5)
        assert scaled_beta == pytest.approx(beta, abs=1e-5)

    @pytest.mark.parametrize("value", [0.0, 0.01, 0.002, 3.7])
    def test_returns_without_variation_raise(self, value: float) -> None:
        # np.full(n, 0.002) demeans to rounding residue, not zeros; the range test still sees it.
        with pytest.raises(UndefinedResultError, match="no variation") as raised:
            fit_garch_11(np.full(100, value))
        assert isinstance(raised.value, ValueError)

    def test_non_convergence_raises_with_the_best_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Unconverged:
            params = np.array([0.1, 0.05, 0.9])
            convergence_flag = 9
            loglikelihood = -123.0
            optimization_result = OptimizeResult(message="stopped by the test")

        class _Model:
            def fit(self, **kwargs: object) -> _Unconverged:
                return _Unconverged()

        monkeypatch.setattr("arch.arch_model", lambda *args, **kwargs: _Model())

        with pytest.raises(ConvergenceError) as raised:
            fit_garch_11(self.RETURNS)

        spread = float(np.std(self.RETURNS - self.RETURNS.mean()))
        assert raised.value.params == pytest.approx((0.1 * spread**2, 0.05, 0.9))
        assert raised.value.optimizer_message == "stopped by the test"


def _conditional_variance_from_arch(returns: np.ndarray, vol: str, o: int) -> np.ndarray:
    """`arch`'s fitted conditional variance of the mean-adjusted returns, in their own units."""
    epsilon = returns - returns.mean()
    spread = float(np.std(epsilon))
    fitted = arch_model(
        epsilon / spread, mean="Zero", vol=vol, p=1, o=o, q=1, dist="normal", rescale=False
    ).fit(disp="off")
    return np.asarray(fitted.conditional_volatility, dtype=np.float64) ** 2 * spread**2


class TestFittedVariancePathsMatchArch:
    """The kernels' default seed reproduces arch's variance path on the fit sample."""

    @pytest.mark.parametrize("scale", [1e-3, 1.0, 1e2])
    def test_garch_11(self, scale: float) -> None:
        returns = _simulate_garch(1000, 0.05, 0.08, 0.9, RNG_SEED) * scale + 0.02 * scale
        epsilon = returns - returns.mean()
        path = garch_11_variance(epsilon, *fit_garch_11(returns))
        reference = _conditional_variance_from_arch(returns, "GARCH", 0)
        np.testing.assert_allclose(path, reference, rtol=1e-10)

    @pytest.mark.parametrize("scale", [1e-3, 1.0, 1e2])
    def test_gjr_garch_11(self, scale: float) -> None:
        returns = _simulate_garch(1000, 0.05, 0.08, 0.9, RNG_SEED + 1) * scale
        epsilon = returns - returns.mean()
        fit = gjr_garch_fit(returns)
        path = gjr_garch_11_variance(epsilon, fit.omega, fit.alpha, fit.gamma, fit.beta)
        reference = _conditional_variance_from_arch(returns, "GARCH", 1)
        np.testing.assert_allclose(path, reference, rtol=1e-10)

    @pytest.mark.parametrize("scale", [1e-3, 1.0, 1e2])
    def test_egarch_11(self, scale: float) -> None:
        returns = _simulate_garch(1000, 0.05, 0.08, 0.9, RNG_SEED + 2) * scale
        epsilon = returns - returns.mean()
        fit = egarch_fit(returns)
        log_path = egarch_11_variance(epsilon, fit.omega, fit.alpha, fit.gamma, fit.beta)
        reference = _conditional_variance_from_arch(returns, "EGARCH", 1)
        np.testing.assert_allclose(np.exp(log_path), reference, rtol=1e-10)

    def test_backcast_is_the_decay_weighted_mean_of_the_first_squares(self) -> None:
        returns = _simulate_garch(300, 0.05, 0.08, 0.9, RNG_SEED)
        epsilon = returns - returns.mean()
        weights = 0.94 ** np.arange(75)
        expected = float(np.sum(weights / weights.sum() * epsilon[:75] ** 2))
        assert variance_backcast(returns) == pytest.approx(expected, rel=1e-12)

    def test_empty_returns_raise(self) -> None:
        with pytest.raises(ValueError):
            variance_backcast(np.array([]))

    def test_non_finite_returns_are_a_plain_value_error(self) -> None:
        with pytest.raises(ValueError, match="finite") as raised:
            variance_backcast(np.array([0.01, np.nan, 0.02]))
        assert not isinstance(raised.value, UndefinedResultError)

    @pytest.mark.filterwarnings("ignore:overflow:RuntimeWarning")
    def test_overflowing_squares_raise_undefined(self) -> None:
        with pytest.raises(UndefinedResultError, match="not finite") as raised:
            variance_backcast(np.array([1e200, -1e200, 1e200]))
        assert isinstance(raised.value, ValueError)

    def test_constant_series_is_defined(self) -> None:
        assert variance_backcast(np.zeros(10)) == 0.0
        assert 0.0 <= variance_backcast(np.full(10, 0.002)) < 1e-30


class TestPresampleVariance:
    def test_default_is_the_backcast_of_the_given_residuals(self) -> None:
        epsilon = _simulate_garch(50, 0.05, 0.08, 0.9, RNG_SEED)
        epsilon = epsilon - epsilon.mean()
        b = variance_backcast(epsilon)
        np.testing.assert_array_equal(
            garch_11_variance(epsilon, 0.05, 0.08, 0.9),
            garch_11_variance(epsilon, 0.05, 0.08, 0.9, presample_variance=b),
        )
        np.testing.assert_array_equal(
            gjr_garch_11_variance(epsilon, 0.05, 0.08, 0.04, 0.85),
            gjr_garch_11_variance(epsilon, 0.05, 0.08, 0.04, 0.85, b),
        )
        np.testing.assert_array_equal(
            egarch_11_variance(epsilon, -0.1, 0.1, -0.05, 0.9),
            egarch_11_variance(epsilon, -0.1, 0.1, -0.05, 0.9, b),
        )

    def test_unconditional_seed_is_available_explicitly(self) -> None:
        epsilon = _simulate_garch(50, 0.05, 0.08, 0.9, RNG_SEED)
        path = garch_11_variance(epsilon, 0.05, 0.08, 0.9, presample_variance=0.05 / 0.02)
        assert path[0] == pytest.approx(0.05 / (1.0 - 0.08 - 0.9))

    def test_pre_sample_value_enters_the_first_variance(self) -> None:
        epsilon = _simulate_garch(50, 0.05, 0.08, 0.9, RNG_SEED)
        path = garch_11_variance(epsilon, 0.05, 0.08, 0.9, presample_variance=2.0)
        assert path[0] == pytest.approx(0.05 + (0.08 + 0.9) * 2.0)
        gjr = gjr_garch_11_variance(epsilon, 0.05, 0.08, 0.04, 0.85, 2.0)
        assert gjr[0] == pytest.approx(0.05 + (0.08 + 0.02 + 0.85) * 2.0)
        egarch = egarch_11_variance(epsilon, -0.1, 0.1, -0.05, 0.9, 2.0)
        assert egarch[0] == pytest.approx(-0.1 + 0.9 * np.log(2.0))

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_presample_variance_raises(self, bad: float) -> None:
        epsilon = _simulate_garch(20, 0.05, 0.08, 0.9, RNG_SEED)
        with pytest.raises(ValueError, match="presample_variance"):
            garch_11_variance(epsilon, 0.05, 0.08, 0.9, presample_variance=bad)
        with pytest.raises(ValueError, match="presample_variance"):
            egarch_11_variance(epsilon, -0.1, 0.1, -0.05, 0.9, bad)
        with pytest.raises(ValueError, match="presample_variance"):
            gjr_garch_11_variance(epsilon, 0.05, 0.08, 0.04, 0.85, bad)


class TestArchConventions:
    RETURNS = _simulate_garch(600, 0.05, 0.08, 0.9, RNG_SEED)

    @pytest.mark.parametrize(
        ("fitter", "vol", "o"), [(egarch_fit, "EGARCH", 1), (gjr_garch_fit, "GARCH", 1)]
    )
    def test_model_options_are_explicit(
        self, monkeypatch: pytest.MonkeyPatch, fitter: object, vol: str, o: int
    ) -> None:
        seen: dict[str, object] = {}

        class _Fitted:
            params = np.array([0.1, 0.05, -0.02, 0.9])
            convergence_flag = 0
            loglikelihood = -1.0

        class _Model:
            def fit(self, **kwargs: object) -> _Fitted:
                seen["fit"] = kwargs
                return _Fitted()

        def spy(data: np.ndarray, **kwargs: object) -> _Model:
            seen["model"] = kwargs
            return _Model()

        monkeypatch.setattr("arch.arch_model", spy)
        fitter(self.RETURNS)  # type: ignore[operator]
        assert seen["model"] == {
            "mean": "Zero",
            "vol": vol,
            "p": 1,
            "o": o,
            "q": 1,
            "dist": "normal",
            "rescale": False,
        }
        assert seen["fit"] == {"disp": "off", "options": {"maxiter": 1000}}

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 1e4])
    def test_units_of_the_returns_only_rescale_the_constant(self, scale: float) -> None:
        # A change of one unit in the last place of the returns moves the fitted GJR omega by up
        # to 1.5e-6 relative and the other parameters by up to 2e-7, so the tolerances are 50 or
        # more times the optimiser's noise. A unit error moves omega by a factor of scale**2.
        base, scaled = gjr_garch_fit(self.RETURNS), gjr_garch_fit(self.RETURNS * scale)
        assert scaled.omega == pytest.approx(base.omega * scale**2, rel=1e-4)
        assert scaled.alpha == pytest.approx(base.alpha, abs=1e-5)
        ebase, escaled = egarch_fit(self.RETURNS), egarch_fit(self.RETURNS * scale)
        assert escaled.omega == pytest.approx(
            ebase.omega + (1.0 - ebase.beta) * np.log(scale**2), abs=1e-4
        )
        assert escaled.beta == pytest.approx(ebase.beta, abs=1e-5)

    @pytest.mark.parametrize("fitter", [egarch_fit, gjr_garch_fit])
    def test_non_convergence_raises_with_the_best_attempt(
        self, monkeypatch: pytest.MonkeyPatch, fitter: object
    ) -> None:
        class _Unconverged:
            params = np.array([0.1, 0.05, -0.02, 0.9])
            convergence_flag = 9
            loglikelihood = -123.0
            optimization_result = OptimizeResult(message="stopped by the test")

        class _Model:
            def fit(self, **kwargs: object) -> _Unconverged:
                return _Unconverged()

        monkeypatch.setattr("arch.arch_model", lambda *args, **kwargs: _Model())
        with pytest.raises(ConvergenceError) as raised:
            fitter(self.RETURNS)  # type: ignore[operator]
        assert raised.value.params is not None and len(raised.value.params) == 4
        assert raised.value.optimizer_message == "stopped by the test"

        restored = pickle.loads(pickle.dumps(raised.value))
        assert type(restored) is ConvergenceError
        assert restored.args == raised.value.args
        assert restored.params == raised.value.params
        assert restored.log_likelihood == raised.value.log_likelihood
        assert restored.optimizer_message == raised.value.optimizer_message

    @pytest.mark.parametrize("fitter", [egarch_fit, gjr_garch_fit])
    def test_returns_without_variation_raise(self, fitter: object) -> None:
        for value in (0.0, 0.01, 0.002):
            with pytest.raises(UndefinedResultError, match="no variation") as raised:
                fitter(np.full(100, value))  # type: ignore[operator]
            assert isinstance(raised.value, ValueError)
