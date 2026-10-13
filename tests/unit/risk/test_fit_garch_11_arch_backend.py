"""Tests for `fit_garch_11`'s default backend, the `arch` package's GARCH(1,1) fit."""

from __future__ import annotations

import numpy as np
import pytest
from arch import arch_model
from scipy.optimize import OptimizeResult

from quantcore.risk import ConvergenceError, fit_garch_11, garch_11_variance

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


class TestArchBackend:
    RETURNS = _simulate_garch(1500, 0.05, 0.08, 0.9, RNG_SEED) + 0.02

    def test_is_the_default(self) -> None:
        assert fit_garch_11(self.RETURNS) == fit_garch_11(self.RETURNS, backend="arch")

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
        omega, alpha, beta = fit_garch_11(self.RETURNS)
        scaled_omega, scaled_alpha, scaled_beta = fit_garch_11(self.RETURNS * scale)
        assert scaled_omega == pytest.approx(omega * scale**2, rel=1e-6)
        assert scaled_alpha == pytest.approx(alpha, abs=1e-6)
        assert scaled_beta == pytest.approx(beta, abs=1e-6)

    def test_reaches_the_likelihood_of_the_quantcore_backend(self) -> None:
        epsilon = self.RETURNS - self.RETURNS.mean()
        from_arch = _gaussian_log_likelihood(epsilon, *fit_garch_11(self.RETURNS))
        own = _gaussian_log_likelihood(epsilon, *fit_garch_11(self.RETURNS, backend="quantcore"))
        assert from_arch == pytest.approx(own, rel=1e-3)

    def test_returns_without_variation_raise(self) -> None:
        with pytest.raises(ValueError, match="no variation"):
            fit_garch_11(np.full(100, 0.01))

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

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="backend"):
            fit_garch_11(self.RETURNS, backend="statsmodels")  # type: ignore[arg-type]

    def test_backend_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            fit_garch_11(self.RETURNS, "quantcore")  # type: ignore[misc]
