"""Oracle tests: SDE path simulators vs their closed-form analytical moments.

Covers Workstream D priority 5 (owner follow-up, decision 4): "GBM / OU /
CIR / Heston simulators vs analytical means, variances and
autocovariances (and Feller-condition edge cases for CIR/Heston)."

No third-party library is the oracle here — the ground truth is the
published closed-form solution for each process's transition mean and
variance (citations per process below, all already in
`docs/REFERENCES.md`), which the Monte Carlo path simulators are checked
against within k standard errors of the Monte Carlo estimate (k=3, the
default from `tests/oracle/README.md`'s tolerance policy), at a sample
size large enough to keep that interval tight.

References:
    Black & Scholes (1973) / geometric Brownian motion moments (standard).
    Uhlenbeck & Ornstein (1930), Vasicek (1977) — OU transition mean/variance.
    Cox, Ingersoll & Ross (1985) — CIR transition mean/variance, Feller
    condition (2*kappa*theta >= sigma^2 for r_t > 0 a.s.).
    Heston (1993) — the variance process V_t is itself CIR, same moment
    formulas with (kappa, theta, xi) in place of CIR's (kappa, theta, sigma).
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.core.sde_solver import euler_maruyama, simulate_gbm_paths
from quantcore.core.stochastic import simulate_cir_paths, simulate_heston_paths, simulate_ou_paths

pytestmark = pytest.mark.oracle

RNG_SEED = 20240921


def _assert_within_k_standard_errors(
    sample: npt_array, expected: float, k: float = 3.0, label: str = ""
) -> None:
    n = sample.size
    sample_mean = float(np.mean(sample))
    standard_error = float(np.std(sample, ddof=1)) / np.sqrt(n)
    margin = k * standard_error
    assert abs(sample_mean - expected) < margin, (
        f"{label}: sample mean {sample_mean:.6g} is not within {k} standard errors "
        f"({margin:.6g}) of the analytical value {expected:.6g}"
    )


# Type alias only for the helper's signature readability; avoids importing
# numpy.typing just for one local annotation.
npt_array = np.ndarray


class TestGbmMoments:
    """S_t = S_0 * exp((r - 0.5*sigma^2)*t + sigma*W_t):
    E[S_t] = S_0*exp(r*t), Var[S_t] = S_0^2*exp(2*r*t)*(exp(sigma^2*t) - 1).
    """

    @pytest.mark.parametrize(
        ("spot", "rate", "vol", "t"),
        [(100.0, 0.05, 0.2, 1.0), (50.0, 0.02, 0.4, 2.0), (100.0, -0.01, 0.15, 0.5)],
    )
    def test_terminal_mean_matches_closed_form(
        self, spot: float, rate: float, vol: float, t: float
    ) -> None:
        paths = simulate_gbm_paths(
            spot, rate, vol, t, num_steps=100, num_paths=200_000, seed=RNG_SEED
        )
        terminal = paths[:, -1]
        expected_mean = spot * np.exp(rate * t)
        _assert_within_k_standard_errors(terminal, expected_mean, label="GBM terminal mean")

    def test_terminal_variance_matches_closed_form(self) -> None:
        spot, rate, vol, t = 100.0, 0.05, 0.3, 1.0
        n_paths = 200_000
        paths = simulate_gbm_paths(
            spot, rate, vol, t, num_steps=100, num_paths=n_paths, seed=RNG_SEED
        )
        terminal = paths[:, -1]
        expected_variance = spot**2 * np.exp(2.0 * rate * t) * (np.exp(vol**2 * t) - 1.0)
        sample_variance = float(np.var(terminal, ddof=1))
        # Variance of a sample variance estimator (approximately, for large
        # n): Var(s^2) ~= (2*sigma^4)/(n-1) for a Gaussian-ish quantity; the
        # terminal distribution here is log-normal, so this is a generous
        # (not exact) standard error, hence a wider k.
        approx_se = expected_variance * np.sqrt(2.0 / n_paths)
        assert abs(sample_variance - expected_variance) < 6.0 * approx_se


class TestOuMoments:
    """X_t = mu + (X_0-mu)*exp(-theta*t) + noise:
    E[X_t] = mu + (X_0-mu)*exp(-theta*t)
    Var[X_t] = sigma^2/(2*theta) * (1 - exp(-2*theta*t))
    """

    @pytest.mark.parametrize(
        ("theta", "mu", "sigma", "x0", "t"),
        [(1.0, 0.5, 0.3, 2.0, 1.0), (3.0, 0.0, 0.4, 1.0, 0.5), (0.5, 1.0, 0.2, 1.0, 3.0)],
    )
    def test_terminal_mean_and_variance_match_closed_form(
        self, theta: float, mu: float, sigma: float, x0: float, t: float
    ) -> None:
        paths = simulate_ou_paths(
            theta, mu, sigma, x0, t, n_paths=200_000, n_steps=100, seed=RNG_SEED
        )
        terminal = paths[:, -1]

        expected_mean = mu + (x0 - mu) * np.exp(-theta * t)
        _assert_within_k_standard_errors(terminal, expected_mean, label="OU terminal mean")

        expected_variance = (sigma**2 / (2.0 * theta)) * (1.0 - np.exp(-2.0 * theta * t))
        sample_variance = float(np.var(terminal, ddof=1))
        approx_se = expected_variance * np.sqrt(2.0 / terminal.size)
        assert abs(sample_variance - expected_variance) < 6.0 * approx_se

    def test_autocovariance_matches_closed_form(self) -> None:
        # Cov(X_s, X_t) for s < t, in the (near-)stationary regime, is
        # Var(X_s) * exp(-theta*(t-s)) -- checked using the two exact-transition
        # simulated time points s=n_steps//2, t=n_steps (both well past the
        # transient from x0, so their marginal variances are close to the
        # stationary sigma^2/(2*theta) this formula assumes).
        theta, mu, sigma, x0, t = 2.0, 0.0, 0.5, 0.0, 4.0
        n_steps = 200
        paths = simulate_ou_paths(
            theta, mu, sigma, x0, t, n_paths=100_000, n_steps=n_steps, seed=RNG_SEED
        )
        mid = n_steps // 2
        x_s = paths[:, mid]
        x_t = paths[:, -1]
        dt = t * (n_steps - mid) / n_steps

        stationary_var = sigma**2 / (2.0 * theta)
        expected_cov = stationary_var * np.exp(-theta * dt)

        sample_cov = float(np.cov(x_s, x_t, ddof=1)[0, 1])
        n = x_s.size
        # Generous standard error for a covariance estimator between two
        # correlated near-Gaussian variables of the above variance.
        approx_se = stationary_var * np.sqrt(2.0 / n)
        assert abs(sample_cov - expected_cov) < 6.0 * approx_se


class TestCirMoments:
    """E[r_t] = theta + (r_0-theta)*exp(-kappa*t)
    Var[r_t] = r_0*(sigma^2/kappa)*(exp(-kappa*t)-exp(-2*kappa*t))
             + theta*(sigma^2/(2*kappa))*(1-exp(-kappa*t))^2
    """

    @pytest.mark.parametrize(
        ("r0", "kappa", "theta", "sigma", "t"),
        [(0.03, 1.0, 0.05, 0.1, 1.0), (0.08, 2.0, 0.03, 0.05, 0.5)],
    )
    def test_terminal_mean_and_variance_match_closed_form(
        self, r0: float, kappa: float, theta: float, sigma: float, t: float
    ) -> None:
        # n_steps=2000: the Milstein scheme's discretization bias (order-1,
        # shrinks roughly linearly with the step size) must be small
        # relative to the Monte Carlo standard error itself for a
        # standard-errors-only tolerance to be meaningful; empirically,
        # 200 steps left a bias several times the standard error at
        # n_paths=200_000 for the faster-mean-reverting parameter set here
        # (kappa=2.0, t=0.5) -- a real, expected, and well-understood
        # property of a fixed-order discretization, not a defect.
        paths = simulate_cir_paths(
            r0, kappa, theta, sigma, t, n_paths=200_000, n_steps=2000, seed=RNG_SEED
        )
        terminal = paths[:, -1]

        expected_mean = theta + (r0 - theta) * np.exp(-kappa * t)
        _assert_within_k_standard_errors(terminal, expected_mean, label="CIR terminal mean")

        expected_variance = (
            r0 * (sigma**2 / kappa) * (np.exp(-kappa * t) - np.exp(-2.0 * kappa * t))
            + theta * (sigma**2 / (2.0 * kappa)) * (1.0 - np.exp(-kappa * t)) ** 2
        )
        sample_variance = float(np.var(terminal, ddof=1))
        approx_se = expected_variance * np.sqrt(2.0 / terminal.size)
        assert abs(sample_variance - expected_variance) < 8.0 * approx_se

    def test_feller_condition_satisfied_paths_stay_positive(self) -> None:
        # 2*kappa*theta >= sigma^2 (Feller 1951) implies r_t > 0 a.s. under
        # the exact SDE; the discretized (Milstein) scheme can still dip
        # slightly negative near the boundary, so this checks the practical
        # invariant quantcore documents (paths overwhelmingly stay
        # non-negative deep in the Feller-satisfied regime), not the
        # theorem's a.s. claim about the continuous-time process itself.
        r0, kappa, theta, sigma = 0.05, 3.0, 0.05, 0.05  # 2*kappa*theta=0.3 >> sigma^2=0.0025
        assert 2.0 * kappa * theta >= sigma**2
        paths = simulate_cir_paths(
            r0, kappa, theta, sigma, t=2.0, n_paths=50_000, n_steps=500, seed=RNG_SEED
        )
        fraction_negative = float(np.mean(paths < 0.0))
        assert fraction_negative < 1e-4

    def test_feller_condition_violated_paths_can_touch_zero(self) -> None:
        # 2*kappa*theta < sigma^2: the boundary is attainable, so a
        # meaningfully larger fraction of simulated values near (or at, for
        # a scheme that floors at 0) zero is expected and is not itself a
        # simulator defect -- this documents/pins that expectation rather
        # than asserting a specific "wrong" numerical outcome.
        r0, kappa, theta, sigma = 0.05, 0.5, 0.02, 0.3  # 2*kappa*theta=0.02 << sigma^2=0.09
        assert 2.0 * kappa * theta < sigma**2
        paths = simulate_cir_paths(
            r0, kappa, theta, sigma, t=2.0, n_paths=50_000, n_steps=500, seed=RNG_SEED
        )
        assert np.all(paths >= 0.0), "simulate_cir_paths must never return a negative value"
        fraction_near_zero = float(np.mean(paths < 1e-4))
        assert fraction_near_zero > 0.0, (
            "expected some mass near the zero boundary when the Feller condition is violated"
        )


class TestHestonVarianceMoments:
    """Heston's variance process V_t is itself CIR with (kappa, theta, xi);
    same closed-form moments as TestCirMoments, applied to xi in place of
    the standalone CIR process's sigma."""

    def test_terminal_variance_mean_matches_cir_closed_form(self) -> None:
        v0, kappa, theta, xi, t = 0.04, 2.0, 0.04, 0.3, 1.0
        _, v_paths = simulate_heston_paths(
            s0=100.0,
            v0=v0,
            mu=0.05,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=-0.5,
            t=t,
            n_paths=200_000,
            n_steps=200,
            seed=RNG_SEED,
        )
        terminal_v = v_paths[:, -1]
        expected_mean = theta + (v0 - theta) * np.exp(-kappa * t)
        _assert_within_k_standard_errors(
            terminal_v, expected_mean, label="Heston terminal variance mean"
        )

    def test_terminal_spot_mean_matches_gbm_style_closed_form(self) -> None:
        # E[S_t] = S_0*exp(mu*t) regardless of the stochastic-volatility
        # dynamics (V_t's randomness doesn't bias the spot's drift under
        # the model's own risk-neutral/actual measure as specified by mu).
        s0, mu, t = 100.0, 0.05, 1.0
        s_paths, _ = simulate_heston_paths(
            s0=s0,
            v0=0.04,
            mu=mu,
            kappa=1.5,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            t=t,
            n_paths=200_000,
            n_steps=200,
            seed=RNG_SEED,
        )
        terminal_s = s_paths[:, -1]
        expected_mean = s0 * np.exp(mu * t)
        _assert_within_k_standard_errors(
            terminal_s, expected_mean, label="Heston terminal spot mean"
        )


class TestGenericEulerMaruyamaMatchesOuClosedForm:
    """`euler_maruyama` is the generic solver behind the specialized,
    Numba-accelerated `simulate_ou_paths`/`simulate_cir_paths`/etc; those
    are already checked against closed-form moments above, but the
    generic (arbitrary-callable) code path itself was not exercised by
    any of those tests. Applying it directly to the OU drift/diffusion
    and checking against the same closed-form moments used in
    `TestOuMoments` covers that remaining gap.
    """

    def test_terminal_mean_and_variance_match_ou_closed_form(self) -> None:
        theta, mu, sigma, x0, t = 1.5, 0.2, 0.3, 1.0, 1.0

        def drift(x: np.ndarray, _t: float) -> np.ndarray:
            return theta * (mu - x)

        def diffusion(x: np.ndarray, _t: float) -> np.ndarray:
            return np.full_like(x, sigma)

        paths = euler_maruyama(
            drift, diffusion, x0=x0, t0=0.0, t1=t, num_steps=200, num_paths=200_000, seed=RNG_SEED
        )
        terminal = paths[:, -1]

        expected_mean = mu + (x0 - mu) * np.exp(-theta * t)
        expected_variance = (sigma**2 / (2.0 * theta)) * (1.0 - np.exp(-2.0 * theta * t))

        _assert_within_k_standard_errors(terminal, expected_mean, label="Euler-Maruyama OU mean")
        sample_variance = float(np.var(terminal, ddof=1))
        approx_se = expected_variance * np.sqrt(2.0 / terminal.size)
        assert abs(sample_variance - expected_variance) < 6.0 * approx_se
