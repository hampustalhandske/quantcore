"""Oracle tests: generator-driven price-path simulators vs their closed-form moments.

No third-party library draws these processes from the same variates, so there is no element-wise
oracle. The ground truth is the published closed-form moments of each process (citations per
class below, all in `docs/REFERENCES.md`), which the simulators are checked against within
k = 3 standard errors of the Monte Carlo estimate at a fixed seed, the default of
`tests/oracle/README.md`'s tolerance policy.

Every statistic is one number per path averaged over independent paths, so its standard error
is the sample standard deviation across paths over sqrt(n_paths), whatever the dependence
between steps inside a path.

Bias is kept apart from Monte Carlo error. Where the expectation of the estimator equals the
closed form exactly (the exact-transition simulators, and statistics that are unbiased by
construction), the estimate is compared with the closed form alone. Where it does not, the
difference is derived from the simulator's own recursion and passed as `bias`, so that the
k-standard-error band stays a statement about sampling noise only. Two sources of bias occur:

- the Heston scheme is a discretisation, so its moments differ from the continuous-time model's
  by a term of order dt;
- the exponential OU and regime-switching processes start from a fixed state instead of their
  stationary law, so averages over a window differ from the stationary moments by a transient
  that decays geometrically.

References:
    Black & Scholes (1973); Glasserman (2004), Chapter 3: geometric Brownian motion moments.
    Uhlenbeck & Ornstein (1930): OU transition mean, variance and autocorrelation.
    Heston (1993): the variance is a square-root diffusion with mean
    theta + (v0 - theta)*exp(-kappa*t). Lord, Koekkoek & van Dijk (2010): full-truncation scheme.
    Bollerslev (1986): GARCH(1,1) unconditional variance omega/(1 - alpha - beta).
    Hamilton (1989): two-state Markov chain, stationary probabilities.
    Merton (1976): jump-diffusion mean and log-return variance.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats

from quantcore.core import (
    simulate_exp_ou_price_paths,
    simulate_garch_t_price_paths,
    simulate_gbm_price_paths,
    simulate_heston_price_paths,
    simulate_jump_diffusion_price_paths,
    simulate_regime_switching_price_paths,
)

pytestmark = pytest.mark.oracle

DT = 1.0 / 252.0


def _assert_within_k_standard_errors(
    per_path: npt.NDArray[np.float64],
    expected: float,
    *,
    label: str,
    bias: float = 0.0,
    k: float = 3.0,
) -> None:
    """Assert the mean of a per-path statistic is within k standard errors of its expectation.

    Args:
        per_path: One value of the statistic per simulated path.
        expected: The closed-form moment the statistic estimates.
        label: Name of the check, for the failure message.
        bias: Known difference between the statistic's expectation and `expected`
            (discretisation, or a start away from the stationary law); zero when the statistic is
            unbiased. The band is centred on `expected + bias`.
        k: Width of the band in standard errors.
    """
    sample_mean = float(np.mean(per_path))
    standard_error = float(np.std(per_path, ddof=1)) / math.sqrt(per_path.size)
    deviation = sample_mean - (expected + bias)
    assert abs(deviation) < k * standard_error, (
        f"{label}: sample mean {sample_mean:.8g} is {deviation / standard_error:+.2f} standard "
        f"errors ({standard_error:.3g} each) from the closed-form value {expected:.8g} "
        f"(known bias {bias:.3g})"
    )


class TestGbmMoments:
    """One-step log-returns are iid Normal((mu - sigma^2/2)*dt, sigma^2*dt) and
    E[S_T] = s0*exp(mu*T).

    The step is the exact transition, so none of the three statistics is biased: the per-path
    mean log-return, the per-path sample variance with ddof=1 (unbiased for iid returns) and the
    terminal price.
    """

    S0, MU, SIGMA, N_STEPS = 100.0, 0.08, 0.25, 252

    @pytest.fixture(scope="class")
    def paths(self) -> npt.NDArray[np.float64]:
        return simulate_gbm_price_paths(
            np.random.default_rng(7), 50_000, self.N_STEPS, DT, s0=self.S0, mu=self.MU,
            sigma=self.SIGMA,
        )  # fmt: skip

    def test_log_return_mean_matches_closed_form(self, paths: npt.NDArray[np.float64]) -> None:
        log_returns = np.diff(np.log(paths), axis=1)
        _assert_within_k_standard_errors(
            log_returns.mean(axis=1),
            (self.MU - 0.5 * self.SIGMA**2) * DT,
            label="GBM mean one-step log-return",
        )

    def test_log_return_variance_matches_closed_form(self, paths: npt.NDArray[np.float64]) -> None:
        log_returns = np.diff(np.log(paths), axis=1)
        _assert_within_k_standard_errors(
            log_returns.var(axis=1, ddof=1),
            self.SIGMA**2 * DT,
            label="GBM one-step log-return variance",
        )

    def test_terminal_mean_matches_closed_form(self, paths: npt.NDArray[np.float64]) -> None:
        _assert_within_k_standard_errors(
            paths[:, -1],
            self.S0 * math.exp(self.MU * self.N_STEPS * DT),
            label="GBM terminal mean",
        )


class TestExpOuMoments:
    """X = log(S) is an OU process: stationary mean theta, stationary variance
    sigma^2/(2*kappa), lag-one autocorrelation exp(-kappa*dt).

    The step is the exact transition. The paths start at X_0 = theta rather than from the
    stationary law, so E[X_k] = theta at every step (no bias in the mean) while
    E[(X_k - theta)^2] = sigma^2/(2*kappa) * (1 - exp(-2*kappa*k*dt)) approaches the stationary
    variance from below. The first 200 of 2,000 steps are dropped and what remains of that
    transient, averaged over the window, is passed as the bias of the variance check (about
    5e-6 of the variance). Deviations are taken from the known theta, not from each path's own
    mean, which would bias the variance of a strongly autocorrelated series downwards.
    """

    S0, KAPPA, SIGMA, N_STEPS, BURN_IN = 100.0, 5.0, 0.3, 2_000, 200

    @pytest.fixture(scope="class")
    def deviation(self) -> npt.NDArray[np.float64]:
        """log(S) - theta after the burn-in, one row per path."""
        paths = simulate_exp_ou_price_paths(
            np.random.default_rng(11), 5_000, self.N_STEPS, DT, s0=self.S0, kappa=self.KAPPA,
            sigma=self.SIGMA,
        )  # fmt: skip
        return np.log(paths[:, self.BURN_IN :]) - math.log(self.S0)

    def test_stationary_mean_matches_closed_form(self, deviation: npt.NDArray[np.float64]) -> None:
        _assert_within_k_standard_errors(
            deviation.mean(axis=1) + math.log(self.S0),
            math.log(self.S0),
            label="exp-OU stationary mean of log-price",
        )

    def test_stationary_variance_matches_closed_form(
        self, deviation: npt.NDArray[np.float64]
    ) -> None:
        stationary_variance = self.SIGMA**2 / (2.0 * self.KAPPA)
        steps = np.arange(self.BURN_IN, self.N_STEPS + 1)
        transient = float(np.mean(np.exp(-2.0 * self.KAPPA * steps * DT)))
        _assert_within_k_standard_errors(
            (deviation**2).mean(axis=1),
            stationary_variance,
            bias=-stationary_variance * transient,
            label="exp-OU stationary variance of log-price",
        )

    def test_lag_one_autocorrelation_matches_closed_form(
        self, deviation: npt.NDArray[np.float64]
    ) -> None:
        # E[d_k * d_{k+1}] = rho * E[d_k^2] holds at every step, stationary or not, so the
        # per-path statistic sum(d_k * d_{k+1}) - rho * sum(d_k^2) has expectation exactly zero.
        # Testing it avoids the small-sample bias of a ratio of sums.
        rho = math.exp(-self.KAPPA * DT)
        cross = (deviation[:, :-1] * deviation[:, 1:]).sum(axis=1)
        square = (deviation[:, :-1] ** 2).sum(axis=1)
        estimate = float(cross.sum() / square.sum())
        assert estimate == pytest.approx(rho, abs=1e-3)
        _assert_within_k_standard_errors(
            cross - rho * square, 0.0, label="exp-OU lag-one autocovariance residual"
        )


class TestHestonMoments:
    """Heston (1993): the variance has mean theta + (v0 - theta)*exp(-kappa*t), so the expected
    average variance over [0, T] is A = theta + (v0 - theta)*(1 - exp(-kappa*T))/(kappa*T),
    E[log(S_T/s0)] = mu*T - A*T/2 and E[S_T] = s0*exp(mu*T).

    The full-truncation Euler scheme reproduces E[S_T] exactly: given the variance at the start
    of a step, the price ratio over the step is log-normal with mean exp(mu*dt). The other two
    moments carry a discretisation bias. While the variance stays non-negative the scheme's mean
    variance follows E[v_k] = theta + (v0 - theta)*(1 - kappa*dt)^k, a geometric decay at the
    Euler rate instead of exp(-kappa*dt), so the scheme's expected average variance is
    A_scheme = theta + (v0 - theta)*(1 - (1 - kappa*dt)^n)/(n*kappa*dt). `A_scheme - A` is about
    4e-4 of A at dt = 1/252 with the parameters below; it is first order in dt and is passed as
    the bias. (With 2*kappa*theta >= xi^2 the simulated variance is negative in a fraction of
    steps too small to move that mean at the precision of these tests.)
    """

    S0, MU, V0, KAPPA, THETA, XI, N_STEPS = 100.0, 0.05, 0.08, 2.0, 0.04, 0.3, 252

    @pytest.fixture(scope="class")
    def paths(self) -> npt.NDArray[np.float64]:
        return simulate_heston_price_paths(
            np.random.default_rng(13), 50_000, self.N_STEPS, DT, s0=self.S0, mu=self.MU,
            v0=self.V0, kappa=self.KAPPA, theta=self.THETA, xi=self.XI,
        )  # fmt: skip

    def _average_variance(self) -> tuple[float, float]:
        """(closed-form expected average variance A, the scheme's bias A_scheme - A)."""
        horizon = self.N_STEPS * DT
        continuous = self.THETA + (self.V0 - self.THETA) * (
            1.0 - math.exp(-self.KAPPA * horizon)
        ) / (self.KAPPA * horizon)
        euler_decay = 1.0 - self.KAPPA * DT
        scheme = self.THETA + (self.V0 - self.THETA) * (1.0 - euler_decay**self.N_STEPS) / (
            self.N_STEPS * self.KAPPA * DT
        )
        return continuous, scheme - continuous

    def test_terminal_mean_matches_closed_form(self, paths: npt.NDArray[np.float64]) -> None:
        _assert_within_k_standard_errors(
            paths[:, -1],
            self.S0 * math.exp(self.MU * self.N_STEPS * DT),
            label="Heston terminal mean",
        )

    def test_mean_log_return_matches_closed_form(self, paths: npt.NDArray[np.float64]) -> None:
        horizon = self.N_STEPS * DT
        average_variance, scheme_bias = self._average_variance()
        _assert_within_k_standard_errors(
            np.log(paths[:, -1] / self.S0),
            self.MU * horizon - 0.5 * average_variance * horizon,
            bias=-0.5 * scheme_bias * horizon,
            label="Heston mean terminal log-return",
        )

    def test_realised_average_variance_matches_closed_form(
        self, paths: npt.NDArray[np.float64]
    ) -> None:
        # The realised variance sum(r_k^2)/T estimates A with a second bias on top of the
        # scheme's: E[r_k^2] holds the squared drift ((mu - v/2)*dt)^2 as well as v*dt. Its
        # contribution is dt * mean(E[(mu - v/2)^2]) = dt * mean((mu - E[v]/2)^2 + Var[v]/4),
        # evaluated with the square-root diffusion's closed-form mean and variance. It is about
        # 3e-6, a few percent of one standard error.
        horizon = self.N_STEPS * DT
        average_variance, scheme_bias = self._average_variance()
        t = np.arange(self.N_STEPS) * DT
        decay = np.exp(-self.KAPPA * t)
        mean_v = self.THETA + (self.V0 - self.THETA) * decay
        var_v = (
            self.V0 * self.XI**2 / self.KAPPA * (decay - decay**2)
            + self.THETA * self.XI**2 / (2.0 * self.KAPPA) * (1.0 - decay) ** 2
        )
        squared_drift = DT * float(np.mean((self.MU - 0.5 * mean_v) ** 2 + 0.25 * var_v))
        log_returns = np.diff(np.log(paths), axis=1)
        _assert_within_k_standard_errors(
            (log_returns**2).sum(axis=1) / horizon,
            average_variance,
            bias=scheme_bias + squared_drift,
            label="Heston realised average variance",
        )

    def test_terminal_mean_holds_when_feller_condition_is_violated(self) -> None:
        # 2*kappa*theta = 0.01 << xi^2 = 2.25: the simulated variance is negative, and so
        # truncated, in a large share of steps. E[S_T] = s0*exp(mu*T) does not depend on the
        # variance dynamics and stays exact for the scheme.
        kappa, theta, xi = 0.5, 0.01, 1.5
        assert 2.0 * kappa * theta < xi**2
        paths = simulate_heston_price_paths(
            np.random.default_rng(13), 50_000, self.N_STEPS, DT, s0=self.S0, mu=self.MU, v0=0.04,
            kappa=kappa, theta=theta, xi=xi, rho=-0.7,
        )  # fmt: skip
        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0.0)
        _assert_within_k_standard_errors(
            paths[:, -1],
            self.S0 * math.exp(self.MU * self.N_STEPS * DT),
            label="Heston terminal mean, Feller condition violated",
        )


class TestGarchTMoments:
    """Bollerslev (1986): with h_0 at the unconditional variance, E[h_k] = omega/(1 - alpha -
    beta) at every step, and the log-returns r_k = mu*dt + sqrt(h_k)*z_k are serially
    uncorrelated with that variance and mean mu*dt.

    Both statistics are unbiased: the per-path mean, and the per-path sample variance with
    ddof=1 (unbiased for uncorrelated returns of constant variance). The standard error of the
    variance needs a finite fourth moment of the returns, which holds for these parameters:
    (alpha + beta)^2 + alpha^2*(kappa_z - 1) = 0.915 < 1 with innovation kurtosis
    kappa_z = 3*(nu - 2)/(nu - 4) = 6 at nu = 6.
    """

    MU, OMEGA, ALPHA, BETA, NU = 0.08, 5e-5, 0.05, 0.90, 6.0

    @pytest.fixture(scope="class")
    def log_returns(self) -> npt.NDArray[np.float64]:
        paths = simulate_garch_t_price_paths(
            np.random.default_rng(17), 50_000, 252, DT, mu=self.MU, omega=self.OMEGA,
            alpha=self.ALPHA, beta=self.BETA, nu=self.NU,
        )  # fmt: skip
        return np.diff(np.log(paths), axis=1)

    def test_log_return_mean_is_drift_times_dt(self, log_returns: npt.NDArray[np.float64]) -> None:
        _assert_within_k_standard_errors(
            log_returns.mean(axis=1), self.MU * DT, label="GARCH-t mean one-step log-return"
        )

    def test_log_return_variance_matches_unconditional_variance(
        self, log_returns: npt.NDArray[np.float64]
    ) -> None:
        _assert_within_k_standard_errors(
            log_returns.var(axis=1, ddof=1),
            self.OMEGA / (1.0 - self.ALPHA - self.BETA),
            label="GARCH-t one-step log-return variance",
        )

    def test_returns_are_heavy_tailed(self, log_returns: npt.NDArray[np.float64]) -> None:
        # The Student-t innovation alone has excess kurtosis 6/(nu - 4) = 3 and the variance
        # recursion adds to it. No standard error is attached: the sample kurtosis of returns
        # whose eighth moment is infinite has no finite variance, so only the sign is asserted.
        assert stats.kurtosis(log_returns.ravel(), fisher=True, bias=False) > 0.0


class TestRegimeSwitchingMoments:
    """Hamilton (1989): the chain's stationary probabilities are pi0 = p10/(p01 + p10) and
    pi1 = p01/(p01 + p10). With regime means m_s = (mu_s - sigma_s^2/2)*dt, a one-step
    log-return has stationary mean pi0*m0 + pi1*m1 and stationary variance
    pi0*sigma0^2*dt + pi1*sigma1^2*dt + pi0*pi1*(m0 - m1)^2.

    Within a regime the step is exact. Every path starts in regime 0, and the return of step k
    is in regime 1 with probability pi1*(1 - lam^k), lam = 1 - p01 - p10, which approaches pi1
    geometrically. The first 50 of 500 steps are dropped; the remaining shortfall of regime 1,
    averaged over the window, changes both moments by about 3e-5 relative and is passed as the
    bias. The variance is measured about the known stationary mean, not about each path's own
    mean, which would bias it because the regime makes returns autocorrelated.
    """

    MU0, SIGMA0, MU1, SIGMA1, P01, P10 = 0.10, 0.15, -0.20, 0.40, 0.02, 0.10
    N_STEPS, BURN_IN = 500, 50

    @pytest.fixture(scope="class")
    def log_returns(self) -> npt.NDArray[np.float64]:
        paths = simulate_regime_switching_price_paths(
            np.random.default_rng(19), 20_000, self.N_STEPS, DT, mu0=self.MU0, sigma0=self.SIGMA0,
            mu1=self.MU1, sigma1=self.SIGMA1, p01=self.P01, p10=self.P10,
        )  # fmt: skip
        return np.diff(np.log(paths), axis=1)[:, self.BURN_IN :]

    def _regime_one_probabilities(self) -> tuple[float, float]:
        """(stationary probability of regime 1, its average over the steps after the burn-in)."""
        pi1 = self.P01 / (self.P01 + self.P10)
        lam = 1.0 - self.P01 - self.P10
        steps = np.arange(self.BURN_IN + 1, self.N_STEPS + 1)
        return pi1, pi1 * (1.0 - float(np.mean(lam**steps)))

    def test_log_return_mean_matches_closed_form(
        self, log_returns: npt.NDArray[np.float64]
    ) -> None:
        m0 = (self.MU0 - 0.5 * self.SIGMA0**2) * DT
        m1 = (self.MU1 - 0.5 * self.SIGMA1**2) * DT
        pi1, window_p1 = self._regime_one_probabilities()
        _assert_within_k_standard_errors(
            log_returns.mean(axis=1),
            (1.0 - pi1) * m0 + pi1 * m1,
            bias=(window_p1 - pi1) * (m1 - m0),
            label="regime-switching mean one-step log-return",
        )

    def test_log_return_variance_matches_closed_form(
        self, log_returns: npt.NDArray[np.float64]
    ) -> None:
        m0 = (self.MU0 - 0.5 * self.SIGMA0**2) * DT
        m1 = (self.MU1 - 0.5 * self.SIGMA1**2) * DT
        pi1, window_p1 = self._regime_one_probabilities()
        pi0 = 1.0 - pi1
        stationary_mean = pi0 * m0 + pi1 * m1
        stationary_variance = (
            pi0 * self.SIGMA0**2 * DT + pi1 * self.SIGMA1**2 * DT + pi0 * pi1 * (m0 - m1) ** 2
        )
        # Second moment about the stationary mean in each regime; weighting the two by the
        # window's occupancy instead of the stationary one gives the estimator's expectation.
        about_mean_0 = self.SIGMA0**2 * DT + (m0 - stationary_mean) ** 2
        about_mean_1 = self.SIGMA1**2 * DT + (m1 - stationary_mean) ** 2
        _assert_within_k_standard_errors(
            ((log_returns - stationary_mean) ** 2).mean(axis=1),
            stationary_variance,
            bias=(window_p1 - pi1) * (about_mean_1 - about_mean_0),
            label="regime-switching one-step log-return variance",
        )


class TestJumpDiffusionMoments:
    """Merton (1976), with lambda the jump intensity, log jump sizes Normal(m, s^2) and
    c = exp(m + s^2/2) - 1: E[S_T] = s0*exp(mu*T),
    E[log(S_T/s0)] = (mu - sigma^2/2 - lambda*c + lambda*m)*T and
    Var[log(S_T/s0)] = (sigma^2 + lambda*(m^2 + s^2))*T.

    The step is the exact transition on the grid, and the variance is measured about the known
    mean, so no statistic is biased. Two regimes are covered: rare jumps, where almost every
    step has no jump or one, and many jumps per step, where the summed jump size
    Normal(N*m, N*s^2) is drawn with N well above 1 and the jump counts come from the sampler
    NumPy uses for Poisson means of 10 and more.
    """

    S0, MU, SIGMA = 100.0, 0.06, 0.2

    def _check(
        self,
        seed: int,
        n_paths: int,
        n_steps: int,
        jump_intensity: float,
        jump_mean_log: float,
        jump_std_log: float,
        regime: str,
    ) -> None:
        horizon = n_steps * DT
        paths = simulate_jump_diffusion_price_paths(
            np.random.default_rng(seed), n_paths, n_steps, DT, s0=self.S0, mu=self.MU,
            sigma=self.SIGMA, jump_intensity=jump_intensity, jump_mean_log=jump_mean_log,
            jump_std_log=jump_std_log,
        )  # fmt: skip
        terminal = paths[:, -1]
        _assert_within_k_standard_errors(
            terminal, self.S0 * math.exp(self.MU * horizon), label=f"jump terminal mean, {regime}"
        )

        compensator = math.exp(jump_mean_log + 0.5 * jump_std_log**2) - 1.0
        expected_mean = (
            self.MU - 0.5 * self.SIGMA**2 - jump_intensity * compensator
        ) * horizon + jump_intensity * jump_mean_log * horizon
        log_return = np.log(terminal / self.S0)
        _assert_within_k_standard_errors(
            log_return, expected_mean, label=f"jump mean terminal log-return, {regime}"
        )
        _assert_within_k_standard_errors(
            (log_return - expected_mean) ** 2,
            (self.SIGMA**2 + jump_intensity * (jump_mean_log**2 + jump_std_log**2)) * horizon,
            label=f"jump terminal log-return variance, {regime}",
        )

    def test_rare_jumps_match_closed_form(self) -> None:
        # lambda*dt = 0.006 jumps per step.
        self._check(23, 50_000, 252, 1.5, -0.02, 0.1, "rare jumps")

    def test_many_jumps_per_step_match_closed_form(self) -> None:
        # lambda*dt = 11.9 jumps per step.
        self._check(23, 20_000, 60, 3000.0, -0.001, 0.004, "many jumps per step")


class TestPricePathSimulatorsContract:
    """What the simulators do with input the closed forms above do not cover.

    Scale parameters, the grid and every drift or level parameter are validated
    (`tests/unit/core/test_price_paths.py`). A non-finite drift raises `ValueError`; a drift that
    overflows the prices raises `FloatingPointError`; one that underflows them returns `0.0`.
    """

    def test_non_finite_drift_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="mu"):
            simulate_gbm_price_paths(np.random.default_rng(0), 2, 5, DT, mu=float("nan"))

    def test_extreme_drift_overflow_raises_and_underflow_returns_zero(self) -> None:
        with pytest.raises(FloatingPointError, match="overflow"):
            simulate_gbm_price_paths(np.random.default_rng(0), 2, 5, DT, mu=1e6)
        down = simulate_gbm_price_paths(np.random.default_rng(0), 2, 5, DT, mu=-1e6)
        assert np.all(down[:, 1:] == 0.0)

    def test_single_path_single_step_follows_the_one_step_formula(self) -> None:
        mu, sigma = 0.07, 0.3
        z = float(np.random.default_rng(3).standard_normal())
        paths = simulate_gbm_price_paths(np.random.default_rng(3), 1, 1, DT, mu=mu, sigma=sigma)
        expected = 100.0 * math.exp((mu - 0.5 * sigma**2) * DT + sigma * math.sqrt(DT) * z)
        assert paths.shape == (1, 2)
        assert paths[0, 1] == pytest.approx(expected, rel=1e-14)

    def test_integer_and_numpy_scalar_arguments_give_float64_prices(self) -> None:
        plain = simulate_gbm_price_paths(np.random.default_rng(4), 3, 6, DT, s0=100.0, mu=1.0)
        other = simulate_gbm_price_paths(
            np.random.default_rng(4), np.int64(3), np.int32(6), np.float64(DT), s0=100, mu=1
        )
        assert other.dtype == np.float64
        np.testing.assert_array_equal(other, plain)
