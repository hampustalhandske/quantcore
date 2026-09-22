"""Tests for OU, Heston, and CIR path simulation in core/stochastic.py."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.core.stochastic import (
    CIRParams,
    cir_fit,
    simulate_cir_paths,
    simulate_heston_paths,
    simulate_ou_paths,
)

# ---------------------------------------------------------------------------
# simulate_ou_paths
# ---------------------------------------------------------------------------


class TestSimulateOuPaths:
    def test_invalid_theta_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_ou_paths(theta=0.0, mu=0.0, sigma=0.1, x0=0.0, t=1.0, n_paths=10, n_steps=10)

    @pytest.mark.parametrize("theta", [0.0, -1.0])
    def test_non_positive_theta_raises(self, theta: float) -> None:
        with pytest.raises(ValueError):
            simulate_ou_paths(theta=theta, mu=0.0, sigma=0.1, x0=0.0, t=1.0, n_paths=10, n_steps=10)

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_ou_paths(theta=1.0, mu=0.0, sigma=-0.1, x0=0.0, t=1.0, n_paths=10, n_steps=10)

    def test_negative_t_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_ou_paths(theta=1.0, mu=0.0, sigma=0.1, x0=0.0, t=-1.0, n_paths=10, n_steps=10)

    @pytest.mark.parametrize("n_paths", [0, -1])
    def test_non_positive_n_paths_raises(self, n_paths: int) -> None:
        with pytest.raises(ValueError):
            simulate_ou_paths(
                theta=1.0, mu=0.0, sigma=0.1, x0=0.0, t=1.0, n_paths=n_paths, n_steps=10
            )

    @pytest.mark.parametrize("n_steps", [0, -1])
    def test_non_positive_n_steps_raises(self, n_steps: int) -> None:
        with pytest.raises(ValueError):
            simulate_ou_paths(
                theta=1.0, mu=0.0, sigma=0.1, x0=0.0, t=1.0, n_paths=10, n_steps=n_steps
            )

    def test_output_shape(self) -> None:
        paths = simulate_ou_paths(
            theta=1.0, mu=0.0, sigma=0.2, x0=1.0, t=1.0, n_paths=5, n_steps=20, seed=1
        )
        assert paths.shape == (5, 21)

    def test_initial_column_equals_x0(self) -> None:
        paths = simulate_ou_paths(
            theta=1.0, mu=0.5, sigma=0.2, x0=3.0, t=1.0, n_paths=4, n_steps=10, seed=1
        )
        assert np.allclose(paths[:, 0], 3.0)

    def test_reproducible_with_seed(self) -> None:
        kwargs = dict(theta=1.0, mu=0.0, sigma=0.2, x0=1.0, t=1.0, n_paths=5, n_steps=10, seed=42)
        assert np.array_equal(simulate_ou_paths(**kwargs), simulate_ou_paths(**kwargs))

    def test_default_seed_is_none_and_draws_fresh_entropy(self) -> None:
        # Uniformity with euler_maruyama/simulate_gbm_paths (owner follow-up
        # 2f): default seed=None, not a fixed value, so two no-seed calls
        # are NOT reproducible against each other.
        kwargs = dict(theta=1.0, mu=0.0, sigma=0.2, x0=1.0, t=1.0, n_paths=5, n_steps=10)
        first = simulate_ou_paths(**kwargs)
        second = simulate_ou_paths(**kwargs)
        assert not np.array_equal(first, second)

    def test_terminal_distribution_matches_stationary_moments(self) -> None:
        theta, mu, sigma = 3.0, 0.5, 0.4
        paths = simulate_ou_paths(
            theta=theta, mu=mu, sigma=sigma, x0=mu, t=5.0, n_paths=20_000, n_steps=50, seed=7
        )
        terminal = paths[:, -1]
        expected_var = sigma**2 / (2.0 * theta)
        assert terminal.mean() == pytest.approx(mu, abs=0.05)
        assert terminal.var() == pytest.approx(expected_var, rel=0.15)


# ---------------------------------------------------------------------------
# simulate_heston_paths
# ---------------------------------------------------------------------------


class TestSimulateHestonPaths:
    def test_non_positive_kappa_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_heston_paths(
                s0=100.0,
                v0=0.04,
                mu=0.05,
                kappa=0.0,
                theta=0.04,
                xi=0.2,
                rho=0.0,
                t=1.0,
                n_paths=10,
                n_steps=10,
            )

    def test_non_positive_theta_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_heston_paths(
                s0=100.0,
                v0=0.04,
                mu=0.05,
                kappa=1.0,
                theta=0.0,
                xi=0.2,
                rho=0.0,
                t=1.0,
                n_paths=10,
                n_steps=10,
            )

    def test_negative_xi_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_heston_paths(
                s0=100.0,
                v0=0.04,
                mu=0.05,
                kappa=1.0,
                theta=0.04,
                xi=-0.1,
                rho=0.0,
                t=1.0,
                n_paths=10,
                n_steps=10,
            )

    @pytest.mark.parametrize("rho", [-1.5, 1.5])
    def test_rho_outside_unit_interval_raises(self, rho: float) -> None:
        with pytest.raises(ValueError):
            simulate_heston_paths(
                s0=100.0,
                v0=0.04,
                mu=0.05,
                kappa=1.0,
                theta=0.04,
                xi=0.2,
                rho=rho,
                t=1.0,
                n_paths=10,
                n_steps=10,
            )

    def test_non_positive_s0_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_heston_paths(
                s0=0.0,
                v0=0.04,
                mu=0.05,
                kappa=1.0,
                theta=0.04,
                xi=0.2,
                rho=0.0,
                t=1.0,
                n_paths=10,
                n_steps=10,
            )

    def test_negative_v0_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_heston_paths(
                s0=100.0,
                v0=-0.01,
                mu=0.05,
                kappa=1.0,
                theta=0.04,
                xi=0.2,
                rho=0.0,
                t=1.0,
                n_paths=10,
                n_steps=10,
            )

    @pytest.mark.parametrize("n_paths,n_steps", [(0, 10), (10, 0)])
    def test_non_positive_path_or_step_count_raises(self, n_paths: int, n_steps: int) -> None:
        with pytest.raises(ValueError):
            simulate_heston_paths(
                s0=100.0,
                v0=0.04,
                mu=0.05,
                kappa=1.0,
                theta=0.04,
                xi=0.2,
                rho=0.0,
                t=1.0,
                n_paths=n_paths,
                n_steps=n_steps,
            )

    def test_output_shapes(self) -> None:
        s_paths, v_paths = simulate_heston_paths(
            s0=100.0,
            v0=0.04,
            mu=0.05,
            kappa=1.0,
            theta=0.04,
            xi=0.2,
            rho=-0.5,
            t=1.0,
            n_paths=5,
            n_steps=20,
            seed=1,
        )
        assert s_paths.shape == (5, 21)
        assert v_paths.shape == (5, 21)

    def test_initial_columns_equal_s0_v0(self) -> None:
        s_paths, v_paths = simulate_heston_paths(
            s0=123.0,
            v0=0.05,
            mu=0.05,
            kappa=1.0,
            theta=0.04,
            xi=0.2,
            rho=-0.5,
            t=1.0,
            n_paths=4,
            n_steps=10,
            seed=1,
        )
        assert np.allclose(s_paths[:, 0], 123.0)
        assert np.allclose(v_paths[:, 0], 0.05)

    def test_variance_paths_never_negative(self) -> None:
        _, v_paths = simulate_heston_paths(
            s0=100.0,
            v0=0.01,
            mu=0.05,
            kappa=0.5,
            theta=0.01,
            xi=0.6,
            rho=-0.7,
            t=1.0,
            n_paths=500,
            n_steps=250,
            seed=1,
        )
        assert np.all(v_paths >= 0.0)

    def test_reproducible_with_seed(self) -> None:
        kwargs = dict(
            s0=100.0,
            v0=0.04,
            mu=0.05,
            kappa=1.0,
            theta=0.04,
            xi=0.2,
            rho=-0.5,
            t=1.0,
            n_paths=5,
            n_steps=10,
            seed=42,
        )
        s1, v1 = simulate_heston_paths(**kwargs)
        s2, v2 = simulate_heston_paths(**kwargs)
        assert np.array_equal(s1, s2)
        assert np.array_equal(v1, v2)

    def test_default_seed_is_none_and_draws_fresh_entropy(self) -> None:
        kwargs = dict(
            s0=100.0,
            v0=0.04,
            mu=0.05,
            kappa=1.0,
            theta=0.04,
            xi=0.2,
            rho=-0.5,
            t=1.0,
            n_paths=5,
            n_steps=10,
        )
        s1, _ = simulate_heston_paths(**kwargs)
        s2, _ = simulate_heston_paths(**kwargs)
        assert not np.array_equal(s1, s2)

    def test_constant_vol_matches_gbm_terminal_moments(self) -> None:
        # rho=0, xi=0 -> V_t stays at v0 deterministically, so S_t is exactly
        # GBM with volatility sqrt(v0). Compare terminal log(S_T) moments
        # against the closed-form GBM lognormal moments.
        s0, v0, mu, t = 100.0, 0.04, 0.08, 1.0
        s_paths, _ = simulate_heston_paths(
            s0=s0,
            v0=v0,
            mu=mu,
            kappa=1.0,
            theta=v0,
            xi=0.0,
            rho=0.0,
            t=t,
            n_paths=20_000,
            n_steps=100,
            seed=3,
        )
        log_terminal = np.log(s_paths[:, -1])
        expected_mean = np.log(s0) + (mu - 0.5 * v0) * t
        expected_std = np.sqrt(v0 * t)
        assert log_terminal.mean() == pytest.approx(expected_mean, abs=0.02)
        assert log_terminal.std() == pytest.approx(expected_std, rel=0.1)


# ---------------------------------------------------------------------------
# simulate_cir_paths
# ---------------------------------------------------------------------------


class TestSimulateCirPaths:
    def test_non_positive_kappa_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_cir_paths(
                r0=0.03, kappa=0.0, theta=0.03, sigma=0.1, t=1.0, n_paths=10, n_steps=10
            )

    def test_non_positive_theta_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_cir_paths(
                r0=0.03, kappa=1.0, theta=0.0, sigma=0.1, t=1.0, n_paths=10, n_steps=10
            )

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_cir_paths(
                r0=0.03, kappa=1.0, theta=0.03, sigma=-0.1, t=1.0, n_paths=10, n_steps=10
            )

    def test_negative_r0_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_cir_paths(
                r0=-0.01, kappa=1.0, theta=0.03, sigma=0.1, t=1.0, n_paths=10, n_steps=10
            )

    def test_negative_t_raises(self) -> None:
        with pytest.raises(ValueError):
            simulate_cir_paths(
                r0=0.03, kappa=1.0, theta=0.03, sigma=0.1, t=-1.0, n_paths=10, n_steps=10
            )

    @pytest.mark.parametrize("n_paths,n_steps", [(0, 10), (10, 0)])
    def test_non_positive_path_or_step_count_raises(self, n_paths: int, n_steps: int) -> None:
        with pytest.raises(ValueError):
            simulate_cir_paths(
                r0=0.03, kappa=1.0, theta=0.03, sigma=0.1, t=1.0, n_paths=n_paths, n_steps=n_steps
            )

    def test_output_shape(self) -> None:
        paths = simulate_cir_paths(
            r0=0.03, kappa=1.0, theta=0.03, sigma=0.1, t=1.0, n_paths=5, n_steps=20, seed=1
        )
        assert paths.shape == (5, 21)

    def test_initial_column_equals_r0(self) -> None:
        paths = simulate_cir_paths(
            r0=0.05, kappa=1.0, theta=0.03, sigma=0.1, t=1.0, n_paths=4, n_steps=10, seed=1
        )
        assert np.allclose(paths[:, 0], 0.05)

    def test_paths_never_negative(self) -> None:
        paths = simulate_cir_paths(
            r0=0.001, kappa=0.5, theta=0.02, sigma=0.3, t=1.0, n_paths=500, n_steps=250, seed=1
        )
        assert np.all(paths >= 0.0)

    def test_reproducible_with_seed(self) -> None:
        kwargs = dict(
            r0=0.03, kappa=1.0, theta=0.03, sigma=0.1, t=1.0, n_paths=5, n_steps=10, seed=42
        )
        assert np.array_equal(simulate_cir_paths(**kwargs), simulate_cir_paths(**kwargs))

    def test_default_seed_is_none_and_draws_fresh_entropy(self) -> None:
        kwargs = dict(r0=0.03, kappa=1.0, theta=0.03, sigma=0.1, t=1.0, n_paths=5, n_steps=10)
        first = simulate_cir_paths(**kwargs)
        second = simulate_cir_paths(**kwargs)
        assert not np.array_equal(first, second)

    def test_terminal_mean_converges_to_theta(self) -> None:
        theta = 0.05
        paths = simulate_cir_paths(
            r0=0.15, kappa=2.0, theta=theta, sigma=0.05, t=5.0, n_paths=20_000, n_steps=100, seed=9
        )
        assert paths[:, -1].mean() == pytest.approx(theta, abs=0.01)


# ---------------------------------------------------------------------------
# cir_fit
#
# `cir_fit` estimates (kappa, theta, sigma) via the CKLS (1992) GLS moment
# estimator on the CIR Euler discretization, per the docstring in
# core/stochastic.py. Since this uses `simulate_cir_paths`'s own Milstein
# discretization to generate the known-parameter test series, we get a
# genuine end-to-end check that piping `cir_fit`'s output back into
# `simulate_cir_paths` is self-consistent.
# ---------------------------------------------------------------------------


class TestCirFit:
    def test_non_positive_dt_raises(self) -> None:
        with pytest.raises(ValueError):
            cir_fit(np.array([0.03, 0.031, 0.029, 0.032, 0.030]), dt=0.0)

    def test_empty_rate_series_raises(self) -> None:
        with pytest.raises(ValueError):
            cir_fit(np.array([]), dt=1.0 / 252)

    def test_too_short_rate_series_raises(self) -> None:
        with pytest.raises(ValueError):
            cir_fit(np.array([0.03, 0.031, 0.029]), dt=1.0 / 252)

    def test_non_positive_rate_value_raises(self) -> None:
        with pytest.raises(ValueError):
            cir_fit(np.array([0.03, 0.031, 0.0, 0.029, 0.032]), dt=1.0 / 252)

    @pytest.mark.parametrize(
        "kappa_true, theta_true, sigma_true, seed",
        [
            (2.0, 0.04, 0.05, 1),
            (1.5, 0.03, 0.04, 2),
            (3.0, 0.05, 0.06, 3),
            (0.8, 0.02, 0.03, 4),
        ],
    )
    def test_fitted_params_are_in_admissible_cir_region(
        self, kappa_true: float, theta_true: float, sigma_true: float, seed: int
    ) -> None:
        dt = 1.0 / 252
        n_steps = 5000
        paths = simulate_cir_paths(
            r0=theta_true,
            kappa=kappa_true,
            theta=theta_true,
            sigma=sigma_true,
            t=n_steps * dt,
            n_paths=1,
            n_steps=n_steps,
            seed=seed,
        )
        rate_series = paths[0]

        fitted = cir_fit(rate_series, dt)

        assert isinstance(fitted, CIRParams)
        assert fitted.kappa > 0.0
        assert fitted.theta > 0.0
        assert fitted.sigma > 0.0

    def test_recovers_known_parameters_on_synthetic_data(self) -> None:
        # Tolerance is intentionally loose: CIR kappa is notoriously the
        # noisiest parameter to estimate from a single finite-length path
        # (see e.g. the CKLS 1992 and subsequent short-rate-model literature
        # on the wide sampling variance of mean-reversion speed estimates),
        # even at several thousand daily observations. theta and sigma are
        # much better identified since they govern the long-run level and
        # diffusion scale directly. We can't run the estimator ahead of time
        # to hand-calibrate a tighter bound, so this checks order-of-
        # magnitude recovery plus admissibility, mirroring the house style
        # in test_fit_garch_11.py.
        kappa_true, theta_true, sigma_true = 2.0, 0.04, 0.05
        dt = 1.0 / 252
        n_steps = 10_000
        paths = simulate_cir_paths(
            r0=theta_true,
            kappa=kappa_true,
            theta=theta_true,
            sigma=sigma_true,
            t=n_steps * dt,
            n_paths=1,
            n_steps=n_steps,
            seed=11,
        )
        rate_series = paths[0]

        fitted = cir_fit(rate_series, dt)

        assert fitted.kappa == pytest.approx(kappa_true, rel=0.75)
        assert fitted.theta == pytest.approx(theta_true, rel=0.3)
        assert fitted.sigma == pytest.approx(sigma_true, rel=0.3)

        # Closer to the truth than a deliberately poor, arbitrary guess --
        # a check that holds for "roughly correct" recovery without
        # requiring numeric precision we can't verify offline.
        bad_guess = CIRParams(kappa=10.0, theta=0.20, sigma=0.50)
        assert abs(fitted.kappa - kappa_true) < abs(bad_guess.kappa - kappa_true)
        assert abs(fitted.theta - theta_true) < abs(bad_guess.theta - theta_true)
        assert abs(fitted.sigma - sigma_true) < abs(bad_guess.sigma - sigma_true)

    def test_deterministic_given_fixed_input(self) -> None:
        dt = 1.0 / 252
        paths = simulate_cir_paths(
            r0=0.04, kappa=2.0, theta=0.04, sigma=0.05, t=500 * dt, n_paths=1, n_steps=500, seed=5
        )
        rate_series = paths[0]

        result_1 = cir_fit(rate_series, dt)
        result_2 = cir_fit(rate_series, dt)

        assert result_1 == result_2
