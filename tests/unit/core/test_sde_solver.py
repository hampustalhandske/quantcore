"""Tests for the Euler-Maruyama SDE solver and GBM path simulator.

Reference: Kloeden & Platen (1992), Ch. 9 — Euler-Maruyama scheme

    X_{k+1} = X_k + a(X_k, t_k) * dt + b(X_k, t_k) * dW_k,  dW_k ~ N(0, dt)

`simulate_gbm_paths` specializes this to a(S, t) = r * S, b(S, t) = sigma * S.

Note: because Euler-Maruyama is a *discretization* of the continuous SDE, the
degenerate zero-volatility GBM path is exactly the discrete compounding
recursion S_{k+1} = S_k * (1 + r * dt), NOT the continuous closed-form
S_0 * exp(r * T). Tests below check against the discrete recursion, which is
exact regardless of step count, rather than the continuous solution (which
Euler-Maruyama only approximates).
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.core.sde_solver import euler_maruyama, simulate_gbm_paths

# ---------------------------------------------------------------------------
# euler_maruyama: shape, initial condition, reproducibility
# ---------------------------------------------------------------------------


def _zero_drift(x: np.ndarray, _t: float) -> np.ndarray:
    return np.zeros_like(x)


def _unit_diffusion(x: np.ndarray, _t: float) -> np.ndarray:
    return np.ones_like(x)


def test_euler_maruyama_output_shape() -> None:
    paths = euler_maruyama(
        drift=_zero_drift,
        diffusion=_unit_diffusion,
        x0=0.0,
        t0=0.0,
        t1=1.0,
        num_steps=10,
        num_paths=5,
        seed=1,
    )
    assert paths.shape == (5, 11)


def test_euler_maruyama_initial_column_equals_x0() -> None:
    x0 = 3.5
    paths = euler_maruyama(
        drift=_zero_drift,
        diffusion=_unit_diffusion,
        x0=x0,
        t0=0.0,
        t1=1.0,
        num_steps=8,
        num_paths=4,
        seed=1,
    )
    assert np.allclose(paths[:, 0], x0)


def test_euler_maruyama_reproducible_with_seed() -> None:
    kwargs = dict(
        drift=_zero_drift,
        diffusion=_unit_diffusion,
        x0=1.0,
        t0=0.0,
        t1=1.0,
        num_steps=10,
        num_paths=5,
        seed=99,
    )
    paths1 = euler_maruyama(**kwargs)
    paths2 = euler_maruyama(**kwargs)
    assert np.array_equal(paths1, paths2)


def test_euler_maruyama_zero_diffusion_is_deterministic_ode_path() -> None:
    # dX = c dt, b = 0 -> X_k = x0 + c * k * dt, exactly, for every path.
    c = 2.0
    x0 = 5.0
    t0, t1, num_steps = 0.0, 2.0, 4
    dt = (t1 - t0) / num_steps

    def const_drift(x: np.ndarray, _t: float) -> np.ndarray:
        return np.full_like(x, c)

    def zero_diffusion(x: np.ndarray, _t: float) -> np.ndarray:
        return np.zeros_like(x)

    paths = euler_maruyama(
        drift=const_drift,
        diffusion=zero_diffusion,
        x0=x0,
        t0=t0,
        t1=t1,
        num_steps=num_steps,
        num_paths=3,
        seed=1,
    )
    expected = x0 + c * dt * np.arange(num_steps + 1)
    for row in range(3):
        assert np.allclose(paths[row], expected)


def test_euler_maruyama_num_steps_one_boundary() -> None:
    paths = euler_maruyama(
        drift=_zero_drift,
        diffusion=_unit_diffusion,
        x0=0.0,
        t0=0.0,
        t1=1.0,
        num_steps=1,
        num_paths=3,
        seed=1,
    )
    assert paths.shape == (3, 2)


@pytest.mark.parametrize("num_steps", [0, -1])
def test_euler_maruyama_non_positive_num_steps_raises(num_steps: int) -> None:
    # num_steps must be >= 1 whenever t1 > t0: dt = (t1 - t0) / num_steps is
    # undefined (division by zero) for num_steps=0, and a negative step
    # count is meaningless.
    with pytest.raises(ValueError):
        euler_maruyama(
            drift=_zero_drift,
            diffusion=_unit_diffusion,
            x0=0.0,
            t0=0.0,
            t1=1.0,
            num_steps=num_steps,
            num_paths=3,
            seed=1,
        )


def test_euler_maruyama_negative_num_paths_raises() -> None:
    # A negative path count is not representable as an array dimension and
    # has no meaningful interpretation.
    with pytest.raises(ValueError):
        euler_maruyama(
            drift=_zero_drift,
            diffusion=_unit_diffusion,
            x0=0.0,
            t0=0.0,
            t1=1.0,
            num_steps=10,
            num_paths=-1,
            seed=1,
        )


def test_euler_maruyama_end_before_start_raises() -> None:
    with pytest.raises(ValueError):
        euler_maruyama(
            drift=_zero_drift,
            diffusion=_unit_diffusion,
            x0=0.0,
            t0=1.0,
            t1=0.0,
            num_steps=10,
            num_paths=3,
            seed=1,
        )


# ---------------------------------------------------------------------------
# simulate_gbm_paths
# ---------------------------------------------------------------------------


def test_gbm_output_shape() -> None:
    paths = simulate_gbm_paths(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_steps=20,
        num_paths=10,
        seed=1,
    )
    assert paths.shape == (10, 21)


def test_gbm_initial_column_equals_spot() -> None:
    spot = 123.4
    paths = simulate_gbm_paths(
        spot=spot,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_steps=10,
        num_paths=6,
        seed=1,
    )
    assert np.allclose(paths[:, 0], spot)


def test_gbm_reproducible_with_seed() -> None:
    kwargs = dict(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_steps=20,
        num_paths=10,
        seed=2024,
    )
    paths1 = simulate_gbm_paths(**kwargs)
    paths2 = simulate_gbm_paths(**kwargs)
    assert np.array_equal(paths1, paths2)


def test_gbm_paths_are_strictly_positive() -> None:
    # GBM is a strictly positive process (log-normal); Euler-Maruyama with a
    # reasonably fine grid should preserve this for realistic vol/horizon.
    paths = simulate_gbm_paths(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_steps=250,
        num_paths=500,
        seed=1,
    )
    assert np.all(paths > 0.0)


def test_gbm_zero_volatility_matches_discrete_compounding() -> None:
    # With sigma=0 there is no diffusion term, so the Euler-Maruyama
    # recursion collapses to the deterministic discrete compounding
    # S_{k+1} = S_k * (1 + r * dt) -- exact, for any num_steps.
    spot, rate, t, num_steps = 100.0, 0.05, 1.0, 12
    dt = t / num_steps
    paths = simulate_gbm_paths(
        spot=spot,
        rate=rate,
        volatility=0.0,
        time_to_maturity=t,
        num_steps=num_steps,
        num_paths=4,
        seed=1,
    )
    expected = spot * (1.0 + rate * dt) ** np.arange(num_steps + 1)
    for row in range(4):
        assert np.allclose(paths[row], expected)


def test_gbm_zero_time_to_maturity_stays_at_spot() -> None:
    # T=0 implies dt=0 for every step, so every column must equal spot
    # exactly regardless of volatility.
    spot = 50.0
    paths = simulate_gbm_paths(
        spot=spot,
        rate=0.05,
        volatility=0.3,
        time_to_maturity=0.0,
        num_steps=5,
        num_paths=3,
        seed=1,
    )
    assert np.allclose(paths, spot)


def test_gbm_num_paths_and_steps_equal_one_boundary() -> None:
    paths = simulate_gbm_paths(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        num_steps=1,
        num_paths=1,
        seed=1,
    )
    assert paths.shape == (1, 2)
    assert paths[0, 0] == pytest.approx(100.0)


def test_gbm_negative_num_paths_raises() -> None:
    with pytest.raises(ValueError):
        simulate_gbm_paths(
            spot=100.0,
            rate=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            num_steps=10,
            num_paths=-1,
            seed=1,
        )


@pytest.mark.parametrize("num_steps", [0, -1])
def test_gbm_non_positive_num_steps_raises(num_steps: int) -> None:
    with pytest.raises(ValueError):
        simulate_gbm_paths(
            spot=100.0,
            rate=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            num_steps=num_steps,
            num_paths=10,
            seed=1,
        )
