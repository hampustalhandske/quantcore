"""Tests for the GBM path simulator.

Reference: Kloeden & Platen (1992), Ch. 9 — Euler-Maruyama scheme in log-price space.

The degenerate zero-volatility GBM path is exactly the discrete compounding
recursion S_{k+1} = S_k * (1 + r * dt), NOT the continuous closed-form
S_0 * exp(r * T); the tests check against the discrete recursion.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.core.sde_solver import simulate_gbm_paths

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


_GBM_ARGS = dict(
    spot=100.0, rate=0.05, volatility=0.2, time_to_maturity=1.0, num_steps=20, num_paths=10
)


@pytest.mark.parametrize("seed", [None, -1, 1.5, True, "1"])
def test_gbm_seed_must_be_a_non_negative_integer(seed: object) -> None:
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        simulate_gbm_paths(**_GBM_ARGS, seed=seed)  # type: ignore[arg-type]


def test_gbm_seed_is_required() -> None:
    with pytest.raises(TypeError, match="seed"):
        simulate_gbm_paths(**_GBM_ARGS)  # type: ignore[call-arg]
