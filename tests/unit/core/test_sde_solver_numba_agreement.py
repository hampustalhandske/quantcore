"""Numba/NumPy numerical agreement for the GBM Euler-Maruyama kernel.

Per CLAUDE.md: "Every numerically heavy function needs two versions: a plain
NumPy reference implementation and a Numba-accelerated one, with a test
asserting they agree within a numerical tolerance (np.allclose)."

``simulate_gbm_paths`` is backed by two GBM Euler-Maruyama kernels that share
the same pre-drawn Brownian increments:

    - ``_simulate_gbm_paths_numpy`` -- plain NumPy reference implementation.
    - ``_simulate_gbm_paths_numba_kernel`` -- Numba-jitted kernel, used by the
      public ``simulate_gbm_paths``/``_simulate_gbm_paths``.

Feeding both the identical increments array isolates the comparison to the
numerical recursion itself (not to any difference in random draws).
"""

from __future__ import annotations

import numpy as np

from quantcore.core.sde_solver import (
    _generate_gbm_increments,
    _simulate_gbm_paths_numba_kernel,
    _simulate_gbm_paths_numpy,
)


def test_numba_and_numpy_gbm_kernels_agree_on_shared_increments() -> None:
    spot, rate, volatility = 100.0, 0.05, 0.2
    num_steps, num_paths, time_to_maturity = 50, 200, 1.0
    dt = time_to_maturity / num_steps

    dw = _generate_gbm_increments(dt, num_steps, num_paths, seed=1234)

    numpy_paths = _simulate_gbm_paths_numpy(spot, rate, volatility, dt, dw)
    numba_paths = _simulate_gbm_paths_numba_kernel(spot, rate, volatility, dt, dw)

    assert numpy_paths.shape == numba_paths.shape
    assert np.allclose(numpy_paths, numba_paths, rtol=1e-10, atol=1e-10)


def test_numba_and_numpy_gbm_kernels_agree_across_parameterizations() -> None:
    dt = 0.01
    num_steps, num_paths = 30, 75

    for spot, rate, volatility, seed in [
        (50.0, 0.0, 0.1, 1),
        (200.0, 0.08, 0.4, 2),
        (10.0, -0.02, 0.5, 3),
    ]:
        dw = _generate_gbm_increments(dt, num_steps, num_paths, seed=seed)
        numpy_paths = _simulate_gbm_paths_numpy(spot, rate, volatility, dt, dw)
        numba_paths = _simulate_gbm_paths_numba_kernel(spot, rate, volatility, dt, dw)
        assert np.allclose(numpy_paths, numba_paths, rtol=1e-10, atol=1e-10)


def test_numba_kernel_matches_public_simulate_gbm_paths() -> None:
    # The public simulate_gbm_paths delegates to the Numba kernel; check that
    # feeding the same increments through the NumPy reference recovers the
    # same result as the public function's output.
    from quantcore.core.sde_solver import simulate_gbm_paths

    spot, rate, volatility = 100.0, 0.03, 0.25
    num_steps, num_paths, time_to_maturity = 40, 100, 0.5
    seed = 999
    dt = time_to_maturity / num_steps

    public_paths = simulate_gbm_paths(
        spot=spot,
        rate=rate,
        volatility=volatility,
        time_to_maturity=time_to_maturity,
        num_steps=num_steps,
        num_paths=num_paths,
        seed=seed,
    )

    dw = _generate_gbm_increments(dt, num_steps, num_paths, seed=seed)
    reference_paths = _simulate_gbm_paths_numpy(spot, rate, volatility, dt, dw)

    assert np.allclose(public_paths, reference_paths, rtol=1e-10, atol=1e-10)
