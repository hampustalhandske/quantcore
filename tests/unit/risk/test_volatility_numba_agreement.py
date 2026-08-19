"""Numba/NumPy numerical agreement for the GARCH(1,1) conditional variance
recursion.

Per CLAUDE.md: "Every numerically heavy function needs two versions: a plain
NumPy reference implementation and a Numba-accelerated one, with a test
asserting they agree within a numerical tolerance (np.allclose)."

``garch_11_variance`` is backed by two GARCH(1,1) recursion kernels, exactly
mirroring how ``simulate_gbm_paths`` in `core/sde_solver.py` colocates its
NumPy/Numba pair:

    - ``_garch_11_variance_numpy`` -- plain NumPy reference implementation
      (the pre-existing ``_garch_11_variance``, renamed).
    - ``_garch_11_variance_numba_kernel`` -- Numba-jitted kernel, used by the
      public ``garch_11_variance`` dispatcher (no silent fallback to the
      NumPy path, per CLAUDE.md's "no silent fallbacks" rule -- same
      precedent as ``_simulate_gbm_paths_numba_kernel``).

Both kernels take the identical `(returns, omega, alpha, beta)` inputs, so
agreement tests need no shared-random-draws plumbing (unlike the GBM kernel
pair, which shares pre-drawn Brownian increments) -- the recursion itself is
fully deterministic given its inputs.
"""

from __future__ import annotations

import numpy as np

from quantcore.risk.volatility import (
    _garch_11_variance_numba_kernel,
    _garch_11_variance_numpy,
    garch_11_variance,
)


def test_numba_and_numpy_garch_kernels_agree_on_hand_picked_series() -> None:
    returns = np.array([0.02, -0.01, 0.03, 0.00, -0.02, 0.01, 0.015, -0.005])
    omega, alpha, beta = 0.00002, 0.10, 0.85

    numpy_variance = _garch_11_variance_numpy(returns, omega, alpha, beta)
    numba_variance = _garch_11_variance_numba_kernel(returns, omega, alpha, beta)

    assert numpy_variance.shape == numba_variance.shape
    assert np.allclose(numpy_variance, numba_variance, rtol=1e-10, atol=1e-10)


def test_numba_and_numpy_garch_kernels_agree_across_parameterizations() -> None:
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0, 0.02, size=50)

    for omega, alpha, beta in [
        (0.00002, 0.10, 0.85),
        (0.00005, 0.05, 0.90),
        (0.0001, 0.20, 0.60),
        (1e-6, 0.08, 0.88),
        (0.0002, 0.0, 0.5),  # alpha == 0 boundary
        (0.0002, 0.5, 0.0),  # beta == 0 boundary
    ]:
        numpy_variance = _garch_11_variance_numpy(returns, omega, alpha, beta)
        numba_variance = _garch_11_variance_numba_kernel(returns, omega, alpha, beta)
        assert np.allclose(numpy_variance, numba_variance, rtol=1e-10, atol=1e-10)


def test_numba_and_numpy_garch_kernels_agree_on_long_series() -> None:
    # Longer series to catch any accumulation/divergence issues the two
    # recursions might exhibit only after many steps.
    rng = np.random.default_rng(2222)
    returns = rng.normal(0.0, 0.015, size=2000)
    omega, alpha, beta = 0.000015, 0.09, 0.87

    numpy_variance = _garch_11_variance_numpy(returns, omega, alpha, beta)
    numba_variance = _garch_11_variance_numba_kernel(returns, omega, alpha, beta)

    assert numpy_variance.shape == numba_variance.shape
    assert np.allclose(numpy_variance, numba_variance, rtol=1e-10, atol=1e-10)


def test_numba_kernel_matches_public_garch_11_variance() -> None:
    # The public garch_11_variance delegates to the Numba kernel; check that
    # feeding the same inputs through the NumPy reference recovers the same
    # result as the public function's output.
    returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.015, 0.008])
    omega, alpha, beta = 0.00003, 0.12, 0.8

    public_variance = garch_11_variance(returns, omega=omega, alpha=alpha, beta=beta)
    reference_variance = _garch_11_variance_numpy(returns, omega, alpha, beta)

    assert np.allclose(public_variance, reference_variance, rtol=1e-10, atol=1e-10)
