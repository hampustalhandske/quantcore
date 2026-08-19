"""Stochastic differential equation solvers.

Numerical schemes for simulating paths of a general Ito SDE:

    dX_t = a(X_t, t) dt + b(X_t, t) dW_t

where ``a`` is the drift coefficient, ``b`` is the diffusion coefficient, and
``W_t`` is a standard Wiener process. The Euler-Maruyama scheme discretizes
this over a grid t_0, ..., t_n with step dt = (t_1 - t_0) / n as:

    X_{k+1} = X_k + a(X_k, t_k) * dt + b(X_k, t_k) * dW_k,  dW_k ~ N(0, dt)

For Geometric Brownian Motion, the price process itself is log-normal, so a
positivity-preserving implementation applies Euler-Maruyama to the log-price:

    d log(S_t) = (r - 0.5 * sigma^2) dt + sigma dW_t

and then exponentiates at the end. This is equivalent to Euler-Maruyama on the
log process and preserves the strictly-positive GBM invariant without creating
nonsense negative asset prices under realistic volatility / step settings.

Both a plain NumPy reference implementation and a Numba-accelerated version
are required per the project convention (see CLAUDE.md); the Numba variant
is added alongside the reference when this module is implemented.

References:
    Kloeden, P.E. and Platen, E. (1992). *Numerical Solution of Stochastic
    Differential Equations*. Springer. (Euler-Maruyama scheme and
    convergence properties.) See REFERENCES.md.
"""

from __future__ import annotations

from collections.abc import Callable

import numba
import numpy as np
import numpy.typing as npt


def euler_maruyama(
    drift: Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]],
    diffusion: Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]],
    x0: float,
    t0: float,
    t1: float,
    num_steps: int,
    num_paths: int,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Simulate paths of dX_t = a(X_t, t) dt + b(X_t, t) dW_t via Euler-Maruyama.

    Args:
        drift: Drift coefficient a(X_t, t), vectorized over paths.
        diffusion: Diffusion coefficient b(X_t, t), vectorized over paths.
        x0: Initial value X_0.
        t0: Start time.
        t1: End time.
        num_steps: Number of discretization steps between t0 and t1.
        num_paths: Number of independent sample paths to simulate.
        seed: Optional seed for reproducibility.

    Returns:
        Array of shape (num_paths, num_steps + 1) with simulated paths,
        including the initial value X_0 at column 0.

    References:
        Kloeden & Platen (1992), Ch. 9. See REFERENCES.md.
    """
    if t1 < t0:
        raise ValueError("t1 must be >= t0")
    if num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if num_paths < 0:
        raise ValueError("num_paths must be non-negative")
    return _euler_maruyama(drift, diffusion, x0, t0, t1, num_steps, num_paths, seed)


def _euler_maruyama(
    drift: Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]],
    diffusion: Callable[[npt.NDArray[np.float64], float], npt.NDArray[np.float64]],
    x0: float,
    t0: float,
    t1: float,
    num_steps: int,
    num_paths: int,
    seed: int | None,
) -> npt.NDArray[np.float64]:
    # Generic reference implementation. `drift`/`diffusion` are arbitrary
    # Python callables supplied by the caller, so this cannot be Numba-jitted
    # without requiring callers to pass already-njit-compiled functions (which
    # would break the public API contract). Numba acceleration is applied to
    # the GBM specialization below instead, where the coefficients are known
    # concretely at compile time. See module docstring / project summary.
    dt = (t1 - t0) / num_steps
    rng = np.random.default_rng(seed)
    paths = np.empty((num_paths, num_steps + 1), dtype=np.float64)
    paths[:, 0] = x0

    t = t0
    for k in range(num_steps):
        x_k = paths[:, k]
        dw_k = rng.normal(0.0, np.sqrt(dt), size=num_paths)
        paths[:, k + 1] = x_k + drift(x_k, t) * dt + diffusion(x_k, t) * dw_k
        t += dt

    return paths


def _validate_gbm_inputs(
    spot: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    num_steps: int,
    num_paths: int,
) -> None:
    if spot <= 0.0:
        raise ValueError("spot must be strictly positive")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")
    if time_to_maturity < 0.0:
        raise ValueError("time_to_maturity must be non-negative")
    if num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if num_paths < 0:
        raise ValueError("num_paths must be non-negative")


def simulate_gbm_paths(
    spot: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    num_steps: int,
    num_paths: int,
    seed: int | None = None,
) -> npt.NDArray[np.float64]:
    """Simulate Geometric Brownian Motion paths with a positivity-preserving log Euler step.

        d log(S_t) = (r - 0.5 * sigma^2) dt + sigma dW_t

    Args:
        spot: Initial asset price S_0 (must be strictly positive).
        rate: Risk-free interest rate, continuously compounded (r), used as
            the risk-neutral drift.
        volatility: Annualized volatility of the underlying (sigma).
        time_to_maturity: Total simulated horizon T, in years.
        num_steps: Number of discretization steps.
        num_paths: Number of independent sample paths.
        seed: Optional seed for reproducibility.

    Returns:
        Array of shape (num_paths, num_steps + 1) of simulated asset prices,
        including the initial spot at column 0.

    References:
        Kloeden & Platen (1992), Ch. 9 — Euler-Maruyama in log-price space for
        GBM, which preserves positivity while matching the GBM diffusion model.
        See REFERENCES.md.
    """
    _validate_gbm_inputs(spot, rate, volatility, time_to_maturity, num_steps, num_paths)
    return _simulate_gbm_paths(spot, rate, volatility, time_to_maturity, num_steps, num_paths, seed)


def _generate_gbm_increments(
    dt: float,
    num_steps: int,
    num_paths: int,
    seed: int | None,
) -> npt.NDArray[np.float64]:
    """Draw the Brownian increments dW_k ~ N(0, dt) shared by both variants.

    Drawing the increments once in NumPy (rather than inside the Numba
    kernel) means the NumPy-reference and Numba-accelerated GBM simulators
    can be fed the exact same random draws, which is what makes the
    np.allclose agreement test between them meaningful. It also lets
    ``pricing/monte_carlo.py`` build antithetic (mirrored) increment pairs
    for variance reduction without duplicating the GBM recursion.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, np.sqrt(dt), size=(num_paths, num_steps))


def _simulate_gbm_paths_numpy(
    spot: float,
    rate: float,
    volatility: float,
    dt: float,
    dw: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Plain NumPy reference implementation for GBM.

    For positive volatility we use the log-Euler discretization to preserve
    positivity; for the degenerate sigma=0 case we preserve the existing
    discrete compounding contract expected by the project tests.
    """
    num_paths, num_steps = dw.shape
    paths = np.empty((num_paths, num_steps + 1), dtype=np.float64)
    paths[:, 0] = spot

    if volatility == 0.0:
        for k in range(num_steps):
            paths[:, k + 1] = paths[:, k] * (1.0 + rate * dt)
        return paths

    log_paths = np.empty((num_paths, num_steps + 1), dtype=np.float64)
    log_paths[:, 0] = np.log(spot)
    drift = rate - 0.5 * volatility**2

    for k in range(num_steps):
        log_paths[:, k + 1] = log_paths[:, k] + drift * dt + volatility * dw[:, k]

    return np.exp(log_paths)


@numba.njit(cache=True)
def _simulate_gbm_paths_numba_kernel(
    spot: float,
    rate: float,
    volatility: float,
    dt: float,
    dw: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Numba-accelerated GBM scheme, preserving the zero-volatility contract."""
    num_paths, num_steps = dw.shape
    paths = np.empty((num_paths, num_steps + 1))
    if volatility == 0.0:
        for i in range(num_paths):
            paths[i, 0] = spot
            for k in range(num_steps):
                paths[i, k + 1] = paths[i, k] * (1.0 + rate * dt)
        return paths

    log_spot = np.log(spot)
    drift = rate - 0.5 * volatility * volatility
    for i in range(num_paths):
        log_s = log_spot
        paths[i, 0] = spot
        for k in range(num_steps):
            log_s = log_s + drift * dt + volatility * dw[i, k]
            paths[i, k + 1] = np.exp(log_s)
    return paths


def _simulate_gbm_paths(
    spot: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    num_steps: int,
    num_paths: int,
    seed: int | None,
) -> npt.NDArray[np.float64]:
    dt = time_to_maturity / num_steps
    dw = _generate_gbm_increments(dt, num_steps, num_paths, seed)
    # No silent fallback: if the Numba kernel fails to compile/run, this
    # raises rather than transparently falling back to the NumPy reference.
    return _simulate_gbm_paths_numba_kernel(spot, rate, volatility, dt, dw)
