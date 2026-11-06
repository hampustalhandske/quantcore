"""Plain-Python/NumPy reference implementations used to check the Numba kernels.

Each function computes the same quantity as the Numba kernel it is named after
(GBM log-Euler paths, and the GARCH(1,1), EGARCH(1,1) and GJR-GARCH(1,1)
variance recursions) with straightforward loops, so the agreement tests can
compare the compiled kernels against an independent, easily read version.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from quantcore.risk.egarch import _E_ABS_Z


def _simulate_gbm_paths_numpy(
    spot: float,
    rate: float,
    volatility: float,
    dt: float,
    dw: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Plain NumPy reference for the GBM log-Euler path kernel.

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


def _garch_11_variance_numpy(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
    initial_variance: float = float("nan"),
) -> npt.NDArray[np.float64]:
    """Plain NumPy reference implementation of the GARCH(1,1) recursion.

    Plain-Python reference used to check the Numba kernel.
    """
    n = returns.shape[0]
    variance = np.empty(n, dtype=np.float64)

    # Seed the recursion with the model-implied long-run (unconditional)
    # variance omega / (1 - alpha - beta), the stationary variance of the
    # GARCH(1,1) process under alpha + beta < 1 (Bollerslev 1986). This is
    # not pinned down by the recursion formula itself, since sigma_t^2
    # depends on sigma_{t-1}^2 and there is no sigma_0^2 in the input data;
    # anchoring to the theoretical stationary variance (rather than, e.g.,
    # the sample variance of `returns`) keeps the series purely a function
    # of the given model parameters.
    variance[0] = omega / (1.0 - alpha - beta) if np.isnan(initial_variance) else initial_variance

    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]

    return variance


def _egarch_11_variance_numpy(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    initial_variance: float = float("nan"),
) -> npt.NDArray[np.float64]:
    """Plain NumPy reference implementation of the EGARCH(1,1) recursion.

    Plain-Python reference used to check the Numba kernel.
    """
    n = returns.shape[0]
    log_variance = np.empty(n, dtype=np.float64)
    log_variance[0] = (
        omega / (1.0 - beta) if np.isnan(initial_variance) else np.log(initial_variance)
    )

    for t in range(1, n):
        sigma_prev = np.sqrt(np.exp(log_variance[t - 1]))
        z_prev = returns[t - 1] / sigma_prev
        log_variance[t] = (
            omega
            + beta * log_variance[t - 1]
            + alpha * (np.abs(z_prev) - _E_ABS_Z)
            + gamma * z_prev
        )

    return log_variance


def _gjr_garch_11_variance_numpy(
    returns: npt.NDArray[np.float64],
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    initial_variance: float = float("nan"),
) -> npt.NDArray[np.float64]:
    """Plain NumPy reference implementation of the GJR-GARCH(1,1) recursion.

    Plain-Python reference used to check the Numba kernel.
    """
    n = returns.shape[0]
    variance = np.empty(n, dtype=np.float64)
    variance[0] = (
        omega / (1.0 - alpha - gamma / 2.0 - beta)
        if np.isnan(initial_variance)
        else initial_variance
    )

    for t in range(1, n):
        bad_news = 1.0 if returns[t - 1] < 0.0 else 0.0
        variance[t] = (
            omega
            + alpha * returns[t - 1] ** 2
            + gamma * returns[t - 1] ** 2 * bad_news
            + beta * variance[t - 1]
        )

    return variance
