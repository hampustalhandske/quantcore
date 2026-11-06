"""Ledoit-Wolf shrinkage covariance estimation.

Ledoit-Wolf shrinkage (Ledoit & Wolf 2004):
    Sigma_LW = (1 - alpha)*S + alpha*mu*I
    where S is the sample covariance (Theorem 1's derivation uses the
    population/biased normalization 1/T, not the unbiased 1/(T-1), so that
    alpha is Theorem 1's asymptotically-optimal value),
    mu = trace(S)/k is the shrinkage target's scale, and alpha in [0, 1] is
    the analytical shrinkage intensity from Theorem 1 of the paper.

References:
    Ledoit, O. and Wolf, M. (2004), "A Well-Conditioned Estimator for
    Large-Dimensional Covariance Matrices."
    See docs/REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _validate_returns(returns: npt.NDArray[np.float64]) -> None:
    if returns.size == 0:
        raise ValueError("returns must be non-empty")
    if returns.ndim != 2:
        raise ValueError("returns must be a 2D array of shape (T, k)")


def _validate_returns_and_enough_observations(returns: npt.NDArray[np.float64]) -> None:
    _validate_returns(returns)
    num_obs, num_assets = returns.shape
    if num_obs <= num_assets:
        raise ValueError("returns must have more observations (T) than assets (k)")


def ledoit_wolf_shrinkage(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the Ledoit-Wolf analytically-shrunk covariance matrix.

    Args:
        returns: Asset returns, shape (T, k).

    Returns:
        Shrunk covariance matrix, shape (k, k), shrunk toward a scaled
        identity target mu_hat * I.
    """
    _validate_returns_and_enough_observations(returns)
    return _ledoit_wolf_shrinkage(returns)


def _ledoit_wolf_shrinkage(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    num_obs, num_assets = returns.shape
    demeaned = returns - returns.mean(axis=0)
    # Theorem 1's own normalization is 1/T (biased/population); ddof=1 would make alpha,
    # and therefore Sigma_LW, a different and not Theorem-1-optimal estimator.
    sample = (demeaned.T @ demeaned) / num_obs
    mu_hat = float(np.trace(sample)) / num_assets
    target = mu_hat * np.eye(num_assets)

    d_squared = float(np.sum((sample - target) ** 2)) / num_assets

    # sum_t ||x_t x_t' - S||_F^2 = sum_t ||x_t||^4 - T*||S||_F^2, because
    # sum_t x_t' S x_t = T*||S||_F^2 for S = (1/T) sum_t x_t x_t'. Rounding can
    # leave a tiny negative value where the exact sum is zero.
    row_norms_squared = np.einsum("ij,ij->i", demeaned, demeaned)
    dispersion_sum = float(np.sum(row_norms_squared**2)) - num_obs * float(np.sum(sample**2))
    b_bar_squared = max(dispersion_sum, 0.0) / num_assets / num_obs**2

    b_squared = min(b_bar_squared, d_squared)
    if d_squared == 0.0:
        alpha = 0.0
    else:
        alpha = b_squared / d_squared

    result: npt.NDArray[np.float64] = alpha * target + (1.0 - alpha) * sample
    return result
