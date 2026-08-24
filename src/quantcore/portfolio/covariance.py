"""Covariance matrix estimation: sample, EWMA, and Ledoit-Wolf shrinkage.

EWMA covariance (RiskMetrics 1996):
    Sigma_t = lambda*Sigma_{t-1} + (1-lambda)*r_{t-1}*r_{t-1}^T
    Seed: Sigma_0 = sample covariance of the full series.

Ledoit-Wolf shrinkage (Ledoit & Wolf 2004):
    Sigma_LW = (1 - alpha)*S + alpha*mu*I
    where S is the sample covariance, mu = trace(S)/k is the shrinkage
    target's scale, and alpha in [0, 1] is the analytical shrinkage
    intensity from Theorem 1 of the paper.

References:
    Ledoit, O. and Wolf, M. (2004), "A Well-Conditioned Estimator for
    Large-Dimensional Covariance Matrices."
    J.P. Morgan/Reuters (1996), *RiskMetrics — Technical Document*.
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


def sample_covariance(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the sample covariance matrix of a returns panel.

    Args:
        returns: Asset returns, shape (T, k).

    Returns:
        Sample covariance matrix, shape (k, k), using ddof=1.
    """
    _validate_returns_and_enough_observations(returns)
    return _sample_covariance(returns)


def _sample_covariance(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    result: npt.NDArray[np.float64] = np.cov(returns.T, ddof=1)
    return np.atleast_2d(result)


def _validate_ewma_inputs(returns: npt.NDArray[np.float64], lambda_: float) -> None:
    _validate_returns(returns)
    if not (0.0 < lambda_ < 1.0):
        raise ValueError("lambda_ must be strictly between 0 and 1")


def ewma_covariance(
    returns: npt.NDArray[np.float64],
    lambda_: float,
) -> npt.NDArray[np.float64]:
    """Compute the terminal EWMA covariance matrix (RiskMetrics).

    Args:
        returns: Asset returns, shape (T, k).
        lambda_: Decay factor, 0 < lambda_ < 1 (e.g. 0.94 for daily data).

    Returns:
        The most recent EWMA covariance estimate, shape (k, k), seeded with
        the sample covariance of `returns`.
    """
    _validate_ewma_inputs(returns, lambda_)
    return _ewma_covariance(returns, lambda_)


def _ewma_covariance(
    returns: npt.NDArray[np.float64],
    lambda_: float,
) -> npt.NDArray[np.float64]:
    num_obs, _ = returns.shape
    if num_obs < 2:
        sigma: npt.NDArray[np.float64] = np.outer(returns[0], returns[0])
        return sigma
    sigma = np.atleast_2d(np.cov(returns.T, ddof=1)).astype(np.float64)
    for t in range(1, num_obs):
        r_prev = returns[t - 1]
        sigma = lambda_ * sigma + (1.0 - lambda_) * np.outer(r_prev, r_prev)
    return sigma


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
    sample = _sample_covariance(returns)
    mu_hat = float(np.trace(sample)) / num_assets
    target = mu_hat * np.eye(num_assets)

    demeaned = returns - returns.mean(axis=0)
    d_squared = float(np.sum((sample - target) ** 2)) / num_assets

    b_bar_squared = 0.0
    for t in range(num_obs):
        outer_t = np.outer(demeaned[t], demeaned[t])
        b_bar_squared += float(np.sum((outer_t - sample) ** 2)) / num_assets
    b_bar_squared /= num_obs**2

    b_squared = min(b_bar_squared, d_squared)
    if d_squared == 0.0:
        alpha = 0.0
    else:
        alpha = b_squared / d_squared

    result: npt.NDArray[np.float64] = alpha * target + (1.0 - alpha) * sample
    return result
