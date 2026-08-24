"""Black-Litterman posterior expected returns and covariance.

    Implied equilibrium returns (He & Litterman 1999):
        Pi = risk_aversion * Sigma @ w_mkt

    Posterior distribution (Black & Litterman 1992):
        Sigma_BL = [(tau*Sigma)^{-1} + P^T Omega^{-1} P]^{-1}
        mu_BL    = Sigma_BL @ [(tau*Sigma)^{-1}*Pi + P^T*Omega^{-1}*Q]

    Computed here via the algebraically equivalent form that only requires
    solving a (v, v) system (v = number of views), avoiding an explicit
    inverse of Sigma or Omega:
        M        = P @ (tau*Sigma) @ P^T + Omega
        mu_BL    = Pi + (tau*Sigma) @ P^T @ solve(M, Q - P @ Pi)
        Sigma_BL = tau*Sigma - (tau*Sigma) @ P^T @ solve(M, P @ (tau*Sigma))

References:
    Black, F. and Litterman, R. (1992). "Global Portfolio Optimization."
    He, G. and Litterman, R. (1999). "The Intuition Behind Black-Litterman
    Model Portfolios." See docs/REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _validate_implied_equilibrium_returns_inputs(
    cov_matrix: npt.NDArray[np.float64],
    market_weights: npt.NDArray[np.float64],
    risk_aversion: float,
) -> None:
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be a square 2D matrix")
    if market_weights.shape != (cov_matrix.shape[0],):
        raise ValueError("market_weights must have shape (k,) matching cov_matrix")
    if risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be strictly positive")


def implied_equilibrium_returns(
    cov_matrix: npt.NDArray[np.float64],
    market_weights: npt.NDArray[np.float64],
    risk_aversion: float,
) -> npt.NDArray[np.float64]:
    """Reverse-optimize implied equilibrium excess returns.

    Args:
        cov_matrix: (k, k) covariance matrix of asset returns.
        market_weights: (k,) market-cap weights, summing to 1.
        risk_aversion: Market risk aversion coefficient (lambda).

    Returns:
        (k,) implied equilibrium excess returns Pi.
    """
    _validate_implied_equilibrium_returns_inputs(cov_matrix, market_weights, risk_aversion)
    return _implied_equilibrium_returns(cov_matrix, market_weights, risk_aversion)


def _implied_equilibrium_returns(
    cov_matrix: npt.NDArray[np.float64],
    market_weights: npt.NDArray[np.float64],
    risk_aversion: float,
) -> npt.NDArray[np.float64]:
    return risk_aversion * cov_matrix @ market_weights


def _validate_black_litterman_inputs(
    cov_matrix: npt.NDArray[np.float64],
    market_weights: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    omega: npt.NDArray[np.float64],
    risk_aversion: float,
    tau: float,
) -> None:
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be a square 2D matrix")
    k = cov_matrix.shape[0]
    if market_weights.shape != (k,):
        raise ValueError("market_weights must have shape (k,) matching cov_matrix")
    if risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be strictly positive")
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive")
    if P.ndim != 2 or P.shape[1] != k:
        raise ValueError("P must have shape (v, k) with k matching cov_matrix")
    v = P.shape[0]
    if Q.shape != (v,):
        raise ValueError("Q must have shape (v,) matching the number of rows in P")
    if omega.shape != (v, v):
        raise ValueError("omega must have shape (v, v) matching the number of views")


def black_litterman(
    cov_matrix: npt.NDArray[np.float64],
    market_weights: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    omega: npt.NDArray[np.float64],
    risk_aversion: float,
    tau: float = 0.025,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the Black-Litterman posterior expected returns and covariance.

    Args:
        cov_matrix: (k, k) covariance matrix of asset returns.
        market_weights: (k,) market-cap weights.
        P: (v, k) view matrix.
        Q: (v,) view expected returns.
        omega: (v, v) view uncertainty covariance.
        risk_aversion: Market risk aversion coefficient (lambda).
        tau: Scalar scaling the prior uncertainty (~1/T).

    Returns:
        Tuple (posterior_returns, posterior_cov) of shape (k,) and (k, k).
    """
    _validate_black_litterman_inputs(cov_matrix, market_weights, P, Q, omega, risk_aversion, tau)
    return _black_litterman(cov_matrix, market_weights, P, Q, omega, risk_aversion, tau)


def _black_litterman(
    cov_matrix: npt.NDArray[np.float64],
    market_weights: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    omega: npt.NDArray[np.float64],
    risk_aversion: float,
    tau: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    pi = _implied_equilibrium_returns(cov_matrix, market_weights, risk_aversion)
    tau_sigma = tau * cov_matrix

    p_tau_sigma = P @ tau_sigma
    m = p_tau_sigma @ P.T + omega

    posterior_returns = pi + tau_sigma @ P.T @ np.linalg.solve(m, Q - P @ pi)
    posterior_cov = tau_sigma - tau_sigma @ P.T @ np.linalg.solve(m, p_tau_sigma)

    return posterior_returns, posterior_cov
