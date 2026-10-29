"""Tier-1/Tier-2 conditional-volatility models: EWMA, GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1).

The variance recursions start from `arch`'s backcast of the residuals, so they are compared with
`arch` from the first step.
"""

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from arch.univariate.mean import HARX, arch_model

import quantcore.risk.egarch as qc_egarch
import quantcore.risk.volatility as qc_volatility
from conformance.compare import (
    ArrayComparator,
    ConformanceTarget,
)

# --- shared: the arch model the variance recursions are compared with ---------------------------


def _build_arch_model(
    epsilon: npt.NDArray[np.float64], vol: Literal["GARCH", "EGARCH"], o: int = 0
) -> HARX:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return arch_model(epsilon, mean="Zero", vol=vol, p=1, o=o, q=1, dist="normal")


# --- EWMA -----------------------------------------------------------------------------------------


def _ewma_variance_quantcore(
    returns: npt.NDArray[np.float64], lambda_: float
) -> npt.NDArray[np.float64]:
    return np.asarray(qc_egarch.ewma_variance(returns, lambda_), dtype=np.float64)


def _ewma_variance_reference(
    returns: npt.NDArray[np.float64], lambda_: float
) -> npt.NDArray[np.float64]:
    """Hand recursion with the *same* seeding as quantcore: `sigma_0^2 = returns[0]**2`.

    `arch` has no standalone EWMA/RiskMetrics volatility filter with a lambda parameter exposed the
    same way, so this is a from-scratch implementation of the documented recursion, not a call into
    an oracle library -- it establishes only that quantcore's formula and seeding match the
    RiskMetrics definition it cites, not a second, independent numerical implementation.
    """
    n = returns.shape[0]
    variance = np.empty(n, dtype=np.float64)
    variance[0] = returns[0] ** 2
    for t in range(1, n):
        variance[t] = lambda_ * variance[t - 1] + (1.0 - lambda_) * returns[t - 1] ** 2
    return variance


_EWMA_VARIANCE = ConformanceTarget(
    name="ewma_variance",
    tier=2,
    oracle="hand recursion (RiskMetrics definition, identical seeding)",
    citation="J.P. Morgan/Reuters (1996), RiskMetrics -- Technical Document (4th ed.).",
    compare=ArrayComparator(),
    quantcore=_ewma_variance_quantcore,
    reference=_ewma_variance_reference,
)


# --- GARCH(1,1) variance ----------------------------------------------------------------------


def _garch_11_variance_quantcore(
    returns: npt.NDArray[np.float64], omega: float, alpha: float, beta: float
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_volatility.garch_11_variance(returns, omega, alpha, beta), dtype=np.float64
    )


def _garch_11_variance_reference(
    returns: npt.NDArray[np.float64], omega: float, alpha: float, beta: float
) -> npt.NDArray[np.float64]:
    model = _build_arch_model(returns, vol="GARCH")
    result = model.fix([omega, alpha, beta])
    variance: npt.NDArray[np.float64] = (
        np.asarray(result.conditional_volatility, dtype=np.float64) ** 2
    )
    return variance


_GARCH_11_VARIANCE = ConformanceTarget(
    name="garch_11_variance",
    tier=1,
    oracle="arch (conditional variance via arch_model(...).fix(params))",
    citation="Bollerslev, T. (1986), 'Generalized Autoregressive Conditional Heteroskedasticity.' "
    "Journal of Econometrics, 31(3), 307-327.",
    compare=ArrayComparator(),
    quantcore=_garch_11_variance_quantcore,
    reference=_garch_11_variance_reference,
)


# --- EGARCH(1,1) log-variance -------------------------------------------------------------------


def _egarch_11_variance_quantcore(
    returns: npt.NDArray[np.float64], omega: float, alpha: float, gamma: float, beta: float
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_egarch.egarch_11_variance(returns, omega, alpha, gamma, beta), dtype=np.float64
    )


def _egarch_11_variance_reference(
    returns: npt.NDArray[np.float64], omega: float, alpha: float, gamma: float, beta: float
) -> npt.NDArray[np.float64]:
    model = _build_arch_model(returns, vol="EGARCH", o=1)
    result = model.fix([omega, alpha, gamma, beta])
    log_variance: npt.NDArray[np.float64] = np.log(
        np.asarray(result.conditional_volatility, dtype=np.float64) ** 2
    )
    return log_variance


_EGARCH_11_VARIANCE = ConformanceTarget(
    name="egarch_11_variance",
    tier=2,
    oracle="arch (conditional variance via arch_model(vol='EGARCH').fix(params))",
    citation="Nelson, D.B. (1991), 'Conditional Heteroskedasticity in Asset Returns: A New "
    "Approach.' Econometrica, 59(2), 347-370.",
    compare=ArrayComparator(),
    quantcore=_egarch_11_variance_quantcore,
    reference=_egarch_11_variance_reference,
)


# --- GJR-GARCH(1,1) variance ---------------------------------------------------------------------


def _gjr_garch_11_variance_quantcore(
    returns: npt.NDArray[np.float64], omega: float, alpha: float, gamma: float, beta: float
) -> npt.NDArray[np.float64]:
    return np.asarray(
        qc_egarch.gjr_garch_11_variance(returns, omega, alpha, gamma, beta), dtype=np.float64
    )


def _gjr_garch_11_variance_reference(
    returns: npt.NDArray[np.float64], omega: float, alpha: float, gamma: float, beta: float
) -> npt.NDArray[np.float64]:
    model = _build_arch_model(returns, vol="GARCH", o=1)
    result = model.fix([omega, alpha, gamma, beta])
    variance: npt.NDArray[np.float64] = (
        np.asarray(result.conditional_volatility, dtype=np.float64) ** 2
    )
    return variance


_GJR_GARCH_11_VARIANCE = ConformanceTarget(
    name="gjr_garch_11_variance",
    tier=2,
    oracle="arch (conditional variance via arch_model(vol='GARCH', o=1).fix(params))",
    citation="Glosten, Jagannathan & Runkle (1993). Full citation in docs/REFERENCES.md.",
    compare=ArrayComparator(),
    quantcore=_gjr_garch_11_variance_quantcore,
    reference=_gjr_garch_11_variance_reference,
)


# --- fit_garch_11, egarch_fit, gjr_garch_fit --------------------------------------------------
#
# Each fitter is one `ConformanceTarget` returning `(log_likelihood, params)`: the likelihood
# alone is judged (the optimum is not unique); `params` are reported, never compared.

FitResult = tuple[float, npt.NDArray[np.float64]]
"""`(log_likelihood, params)`, as every fitter's quantcore adapter and oracle return it."""
