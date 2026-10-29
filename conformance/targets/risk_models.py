"""Tier-1/Tier-2 conditional-volatility models: EWMA, GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1).

The recursions are seeded at the model-implied long-run variance, whereas `arch` seeds from a
data-driven backcast; they converge, so conformance compares after a persistence-dependent
burn-in. Maximum-likelihood fitters are compared on the attained Gaussian log-likelihood under
one shared function, not on raw parameters (the likelihood is flat near its optimum); `params`
are reported alongside but never compared.
"""

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from arch.univariate.base import ARCHModelResult
from arch.univariate.mean import HARX, arch_model

import quantcore.risk.egarch as qc_egarch
import quantcore.risk.volatility as qc_volatility
from conformance.compare import (
    ArrayComparator,
    ConformanceTarget,
    Tolerance,
)

# --- shared: persistence-dependent burn-in, and the Gaussian log-likelihood evaluator -----------


@dataclass(frozen=True)
class _BurnInArrayComparator:
    """`Comparator[FloatArray]` that ignores a leading transient before comparing two arrays.

    Applies `ArrayComparator` to `value[burn_in_cap:]` (clipped to the length); `is_finite` still
    checks the whole array. The oracle's different initial-variance convention gives a geometrically
    decaying difference at the start. `burn_in_cap` is fixed at construction, generous for the
    persistence the conformance tests draw (`alpha + beta <= 0.9`).

    Attributes:
        burn_in_cap: Number of leading points to exclude, before clipping to the array's length.
    """

    burn_in_cap: int

    def is_finite(self, value: npt.NDArray[np.float64]) -> bool:
        return ArrayComparator().is_finite(value)

    def max_errors(
        self, actual: npt.NDArray[np.float64], expected: npt.NDArray[np.float64]
    ) -> tuple[float, float]:
        start = min(self.burn_in_cap, max(actual.shape[0] - 1, 0))
        return ArrayComparator().max_errors(actual[start:], expected[start:])

    def assert_close(
        self, actual: npt.NDArray[np.float64], expected: npt.NDArray[np.float64], tol: Tolerance
    ) -> None:
        start = min(self.burn_in_cap, max(actual.shape[0] - 1, 0))
        ArrayComparator().assert_close(actual[start:], expected[start:], tol)


_BURN_IN_CAP = 300


def _garch_gaussian_loglik(
    epsilon: npt.NDArray[np.float64], omega: float, alpha: float, beta: float
) -> float:
    """Gaussian log-likelihood of `epsilon` under the GARCH(1,1)/GJR-GARCH(1,1) variance path.

    Built from `garch_11_variance`/`gjr_garch_11_variance` (both conformance-tested against `arch`
    elsewhere in this module) rather than re-deriving the recursion, so the only new arithmetic is
    the standard Gaussian log-density sum -- the same formula quantcore's own fitter documents.
    """
    variance = qc_volatility.garch_11_variance(epsilon, omega, alpha, beta)
    return float(-0.5 * np.sum(np.log(2.0 * np.pi) + np.log(variance) + epsilon**2 / variance))


_STATIONARITY_CLIP_MARGIN = 1e-6
_MIN_OMEGA_CLIP = 1e-10


def _build_arch_model(
    epsilon: npt.NDArray[np.float64], vol: Literal["GARCH", "EGARCH"], o: int = 0
) -> HARX:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return arch_model(epsilon, mean="Zero", vol=vol, p=1, o=o, q=1, dist="normal")


def _fit_with_arch(
    epsilon: npt.NDArray[np.float64], vol: Literal["GARCH", "EGARCH"], o: int = 0
) -> ARCHModelResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _build_arch_model(epsilon, vol, o)
        return model.fit(disp="off", show_warning=False)


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
    compare=_BurnInArrayComparator(_BURN_IN_CAP),
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
    compare=_BurnInArrayComparator(_BURN_IN_CAP),
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
    compare=_BurnInArrayComparator(_BURN_IN_CAP),
    quantcore=_gjr_garch_11_variance_quantcore,
    reference=_gjr_garch_11_variance_reference,
)


# --- fit_garch_11, egarch_fit, gjr_garch_fit --------------------------------------------------
#
# Each fitter is one `ConformanceTarget` returning `(log_likelihood, params)`: the likelihood
# alone is judged (the optimum is not unique); `params` are reported, never compared.

FitResult = tuple[float, npt.NDArray[np.float64]]
"""`(log_likelihood, params)`, as every fitter's quantcore adapter and oracle return it."""
