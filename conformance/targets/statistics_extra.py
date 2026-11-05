"""Tier-1 statistics targets: OLS, Newey-West lags, Jarque-Bera, HAC t-test, PBO, effective N, CI.

OLS is two targets, `ols_coefficients` and `ols_r_squared` (centred R-squared with a constant
column, else uncentred). The Newey-West lag count is the rule of thumb
`floor(4 * (n/100)^(2/9))`, not the data-driven procedure. `jarque_bera` is checked
against `scipy.stats.jarque_bera`. `hac_t_test` is checked against a `statsmodels` constant-only
OLS with `cov_type="HAC"`, Bartlett kernel, `use_correction=True` and `use_t=True`.
`cscv_pbo` is checked against an independent implementation of the CSCV procedure of Bailey,
Borwein, Lopez de Prado and Zhu (2016) with `scipy.stats.rankdata` mid-ranks, and
`effective_n_eigen` against the trace identity `trace(C)^2 / ||C||_F^2`.
`stationary_bootstrap_ci` is checked against percentile and Efron (1987) BCa intervals computed by
hand from the same `arch` stationary-bootstrap replicates of the mean.
"""

import math
from itertools import combinations
from statistics import NormalDist
from typing import Literal

import numpy as np
import numpy.typing as npt
import statsmodels.api as sm
from arch.bootstrap import StationaryBootstrap
from scipy.stats import jarque_bera as _scipy_jarque_bera
from scipy.stats import rankdata as _rankdata
from scipy.stats import t as _student_t

import quantcore.statistics as qc_statistics
from conformance.compare import (
    ArrayComparator,
    ConformanceTarget,
    ScalarComparator,
    TupleComparator,
)

# OLS: two targets (coefficients and R-squared), as their comparators differ.


def _ols_coefficients_quantcore(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    coefficients, _residuals, _r_squared = qc_statistics.ols(y, x)
    result: npt.NDArray[np.float64] = coefficients
    return result


def _ols_coefficients_reference(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    fit = sm.OLS(y, x).fit()
    return np.asarray(fit.params, dtype=np.float64)


_OLS_COEFFICIENTS = ConformanceTarget(
    name="ols_coefficients",
    tier=1,
    oracle="statsmodels",
    citation="Gauss-Markov theorem; standard OLS estimator.",
    compare=ArrayComparator(),
    quantcore=_ols_coefficients_quantcore,
    reference=_ols_coefficients_reference,
)


def _ols_r_squared_quantcore(y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> float:
    _coefficients, _residuals, r_squared = qc_statistics.ols(y, x)
    return float(r_squared)


def _ols_r_squared_reference(y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> float:
    return float(sm.OLS(y, x).fit().rsquared)


_OLS_R_SQUARED = ConformanceTarget(
    name="ols_r_squared",
    tier=1,
    oracle="statsmodels",
    citation="Gauss-Markov theorem; standard OLS estimator's coefficient of determination.",
    compare=ScalarComparator(),
    quantcore=_ols_r_squared_quantcore,
    reference=_ols_r_squared_reference,
)


# --- Newey-West optimal lag rule of thumb --------------------------------------------------------


def _newey_west_optimal_lags_quantcore(n: int) -> float:
    return float(qc_statistics.newey_west_optimal_lags(np.zeros(n, dtype=np.float64)))


def _newey_west_optimal_lags_reference(n: int) -> float:
    return float(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


_NEWEY_WEST_OPTIMAL_LAGS = ConformanceTarget(
    name="newey_west_optimal_lags",
    tier=1,
    oracle="hand-computed (the rule of thumb itself)",
    citation=(
        "Newey, W.K. and West, K.D. (1994), 'Automatic Lag Selection in Covariance Matrix "
        "Estimation.' Review of Economic Studies, 61(4), 631-653."
    ),
    compare=ScalarComparator(),
    quantcore=_newey_west_optimal_lags_quantcore,
    reference=_newey_west_optimal_lags_reference,
)


# --- Ljung-Box ------------------------------------------------------------------------------------


# --- Jarque-Bera ----------------------------------------------------------------------------------


def _jarque_bera_quantcore(series: npt.NDArray[np.float64]) -> tuple[float, float]:
    statistic, p_value = qc_statistics.jarque_bera_test(series)
    return float(statistic), float(p_value)


def _jarque_bera_reference(series: npt.NDArray[np.float64]) -> tuple[float, float]:
    result = _scipy_jarque_bera(series)
    return float(result.statistic), float(result.pvalue)


_JARQUE_BERA = ConformanceTarget(
    name="jarque_bera",
    tier=1,
    oracle="scipy.stats.jarque_bera",
    citation=(
        "Jarque, C.M. and Bera, A.K. (1980), 'Efficient Tests for Normality, Homoscedasticity "
        "and Serial Independence of Regression Residuals.' Economics Letters, 6(3), 255-259; "
        "Jarque, C.M. and Bera, A.K. (1987), 'A Test for Normality of Observations and "
        "Regression Residuals.' International Statistical Review, 55(2), 163-172."
    ),
    compare=TupleComparator(ScalarComparator(), ScalarComparator()),
    quantcore=_jarque_bera_quantcore,
    reference=_jarque_bera_reference,
)


# --- HAC t-test -----------------------------------------------------------------------------------


def _hac_t_test_quantcore(
    x: npt.NDArray[np.float64],
    *,
    lags: int,
    alternative: Literal["two-sided", "greater", "less"],
    null: float,
) -> tuple[float, float, float]:
    result = qc_statistics.hac_t_test(x, lags=lags, alternative=alternative, null=null)
    return result.t, result.p_value, result.se


def _hac_t_test_reference(
    x: npt.NDArray[np.float64],
    *,
    lags: int,
    alternative: Literal["two-sided", "greater", "less"],
    null: float,
) -> tuple[float, float, float]:
    fit = sm.OLS(x, np.ones(x.size)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lags, "use_correction": True, "kernel": "bartlett"},
        use_t=True,
    )
    test = fit.t_test((np.ones((1, 1)), np.array([null])), use_t=True)
    t_stat = float(np.squeeze(test.tvalue))
    if alternative == "two-sided":
        p_value = float(np.squeeze(test.pvalue))
    else:
        sign = 1.0 if alternative == "greater" else -1.0
        p_value = float(_student_t.sf(sign * t_stat, fit.df_resid))
    return t_stat, p_value, float(fit.bse[0])


_HAC_T_TEST = ConformanceTarget(
    name="hac_t_test",
    tier=1,
    oracle="statsmodels OLS cov_type='HAC'",
    citation=(
        "Newey, W.K. and West, K.D. (1987), 'A Simple, Positive Semi-Definite, Heteroskedasticity "
        "and Autocorrelation Consistent Covariance Matrix.' Econometrica, 55(3), 703-708."
    ),
    compare=TupleComparator(ScalarComparator(), ScalarComparator(), ScalarComparator()),
    quantcore=_hac_t_test_quantcore,
    reference=_hac_t_test_reference,
)


# --- Multiple testing: CSCV probability of backtest overfitting and effective N -------------------


def _cscv_pbo_quantcore(returns: npt.NDArray[np.float64], *, n_splits: int) -> float:
    return qc_statistics.cscv_pbo(returns, n_splits=n_splits).pbo


def _column_sharpe(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Per-period Sharpe ratio of every column, 0.0 where the standard deviation is 0."""
    sigma = x.std(axis=0, ddof=1)
    safe = np.where(sigma == 0.0, 1.0, sigma)
    sharpe: npt.NDArray[np.float64] = np.where(sigma == 0.0, 0.0, x.mean(axis=0) / safe)
    return sharpe


def _cscv_pbo_reference(returns: npt.NDArray[np.float64], *, n_splits: int) -> float:
    n_trials = returns.shape[1]
    block = returns.shape[0] // n_splits
    m = returns[: block * n_splits]
    block_of_row = np.arange(m.shape[0]) // block
    overfit = 0
    total = 0
    for train in combinations(range(n_splits), n_splits // 2):
        in_train = np.isin(block_of_row, train)
        best = int(np.argmax(_column_sharpe(m[in_train])))
        omega = float(_rankdata(_column_sharpe(m[~in_train]), method="average")[best])
        omega /= n_trials + 1
        overfit += math.log(omega / (1.0 - omega)) <= 0.0
        total += 1
    return overfit / total


_CSCV_PBO = ConformanceTarget(
    name="cscv_pbo",
    tier=1,
    oracle="CSCV procedure by hand with scipy.stats.rankdata(method='average')",
    citation=(
        "Bailey, D.H., Borwein, J.M., Lopez de Prado, M. and Zhu, Q.J. (2016), 'The Probability "
        "of Backtest Overfitting.' Journal of Computational Finance, 20(4), 39-69."
    ),
    compare=ScalarComparator(),
    quantcore=_cscv_pbo_quantcore,
    reference=_cscv_pbo_reference,
)


def _effective_n_eigen_quantcore(returns: npt.NDArray[np.float64]) -> float:
    return qc_statistics.effective_n_eigen(returns)


def _effective_n_eigen_reference(returns: npt.NDArray[np.float64]) -> float:
    centred = returns - returns.mean(axis=0)
    norms = np.sqrt((centred**2).sum(axis=0))
    unit = np.divide(centred, norms, out=np.zeros_like(centred), where=norms > 0.0)
    corr = unit.T @ unit
    np.fill_diagonal(corr, (norms > 0.0).astype(np.float64))
    frobenius_sq = float((corr**2).sum())
    return float(np.trace(corr)) ** 2 / frobenius_sq if frobenius_sq > 0.0 else 1.0


_EFFECTIVE_N_EIGEN = ConformanceTarget(
    name="effective_n_eigen",
    tier=1,
    oracle="trace identity trace(C)^2 / ||C||_F^2",
    citation=(
        "Bailey, D.H. and Lopez de Prado, M. (2014), 'The Deflated Sharpe Ratio: Correcting for "
        "Selection Bias, Backtest Overfitting and Non-Normality.' Journal of Portfolio "
        "Management, 40(5), 94-107."
    ),
    compare=ScalarComparator(),
    quantcore=_effective_n_eigen_quantcore,
    reference=_effective_n_eigen_reference,
)


# --- Stationary-bootstrap confidence interval -----------------------------------------------------


def _sample_mean(values: npt.NDArray[np.float64]) -> float:
    """The statistic both sides bootstrap: the sample mean."""
    return float(np.mean(values))


def _stationary_bootstrap_ci_quantcore(
    x: npt.NDArray[np.float64],
    *,
    rng_seed: int,
    block_length: float,
    method: Literal["percentile", "bca"],
    ci: float,
    n_boot: int,
) -> npt.NDArray[np.float64]:
    result = qc_statistics.stationary_bootstrap_ci(
        x,
        _sample_mean,
        rng=np.random.default_rng(rng_seed),
        n_boot=n_boot,
        ci=ci,
        method=method,
        block_length=block_length,
    )
    return np.array([result.ci_low, result.ci_high])


def _linear_quantile(sorted_values: npt.NDArray[np.float64], level: float) -> float:
    """Quantile at `level` of an ascending sample, interpolating at position `(m - 1) * level`."""
    position = (sorted_values.shape[0] - 1) * level
    below = math.floor(position)
    above = min(below + 1, sorted_values.shape[0] - 1)
    weight = position - below
    return float(sorted_values[below] + weight * (sorted_values[above] - sorted_values[below]))


def _stationary_bootstrap_ci_reference(
    x: npt.NDArray[np.float64],
    *,
    rng_seed: int,
    block_length: float,
    method: Literal["percentile", "bca"],
    ci: float,
    n_boot: int,
) -> npt.NDArray[np.float64]:
    """Percentile or BCa interval of the mean, computed by hand from arch's resamples.

    The resamples are drawn by `arch.bootstrap.StationaryBootstrap` with integer block size
    `max(1, round(block_length))` from `default_rng(rng_seed)`, as quantcore draws them. Everything
    after the draws is independent of quantcore: the normal distribution is
    `statistics.NormalDist`, the quantiles interpolate the sorted replicates by hand, and the
    BCa constants follow Efron (1987): `z0 = Phi^-1(#{theta* < theta} / B)`
    (the share clipped to `[1e-10, 1 - 1e-10]`), `a = sum(d^3) / (6 * (sum(d^2))^1.5)` with
    `d = mean(theta_(.)) - theta_(i)` over the leave-one-out means (0 when all are equal), and
    levels `Phi(z0 + (z0 + z) / (1 - a * (z0 + z)))`.
    """
    normal = NormalDist()
    bootstrap = StationaryBootstrap(
        max(1, round(block_length)), x, seed=np.random.default_rng(rng_seed)
    )
    replicates = np.sort(
        np.array([_sample_mean(data[0]) for data, _kw in bootstrap.bootstrap(n_boot)])
    )
    tail = (1.0 - ci) / 2.0
    if method == "percentile":
        levels = [tail, 1.0 - tail]
    else:
        theta = _sample_mean(x)
        share = np.count_nonzero(replicates < theta) / replicates.shape[0]
        z0 = normal.inv_cdf(min(max(share, 1e-10), 1.0 - 1e-10))
        rows = np.arange(x.shape[0])
        leave_one_out = np.array([_sample_mean(x[rows != i]) for i in range(x.shape[0])])
        d = leave_one_out.mean() - leave_one_out
        sum_sq = float(np.dot(d, d))
        a = float(np.dot(d * d, d)) / (6.0 * sum_sq**1.5) if sum_sq > 0.0 else 0.0
        levels = [
            min(max(normal.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z))), 0.0), 1.0)
            for z in (normal.inv_cdf(tail), normal.inv_cdf(1.0 - tail))
        ]
    return np.array(sorted(_linear_quantile(replicates, level) for level in levels))


_STATIONARY_BOOTSTRAP_CI = ConformanceTarget(
    name="stationary_bootstrap_ci",
    tier=1,
    oracle="percentile and Efron (1987) BCa by hand on arch.bootstrap.StationaryBootstrap draws",
    citation=(
        "Efron, B. (1987), 'Better Bootstrap Confidence Intervals.' Journal of the American "
        "Statistical Association, 82(397), 171-185; Politis, D.N. and Romano, J.P. (1994), 'The "
        "Stationary Bootstrap.' Journal of the American Statistical Association, 89(428), "
        "1303-1313."
    ),
    compare=ArrayComparator(),
    quantcore=_stationary_bootstrap_ci_quantcore,
    reference=_stationary_bootstrap_ci_reference,
)


# --- ARCH-LM --------------------------------------------------------------------------------------


# --- KPSS -----------------------------------------------------------------------------------------
