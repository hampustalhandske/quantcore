"""Wall-clock benchmarks for quantcore.

Two groups of measurements, both printed as Markdown tables after a header
that records the machine, the Python and package versions, the timing method
and the baseline every number is measured against:

1. quantcore's Numba kernels against the NumPy reference kernels that live
   beside them in `src/quantcore`, plus end-to-end Monte Carlo throughput.
2. quantcore's public functions against the external library each one is
   oracle-tested against (`arch`, `statsmodels`, `scipy`, `QuantLib`, `scikit-learn`,
   `empyrical`, `cvxpy` with the CLARABEL solver, `quantstats` on `pandas` Series,
   `filterpy`, and `skfolio` when it is installed), on identical inputs.

Every timing is the median of at least `MIN_REPEATS` runs of an already
warmed-up call, so Numba JIT compilation, imports and first-call caches are
excluded on both sides. Calls faster than `MIN_RUN_SECONDS` are looped inside
each run and reported per call.

The "agree" column re-runs both sides once on the timed input and compares
the results at the tolerance of the corresponding oracle test; the tolerance
is printed in the row. A row whose two sides do different work says so in its
"like for like" column, and its ratio is not a speed comparison.

Needs the `oracle` extra:

    uv sync --extra dev --extra oracle
    uv run python benchmarks/bench_quantcore.py
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import empyrical
import numpy as np
import numpy.typing as npt
import pandas as pd
import QuantLib
import quantstats.stats as qs_stats
import statsmodels.api as sm
from arch import arch_model
from filterpy.kalman import KalmanFilter
from scipy.stats import jarque_bera
from sklearn.covariance import LedoitWolf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.stattools import kpss as sm_kpss

import quantcore
from quantcore.core.price_paths import (
    simulate_exp_ou_price_paths,
    simulate_garch_t_price_paths,
    simulate_gbm_price_paths,
    simulate_heston_price_paths,
    simulate_jump_diffusion_price_paths,
    simulate_regime_switching_price_paths,
)
from quantcore.core.sde_solver import (
    _generate_gbm_increments,
    _simulate_gbm_paths_numba_kernel,
    _simulate_gbm_paths_numpy,
    simulate_gbm_paths,
)
from quantcore.core.stochastic import simulate_heston_paths
from quantcore.filtering import dynamic_hedge_ratio
from quantcore.performance import (
    common_sense_ratio,
    cpc_index,
    gain_to_pain_ratio,
    outlier_loss_ratio,
    outlier_win_ratio,
    risk_of_ruin,
    serenity_index,
    ulcer_index,
    ulcer_performance_index,
)
from quantcore.portfolio import (
    l1_turnover_penalized_weights,
    ledoit_wolf_shrinkage,
    mean_variance_weights,
    min_variance_weights,
)
from quantcore.pricing import black_scholes_call, heston_cos_call, monte_carlo_call_price
from quantcore.risk import (
    conditional_value_at_risk,
    fit_garch_11,
    garch_11_variance,
    value_at_risk,
)
from quantcore.risk.volatility import _garch_11_variance_numba_kernel, _garch_11_variance_numpy
from quantcore.statistics import (
    adf_test,
    arch_lm_test,
    engle_granger_test,
    jarque_bera_test,
    kpss_test,
    newey_west_cov,
    ols,
)
from quantcore.time_series import arima_fit, arima_forecast

FloatArray = npt.NDArray[np.float64]

MIN_REPEATS = 5
DEFAULT_REPEATS = 7
MIN_RUN_SECONDS = 2e-3
SEED = 20240921

REPORTED_PACKAGES = (
    "quantcore",
    "numpy",
    "scipy",
    "numba",
    "pandas",
    "statsmodels",
    "arch",
    "QuantLib",
    "scikit-learn",
    "empyrical-reloaded",
    "cvxpy",
    "clarabel",
    "quantstats",
    "filterpy",
    "skfolio",
)


@dataclass(frozen=True)
class Row:
    """One quantcore-versus-library comparison.

    `library_seconds` is None when no library timing exists for the row.
    `agree` and `like_for_like` are printed verbatim.
    """

    function: str
    size: str
    quantcore_seconds: float
    library: str
    library_seconds: float | None
    agree: str
    like_for_like: str


def median_seconds(fn: Callable[[], object], repeats: int = DEFAULT_REPEATS) -> float:
    """Median per-call wall-clock time of `fn`, in seconds, over `repeats` runs.

    `fn` is called once before timing (warm-up). If that call takes less than
    `MIN_RUN_SECONDS`, each run loops `fn` enough times to last about that
    long and the run's time is divided by the loop count.
    """
    if repeats < MIN_REPEATS:
        raise ValueError(f"repeats must be at least {MIN_REPEATS}")
    fn()
    start = time.perf_counter()
    fn()
    single = time.perf_counter() - start
    number = max(1, int(MIN_RUN_SECONDS / max(single, 1e-9)))

    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(number):
            fn()
        runs.append((time.perf_counter() - start) / number)
    return statistics.median(runs)


def format_seconds(seconds: float | None) -> str:
    """Render a duration with a unit that keeps three significant figures."""
    if seconds is None:
        return "-"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3g} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds:.3g} s"


def print_table(title: str, rows: list[Row]) -> None:
    """Print comparison rows as a Markdown table.

    The ratio column is library time divided by quantcore time: above 1 means
    quantcore is faster, below 1 means the library is faster.
    """
    print(f"\n### {title}\n")
    print(
        "| function | size | quantcore | library | library time | "
        "ratio (library / quantcore) | agree | like for like |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        if row.library_seconds is None:
            ratio = "-"
        else:
            ratio = f"{row.library_seconds / row.quantcore_seconds:.3g}x"
        print(
            f"| {row.function} | {row.size} | {format_seconds(row.quantcore_seconds)} | "
            f"{row.library} | {format_seconds(row.library_seconds)} | {ratio} | "
            f"{row.agree} | {row.like_for_like} |"
        )


def yes_no(condition: bool, tolerance: str) -> str:
    """Render an agreement verdict together with the tolerance it was judged at."""
    return f"{'yes' if condition else 'NO'} ({tolerance})"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or "unknown"


def _load_average() -> str:
    """One-minute load average, which shows whether other work competed for the CPU."""
    try:
        one_minute = os.getloadavg()[0]
    except (AttributeError, OSError):
        return "unknown"
    return (
        f"{one_minute:.2f} one-minute load average on {os.cpu_count()} cores "
        "(timings are inflated when this is not well below the core count)"
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _source_revision() -> str:
    """Git commit of the quantcore source being imported, or why it is unknown.

    Reads the enclosing git checkout when quantcore is imported from one, and
    otherwise the commit recorded by the installer for a git-URL install.
    """
    source_dir = Path(quantcore.__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=source_dir, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        git("ls-files", "--error-unmatch", "__init__.py")
        sha = git("rev-parse", "--short=12", "HEAD")
        dirty = git("status", "--porcelain", "--", ".")
        return f"{sha}{' with uncommitted changes under src/quantcore' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        pass

    try:
        direct_url = importlib.metadata.distribution("quantcore").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        direct_url = None
    if direct_url:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        if commit:
            return f"{commit[:12]} (installed from git)"
    return "unknown (not a git checkout and no install commit recorded)"


def _blas_threads() -> str:
    try:
        threadpoolctl = importlib.import_module("threadpoolctl")
    except ImportError:
        return "unknown (threadpoolctl not installed)"
    pools = threadpoolctl.threadpool_info()
    return ", ".join(
        f"{pool.get('internal_api', '?')} {pool.get('version', '?')} x{pool.get('num_threads')}"
        for pool in pools
    )


def print_header() -> None:
    """Print the machine, versions, timing method and baselines for this run."""
    print("# quantcore benchmark\n")
    print("## Environment\n")
    print(f"- Machine: {_cpu_name()}, {os.cpu_count()} logical cores, {platform.machine()}")
    print(f"- OS: {platform.platform()}")
    print(f"- System load at start: {_load_average()}")
    print(f"- Python: {sys.version.split()[0]} ({platform.python_implementation()})")
    for name in REPORTED_PACKAGES:
        print(f"- {name}: {_package_version(name)}")
    print(f"- quantcore source revision: {_source_revision()}")
    print(f"- BLAS/OpenMP thread pools: {_blas_threads()}")
    print("\n## Method\n")
    print(
        f"- Timing: time.perf_counter, median of {DEFAULT_REPEATS} runs "
        f"(never fewer than {MIN_REPEATS}); calls under {MIN_RUN_SECONDS * 1e3:g} ms are "
        "looped inside each run and reported per call."
    )
    print(
        "- JIT compilation time: EXCLUDED. Every timed callable, quantcore and library alike, "
        "is called once on the timed input before the first timed run."
    )
    print("- Inputs: both sides of a row receive the same arrays, generated from a fixed seed.")
    print(
        "- Ratio: library time / quantcore time. Above 1 quantcore is faster, "
        "below 1 the library is faster."
    )
    print("\n## Baselines\n")
    print(
        "- GARCH(1,1) recursion kernel: `_garch_11_variance_numpy`, an interpreted Python loop "
        "writing into a NumPy array (not vectorised NumPy)."
    )
    print(
        "- GBM path kernel: `_simulate_gbm_paths_numpy`, vectorised across paths with a Python "
        "loop over time steps, on the same pre-drawn increments."
    )
    print("- Throughput rows: no baseline; absolute paths per second including random draws.")
    print(
        "- GARCH(1,1) fit: `arch_model(mean='Zero', vol='GARCH', p=1, q=1, dist='normal', "
        "rescale=False).fit(disp='off')` on the demeaned series, model construction included."
    )
    print(
        "- ARIMA fit: `statsmodels` `ARIMA(order, trend='n').fit()` (exact Gaussian likelihood "
        "by Kalman filter); quantcore minimises the conditional sum of squares."
    )
    print("- ARIMA forecast: `ARIMAResults.forecast(h)` on an already fitted result.")
    print(
        "- Heston call: `QuantLib.AnalyticHestonEngine`, both with its default integration and "
        "with the 1e-12 adaptive Gauss-Lobatto setting the oracle test uses; quantcore uses its "
        "default n_terms=2048."
    )
    print("- OLS: `sm.OLS(y, x).fit()` reading params, resid and rsquared.")
    print(
        "- Newey-West: `sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': L}).cov_params()` "
        "against quantcore `ols` followed by `newey_west_cov`."
    )
    print("- Engle-Granger: `statsmodels` `coint(y, x, trend='c', maxlag=1, autolag=None)`.")
    print(
        "- VaR / CVaR: `empyrical.value_at_risk` / `conditional_value_at_risk` (historical). "
        "scikit-learn has no VaR or CVaR function. `skfolio.measures` is timed when installed."
    )
    print("- Covariance shrinkage: `sklearn.covariance.LedoitWolf().fit(returns).covariance_`.")
    print(
        "- Portfolio optimisers: the same convex program in `cvxpy` solved by CLARABEL, timed "
        "twice: building the problem and solving it on every call, and `problem.solve` alone on "
        "a problem built outside the timed call."
    )
    print("- ADF test: `statsmodels` `adfuller(series, maxlag=L, regression='c', autolag=None)`.")
    print(
        "- Diagnostics: `statsmodels` `kpss(regression='c', nlags='auto')`, "
        "`het_arch(nlags=None)` and `scipy.stats.jarque_bera`."
    )
    print(
        "- Performance ratios: the `quantstats.stats` function of the same name on a "
        "`pandas.Series` built outside the timed call, with `rf=0` where it takes one."
    )
    print(
        "- Dynamic hedge ratio: a `filterpy.kalman.KalmanFilter` loop that sets the observation "
        "matrix to x[t] and calls `predict` and `update` once per observation."
    )


# ---------------------------------------------------------------------------
# Numba kernels against their NumPy reference kernels
# ---------------------------------------------------------------------------


def bench_kernels() -> None:
    """Time the two Numba kernels that ship with a NumPy reference implementation."""
    print("\n### Numba kernels against the in-repo NumPy reference kernels\n")
    print("| kernel | size | Numba | NumPy reference | speedup | agree (rtol 1e-10) |")
    print("|---|---|---|---|---|---|")

    omega, alpha, beta = 0.000015, 0.09, 0.87
    for n in (2_520, 100_000, 1_000_000):
        returns = np.random.default_rng(SEED).normal(0.0, 0.015, size=n)

        def numba_garch(returns: FloatArray = returns) -> FloatArray:
            return _garch_11_variance_numba_kernel(returns, omega, alpha, beta)

        def numpy_garch(returns: FloatArray = returns) -> FloatArray:
            return _garch_11_variance_numpy(returns, omega, alpha, beta)

        t_numba = median_seconds(numba_garch)
        t_numpy = median_seconds(numpy_garch)
        agree = np.allclose(numba_garch(), numpy_garch(), rtol=1e-10, atol=1e-10)
        print(
            f"| GARCH(1,1) variance recursion | {n:,} points | {format_seconds(t_numba)} | "
            f"{format_seconds(t_numpy)} | {t_numpy / t_numba:.3g}x | {'yes' if agree else 'NO'} |"
        )

    spot, rate, vol, maturity = 100.0, 0.05, 0.2, 1.0
    for num_paths, num_steps in ((10_000, 252), (100_000, 252), (1_000, 2_520)):
        dt = maturity / num_steps
        dw = _generate_gbm_increments(dt, num_steps, num_paths, seed=SEED)

        def numba_gbm(dt: float = dt, dw: FloatArray = dw) -> FloatArray:
            return _simulate_gbm_paths_numba_kernel(spot, rate, vol, dt, dw)

        def numpy_gbm(dt: float = dt, dw: FloatArray = dw) -> FloatArray:
            return _simulate_gbm_paths_numpy(spot, rate, vol, dt, dw)

        t_numba = median_seconds(numba_gbm)
        t_numpy = median_seconds(numpy_gbm)
        agree = np.allclose(numba_gbm(), numpy_gbm(), rtol=1e-10, atol=1e-10)
        print(
            f"| GBM log-Euler paths | {num_paths:,} paths x {num_steps:,} steps | "
            f"{format_seconds(t_numba)} | {format_seconds(t_numpy)} | "
            f"{t_numpy / t_numba:.3g}x | {'yes' if agree else 'NO'} |"
        )


def bench_throughput() -> None:
    """Report end-to-end simulation throughput, random number generation included."""
    print("\n### Monte Carlo throughput (random draws included, no baseline)\n")
    print("| function | size | time | paths / s | path-steps / s | check |")
    print("|---|---|---|---|---|---|")
    spot, strike, rate, vol, maturity = 100.0, 105.0, 0.03, 0.2, 1.0
    closed_form = black_scholes_call(spot, strike, rate, vol, maturity)

    for num_paths, num_steps in ((100_000, 252), (1_000_000, 1)):

        def price(num_paths: int = num_paths, num_steps: int = num_steps) -> tuple[float, float]:
            return monte_carlo_call_price(
                spot, strike, rate, vol, maturity, num_paths, num_steps, seed=SEED
            )

        seconds = median_seconds(price)
        estimate, std_error = price()
        z = (estimate - closed_form) / std_error
        print(
            f"| monte_carlo_call_price (antithetic) | {num_paths:,} paths x {num_steps} steps | "
            f"{format_seconds(seconds)} | {num_paths / seconds:,.0f} | "
            f"{num_paths * num_steps / seconds:,.0f} | "
            f"{z:+.2f} standard errors from Black-Scholes |"
        )

    num_paths, num_steps = 100_000, 252
    seconds = median_seconds(
        lambda: simulate_gbm_paths(spot, rate, vol, maturity, num_steps, num_paths, seed=SEED)
    )
    print(
        f"| simulate_gbm_paths | {num_paths:,} paths x {num_steps} steps | "
        f"{format_seconds(seconds)} | {num_paths / seconds:,.0f} | "
        f"{num_paths * num_steps / seconds:,.0f} | - |"
    )
    seconds = median_seconds(
        lambda: simulate_heston_paths(
            100.0, 0.04, 0.03, 2.0, 0.04, 0.3, -0.7, 1.0, num_paths, num_steps, seed=SEED
        )
    )
    print(
        f"| simulate_heston_paths | {num_paths:,} paths x {num_steps} steps | "
        f"{format_seconds(seconds)} | {num_paths / seconds:,.0f} | "
        f"{num_paths * num_steps / seconds:,.0f} | - |"
    )
    _throughput_price_path_rows()


def _throughput_price_path_rows() -> None:
    """Print one throughput row per `simulate_*_price_paths` function.

    Each function is timed on 10,000 paths x 252 steps with one
    `default_rng(SEED)` created before timing and shared by every call of that
    function, so each call draws new variates. The check column compares the
    mean terminal price of a separate `default_rng(SEED)` run with its
    closed-form expectation, in standard errors of that mean, where one
    exists: `s0 * exp(mu * T)` for GBM and the jump diffusion, and
    `s0 * exp(Var[log S_T] / 2)` for the exp-OU process started at its
    long-run mean. The Heston, GARCH-t and regime-switching processes have no
    closed-form mean here and show "-".
    """
    num_paths, num_steps = 10_000, 252
    dt = 1.0 / num_steps
    horizon = num_steps * dt
    s0, mu, sigma, kappa = 100.0, 0.05, 0.2, 1.0
    exp_ou_variance = sigma**2 / (2.0 * kappa) * (1.0 - np.exp(-2.0 * kappa * horizon))
    simulators: list[tuple[str, Callable[..., FloatArray], float | None]] = [
        ("simulate_gbm_price_paths", simulate_gbm_price_paths, s0 * np.exp(mu * horizon)),
        (
            "simulate_exp_ou_price_paths",
            simulate_exp_ou_price_paths,
            s0 * np.exp(exp_ou_variance / 2.0),
        ),
        ("simulate_heston_price_paths", simulate_heston_price_paths, None),
        ("simulate_garch_t_price_paths", simulate_garch_t_price_paths, None),
        ("simulate_regime_switching_price_paths", simulate_regime_switching_price_paths, None),
        (
            "simulate_jump_diffusion_price_paths",
            simulate_jump_diffusion_price_paths,
            s0 * np.exp(mu * horizon),
        ),
    ]
    for name, simulate, expected in simulators:
        rng = np.random.default_rng(SEED)
        seconds = median_seconds(
            lambda simulate=simulate, rng=rng: simulate(rng, num_paths, num_steps, dt)
        )
        if expected is None:
            check = "- (rng shared by timed calls)"
        else:
            terminal = simulate(np.random.default_rng(SEED), num_paths, num_steps, dt)[:, -1]
            z = (terminal.mean() - expected) / (terminal.std(ddof=1) / np.sqrt(num_paths))
            check = (
                f"{z:+.2f} standard errors from the closed-form mean (rng shared by timed calls)"
            )
        print(
            f"| {name} | {num_paths:,} paths x {num_steps} steps | "
            f"{format_seconds(seconds)} | {num_paths / seconds:,.0f} | "
            f"{num_paths * num_steps / seconds:,.0f} | {check} |"
        )


# ---------------------------------------------------------------------------
# quantcore against external libraries
# ---------------------------------------------------------------------------


def _simulate_garch_percent_returns(n: int, seed: int) -> FloatArray:
    """Simulate a GARCH(1,1) return series in percent units (daily volatility near 1)."""
    omega, alpha, beta = 0.02, 0.09, 0.87
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n)
    returns = np.empty(n)
    variance = omega / (1.0 - alpha - beta)
    for t in range(n):
        returns[t] = np.sqrt(variance) * shocks[t]
        variance = omega + alpha * returns[t] ** 2 + beta * variance
    return returns


def _gaussian_garch_loglik(epsilon: FloatArray, omega: float, alpha: float, beta: float) -> float:
    variance = garch_11_variance(epsilon, omega=omega, alpha=alpha, beta=beta)
    return float(-0.5 * np.sum(np.log(2.0 * np.pi) + np.log(variance) + epsilon**2 / variance))


def bench_garch_fit() -> list[Row]:
    """`fit_garch_11` against `arch` on the same demeaned percent-return series.

    Agreement follows the edgelab conformance rule for this estimator: both
    fitted parameter sets are scored under one shared Gaussian log-likelihood
    built from `garch_11_variance`, and the two scores must match to
    rtol 1e-3 / atol 1e-2. Raw parameters are not compared, because the two
    optimisers differ and the likelihood is flat near its maximum.
    """
    rows = []
    for n in (1_000, 2_520, 10_000):
        returns = _simulate_garch_percent_returns(n, SEED)
        epsilon = returns - returns.mean()

        def fit_quantcore(returns: FloatArray = returns) -> tuple[float, float, float]:
            return fit_garch_11(returns)

        def fit_arch(epsilon: FloatArray = epsilon) -> FloatArray:
            model = arch_model(
                epsilon, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
            )
            return np.asarray(model.fit(disp="off").params, dtype=np.float64)

        t_quantcore = median_seconds(fit_quantcore)
        t_arch = median_seconds(fit_arch)
        ll_quantcore = _gaussian_garch_loglik(epsilon, *fit_quantcore())
        ll_arch = _gaussian_garch_loglik(epsilon, *fit_arch())
        agree = bool(np.isclose(ll_quantcore, ll_arch, rtol=1e-3, atol=1e-2))
        rows.append(
            Row(
                "fit_garch_11",
                f"{n:,} obs",
                t_quantcore,
                "arch",
                t_arch,
                yes_no(agree, f"log-lik {ll_quantcore:.3f} vs {ll_arch:.3f}, rtol 1e-3"),
                "yes: Gaussian MLE of the same 3 parameters; quantcore runs several "
                "SLSQP starts, arch one; start-up variance conventions differ",
            )
        )
    return rows


def _simulate_arma(n: int, phi: float, theta: float, integrate: bool, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(scale=0.2, size=n)
    series = np.empty(n)
    series[0] = eps[0]
    for t in range(1, n):
        series[t] = phi * series[t - 1] + eps[t] + theta * eps[t - 1]
    return np.cumsum(series) + 100.0 if integrate else series


def bench_arima() -> list[Row]:
    """`arima_fit` and `arima_forecast` against `statsmodels` `ARIMA`.

    The fit rows compare two exact Gaussian maximum-likelihood fits, each
    stopped by its own optimizer; agreement uses the oracle test's parameter
    bounds for that case (coefficients within 1e-3, innovation variance
    within 0.1%).

    The forecast rows feed the statsmodels-fitted coefficients to
    `arima_forecast` and compare with `ARIMAResults.forecast` at
    rtol 1e-9 / atol 1e-10. Both are the conditional mean of the future
    values given the whole series.
    """
    horizon = 10
    cases = (
        ("ARMA(1,1)", (1, 0, 1), 1_000, 0.5, 0.3, False),
        ("ARMA(1,1)", (1, 0, 1), 10_000, 0.5, 0.3, False),
        ("ARIMA(1,1,0)", (1, 1, 0), 1_000, 0.4, 0.0, True),
        ("AR(1)", (1, 0, 0), 1_000, 0.6, 0.0, False),
    )
    rows = []
    for label, order, n, phi, theta, integrate in cases:
        p, d, q = order
        series = _simulate_arma(n, phi, theta, integrate, SEED)

        def fit_quantcore(
            series: FloatArray = series, p: int = p, d: int = d, q: int = q
        ) -> tuple[FloatArray, FloatArray, float]:
            return arima_fit(series, p=p, d=d, q=q)

        def fit_statsmodels(
            series: FloatArray = series, order: tuple[int, int, int] = order
        ) -> object:
            return ARIMA(series, order=order, trend="n").fit()

        t_quantcore = median_seconds(fit_quantcore)
        t_statsmodels = median_seconds(fit_statsmodels, repeats=MIN_REPEATS)
        ar, ma, sigma2 = fit_quantcore()
        result = ARIMA(series, order=order, trend="n").fit()
        sm_params = np.asarray(result.params, dtype=np.float64)
        sm_ar, sm_ma, sm_sigma2 = sm_params[:p], sm_params[p : p + q], float(sm_params[-1])
        close = (
            np.allclose(ar, sm_ar, rtol=0.0, atol=1e-3)
            and np.allclose(ma, sm_ma, rtol=0.0, atol=1e-3)
            and bool(np.isclose(sigma2, sm_sigma2, rtol=1e-3, atol=0.0))
        )
        rows.append(
            Row(
                "arima_fit",
                f"{label}, {n:,} obs",
                t_quantcore,
                "statsmodels",
                t_statsmodels,
                yes_no(close, "coefs abs 1e-3, sigma2 rel 1e-3"),
                "yes for the estimate: both maximize the exact Gaussian likelihood by a "
                "Kalman filter; statsmodels also prepares standard errors",
            )
        )

        def forecast_quantcore(
            series: FloatArray = series,
            sm_ar: FloatArray = sm_ar,
            sm_ma: FloatArray = sm_ma,
            d: int = d,
        ) -> FloatArray:
            return arima_forecast(series, ar_coefs=sm_ar, ma_coefs=sm_ma, d=d, h=horizon)

        def forecast_statsmodels(result: object = result) -> FloatArray:
            return np.asarray(result.forecast(horizon), dtype=np.float64)  # type: ignore[attr-defined]

        t_quantcore = median_seconds(forecast_quantcore)
        t_statsmodels = median_seconds(forecast_statsmodels)
        agree = np.allclose(forecast_quantcore(), forecast_statsmodels(), rtol=1e-9, atol=1e-10)
        if q > 0:
            like_for_like = (
                "yes for the forecast values; statsmodels reuses its stored filter state, "
                "quantcore runs its Kalman filter over the raw series"
            )
        else:
            like_for_like = (
                "yes for the forecast values; statsmodels reuses its stored filter state, "
                "quantcore starts from the raw series"
            )
        rows.append(
            Row(
                "arima_forecast",
                f"{label}, {n:,} obs, h={horizon}",
                t_quantcore,
                "statsmodels",
                t_statsmodels,
                yes_no(agree, "same coefs, rtol 1e-9"),
                like_for_like,
            )
        )
    return rows


HESTON_SPOT, HESTON_RATE = 100.0, 0.0
HESTON_V0, HESTON_KAPPA, HESTON_THETA = 0.04, 1.0, 0.04
HESTON_XI, HESTON_RHO = 0.3, -0.7
HESTON_MONEYNESS = (0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0)
HESTON_MATURITY_DAYS = (7, 37, 183, 365, 730)


def _quantlib_heston_calls(
    contracts: list[tuple[float, int]], oracle_grade: bool
) -> npt.NDArray[np.float64]:
    """Price Heston calls with QuantLib, building the model once for all contracts.

    Args:
        contracts: (strike, days to expiry) pairs.
        oracle_grade: True uses adaptive Gauss-Lobatto at 1e-12 relative
            tolerance, the oracle test's setting; False uses the engine's
            default integration.
    """
    today = QuantLib.Date(1, 1, 2024)
    QuantLib.Settings.instance().evaluationDate = today
    day_count = QuantLib.Actual365Fixed()
    rate_ts = QuantLib.YieldTermStructureHandle(QuantLib.FlatForward(today, HESTON_RATE, day_count))
    div_ts = QuantLib.YieldTermStructureHandle(QuantLib.FlatForward(today, 0.0, day_count))
    process = QuantLib.HestonProcess(
        rate_ts,
        div_ts,
        QuantLib.QuoteHandle(QuantLib.SimpleQuote(HESTON_SPOT)),
        HESTON_V0,
        HESTON_KAPPA,
        HESTON_THETA,
        HESTON_XI,
        HESTON_RHO,
    )
    model = QuantLib.HestonModel(process)
    if oracle_grade:
        engine = QuantLib.AnalyticHestonEngine(model, 1e-12, 1_000_000)
    else:
        engine = QuantLib.AnalyticHestonEngine(model)

    prices = np.empty(len(contracts))
    for i, (strike, days) in enumerate(contracts):
        option = QuantLib.VanillaOption(
            QuantLib.PlainVanillaPayoff(QuantLib.Option.Call, strike),
            QuantLib.EuropeanExercise(today + QuantLib.Period(days, QuantLib.Days)),
        )
        option.setPricingEngine(engine)
        prices[i] = option.NPV()
    return prices


def _quantcore_heston_calls(contracts: list[tuple[float, int]]) -> npt.NDArray[np.float64]:
    return np.array(
        [
            heston_cos_call(
                HESTON_SPOT,
                strike,
                HESTON_RATE,
                days / 365.0,
                HESTON_V0,
                HESTON_KAPPA,
                HESTON_THETA,
                HESTON_XI,
                HESTON_RHO,
            )
            for strike, days in contracts
        ]
    )


def _heston_prices_agree(actual: FloatArray, expected: FloatArray) -> bool:
    """Oracle-test rule: rtol 1e-6 for prices above 1e-4, atol 1e-8 below."""
    large = np.abs(expected) > 1e-4
    return bool(
        np.allclose(actual[large], expected[large], rtol=1e-6, atol=0.0)
        and np.allclose(actual[~large], expected[~large], rtol=0.0, atol=1e-8)
    )


def bench_heston() -> list[Row]:
    """`heston_cos_call` against `QuantLib.AnalyticHestonEngine`.

    QuantLib time includes building the term structures, process, model and
    engine, since quantcore's function starts from the same scalar inputs. On
    the 40-contract grid QuantLib builds them once and reprices; quantcore is
    called once per contract.
    """
    single = [(HESTON_SPOT, 365)]
    grid = [(HESTON_SPOT * m, days) for m in HESTON_MONEYNESS for days in HESTON_MATURITY_DAYS]
    rows = []
    for label, contracts in (("1 call, ATM, 1y", single), ("40 calls, K/S 0.5-3, 7d-2y", grid)):
        reference = _quantlib_heston_calls(contracts, oracle_grade=True)
        t_quantcore = median_seconds(lambda contracts=contracts: _quantcore_heston_calls(contracts))
        agree = _heston_prices_agree(_quantcore_heston_calls(contracts), reference)
        for engine_label, oracle_grade in (
            ("QuantLib, default integration", False),
            ("QuantLib, 1e-12 adaptive", True),
        ):
            t_quantlib = median_seconds(
                lambda contracts=contracts, oracle_grade=oracle_grade: _quantlib_heston_calls(
                    contracts, oracle_grade
                )
            )
            if oracle_grade:
                verdict = yes_no(agree, "rtol 1e-6, atol 1e-8 below 1e-4")
            else:
                default_agrees = _heston_prices_agree(
                    _quantlib_heston_calls(contracts, oracle_grade=False), reference
                )
                verdict = yes_no(
                    agree and default_agrees,
                    "both vs the 1e-12 engine, rtol 1e-6"
                    if default_agrees
                    else "quantcore agrees with the 1e-12 engine; this engine setting does not",
                )
            rows.append(
                Row(
                    "heston_cos_call",
                    label,
                    t_quantcore,
                    engine_label,
                    t_quantlib,
                    verdict,
                    "yes: same price from the same scalar inputs; COS series (2048 terms) "
                    "against characteristic-function quadrature",
                )
            )
    return rows


def _regression_data(n_obs: int, n_regressors: int, seed: int) -> tuple[FloatArray, FloatArray]:
    """Design matrix with an intercept column and a response with AR(1) errors."""
    rng = np.random.default_rng(seed)
    design = np.column_stack([np.ones(n_obs), rng.normal(size=(n_obs, n_regressors - 1))])
    shocks = rng.normal(scale=0.5, size=n_obs)
    errors = np.empty(n_obs)
    errors[0] = shocks[0]
    for t in range(1, n_obs):
        errors[t] = 0.5 * errors[t - 1] + shocks[t]
    y = design @ np.linspace(0.5, 1.5, n_regressors) + errors
    return y, design


def bench_regression() -> list[Row]:
    """`ols` and `ols` + `newey_west_cov` against `statsmodels` OLS and HAC."""
    n_lags = 5
    rows = []
    for n_obs, n_regressors in ((1_000, 4), (100_000, 10)):
        y, design = _regression_data(n_obs, n_regressors, SEED)
        size = f"{n_obs:,} obs x {n_regressors} regressors"

        def ols_quantcore(
            y: FloatArray = y, design: FloatArray = design
        ) -> tuple[FloatArray, FloatArray, float]:
            return ols(y, design)

        def ols_statsmodels(
            y: FloatArray = y, design: FloatArray = design
        ) -> tuple[FloatArray, FloatArray, float]:
            result = sm.OLS(y, design).fit()
            return result.params, result.resid, result.rsquared

        t_quantcore = median_seconds(ols_quantcore)
        t_statsmodels = median_seconds(ols_statsmodels)
        coefs, resid, r_squared = ols_quantcore()
        sm_coefs, sm_resid, sm_r_squared = ols_statsmodels()
        agree = (
            np.allclose(coefs, sm_coefs, rtol=1e-8, atol=0.0)
            and np.allclose(resid, sm_resid, rtol=1e-6, atol=1e-10)
            and bool(np.isclose(r_squared, sm_r_squared, rtol=1e-8))
        )
        rows.append(
            Row(
                "ols",
                size,
                t_quantcore,
                "statsmodels",
                t_statsmodels,
                yes_no(agree, "coefs and R2 rtol 1e-8"),
                "yes: coefficients, residuals and R2 from both",
            )
        )

        def hac_quantcore(y: FloatArray = y, design: FloatArray = design) -> FloatArray:
            _, residuals, _ = ols(y, design)
            return newey_west_cov(design, residuals, n_lags=n_lags)

        def hac_statsmodels(y: FloatArray = y, design: FloatArray = design) -> FloatArray:
            result = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": n_lags})
            return np.asarray(result.cov_params(), dtype=np.float64)

        t_quantcore = median_seconds(hac_quantcore)
        t_statsmodels = median_seconds(hac_statsmodels)
        agree = np.allclose(hac_quantcore(), hac_statsmodels(), rtol=1e-6, atol=0.0)
        rows.append(
            Row(
                "ols + newey_west_cov",
                f"{size}, {n_lags} lags",
                t_quantcore,
                "statsmodels",
                t_statsmodels,
                yes_no(agree, "rtol 1e-6"),
                "yes: OLS fit plus Bartlett-kernel HAC covariance from both",
            )
        )
    return rows


def bench_engle_granger() -> list[Row]:
    """`engle_granger_test` against `statsmodels` `coint` with matching settings."""
    rows = []
    for n_obs in (500, 5_000, 50_000):
        rng = np.random.default_rng(SEED)
        x = np.cumsum(rng.normal(size=n_obs))
        spread = np.empty(n_obs)
        spread[0] = 0.0
        noise = rng.normal(size=n_obs)
        for t in range(1, n_obs):
            spread[t] = 0.5 * spread[t - 1] + noise[t]
        y = 2.0 * x + spread

        def test_quantcore(y: FloatArray = y, x: FloatArray = x) -> tuple[float, float, float]:
            return engle_granger_test(y, x)

        def test_statsmodels(y: FloatArray = y, x: FloatArray = x) -> tuple[float, float]:
            statistic, p_value, _ = coint(y, x, trend="c", maxlag=1, autolag=None)
            return float(statistic), float(p_value)

        t_quantcore = median_seconds(test_quantcore)
        t_statsmodels = median_seconds(test_statsmodels)
        _, statistic, p_value = test_quantcore()
        sm_statistic, sm_p_value = test_statsmodels()
        agree = bool(
            np.isclose(statistic, sm_statistic, rtol=1e-10, atol=1e-10)
            and np.isclose(p_value, sm_p_value, rtol=1e-10, atol=1e-10)
        )
        rows.append(
            Row(
                "engle_granger_test",
                f"{n_obs:,} obs",
                t_quantcore,
                "statsmodels",
                t_statsmodels,
                yes_no(agree, "statistic and p-value 1e-10"),
                "yes: same two-step test; statsmodels also returns critical values",
            )
        )
    return rows


def bench_var_cvar() -> list[Row]:
    """Historical `value_at_risk` / `conditional_value_at_risk` against `empyrical`.

    `skfolio.measures` is added when installed. Its VaR is an order statistic
    where quantcore and empyrical interpolate the empirical quantile linearly,
    so that row is flagged as not comparable. Its CVaR is the
    Rockafellar-Uryasev tail average, which equals quantcore's tail mean when
    (1 - confidence) x observations is an integer, as it is for both sizes
    timed here.
    """
    confidence = 0.95
    try:
        skfolio_measures = importlib.import_module("skfolio.measures")
    except ImportError:
        skfolio_measures = None

    rows = []
    for n_obs in (2_520, 1_000_000):
        returns = np.random.default_rng(SEED).standard_t(df=4, size=n_obs) * 0.01 + 0.001
        size = f"{n_obs:,} obs, 95%"

        def var_quantcore(returns: FloatArray = returns) -> float:
            return value_at_risk(returns, confidence, method="historical")

        def cvar_quantcore(returns: FloatArray = returns) -> float:
            return conditional_value_at_risk(returns, confidence, method="historical")

        def var_empyrical(returns: FloatArray = returns) -> float:
            return -float(empyrical.value_at_risk(returns, cutoff=1.0 - confidence))

        def cvar_empyrical(returns: FloatArray = returns) -> float:
            return -float(empyrical.conditional_value_at_risk(returns, cutoff=1.0 - confidence))

        comparisons: list[tuple[str, Callable[[], float], str, Callable[[], float], str]] = [
            (
                "value_at_risk (historical)",
                var_quantcore,
                "empyrical",
                var_empyrical,
                "yes: linearly interpolated empirical quantile",
            ),
            (
                "conditional_value_at_risk (historical)",
                cvar_quantcore,
                "empyrical",
                cvar_empyrical,
                "yes: mean of returns at or below the VaR threshold",
            ),
        ]
        if skfolio_measures is not None:

            def var_skfolio(returns: FloatArray = returns) -> float:
                return float(skfolio_measures.value_at_risk(returns, beta=confidence))

            def cvar_skfolio(returns: FloatArray = returns) -> float:
                return float(skfolio_measures.cvar(returns, beta=confidence))

            comparisons += [
                (
                    "value_at_risk (historical)",
                    var_quantcore,
                    "skfolio",
                    var_skfolio,
                    "NOT COMPARABLE: skfolio returns an order statistic, no interpolation",
                ),
                (
                    "conditional_value_at_risk (historical)",
                    cvar_quantcore,
                    "skfolio",
                    cvar_skfolio,
                    "yes on this input: skfolio's Rockafellar-Uryasev tail average equals the "
                    "tail mean when (1 - confidence) x obs is an integer, and differs otherwise",
                ),
            ]

        for function, quantcore_fn, library, library_fn, like_for_like in comparisons:
            t_quantcore = median_seconds(quantcore_fn)
            t_library = median_seconds(library_fn)
            agree = bool(np.isclose(quantcore_fn(), library_fn(), rtol=1e-9, atol=0.0))
            rows.append(
                Row(
                    function,
                    size,
                    t_quantcore,
                    library,
                    t_library,
                    yes_no(agree, "rtol 1e-9"),
                    like_for_like,
                )
            )

    if skfolio_measures is None:
        print("\nskfolio is not installed: no skfolio VaR / CVaR rows in this run.")
    return rows


def bench_shrinkage() -> list[Row]:
    """`ledoit_wolf_shrinkage` against `sklearn.covariance.LedoitWolf`."""
    rows = []
    for n_obs, n_assets in ((252, 10), (2_520, 100), (2_520, 500)):
        rng = np.random.default_rng(SEED)
        loadings = rng.normal(size=(n_assets, n_assets)) / np.sqrt(n_assets)
        returns = rng.normal(size=(n_obs, n_assets)) @ loadings.T * 0.01

        def shrink_quantcore(returns: FloatArray = returns) -> FloatArray:
            return ledoit_wolf_shrinkage(returns)

        def shrink_sklearn(returns: FloatArray = returns) -> FloatArray:
            return np.asarray(LedoitWolf().fit(returns).covariance_, dtype=np.float64)

        t_quantcore = median_seconds(shrink_quantcore)
        t_sklearn = median_seconds(shrink_sklearn)
        agree = np.allclose(shrink_quantcore(), shrink_sklearn(), rtol=1e-8, atol=1e-12)
        rows.append(
            Row(
                "ledoit_wolf_shrinkage",
                f"{n_obs:,} obs x {n_assets} assets",
                t_quantcore,
                "scikit-learn",
                t_sklearn,
                yes_no(agree, "rtol 1e-8"),
                "yes: Ledoit-Wolf (2004) shrinkage toward scaled identity; "
                "scikit-learn also validates input and stores the mean",
            )
        )
    return rows


def _factor_model_covariance(n_assets: int, seed: int) -> FloatArray:
    """Well-conditioned daily-return covariance from a random three-factor model.

    A market factor (loadings near 1) and two further factors, each with
    daily variance of order 1e-5, plus idiosyncratic variances between
    0.5e-4 and 1.5e-4, so asset variances are about 1e-4 to 2e-4.
    """
    rng = np.random.default_rng(seed)
    loadings = np.column_stack(
        [rng.normal(1.0, 0.3, n_assets), rng.normal(0.0, 0.5, (n_assets, 2))]
    )
    factor_variance = np.array([0.4e-4, 0.15e-4, 0.1e-4])
    idiosyncratic = rng.uniform(0.5e-4, 1.5e-4, n_assets)
    cov = (loadings * factor_variance) @ loadings.T + np.diag(idiosyncratic)
    return np.asarray(0.5 * (cov + cov.T), dtype=np.float64)


def _solved_weights(problem: cp.Problem, weights: cp.Variable, name: str) -> FloatArray:
    """Solve `problem` with CLARABEL and return the weights, raising unless optimal."""
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"{name}: cvxpy solver did not reach optimality: {problem.status}")
    return np.asarray(weights.value, dtype=np.float64)


def bench_portfolio_optimisers() -> list[Row]:
    """`min_variance_weights`, `mean_variance_weights` and `l1_turnover_penalized_weights`
    against the identical program in `cvxpy` solved by CLARABEL.

    All three are long-only, fully invested (`sum(w) = 1`, `w >= 0`), on a
    random factor-model covariance at daily-return scale. The programs are
    `min w'Sigma w`; `max w'mu - 0.5*lambda*w'Sigma w` with lambda = 5; and
    the same utility minus `(cost_bps / 1e4) * ||w - w_prev||_1` with
    cost_bps = 10 and random long-only `w_prev`. cvxpy gets the L1 term as
    `norm1` where quantcore splits it into purchases and sales.

    Each cvxpy program is timed two ways, one library row each: "build +
    solve" constructs the problem and solves it on every call, as quantcore's
    function takes raw arrays; "solve only" calls `problem.solve` on a
    problem built outside the timed call, the least cvxpy can cost per call.

    Agreement is on the attained objective, evaluated by one NumPy function
    at each side's weights, at abs 1e-8 + rel 1e-6 (CLARABEL's default
    gap tolerance is of that order). The weights themselves are not compared:
    they can differ along flat directions of the objective.
    """
    risk_aversion, cost_bps = 5.0, 10.0
    rows = []
    for n_assets in (10, 50, 200):
        cov = _factor_model_covariance(n_assets, SEED)
        rng = np.random.default_rng(SEED + n_assets)
        mu = rng.normal(4e-4, 3e-4, n_assets)
        previous = rng.dirichlet(np.ones(n_assets))
        penalty = cost_bps / 10_000.0

        def min_variance_objective(w: FloatArray, cov: FloatArray = cov) -> float:
            return float(w @ cov @ w)

        def utility(w: FloatArray, cov: FloatArray = cov, mu: FloatArray = mu) -> float:
            return float(w @ mu - 0.5 * risk_aversion * (w @ cov @ w))

        def turnover_utility(
            w: FloatArray, previous: FloatArray = previous, penalty: float = penalty
        ) -> float:
            return utility(w) - penalty * float(np.sum(np.abs(w - previous)))

        def build_min_variance(cov: FloatArray = cov) -> tuple[cp.Problem, cp.Variable]:
            w = cp.Variable(cov.shape[0])
            problem = cp.Problem(
                cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), [cp.sum(w) == 1, w >= 0]
            )
            return problem, w

        def build_mean_variance(
            cov: FloatArray = cov, mu: FloatArray = mu
        ) -> tuple[cp.Problem, cp.Variable]:
            w = cp.Variable(cov.shape[0])
            objective = w @ mu - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
            return cp.Problem(cp.Maximize(objective), [cp.sum(w) == 1, w >= 0]), w

        def build_turnover(
            cov: FloatArray = cov,
            mu: FloatArray = mu,
            previous: FloatArray = previous,
            penalty: float = penalty,
        ) -> tuple[cp.Problem, cp.Variable]:
            w = cp.Variable(cov.shape[0])
            objective = (
                w @ mu
                - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
                - penalty * cp.norm1(w - previous)
            )
            return cp.Problem(cp.Maximize(objective), [cp.sum(w) == 1, w >= 0]), w

        def quantcore_min_variance(cov: FloatArray = cov) -> FloatArray:
            return min_variance_weights(cov)

        def quantcore_mean_variance(cov: FloatArray = cov, mu: FloatArray = mu) -> FloatArray:
            return mean_variance_weights(mu, cov, risk_aversion)

        def quantcore_turnover(
            cov: FloatArray = cov, mu: FloatArray = mu, previous: FloatArray = previous
        ) -> FloatArray:
            return l1_turnover_penalized_weights(mu, cov, previous, cost_bps, risk_aversion)

        cases: list[
            tuple[
                str,
                Callable[[], FloatArray],
                Callable[[], tuple[cp.Problem, cp.Variable]],
                Callable[[FloatArray], float],
            ]
        ] = [
            (
                "min_variance_weights",
                quantcore_min_variance,
                build_min_variance,
                min_variance_objective,
            ),
            ("mean_variance_weights", quantcore_mean_variance, build_mean_variance, utility),
            ("l1_turnover_penalized_weights", quantcore_turnover, build_turnover, turnover_utility),
        ]
        for name, quantcore_fn, build, objective in cases:
            prebuilt_problem, prebuilt_weights = build()

            def build_and_solve(
                build: Callable[[], tuple[cp.Problem, cp.Variable]] = build, name: str = name
            ) -> FloatArray:
                problem, weights = build()
                return _solved_weights(problem, weights, name)

            def solve_only(
                problem: cp.Problem = prebuilt_problem,
                weights: cp.Variable = prebuilt_weights,
                name: str = name,
            ) -> FloatArray:
                return _solved_weights(problem, weights, name)

            t_quantcore = median_seconds(quantcore_fn)
            t_build = median_seconds(build_and_solve)
            t_solve = median_seconds(solve_only)
            value_quantcore = objective(quantcore_fn())
            value_cvxpy = objective(build_and_solve())
            gap = abs(value_quantcore - value_cvxpy)
            agree = bool(np.isclose(value_quantcore, value_cvxpy, rtol=1e-6, atol=1e-8))
            verdict = yes_no(agree, f"objective abs 1e-8 + rel 1e-6, gap {gap:.1e}")
            like_for_like = (
                "yes: the same convex program; objectives agree, weights may differ along "
                "flat directions"
            )
            for library, t_library in (
                ("cvxpy/CLARABEL, build + solve per call", t_build),
                ("cvxpy/CLARABEL, solve only (problem prebuilt)", t_solve),
            ):
                rows.append(
                    Row(
                        name,
                        f"{n_assets} assets",
                        t_quantcore,
                        library,
                        t_library,
                        verdict,
                        like_for_like,
                    )
                )
    return rows


def bench_adf() -> list[Row]:
    """`adf_test` against `statsmodels` `adfuller` on a random walk.

    Both use a constant, 4 lagged differences and no lag search
    (`autolag=None`). Agreement is the statistic and the MacKinnon p-value at
    the oracle test's rtol 1e-8 / atol 1e-11.
    """
    max_lags = 4
    rows = []
    for n_obs in (500, 5_000, 50_000):
        series = np.cumsum(np.random.default_rng(SEED).normal(size=n_obs))

        def test_quantcore(series: FloatArray = series) -> tuple[float, float]:
            return adf_test(series, max_lags, trend="c", autolag=None)

        def test_statsmodels(series: FloatArray = series) -> tuple[float, float]:
            result = adfuller(series, maxlag=max_lags, regression="c", autolag=None)
            return float(result[0]), float(result[1])

        t_quantcore = median_seconds(test_quantcore)
        t_statsmodels = median_seconds(test_statsmodels)
        agree = bool(np.allclose(test_quantcore(), test_statsmodels(), rtol=1e-8, atol=1e-11))
        rows.append(
            Row(
                "adf_test",
                f"{n_obs:,} obs, {max_lags} lags, constant",
                t_quantcore,
                "statsmodels",
                t_statsmodels,
                yes_no(agree, "statistic and p-value rtol 1e-8"),
                "yes: same regression and MacKinnon p-value; statsmodels also returns "
                "critical values and the regression sample size",
            )
        )
    return rows


def bench_diagnostics() -> list[Row]:
    """`kpss_test`, `arch_lm_test` and `jarque_bera_test` against their oracles.

    - `kpss_test(series, None, "c")` against `statsmodels` `kpss(regression="c",
      nlags="auto")` on an AR(1) series with coefficient 0.9; statistic and
      p-value at rtol 1e-9. quantcore also returns whether the p-value sits at
      an end of the critical-value table.
    - `arch_lm_test(residuals, None)` against `het_arch(nlags=None)` on a
      simulated GARCH(1,1) series; both default to min(10, n // 5) lags and
      neither demeans. Statistic and p-value at rtol 1e-8 / atol 1e-8.
    - `jarque_bera_test(series)` against `scipy.stats.jarque_bera` on
      Student-t(4) returns; both use population moments. Statistic and
      p-value at rtol 1e-9 / atol 1e-12.
    """
    rows = []
    for n_obs in (2_520, 100_000):
        rng = np.random.default_rng(SEED)
        noise = rng.normal(0.0, 0.01, size=n_obs)
        ar1 = np.empty(n_obs)
        ar1[0] = noise[0]
        for t in range(1, n_obs):
            ar1[t] = 0.9 * ar1[t - 1] + noise[t]
        garch = _simulate_garch_percent_returns(n_obs, SEED) * 0.01
        heavy_tailed = rng.standard_t(df=4, size=n_obs) * 0.01

        def kpss_quantcore(series: FloatArray = ar1) -> tuple[float, float]:
            statistic, p_value, _ = kpss_test(series, None, "c")
            return statistic, p_value

        def kpss_statsmodels(series: FloatArray = ar1) -> tuple[float, float]:
            statistic, p_value, _, _ = sm_kpss(series, regression="c", nlags="auto")
            return float(statistic), float(p_value)

        def arch_quantcore(series: FloatArray = garch) -> tuple[float, float]:
            return arch_lm_test(series, None)

        def arch_statsmodels(series: FloatArray = garch) -> tuple[float, float]:
            lm_statistic, lm_p_value, _, _ = het_arch(series, nlags=None)
            return float(lm_statistic), float(lm_p_value)

        def jb_quantcore(series: FloatArray = heavy_tailed) -> tuple[float, float]:
            return jarque_bera_test(series)

        def jb_scipy(series: FloatArray = heavy_tailed) -> tuple[float, float]:
            result = jarque_bera(series)
            return float(result.statistic), float(result.pvalue)

        comparisons: list[
            tuple[
                str,
                Callable[[], tuple[float, float]],
                str,
                Callable[[], tuple[float, float]],
                float,
                float,
                str,
                str,
            ]
        ] = [
            (
                "kpss_test",
                kpss_quantcore,
                "statsmodels",
                kpss_statsmodels,
                1e-9,
                0.0,
                "statistic and p-value rtol 1e-9",
                "yes: level-stationarity null, Bartlett long-run variance with the automatic "
                "bandwidth, p-value interpolated in the same table",
            ),
            (
                "arch_lm_test",
                arch_quantcore,
                "statsmodels",
                arch_statsmodels,
                1e-8,
                1e-8,
                "statistic and p-value rtol 1e-8",
                "yes: (n - lags) R^2 of the squares on their lags, default lag count; "
                "statsmodels also returns the F test",
            ),
            (
                "jarque_bera_test",
                jb_quantcore,
                "scipy",
                jb_scipy,
                1e-9,
                1e-12,
                "statistic and p-value rtol 1e-9",
                "yes: n/6 (S^2 + K^2/4) and its chi-squared(2) p-value",
            ),
        ]
        for name, quantcore_fn, library, library_fn, rtol, atol, tolerance, like in comparisons:
            t_quantcore = median_seconds(quantcore_fn)
            t_library = median_seconds(library_fn)
            agree = bool(np.allclose(quantcore_fn(), library_fn(), rtol=rtol, atol=atol))
            rows.append(
                Row(
                    name,
                    f"{n_obs:,} obs",
                    t_quantcore,
                    library,
                    t_library,
                    yes_no(agree, tolerance),
                    like,
                )
            )
    return rows


def bench_performance_ratios() -> list[Row]:
    """Nine `quantcore.performance` ratios against the `quantstats.stats` function of the same name.

    The input is 2,520 normal daily returns (mean 4e-4, standard deviation 0.01).
    quantstats takes a `pandas.Series` with a DatetimeIndex, built once
    outside the timed call; quantcore takes the NumPy array. quantstats is
    called with `rf=0` where it takes a risk-free rate, and none of the nine
    quantities is annualised on either side. Agreement is rtol 1e-10 / atol
    1e-12, the closed-form tolerance of the oracle tests.
    """
    returns = np.random.default_rng(SEED).normal(4e-4, 0.01, size=2_520)
    series = pd.Series(returns, index=pd.date_range("2000-01-01", periods=returns.size, freq="D"))
    comparisons: list[tuple[str, Callable[[FloatArray], float], Callable[[pd.Series], float]]] = [
        ("ulcer_index", ulcer_index, qs_stats.ulcer_index),
        (
            "ulcer_performance_index",
            ulcer_performance_index,
            lambda s: qs_stats.ulcer_performance_index(s, 0),
        ),
        ("serenity_index", serenity_index, lambda s: qs_stats.serenity_index(s, 0)),
        ("gain_to_pain_ratio", gain_to_pain_ratio, lambda s: qs_stats.gain_to_pain_ratio(s, 0)),
        ("common_sense_ratio", common_sense_ratio, qs_stats.common_sense_ratio),
        ("cpc_index", cpc_index, qs_stats.cpc_index),
        ("outlier_win_ratio", outlier_win_ratio, qs_stats.outlier_win_ratio),
        ("outlier_loss_ratio", outlier_loss_ratio, qs_stats.outlier_loss_ratio),
        ("risk_of_ruin", risk_of_ruin, qs_stats.risk_of_ruin),
    ]
    rows = []
    for name, quantcore_fn, quantstats_fn in comparisons:

        def run_quantcore(fn: Callable[[FloatArray], float] = quantcore_fn) -> float:
            return float(fn(returns))

        def run_quantstats(fn: Callable[[pd.Series], float] = quantstats_fn) -> float:
            return float(fn(series))

        t_quantcore = median_seconds(run_quantcore)
        t_quantstats = median_seconds(run_quantstats)
        agree = bool(np.isclose(run_quantcore(), run_quantstats(), rtol=1e-10, atol=1e-12))
        rows.append(
            Row(
                name,
                "2,520 obs",
                t_quantcore,
                "quantstats",
                t_quantstats,
                yes_no(agree, "rtol 1e-10, atol 1e-12"),
                "yes: same definition on the same returns; quantstats reads a pandas.Series "
                "(built outside the timed call), quantcore a NumPy array",
            )
        )
    return rows


def bench_dynamic_hedge_ratio() -> list[Row]:
    """`dynamic_hedge_ratio` against a `filterpy` `KalmanFilter` loop with per-step H.

    As in the oracle test: a scalar state started from the prior N(0, 1),
    F = 1, Q = 1e-4, R = 0.01, and H set to x[t] before each `predict` /
    `update` pair. The series are a drifting-beta pair, y = beta_t x + noise.
    Agreement is the whole beta path at atol 1e-10.
    """
    obs_var, proc_var = 0.01, 1e-4
    rows = []
    for n_obs in (2_520, 25_200):
        rng = np.random.default_rng(SEED)
        x = rng.normal(size=n_obs)
        true_beta = 1.0 + 0.01 * np.cumsum(rng.normal(scale=0.1, size=n_obs))
        y = true_beta * x + rng.normal(scale=0.1, size=n_obs)

        def beta_quantcore(y: FloatArray = y, x: FloatArray = x) -> FloatArray:
            return dynamic_hedge_ratio(y, x, obs_var=obs_var, proc_var=proc_var)

        def beta_filterpy(y: FloatArray = y, x: FloatArray = x) -> FloatArray:
            kf = KalmanFilter(dim_x=1, dim_z=1)
            kf.x = np.array([0.0])
            kf.P = np.array([[1.0]])
            kf.F = np.array([[1.0]])
            kf.Q = np.array([[proc_var]])
            kf.R = np.array([[obs_var]])
            beta = np.empty(y.shape[0])
            for t in range(y.shape[0]):
                kf.H = np.array([[x[t]]])
                kf.predict()
                kf.update(np.array([y[t]]))
                beta[t] = kf.x[0]
            return beta

        t_quantcore = median_seconds(beta_quantcore)
        t_filterpy = median_seconds(beta_filterpy)
        agree = bool(np.allclose(beta_quantcore(), beta_filterpy(), rtol=0.0, atol=1e-10))
        rows.append(
            Row(
                "dynamic_hedge_ratio",
                f"{n_obs:,} obs",
                t_quantcore,
                "filterpy",
                t_filterpy,
                yes_no(agree, "beta path atol 1e-10"),
                "yes: same prior and noise variances; filterpy builds the filter and "
                "allocates its matrices at every step",
            )
        )
    return rows


def main() -> None:
    """Run every benchmark and print the report to stdout."""
    warnings.simplefilter("ignore")
    print_header()
    print("\n## Results")
    bench_kernels()
    bench_throughput()
    print_table("GARCH(1,1) fit: quantcore vs arch", bench_garch_fit())
    print_table("ARIMA fit and forecast: quantcore vs statsmodels", bench_arima())
    print_table("Heston COS call price: quantcore vs QuantLib", bench_heston())
    print_table("OLS and Newey-West: quantcore vs statsmodels", bench_regression())
    print_table("Engle-Granger: quantcore vs statsmodels", bench_engle_granger())
    print_table("Historical VaR / CVaR: quantcore vs empyrical and skfolio", bench_var_cvar())
    print_table("Ledoit-Wolf shrinkage: quantcore vs scikit-learn", bench_shrinkage())
    print_table("Portfolio optimisers: quantcore vs cvxpy/CLARABEL", bench_portfolio_optimisers())
    print_table("ADF test: quantcore vs statsmodels", bench_adf())
    print_table(
        "Diagnostics (KPSS, ARCH-LM, Jarque-Bera): quantcore vs statsmodels and scipy",
        bench_diagnostics(),
    )
    print_table("Performance ratios: quantcore vs quantstats", bench_performance_ratios())
    print_table("Dynamic hedge ratio: quantcore vs filterpy", bench_dynamic_hedge_ratio())


if __name__ == "__main__":
    main()
