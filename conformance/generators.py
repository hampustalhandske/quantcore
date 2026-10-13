"""Hypothesis strategies for the conformance differential-testing suite.

Every strategy returns a small tagged dataclass carrying the drawn array(s), the *family* it came
from (so failures and the report can be grouped), and the integer seed that produced it (so a
failing example can be replayed outside hypothesis: `numpy.random.default_rng(seed)`). Stochastic
families are always generated from a hypothesis-drawn seed through `numpy.random.default_rng`,
never from hypothesis's own float/array strategies directly, so shrinking operates on the seed and
`n` rather than on individual floats, and a shrunk failure still replays exactly.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from hypothesis import strategies as st

type FloatArray = npt.NDArray[np.float64]

type ReturnFamily = Literal[
    "normal",
    "student_t3",
    "skewed",
    "ar1",
    "garch11",
    "near_constant",
    "all_positive",
    "all_negative",
    "tiny_scale",
    "huge_scale",
]

RETURN_FAMILIES: tuple[ReturnFamily, ...] = (
    "normal",
    "student_t3",
    "skewed",
    "ar1",
    "garch11",
    "near_constant",
    "all_positive",
    "all_negative",
    "tiny_scale",
    "huge_scale",
)

type RegressionFamily = Literal["homoskedastic", "heteroskedastic", "autocorrelated"]
REGRESSION_FAMILIES: tuple[RegressionFamily, ...] = (
    "homoskedastic",
    "heteroskedastic",
    "autocorrelated",
)

type CointegrationFamily = Literal["cointegrated", "independent_rw", "near_unit_root"]
COINTEGRATION_FAMILIES: tuple[CointegrationFamily, ...] = (
    "cointegrated",
    "independent_rw",
    "near_unit_root",
)

# Returns are clipped to this floor so a compounded path (cumprod(1 + r)) never touches or
# crosses zero; -0.999 keeps even the "huge_scale" family's extreme draws strictly > -1.
_MIN_RETURN = -0.999

type LengthBucket = Literal["short", "medium", "long"]

# Boundaries of the three declared coverage buckets: short 2-10, medium 11-500, long 501-5000.
_SHORT_MAX = 10
_MEDIUM_MAX = 500


def length_bucket_for(n: int) -> LengthBucket:
    """Classify an input length into the coverage bucket the conformance report tracks.

    Every target's conformance test must record at least one case in each bucket it
    declares; `build_report.py` fails the build otherwise.

    Args:
        n: Number of observations in a generated input.

    Returns:
        `"short"` for 2-10, `"medium"` for 11-500, `"long"` for 501 and above.
    """
    if n <= _SHORT_MAX:
        return "short"
    if n <= _MEDIUM_MAX:
        return "medium"
    return "long"


# (lo, hi) bounds of each declared coverage bucket, matching `length_bucket_for`'s thresholds.
_BUCKET_RANGES: tuple[tuple[int, int], ...] = (
    (2, _SHORT_MAX),
    (_SHORT_MAX + 1, _MEDIUM_MAX),
    (_MEDIUM_MAX + 1, 5000),
)


def _length_strategy(min_len: int, max_len: int) -> st.SearchStrategy[int]:
    """Draw a length from `one_of` the short/medium/long buckets, clipped to `[min_len, max_len]`.

    Coverage is enforced, not hoped for: a plain `st.integers(min_len, max_len)` draw is
    overwhelmingly likely to land in the "long" bucket once `max_len` is in the thousands, so short
    series would almost never appear. Drawing `one_of` the three buckets instead gives each an
    equal chance regardless of how wide the overall range is.

    A bucket whose clipped range comes out empty (its `hi` falls below `min_len`, e.g. the short
    bucket when `min_len` exceeds 10) is dropped from the `one_of`, so a function whose true
    minimum length excludes a whole bucket simply never draws from it — that bucket is not among
    the ones such a function declares.

    Args:
        min_len: Smallest length any drawn value may take (a function's true minimum input length).
        max_len: Largest length any drawn value may take.

    Returns:
        A strategy drawing an integer length, biased toward covering all three buckets rather than
        toward the widest one.
    """
    strategies = [
        st.integers(min_value=clipped_lo, max_value=clipped_hi)
        for lo, hi in _BUCKET_RANGES
        for clipped_lo, clipped_hi in [(max(lo, min_len), min(hi, max_len))]
        if clipped_lo <= clipped_hi
    ]
    if not strategies:
        strategies = [st.integers(min_value=min_len, max_value=max(min_len, max_len))]
    return st.one_of(strategies)


@dataclass(frozen=True)
class TaggedReturns:
    """A returns array drawn by `returns_series`, tagged with its family and seed.

    Attributes:
        values: The drawn returns, a finite, C-contiguous `float64` array, each element > -1.
        family: Which generator family produced `values`.
        seed: The `numpy.random.default_rng` seed that reproduces `values` exactly (for the
            stochastic families; deterministic for `near_constant`).
        n: Number of observations, `values.shape[0]`.
    """

    values: FloatArray
    family: str
    seed: int
    n: int


@dataclass(frozen=True)
class TaggedRegression:
    """A regression problem drawn by `regression_problem`, tagged with its error family.

    Attributes:
        y: Dependent variable, shape `(T,)`.
        x: Regressors including an intercept column, shape `(T, k)`.
        family: Which error-generating process produced `y`.
        seed: Seed reproducing the draw.
    """

    y: FloatArray
    x: FloatArray
    family: str
    seed: int


@dataclass(frozen=True)
class TaggedPair:
    """A pair of series drawn by `cointegration_pair`, tagged with its relationship family.

    Attributes:
        y: First series, shape `(T,)`.
        x: Second series, shape `(T,)`.
        family: Which data-generating process produced the pair.
        seed: Seed reproducing the draw.
    """

    y: FloatArray
    x: FloatArray
    family: str
    seed: int


def _simulate_returns(family: ReturnFamily, n: int, rng: np.random.Generator) -> FloatArray:
    if family == "normal":
        values = rng.normal(0.0005, 0.01, n)
    elif family == "student_t3":
        values = rng.standard_t(3, n) * 0.01
    elif family == "skewed":
        values = 0.02 * (rng.lognormal(mean=0.0, sigma=0.5, size=n) - 1.0)
    elif family == "ar1":
        eps = rng.normal(0.0, 0.01, n)
        values = np.empty(n, dtype=np.float64)
        values[0] = eps[0]
        phi = 0.6
        for t in range(1, n):
            values[t] = phi * values[t - 1] + eps[t]
    elif family == "garch11":
        omega, alpha, beta = 1e-6, 0.1, 0.85
        variance = np.empty(n, dtype=np.float64)
        values = np.empty(n, dtype=np.float64)
        z = rng.normal(0.0, 1.0, n)
        variance[0] = omega / (1.0 - alpha - beta)
        values[0] = np.sqrt(variance[0]) * z[0]
        for t in range(1, n):
            variance[t] = omega + alpha * values[t - 1] ** 2 + beta * variance[t - 1]
            values[t] = np.sqrt(variance[t]) * z[t]
    elif family == "near_constant":
        values = np.full(n, 0.0004, dtype=np.float64) + rng.normal(0.0, 1e-9, n)
    elif family == "all_positive":
        values = np.abs(rng.normal(0.005, 0.01, n))
    elif family == "all_negative":
        values = -np.abs(rng.normal(0.005, 0.01, n))
    elif family == "tiny_scale":
        values = rng.normal(0.0, 1e-8, n)
    elif family == "huge_scale":
        values = rng.normal(0.0, 1e3, n)
    return np.clip(np.asarray(values, dtype=np.float64), _MIN_RETURN, None)


@st.composite
def returns_series(
    draw: st.DrawFn,
    min_len: int = 2,
    max_len: int = 5000,
    families: tuple[ReturnFamily, ...] | None = None,
) -> TaggedReturns:
    """Draw a finite `float64` returns array from one of the return-family generators.

    Args:
        draw: Hypothesis's draw function (supplied automatically by `@st.composite`).
        min_len: Minimum number of observations.
        max_len: Maximum number of observations.
        families: Subset of `RETURN_FAMILIES` to draw from; defaults to all of them.

    Returns:
        A `TaggedReturns` carrying the array, its family, and the seed that reproduces it.
    """
    family = draw(st.sampled_from(families or RETURN_FAMILIES))
    n = draw(_length_strategy(min_len, max_len))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    values = _simulate_returns(family, n, np.random.default_rng(seed))
    return TaggedReturns(values=values, family=family, seed=seed, n=n)


@st.composite
def regression_problem(draw: st.DrawFn, min_len: int = 20, max_len: int = 5000) -> TaggedRegression:
    """Draw an OLS regression problem `(y, x)` with an intercept column and one error process.

    Args:
        draw: Hypothesis's draw function.
        min_len: Minimum number of observations.
        max_len: Maximum number of observations.

    Returns:
        A `TaggedRegression` with `x`'s first column all ones, a second column of regressor
        values, and `y` generated under a homoskedastic, heteroskedastic, or autocorrelated error
        process.
    """
    family = draw(st.sampled_from(REGRESSION_FAMILIES))
    n = draw(_length_strategy(min_len, max_len))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    regressor = rng.normal(0.0, 1.0, n)
    intercept, slope = 0.5, 1.5

    if family == "homoskedastic":
        errors = rng.normal(0.0, 1.0, n)
    elif family == "heteroskedastic":
        scale = 0.2 + np.abs(regressor)
        errors = rng.normal(0.0, 1.0, n) * scale
    else:  # autocorrelated
        innovations = rng.normal(0.0, 1.0, n)
        errors = np.empty(n, dtype=np.float64)
        errors[0] = innovations[0]
        phi = 0.7
        for t in range(1, n):
            errors[t] = phi * errors[t - 1] + innovations[t]

    y = intercept + slope * regressor + errors
    x = np.column_stack([np.ones(n), regressor]).astype(np.float64)
    return TaggedRegression(y=y.astype(np.float64), x=x, family=family, seed=seed)


@st.composite
def cointegration_pair(draw: st.DrawFn, min_len: int = 50, max_len: int = 5000) -> TaggedPair:
    """Draw a pair of series `(y, x)` that is cointegrated, independent, or near-unit-root.

    Args:
        draw: Hypothesis's draw function.
        min_len: Minimum number of observations.
        max_len: Maximum number of observations.

    Returns:
        A `TaggedPair` whose `family` is `"cointegrated"` (`y` and `x` share a common random-walk
        component plus a stationary spread), `"independent_rw"` (two unrelated random walks, not
        cointegrated), or `"near_unit_root"` (a highly persistent but stationary AR(1) pair, the
        boundary case a residual-based test must not mistake for either extreme).
    """
    family = draw(st.sampled_from(COINTEGRATION_FAMILIES))
    n = draw(_length_strategy(min_len, max_len))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    if family == "cointegrated":
        common = np.cumsum(rng.normal(0.0, 1.0, n))
        x = common + rng.normal(0.0, 0.1, n)
        y = 1.5 * common + rng.normal(0.0, 0.3, n)
    elif family == "independent_rw":
        x = np.cumsum(rng.normal(0.0, 1.0, n))
        y = np.cumsum(rng.normal(0.0, 1.0, n))
    else:  # near_unit_root
        phi = 0.995
        x = np.empty(n, dtype=np.float64)
        y = np.empty(n, dtype=np.float64)
        x[0] = rng.normal(0.0, 1.0)
        y[0] = rng.normal(0.0, 1.0)
        ex = rng.normal(0.0, 1.0, n)
        ey = rng.normal(0.0, 1.0, n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + ex[t]
            y[t] = phi * y[t - 1] + ey[t]

    return TaggedPair(
        y=np.asarray(y, dtype=np.float64),
        x=np.asarray(x, dtype=np.float64),
        family=family,
        seed=seed,
    )


# --- portfolio family: covariance matrices and returns panels -----------------------------------

type CovarianceFamily = Literal[
    "well_conditioned", "near_singular", "equal_vol", "one_dominant", "high_condition"
]
COVARIANCE_FAMILIES: tuple[CovarianceFamily, ...] = (
    "well_conditioned",
    "near_singular",
    "equal_vol",
    "one_dominant",
    "high_condition",
)

# Congruence-transform scale applied to the first asset for the "one_dominant" family: cov' =
# D@cov@D with D = diag(scale, 1, ..., 1) stays PSD for any base PSD cov (a congruence transform
# preserves definiteness), and makes asset 0's variance ~scale**2 times the others'.
_ONE_DOMINANT_SCALE = 50.0


@dataclass(frozen=True)
class TaggedCovariance:
    """A `(k, k)` PSD covariance matrix drawn by `covariance_matrix`, tagged with its family.

    Attributes:
        matrix: The drawn covariance matrix, symmetric PSD, `float64`.
        family: Which generator family produced `matrix`.
        seed: The `numpy.random.default_rng` seed that reproduces `matrix` exactly.
        k: Dimension, `matrix.shape[0]`.
        condition_number: `numpy.linalg.cond(matrix)`, recorded at draw time so a conformance test
            can report or filter on it without recomputing.
    """

    matrix: FloatArray
    family: str
    seed: int
    k: int
    condition_number: float


def _simulate_covariance(family: CovarianceFamily, k: int, rng: np.random.Generator) -> FloatArray:
    """Build a `(k, k)` PSD covariance matrix with the conditioning `family` names.

    Every family is built as `Q @ diag(eigvals) @ Q.T` for a random orthogonal `Q` (from the QR
    decomposition of a random Gaussian matrix), so the result is exactly PSD by construction
    (eigenvalues drawn strictly positive) rather than merely "usually" PSD, and its condition
    number is `max(eigvals) / min(eigvals)` by construction, not a downstream measurement.
    """
    orthogonal, _ = np.linalg.qr(rng.normal(0.0, 1.0, (k, k)))
    if family == "well_conditioned":
        eigvals = rng.uniform(0.5, 2.0, k)
    elif family == "near_singular":
        eigvals = rng.uniform(0.5, 2.0, k)
        # 1e-6, not smaller: a condition number near 1e6 already stresses a numerical solver
        # while staying within reach of double-precision optimisation (kappa near 1e10+ makes
        # even the CORRECT optimum numerically unrecoverable by any general-purpose solver, which
        # would make this family untestable rather than merely hard).
        eigvals[0] = 1e-6
    elif family == "equal_vol":
        eigvals = rng.uniform(0.3, 3.0, k)
    elif family == "one_dominant":
        eigvals = rng.uniform(0.5, 2.0, k)
    else:  # high_condition
        # Spans condition numbers up to ~1e6 (not 1e9+): the same double-precision-recoverability
        # limit as "near_singular" above -- a log-barrier risk-parity solve along the smallest
        # eigenvalue's direction has an optimal weight scaling like 1/sqrt(kappa), so a condition
        # number much beyond this pushes even the mathematically-correct optimum outside what any
        # general-purpose double-precision solver can locate reliably.
        eigvals = np.exp(rng.uniform(np.log(1e-3), np.log(1e3), k))

    cov = (orthogonal * eigvals) @ orthogonal.T
    if family == "equal_vol":
        # Rescale to a correlation matrix (unit diagonal): every asset has the same variance, only
        # the correlation structure varies, per the family's name.
        scale = np.sqrt(np.diag(cov))
        cov = cov / np.outer(scale, scale)
    elif family == "one_dominant":
        scale_vector = np.ones(k)
        scale_vector[0] = _ONE_DOMINANT_SCALE
        cov = np.outer(scale_vector, scale_vector) * cov
    return np.asarray(cov, dtype=np.float64)


@st.composite
def covariance_matrix(
    draw: st.DrawFn,
    min_k: int = 2,
    max_k: int = 10,
    families: tuple[CovarianceFamily, ...] | None = None,
) -> TaggedCovariance:
    """Draw a `(k, k)` PSD covariance matrix from one of the conditioning-family generators.

    Args:
        draw: Hypothesis's draw function.
        min_k: Minimum matrix dimension.
        max_k: Maximum matrix dimension.
        families: Subset of `COVARIANCE_FAMILIES` to draw from; defaults to all of them.

    Returns:
        A `TaggedCovariance` carrying the matrix, its family, dimension, seed, and condition
        number.
    """
    family = draw(st.sampled_from(families or COVARIANCE_FAMILIES))
    k = draw(st.integers(min_value=min_k, max_value=max_k))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    matrix = _simulate_covariance(family, k, np.random.default_rng(seed))
    return TaggedCovariance(
        matrix=matrix, family=family, seed=seed, k=k, condition_number=float(np.linalg.cond(matrix))
    )


@dataclass(frozen=True)
class TaggedPanel:
    """A `(T, k)` returns panel drawn by `returns_panel`, tagged with its generating family.

    Attributes:
        values: The drawn panel, finite, C-contiguous `float64`, shape `(n, k)`.
        family: Which `COVARIANCE_FAMILIES` member generated the panel's covariance structure.
        seed: The `numpy.random.default_rng` seed that reproduces `values` exactly.
        n: Number of observations, `values.shape[0]`.
        k: Number of assets, `values.shape[1]`.
    """

    values: FloatArray
    family: str
    seed: int
    n: int
    k: int


@st.composite
def returns_panel(
    draw: st.DrawFn,
    min_len: int = 20,
    max_len: int = 2000,
    min_k: int = 2,
    max_k: int = 8,
    families: tuple[CovarianceFamily, ...] | None = None,
) -> TaggedPanel:
    """Draw a `(T, k)` multivariate-normal returns panel with a declared conditioning family.

    Args:
        draw: Hypothesis's draw function.
        min_len: Minimum number of observations `T`.
        max_len: Maximum number of observations `T`.
        min_k: Minimum number of assets `k`.
        max_k: Maximum number of assets `k`.
        families: Subset of `COVARIANCE_FAMILIES` to draw the generating covariance from;
            defaults to all of them.

    Returns:
        A `TaggedPanel` carrying the simulated panel, its generating covariance's family, `n`,
        `k`, and the seed reproducing it.
    """
    family = draw(st.sampled_from(families or COVARIANCE_FAMILIES))
    k = draw(st.integers(min_value=min_k, max_value=max_k))
    n = draw(_length_strategy(max(min_len, k + 2), max_len))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    cov = _simulate_covariance(family, k, rng)
    # A small diagonal jitter keeps the Cholesky factor well-defined for "near_singular" (whose
    # smallest eigenvalue is 1e-10 by construction) without perturbing any other family's already
    # well-conditioned matrix in any way that matters at float64 precision.
    chol = np.linalg.cholesky(cov + np.eye(k) * 1e-9)
    innovations = rng.normal(0.0, 1.0, (n, k))
    values = (innovations @ chol.T) * 0.02
    return TaggedPanel(
        values=np.asarray(values, dtype=np.float64), family=family, seed=seed, n=n, k=k
    )


# --- time_series family: ARMA series and small VAR systems (conformance.targets.time_series)

type ArmaFamily = Literal["white_noise", "ar1", "ar2", "ma1", "ma2", "arma11", "near_unit_root"]
ARMA_FAMILIES: tuple[ArmaFamily, ...] = (
    "white_noise",
    "ar1",
    "ar2",
    "ma1",
    "ma2",
    "arma11",
    "near_unit_root",
)

# True (ar_coefs, ma_coefs) used by `_simulate_arma` for each family, known-order data-generating
# processes so an order-selection test can check whether the search recovers them.
_ARMA_TRUE_PARAMS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "white_noise": ((), ()),
    "ar1": ((0.6,), ()),
    "ar2": ((0.5, -0.3), ()),
    "ma1": ((), (0.5,)),
    "ma2": ((), (0.4, -0.2)),
    "arma11": ((0.5,), (0.3,)),
    "near_unit_root": ((0.97,), ()),
}


@dataclass(frozen=True)
class TaggedArma:
    """An ARMA(p, q) series drawn by `arma_series`, tagged with the process that produced it.

    Attributes:
        values: The drawn series, a finite, C-contiguous `float64` array.
        family: Which generator family produced `values` (see `ARMA_FAMILIES`).
        seed: The `numpy.random.default_rng` seed that reproduces `values` exactly.
        n: Number of observations, `values.shape[0]`.
        true_ar: The AR coefficients actually used to simulate `values` (empty for a pure-MA or
            white-noise family).
        true_ma: The MA coefficients actually used to simulate `values` (empty for a pure-AR or
            white-noise family).
    """

    values: FloatArray
    family: str
    seed: int
    n: int
    true_ar: FloatArray
    true_ma: FloatArray


def _simulate_arma(family: ArmaFamily, n: int, rng: np.random.Generator) -> FloatArray:
    """Simulate a stationary/near-unit-root ARMA(p, q) series with the sign convention.

    `y_t = sum_i phi_i*y_{t-i} + e_t + sum_j theta_j*e_{t-j}`, matching the `theta(B) = 1 +
    theta_1*B + ...` polynomial `quantcore.time_series.arima`'s module docstring documents, so a
    fitted `(ar_coefs, ma_coefs)` can be compared directly against `_ARMA_TRUE_PARAMS`.
    """
    ar, ma = _ARMA_TRUE_PARAMS[family]
    burn_in = 200
    total = n + burn_in
    innovations = rng.normal(0.0, 1.0, total)
    values = np.zeros(total, dtype=np.float64)
    p, q = len(ar), len(ma)
    for t in range(total):
        ar_term = sum(ar[i] * values[t - 1 - i] for i in range(p) if t - 1 - i >= 0)
        ma_term = sum(ma[j] * innovations[t - 1 - j] for j in range(q) if t - 1 - j >= 0)
        values[t] = ar_term + innovations[t] + ma_term
    return np.asarray(values[burn_in:], dtype=np.float64)


@st.composite
def arma_series(
    draw: st.DrawFn,
    min_len: int = 20,
    max_len: int = 2000,
    families: tuple[ArmaFamily, ...] | None = None,
) -> TaggedArma:
    """Draw a finite `float64` series from a known-order stationary/near-unit-root ARMA process.

    Args:
        draw: Hypothesis's draw function.
        min_len: Minimum number of observations.
        max_len: Maximum number of observations.
        families: Subset of `ARMA_FAMILIES` to draw from; defaults to all of them.

    Returns:
        A `TaggedArma` carrying the series, its family, the seed that reproduces it, and the true
        `(ar, ma)` coefficients used to simulate it.
    """
    family = draw(st.sampled_from(families or ARMA_FAMILIES))
    n = draw(_length_strategy(min_len, max_len))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    values = _simulate_arma(family, n, np.random.default_rng(seed))
    true_ar, true_ma = _ARMA_TRUE_PARAMS[family]
    return TaggedArma(
        values=values,
        family=family,
        seed=seed,
        n=n,
        true_ar=np.array(true_ar, dtype=np.float64),
        true_ma=np.array(true_ma, dtype=np.float64),
    )


type VarFamily = Literal["var2_stationary", "var3_stationary", "var2_near_unit_root"]
VAR_FAMILIES: tuple[VarFamily, ...] = ("var2_stationary", "var3_stationary", "var2_near_unit_root")

# Fixed, stable coefficient matrices (spectral radius of the companion form < 1) per family, so
# every draw from a given family shares the same known data-generating process and differs only
# in its innovations and length.
_VAR2_STATIONARY_A = np.array([[0.5, 0.2], [0.1, 0.4]], dtype=np.float64)
_VAR3_STATIONARY_A = np.array([[0.4, 0.1, 0.0], [0.0, 0.3, 0.2], [0.1, 0.0, 0.5]], dtype=np.float64)
_VAR2_NEAR_UNIT_ROOT_A = np.array([[0.9, 0.05], [0.0, 0.9]], dtype=np.float64)


@dataclass(frozen=True)
class TaggedVarSystem:
    """A small VAR(1) system drawn by `var_system`, tagged with the process that produced it.

    Attributes:
        data: Observations, shape `(T, k)`.
        family: Which generator family produced `data` (see `VAR_FAMILIES`).
        seed: The `numpy.random.default_rng` seed that reproduces `data` exactly.
        n: Number of observations, `data.shape[0]`.
        k: Number of variables, `data.shape[1]`.
        true_a: The `(k, k)` VAR(1) coefficient matrix actually used to simulate `data`.
    """

    data: FloatArray
    family: str
    seed: int
    n: int
    k: int
    true_a: FloatArray


def _var1_coef_matrix(family: VarFamily) -> FloatArray:
    if family == "var2_stationary":
        return _VAR2_STATIONARY_A
    if family == "var3_stationary":
        return _VAR3_STATIONARY_A
    return _VAR2_NEAR_UNIT_ROOT_A


@st.composite
def var_system(
    draw: st.DrawFn,
    min_len: int = 30,
    max_len: int = 2000,
    families: tuple[VarFamily, ...] | None = None,
) -> TaggedVarSystem:
    """Draw a stationary or near-unit-root VAR(1) system with 2 or 3 variables.

    Args:
        draw: Hypothesis's draw function.
        min_len: Minimum number of observations.
        max_len: Maximum number of observations.
        families: Subset of `VAR_FAMILIES` to draw from; defaults to all of them.

    Returns:
        A `TaggedVarSystem` carrying the `(T, k)` data, its family, the seed that reproduces it,
        and the true `(k, k)` VAR(1) coefficient matrix (zero intercept) used to simulate it.
    """
    family = draw(st.sampled_from(families or VAR_FAMILIES))
    n = draw(_length_strategy(min_len, max_len))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    a = _var1_coef_matrix(family)
    k = a.shape[0]
    burn_in = 200
    total = n + burn_in
    innovations = rng.normal(0.0, 1.0, size=(total, k))
    data = np.zeros((total, k), dtype=np.float64)
    for t in range(1, total):
        data[t] = a @ data[t - 1] + innovations[t]

    return TaggedVarSystem(
        data=np.asarray(data[burn_in:], dtype=np.float64),
        family=family,
        seed=seed,
        n=n,
        k=k,
        true_a=a,
    )
