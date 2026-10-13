"""Ljung-Box autocorrelation testing and ARIMA(p, d, q) fitting/forecasting.

ARIMA(p, d, q) (Box, Jenkins & Reinsel 2015):
    phi(B) * Delta^d(y_t) = theta(B) * e_t,   e_t ~ N(0, sigma2) i.i.d.
    where phi(B) = 1 - phi_1*B - ... - phi_p*B^p  (AR polynomial)
          theta(B) = 1 + theta_1*B + ... + theta_q*B^q (MA polynomial)
          Delta = (1 - B) (difference operator)
    No constant is estimated: the differenced series w_t = Delta^d(y_t) is
    modeled as a zero-mean ARMA(p, q).

Exact maximum likelihood (`arima_fit`'s default; Gardner, Harvey &
Phillips 1980; Harvey 1989, ch. 3):
    w_t is put in state-space form with r = max(p, q + 1) states,

        alpha_t = T * alpha_{t-1} + R * e_t,    w_t = alpha_t[0]
        T = [[phi_1, 1, 0, ..], [phi_2, 0, 1, ..], .., [phi_r, 0, .., 0]]
        R = (1, theta_1, ..., theta_{r-1})'
        (phi_j = 0 for j > p, theta_j = 0 for j > q)

    and the pre-sample state is given its stationary distribution,
    alpha_0 ~ N(0, sigma2 * P0) with P0 = T*P0*T' + R*R'. The Kalman filter
    then yields the one-step prediction errors v_t and their variances
    sigma2 * F_t, and with them the likelihood of all n observations of
    w_t (the prediction-error decomposition):

        log L = -n/2 * log(2*pi*sigma2) - 1/2 * sum_t log(F_t)
                - 1/(2*sigma2) * sum_t v_t^2 / F_t

    sigma2 is maximized out in closed form, sigma2_hat = (1/n) * sum_t
    v_t^2 / F_t, which leaves a function of (phi, theta) alone to maximize
    numerically. phi(B) is kept stationary and theta(B) invertible by
    optimizing over unconstrained variables u whose tanh are the partial
    autocorrelations of the AR part and of the MA part, mapped to
    coefficients by the Durbin-Levinson recursion (Jones 1980;
    Barndorff-Nielsen & Schou 1973). The search starts from the two-stage
    regression estimate of Hannan & Rissanen (1982).

Conditional sum of squares (`arima_fit(..., method="css")`; Box, Jenkins &
Reinsel 2015, ch. 7):
    residuals are computed recursively from the AR/MA polynomials with
    pre-sample residuals (and, for the CSS objective, pre-sample y-lags)
    treated as 0, and the parameters minimize sum_t e_t^2. It approximates
    the exact likelihood by conditioning on the pre-sample values instead
    of integrating over them, and agrees with it only as n grows.

Ljung-Box Q-statistic (Ljung & Box 1978):
    Q(m) = T*(T+2) * sum_{k=1}^m rho_k^2 / (T-k)
    ~ chi^2(m) under H0: no autocorrelation up to lag m

References:
    Barndorff-Nielsen, O. and Schou, G. (1973). "On the Parametrization of
    Autoregressive Models by Partial Autocorrelations." *Journal of
    Multivariate Analysis*, 3(4), 408-419.

    Box, G.E.P., Jenkins, G.M., Reinsel, G.C., and Ljung, G.M. (2015). *Time
    Series Analysis: Forecasting and Control* (5th ed.). Wiley.

    Gardner, G., Harvey, A.C., and Phillips, G.D.A. (1980). "Algorithm AS
    154: An Algorithm for Exact Maximum Likelihood Estimation of
    Autoregressive-Moving Average Models by Means of Kalman Filtering."
    *Applied Statistics*, 29(3), 311-322.

    Hannan, E.J. and Rissanen, J. (1982). "Recursive Estimation of Mixed
    Autoregressive-Moving Average Order." *Biometrika*, 69(1), 81-94.

    Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and
    the Kalman Filter*. Cambridge University Press.

    Jones, R.H. (1980). "Maximum Likelihood Fitting of ARMA Models to Time
    Series with Missing Observations." *Technometrics*, 22(3), 389-395.

    Ljung, G.M. and Box, G.E.P. (1978). "On a Measure of Lack of Fit in Time
    Series Models." *Biometrika*, 65(2), 297-303.

    See docs/REFERENCES.md.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, NamedTuple

import numba
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import chi2

from quantcore.risk.volatility import ConvergenceError


def _validate_ljung_box_inputs(residuals: npt.NDArray[np.float64], n_lags: int) -> None:
    if residuals.size == 0:
        raise ValueError("residuals must be non-empty")
    if n_lags <= 0:
        raise ValueError("n_lags must be a positive integer")
    if n_lags >= residuals.size:
        raise ValueError("n_lags must be smaller than the number of residuals")


def ljung_box_test(
    residuals: npt.NDArray[np.float64],
    n_lags: int,
    model_df: int = 0,
) -> tuple[float, float]:
    """Ljung-Box Q-test for autocorrelation in a residual series.

    Args:
        residuals: Residual series to test.
        n_lags: Number of lags to include in the test statistic.
        model_df: Number of fitted ARMA parameters (p + q) to subtract from
            the chi-squared degrees of freedom. 0 (default) is the
            original, only behavior: `df = n_lags`, appropriate when
            testing a raw series that wasn't fit by an ARMA model.
            When `residuals` come from an ARMA(p, q) fit and you're
            testing whether autocorrelation remains, the standard
            adjustment is `df = n_lags - p - q` (matches `statsmodels`'
            `acorr_ljungbox`'s `model_df`) — using the unadjusted `n_lags`
            in that case overstates the degrees of freedom, understating
            significance (a Q-statistic that should reject the
            no-autocorrelation null more easily reports a p-value that's
            too large). Must satisfy `model_df < n_lags`.

    Returns:
        Tuple (q_statistic, p_value), with p_value from
        `chi2.sf(Q, df=n_lags - model_df)`.

    References:
        Ljung, G.M. and Box, G.E.P. (1978). "On a Measure of Lack of Fit
        in Time Series Models." *Biometrika*, 65(2), 297-303. See
        docs/REFERENCES.md.
    """
    _validate_ljung_box_inputs(residuals, n_lags)
    if model_df < 0:
        raise ValueError("model_df must be non-negative")
    if model_df >= n_lags:
        raise ValueError("model_df must be strictly less than n_lags")
    return _ljung_box_test(residuals, n_lags, model_df)


def _ljung_box_test(
    residuals: npt.NDArray[np.float64], n_lags: int, model_df: int = 0
) -> tuple[float, float]:
    n = residuals.size
    centered = residuals - residuals.mean()
    denom = float(np.sum(centered**2))

    autocorr = np.empty(n_lags, dtype=np.float64)
    for k in range(1, n_lags + 1):
        autocorr[k - 1] = float(np.sum(centered[k:] * centered[:-k])) / denom

    lags = np.arange(1, n_lags + 1, dtype=np.float64)
    q_statistic = float(n * (n + 2) * np.sum(autocorr**2 / (n - lags)))
    p_value = float(chi2.sf(q_statistic, df=n_lags - model_df))
    return q_statistic, p_value


def _validate_arima_series(series: npt.NDArray[np.float64]) -> None:
    if series.size == 0:
        raise ValueError("series must be non-empty")
    if series.ndim != 1:
        raise ValueError("series must be a 1-D array")
    if not np.all(np.isfinite(series)):
        raise ValueError("series must contain only finite values")


def _validate_arima_fit_inputs(series: npt.NDArray[np.float64], p: int, d: int, q: int) -> None:
    _validate_arima_series(series)
    if p < 0:
        raise ValueError("p must be non-negative")
    if q < 0:
        raise ValueError("q must be non-negative")
    if d not in (0, 1, 2):
        raise ValueError("d must be 0, 1, or 2")
    if p == 0 and q == 0:
        raise ValueError("p and q cannot both be 0")
    if series.size <= d + max(p, q):
        raise ValueError("series is too short for the requested (p, d, q) order")


def arima_fit(
    series: npt.NDArray[np.float64],
    p: int,
    d: int,
    q: int,
    *,
    method: Literal["mle", "css"] = "mle",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Estimate ARIMA(p, d, q) parameters, by exact maximum likelihood by default.

    The series is differenced `d` times and a zero-mean ARMA(p, q) is fit
    to the result, `w_t`; no constant is estimated.

    `method="mle"` maximizes the exact Gaussian likelihood of all
    `n = series.size - d` values of `w_t`, the standard definition of the
    ARMA maximum-likelihood estimate: the pre-sample state has its
    stationary distribution and the likelihood is evaluated by a Kalman
    filter (see the module docstring). The AR polynomial is constrained to
    be stationary and the MA polynomial to be invertible. The likelihood
    of a mixed model can have several local maxima, so L-BFGS-B is run
    from the Hannan-Rissanen regression estimate, from white noise (all
    coefficients 0), and from the fits of orders (p - 1, q) and
    (p, q - 1), themselves obtained this way, each extended by a zero
    coefficient; the best optimum is returned. The result therefore never
    has a lower likelihood than the fit of an order nested in it, at the
    cost of fitting every order (i, j) with i <= p and j <= q. These
    starting points do not guarantee the global maximum.

    `method="css"` minimizes the conditional sum of squares instead: the
    residual recursion is started with every pre-sample value set to 0 and
    the first `max(p, q)` values of `w_t` are used only as lags. It is an
    approximation to the estimator above that converges to it as `n`
    grows, places no constraint on the coefficients, and attains a lower
    exact likelihood on any finite sample.

    Args:
        series: Raw (undifferenced) time series, 1-D and finite.
        p: AR order.
        d: Integration order (0, 1, or 2).
        q: MA order.
        method: Estimator, `"mle"` (exact Gaussian maximum likelihood) or
            `"css"` (conditional sum of squares). Keyword-only.

    Returns:
        Tuple (ar_coefficients, ma_coefficients, sigma2_hat):
        `ar_coefficients` is `phi_1..phi_p`, shape (p,); `ma_coefficients`
        is `theta_1..theta_q`, shape (q,), with the sign convention
        `theta(B) = 1 + theta_1*B + ...`; `sigma2_hat` is the innovation
        variance estimate.

        With `method="mle"`, `sigma2_hat` is the maximum-likelihood
        variance `(1/n) * sum_t v_t^2 / F_t` (divisor `n`, no
        degrees-of-freedom correction). The returned AR coefficients are
        stationary and the MA coefficients invertible, exactly, as the
        floating-point numbers they are: every partial autocorrelation of
        either part is at most `1 - 1e-6` in magnitude and the rounded
        coefficients are verified in rational arithmetic before they are
        returned. When the likelihood has no maximum inside that region
        (for example a differenced series that is constant, which an AR
        root of 1 would reproduce exactly), the estimate lies on its edge.
        A differenced series that is identically zero carries no
        information about the coefficients and returns all-zero
        coefficients with `sigma2_hat = 0.0`.

        With `method="css"`, `sigma2_hat = SSE / (n - max(p, q))`.

    Raises:
        ValueError: If `series` is empty, not 1-D, or contains a non-finite
            value; if `p` or `q` is negative, both are 0, or `d` is not 0,
            1, or 2; if `series` has at most `d + max(p, q)` values; or if
            `method` is not `"mle"` or `"css"`.
        ConvergenceError: With `method="css"`, if Nelder-Mead stops on its
            iteration or evaluation limit; `params` then holds the point
            it stopped at and `log_likelihood` the conditional Gaussian
            log-likelihood there. With `method="mle"`, if the optimizer
            converges from no starting point (`quantcore.risk.ConvergenceError`,
            a `RuntimeError` subclass, the same exception the GARCH-family
            fitters raise). No partial result is returned; the exception's
            `params` holds the best attempt as
            `(*ar_coefficients, *ma_coefficients, sigma2_hat)`, with its
            exact `log_likelihood` and the `optimizer_message`, or `None`
            for all three if the optimizer stopped at no usable point.
            Non-convergence happens when a sample is too short to
            identify the model (about as many observations as
            coefficients). No usable point is left when the values are
            too large or too small for their squares to be represented,
            or when the likelihood peaks where three or more partial
            autocorrelations of one polynomial reach their bound
            together: the stationary region is then thinner than
            floating-point resolution and has no representable point.

    References:
        Gardner, G., Harvey, A.C., and Phillips, G.D.A. (1980). "Algorithm
        AS 154: An Algorithm for Exact Maximum Likelihood Estimation of
        Autoregressive-Moving Average Models by Means of Kalman
        Filtering." *Applied Statistics*, 29(3), 311-322. (State-space
        form, stationary initial covariance, exact likelihood.)

        Harvey, A.C. (1989). *Forecasting, Structural Time Series Models
        and the Kalman Filter*. Cambridge University Press. (Ch. 3:
        Kalman filter, prediction-error decomposition, concentrated
        likelihood.)

        Jones, R.H. (1980). "Maximum Likelihood Fitting of ARMA Models to
        Time Series with Missing Observations." *Technometrics*, 22(3),
        389-395. (Partial-autocorrelation reparameterization.)

        Hannan, E.J. and Rissanen, J. (1982). "Recursive Estimation of
        Mixed Autoregressive-Moving Average Order." *Biometrika*, 69(1),
        81-94. (Starting values.)

        Box, G.E.P., Jenkins, G.M., Reinsel, G.C., and Ljung, G.M. (2015).
        *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
        (The model; ch. 7 for the conditional sum of squares.) See
        docs/REFERENCES.md.
    """
    if method not in ("mle", "css"):
        raise ValueError('method must be "mle" or "css"')
    _validate_arima_fit_inputs(series, p, d, q)
    if method == "css":
        return _arima_fit_css(series, p, d, q)
    return _arima_fit_exact_mle(series, p, d, q)


def _validate_arima_residuals_inputs(
    series: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    ma_coefs: npt.NDArray[np.float64],
    d: int,
) -> None:
    if series.size == 0:
        raise ValueError("series must be non-empty")
    if ar_coefs.ndim != 1:
        raise ValueError("ar_coefs must be a 1-D array")
    if ma_coefs.ndim != 1:
        raise ValueError("ma_coefs must be a 1-D array")
    if d not in (0, 1, 2):
        raise ValueError("d must be 0, 1, or 2")
    if series.size <= d + max(ar_coefs.size, ma_coefs.size):
        raise ValueError("series is too short for the given (ar_coefs, ma_coefs, d)")


def arima_residuals(
    series: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    ma_coefs: npt.NDArray[np.float64],
    d: int,
) -> npt.NDArray[np.float64]:
    """Compute the conditional (CSS) residual series of an ARIMA model at given coefficients.

    `arima_fit` returns only the fitted coefficients and `sigma2_hat`, with
    no residual series to diagnose the fit with. Pass this function's
    output straight into `ljung_box_test` to check whether the fit actually
    whitened the series -- passing `model_df=p + q` (the fitted ARMA
    parameter count), since these residuals come from a fitted model, not a
    raw series:

        ar_coefs, ma_coefs, _ = arima_fit(series, p, d, q)
        residuals = arima_residuals(series, ar_coefs, ma_coefs, d)
        _, p_value = ljung_box_test(residuals, n_lags, model_df=p + q)

    A high `p_value` means the residuals are consistent with white noise, so
    the chosen (p, d, q) order adequately removed the series' autocorrelation.
    A low `p_value` means significant autocorrelation remains and the order
    (or the fit) should be reconsidered -- this is the proper POST-fit
    analogue of a cruder pre-fit Ljung-Box gate run on the raw series.

    These are conditional residuals: the recursion
    `e_t = w_t - sum_i phi_i*w_{t-i} - sum_j theta_j*e_{t-j}` on the
    differenced series `w_t`, started at index `max(p, q)` with every
    earlier residual set to 0, i.e. conditional on the unobserved
    pre-sample innovations being zero. They are not the exact one-step
    prediction errors `v_t = w_t - E[w_t | w_1..w_{t-1}]` that the Kalman
    filter of the exact likelihood produces, which treat the pre-sample
    state as random. The two differ as follows:

    - There are `n - max(p, q)` conditional residuals, all taken to have
      variance sigma2. There are `n` prediction errors, one per
      observation, with variances `sigma2 * F_t` that start above sigma2
      and, for an invertible MA part, decrease to it.
    - With no MA part the conditional residuals equal the prediction
      errors from index p onward; only the first p errors are missing.
    - With an invertible MA part they differ at every index, by an amount
      that shrinks geometrically at the rate of the largest MA root
      modulus: visible in the first observations, negligible after a few
      dozen unless a root is close to the unit circle.

    For coefficients from `arima_fit(..., method="css")` these are exactly
    the residuals whose sum of squares that fit minimized, and
    `sum(residuals**2) / residuals.size` is its `sigma2_hat`. For
    coefficients from the default exact-likelihood fit their mean square
    is close to but not equal to its `sigma2_hat`.

    Args:
        series: Raw (undifferenced) series, on the same scale `arima_fit`
            was called with.
        ar_coefs: Fitted AR coefficients, shape (p,) -- e.g. `arima_fit`'s
            first return value.
        ma_coefs: Fitted MA coefficients, shape (q,) -- e.g. `arima_fit`'s
            second return value.
        d: Integration order used when fitting.

    Returns:
        CSS residual series on the differenced scale, from index
        `max(p, q)` onward, shape `(series.size - d - max(p, q),)`: the
        first `max(p, q)` differenced values serve only as lags.

    References:
        Box, G.E.P., Jenkins, G.M., Reinsel, G.C., and Ljung, G.M. (2015).
        *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
        (Ch. 7: conditional sum-of-squares residual recursion.) See
        docs/REFERENCES.md.
    """
    _validate_arima_residuals_inputs(series, ar_coefs, ma_coefs, d)
    differenced = _difference(series, d)
    phi = np.ascontiguousarray(ar_coefs, dtype=np.float64)
    ma = np.ascontiguousarray(ma_coefs, dtype=np.float64)
    return _css_residuals_numba_kernel(phi, ma, differenced)


def select_arima_order(
    series: npt.NDArray[np.float64],
    max_p: int,
    max_d: int,
    max_q: int,
    criterion: str = "aic",
) -> tuple[int, int, int]:
    """Grid-search ARIMA(p, d, q) orders and return the one minimizing an information criterion.

    Each candidate is fit by exact Gaussian maximum likelihood, exactly as
    `arima_fit` fits it by default, and scored on its maximized exact
    log-likelihood `log L` over the `n = series.size - d` differenced
    observations, with `k = p + q + 1` estimated parameters (the
    coefficients and the innovation variance):

        AIC = 2*k - 2*log L,    BIC = k*log(n) - 2*log L

    All (p, q) candidates at one `d` are scored on the same `n`
    observations, so their criteria are directly comparable. Candidates
    with different `d` are not: differencing changes both the data being
    modeled and its length, so their likelihoods are densities of
    different random vectors, and the comparison of criteria across `d`
    is a heuristic rather than a likelihood comparison. To avoid relying
    on it, fix the integration order beforehand (for example with a
    unit-root test) and call this function for that single `d`.

    Every candidate at a `d` whose differenced series has no variation
    (all values equal: any difference of a constant series, the first
    difference of a linear trend) is skipped. A zero-mean ARMA has no
    maximum-likelihood fit to such a series, its likelihood growing
    without bound, so there is no finite criterion to compare.

    Args:
        series: Raw (undifferenced) time series, 1-D and finite.
        max_p: Largest AR order to try (inclusive), searched from 0.
        max_d: Largest integration order to try (inclusive), searched from 0.
        max_q: Largest MA order to try (inclusive), searched from 0.
        criterion: "aic" or "bic".

    Returns:
        The (p, d, q) triple minimizing the chosen criterion among candidates
        that `arima_fit` accepts. Candidates rejected by `arima_fit` (e.g.
        p == q == 0, a series too short for the order, or a fit that does
        not converge) are skipped, as are those whose differenced series
        has no variation, so the returned order is always one `arima_fit`
        fits without raising. Ties go to the candidate reached first in
        the order d, then p, then q, each ascending.

    Raises:
        ValueError: If `criterion` is not "aic" or "bic"; if `series` is
            empty, not 1-D, or contains a non-finite value; or if no
            candidate is left to compare, every one being too short for
            its order, without variation after differencing, or not
            converging.

    References:
        Akaike, H. (1974). "A New Look at the Statistical Model
        Identification." *IEEE Transactions on Automatic Control*, 19(6),
        716-723. (AIC.)

        Schwarz, G. (1978). "Estimating the Dimension of a Model." *The
        Annals of Statistics*, 6(2), 461-464. (BIC.)

        See `arima_fit` for the likelihood, and docs/REFERENCES.md.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError('criterion must be "aic" or "bic"')
    _validate_arima_series(series)

    best_order: tuple[int, int, int] | None = None
    best_score = np.inf

    for d in range(max_d + 1):
        differenced: npt.NDArray[np.float64] | None = None
        fitted: dict[tuple[int, int], _ExactMleFit | None] = {}
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    # The series itself is valid; this rejects the order.
                    _validate_arima_fit_inputs(series, p, d, q)
                except ValueError:
                    continue

                if differenced is None:
                    differenced = _differenced_for_exact_mle(series, d)
                if np.ptp(differenced) == 0.0:
                    continue
                fit = _fit_exact_mle(differenced, p, q, fitted)
                if fit is None or not fit.converged:
                    continue

                n_obs = differenced.size
                k = p + q + 1
                score = (
                    2.0 * k - 2.0 * fit.log_likelihood
                    if criterion == "aic"
                    else k * np.log(n_obs) - 2.0 * fit.log_likelihood
                )

                if score < best_score:
                    best_score = score
                    best_order = (p, d, q)

    if best_order is None:
        raise ValueError(
            "no feasible (p, d, q) order found for the given series and bounds: every "
            "candidate is too short for its order, has no variation after differencing, "
            "or has a fit that does not converge"
        )

    return best_order


def _difference(series: npt.NDArray[np.float64], d: int) -> npt.NDArray[np.float64]:
    diffed = series
    for _ in range(d):
        diffed = np.diff(diffed)
    return diffed


# phi/ma are pre-sliced by callers (rather than a combined `params` array
# sliced inside) because Numba needs simple, directly-typed array arguments
# -- slicing `params[:p]`/`params[p:p+q]` outside the njit boundary keeps the
# kernel's signature trivial to type and lets every caller (the Nelder-Mead
# objective and `arima_residuals`) share the exact same compiled code path,
# which is what makes their outputs bit-identical by construction.
@numba.njit(cache=True)
def _css_residuals_numba_kernel(
    phi: npt.NDArray[np.float64],
    ma: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    p = phi.shape[0]
    q = ma.shape[0]
    n = y.shape[0]
    start = max(p, q)
    residuals = np.zeros(n, dtype=np.float64)

    for t in range(start, n):
        ar_term = 0.0
        for i in range(p):
            ar_term += phi[i] * y[t - 1 - i]
        ma_term = 0.0
        for i in range(q):
            ma_term += ma[i] * residuals[t - 1 - i]
        residuals[t] = y[t] - ar_term - ma_term

    return residuals[start:]


def _css_objective(
    params: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    p: int,
    q: int,
) -> float:
    phi = params[:p]
    ma = params[p : p + q]
    residuals = _css_residuals_numba_kernel(phi, ma, y)
    return float(np.sum(residuals**2))


def _arima_fit_css(
    series: npt.NDArray[np.float64],
    p: int,
    d: int,
    q: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    y = _difference(series, d)
    x0 = np.zeros(p + q, dtype=np.float64)

    result = minimize(
        _css_objective,
        x0,
        args=(y, p, q),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000, "maxfev": 5000},
    )

    params = result.x
    ar_coefs = params[:p]
    ma_coefs = params[p : p + q]
    n_used = y.size - max(p, q)
    sigma2_hat = float(result.fun) / n_used

    if not result.success:
        with np.errstate(divide="ignore"):
            conditional_log_likelihood = -0.5 * n_used * (np.log(2.0 * np.pi * sigma2_hat) + 1.0)
        raise ConvergenceError(
            f"arima_fit: ARIMA({p},{d},{q}) conditional sum-of-squares minimization "
            "did not converge",
            params=(*ar_coefs.tolist(), *ma_coefs.tolist(), sigma2_hat),
            log_likelihood=float(conditional_log_likelihood),
            optimizer_message=str(result.message),
        )

    return ar_coefs, ma_coefs, sigma2_hat


# --- exact Gaussian maximum likelihood -----------------------------------

# Largest |partial autocorrelation| the optimizer may reach, as a distance
# from 1: keeps the fitted AR polynomial strictly stationary and the MA
# polynomial strictly invertible, by the same margin the GARCH fitters keep
# from their stationarity boundary.
_PACF_MARGIN = 1e-6
_MAX_ABS_UNCONSTRAINED = float(np.arctanh(1.0 - _PACF_MARGIN))

# Largest |partial autocorrelation| of a starting value, so that the search
# never starts on the flat part of tanh, where the gradient vanishes.
_START_PACF_CLIP = 0.99

# Objective value reported to the optimizer at a point where the likelihood
# cannot be evaluated, in place of inf/NaN, which L-BFGS-B's line search
# cannot recover from.
_EXACT_MLE_INFEASIBLE_PENALTY = 1e10

# Central-difference step for the gradient of the objective, which is the
# negative log-likelihood per observation and so of order 1: the cube root
# of the float64 machine epsilon, which balances truncation against
# rounding error.
_GRADIENT_STEP = 6.0e-6

# The Kalman recursion switches to its steady-state form (F_t = 1, gain R)
# once every element of the filtered state covariance is below this,
# relative to R'R = 1 + sum_j theta_j^2.
_STEADY_STATE_TOL = 1e-14

# L-BFGS-B stopping rules, on the per-observation objective.
_MAX_ITERATIONS = 500
_MAX_EVALUATIONS = 5000
_FTOL = 1e-12
_GTOL = 1e-7

# A run that L-BFGS-B ends without declaring success (a line search that
# can make no further progress at the optimum, or an iteration cap) still
# counts as converged when the max-norm of its projected gradient is below
# this.
_CONVERGED_GRADIENT_NORM = 1e-5


@numba.njit(cache=True)
def _pacf_to_coefs_kernel(pacf: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map partial autocorrelations in (-1, 1) to stationary AR coefficients.

    Durbin-Levinson step-up recursion: with `a_k = pacf[k - 1]`, the AR(k)
    coefficients follow from the AR(k - 1) ones by `phi_k^(k) = a_k` and
    `phi_j^(k) = phi_j^(k-1) - a_k * phi_{k-j}^(k-1)` for j < k. The map
    is one-to-one from (-1, 1)^k onto the stationary region
    (Barndorff-Nielsen & Schou 1973).

    Args:
        pacf: Partial autocorrelations `a_1..a_k`, shape (k,).

    Returns:
        AR coefficients `phi_1..phi_k` of `1 - phi_1*B - ... - phi_k*B^k`,
        shape (k,).
    """
    k = pacf.shape[0]
    coefs = np.zeros(k, dtype=np.float64)
    prev = np.zeros(k, dtype=np.float64)
    for j in range(k):
        a = pacf[j]
        for i in range(j):
            prev[i] = coefs[i]
        for i in range(j):
            coefs[i] = prev[i] - a * prev[j - 1 - i]
        coefs[j] = a
    return coefs


@numba.njit(cache=True)
def _coefs_to_pacf_kernel(
    coefs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Map AR coefficients to their partial autocorrelations (step-down recursion).

    The inverse of `_pacf_to_coefs_kernel`.

    Args:
        coefs: AR coefficients `phi_1..phi_k`, shape (k,).

    Returns:
        `(pacf, is_stationary)`. `is_stationary` is False as soon as a
        partial autocorrelation with magnitude 1 or more appears, i.e.
        `coefs` is outside the stationary region; `pacf` is then not
        meaningful.
    """
    k = coefs.shape[0]
    work = coefs.copy()
    prev = np.zeros(k, dtype=np.float64)
    pacf = np.zeros(k, dtype=np.float64)
    for j in range(k - 1, -1, -1):
        a = work[j]
        pacf[j] = a
        if not (abs(a) < 1.0):
            return pacf, False
        denom = 1.0 - a * a
        for i in range(j):
            prev[i] = work[i]
        for i in range(j):
            work[i] = (prev[i] + a * prev[j - 1 - i]) / denom
    return pacf, True


def _is_strictly_stationary(coefs: npt.NDArray[np.float64]) -> bool:
    """Decide exactly whether `1 - c_1*z - ... - c_k*z^k` has every root outside the unit circle.

    Runs the step-down recursion of `_coefs_to_pacf_kernel` in rational
    arithmetic on the exact values of the given doubles, so the answer
    carries no rounding error. A floating-point test cannot decide this
    near the edge of the stationary region: where several partial
    autocorrelations are within 1e-6 of 1 in magnitude the region is
    thinner than the spacing of doubles, and rounding the coefficients
    can move them across its edge.

    Args:
        coefs: Finite AR coefficients `c_1..c_k`, shape (k,). Pass `-theta`
            to test the invertibility of `1 + theta_1*z + ...`.

    Returns:
        True if the polynomial is stationary, i.e. every partial
        autocorrelation is below 1 in magnitude.
    """
    work = [Fraction(float(c)) for c in coefs]
    for j in range(len(work) - 1, -1, -1):
        a = work[j]
        if not abs(a) < 1:
            return False
        work = [(work[i] + a * work[j - 1 - i]) / (1 - a * a) for i in range(j)]
    return True


@numba.njit(cache=True)
def _unconstrained_to_arma_kernel(
    u: npt.NDArray[np.float64], p: int, q: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Map the optimizer's unconstrained variables to a stationary phi and an invertible theta.

    `tanh(u[:p])` are the partial autocorrelations of the AR part.
    `theta(B) = 1 + theta_1*B + ...` is invertible exactly when `-theta`
    is a stationary AR coefficient vector, so the MA part is
    `theta = -step_up(-tanh(u[p:]))`; the signs make `theta_1 = tanh(u)`
    for q = 1.

    Args:
        u: Unconstrained variables, shape (p + q,).
        p: AR order.
        q: MA order.

    Returns:
        `(phi, theta)`, shapes (p,) and (q,).
    """
    phi = _pacf_to_coefs_kernel(np.tanh(u[:p]))
    theta = -_pacf_to_coefs_kernel(-np.tanh(u[p : p + q]))
    return phi, theta


@numba.njit(cache=True)
def _solve_linear_system_kernel(
    matrix: npt.NDArray[np.float64], rhs: npt.NDArray[np.float64]
) -> bool:
    """Solve `matrix @ x = rhs` in place by Gaussian elimination with partial pivoting.

    Args:
        matrix: Square coefficient matrix, shape (m, m); overwritten.
        rhs: Right-hand side, shape (m,); holds the solution `x` on return.

    Returns:
        False if a pivot is zero or not a number, i.e. the system is
        singular to working precision; `rhs` is then not meaningful.
    """
    m = rhs.shape[0]
    for col in range(m):
        pivot_row = col
        largest = abs(matrix[col, col])
        for row in range(col + 1, m):
            candidate = abs(matrix[row, col])
            if candidate > largest:
                largest = candidate
                pivot_row = row
        if not (largest > 0.0):
            return False
        if pivot_row != col:
            for k in range(col, m):
                held = matrix[col, k]
                matrix[col, k] = matrix[pivot_row, k]
                matrix[pivot_row, k] = held
            held = rhs[col]
            rhs[col] = rhs[pivot_row]
            rhs[pivot_row] = held
        pivot = matrix[col, col]
        for row in range(col + 1, m):
            factor = matrix[row, col] / pivot
            if factor != 0.0:
                for k in range(col + 1, m):
                    matrix[row, k] -= factor * matrix[col, k]
                rhs[row] -= factor * rhs[col]
    for row in range(m - 1, -1, -1):
        total = rhs[row]
        for k in range(row + 1, m):
            total -= matrix[row, k] * rhs[k]
        rhs[row] = total / matrix[row, row]
    return True


@numba.njit(cache=True)
def _stationary_state_covariance_kernel(
    phi_full: npt.NDArray[np.float64], rvec: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], bool]:
    """Solve `P = T*P*T' + R*R'` for the stationary state covariance, in units of sigma2.

    `T` is the companion matrix of the module docstring, with `phi_full`
    in column 0 and ones on the superdiagonal, so
    `(T*P*T')[i, j] = phi_i*phi_j*P[0, 0] + phi_i*P[0, j+1]
    + phi_j*P[i+1, 0] + P[i+1, j+1]` (terms with an index of r dropped).
    The r^2 equations are solved as the linear system
    `(I - T kron T) vec(P) = vec(R*R')`.

    Args:
        phi_full: AR coefficients padded with zeros to length r.
        rvec: `(1, theta_1, ..., theta_{r-1})`, padded with zeros to
            length r.

    Returns:
        `(P, ok)` with `P` of shape (r, r). `ok` is False when the system
        is singular, which happens only outside the stationary region.
    """
    r = phi_full.shape[0]
    m = r * r
    lhs = np.zeros((m, m), dtype=np.float64)
    rhs = np.zeros(m, dtype=np.float64)
    for i in range(r):
        for j in range(r):
            row = i * r + j
            lhs[row, row] += 1.0
            rhs[row] = rvec[i] * rvec[j]
            lhs[row, 0] -= phi_full[i] * phi_full[j]
            if j + 1 < r:
                lhs[row, j + 1] -= phi_full[i]
            if i + 1 < r:
                lhs[row, (i + 1) * r] -= phi_full[j]
            if i + 1 < r and j + 1 < r:
                lhs[row, (i + 1) * r + j + 1] -= 1.0
    ok = _solve_linear_system_kernel(lhs, rhs)
    cov = np.empty((r, r), dtype=np.float64)
    for i in range(r):
        for j in range(r):
            cov[i, j] = 0.5 * (rhs[i * r + j] + rhs[j * r + i])
    return cov, ok


@numba.njit(cache=True)
def _arma_kalman_filter_kernel(
    phi: npt.NDArray[np.float64],
    theta: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
) -> tuple[float, float, npt.NDArray[np.float64], bool]:
    """Kalman filter of a zero-mean ARMA(p, q) sample from its stationary initial state.

    Runs the filter of the module docstring over all of `w` and returns
    the two sums of the prediction-error decomposition together with the
    state predicted for the first time point after the sample. Because
    `w_t` is the first state element observed without noise, the filtered
    state has `alpha_t[0] = w_t` exactly and row and column 0 of its
    covariance are zero, so the covariance prediction `T*P*T' + R*R'`
    reduces to a shift of the filtered covariance plus `R*R'`. Once the
    filtered covariance has decayed to zero the prediction is `R*R'` at
    every later step, `F_t = 1` and the gain is `R`: the recursion then
    continues in that steady-state form (the quick recursion of Gardner,
    Harvey & Phillips 1980). A pure AR(p) reaches it after p observations.

    Args:
        phi: AR coefficients, shape (p,).
        theta: MA coefficients, shape (q,).
        w: Observed (already differenced) series, shape (n,).

    Returns:
        `(sum_sq, sum_log, next_state, ok)`: `sum_sq = sum_t v_t^2 / F_t`
        and `sum_log = sum_t log(F_t)`, with the prediction-error variances
        `F_t` in units of sigma2; `next_state`, shape (r,), is the
        conditional mean of `alpha_{n+1}` given all of `w`, whose first
        element is the one-step forecast of `w`. `ok` is False when the
        filter cannot be run at `(phi, theta)`: the stationary covariance
        does not exist or a prediction-error variance is not a positive
        finite number. The other three values are then not meaningful.
    """
    p = phi.shape[0]
    q = theta.shape[0]
    n = w.shape[0]
    r = max(p, q + 1)

    phi_full = np.zeros(r, dtype=np.float64)
    for i in range(p):
        phi_full[i] = phi[i]
    rvec = np.zeros(r, dtype=np.float64)
    rvec[0] = 1.0
    r_norm = 1.0
    for i in range(q):
        rvec[i + 1] = theta[i]
        r_norm += theta[i] * theta[i]

    state = np.zeros(r, dtype=np.float64)
    pred_cov, ok = _stationary_state_covariance_kernel(phi_full, rvec)
    if not ok:
        return 0.0, 0.0, state, False
    filt_cov = np.zeros((r, r), dtype=np.float64)
    gain = np.zeros(r, dtype=np.float64)

    sum_sq = 0.0
    sum_log = 0.0
    steady = False
    steady_tol = _STEADY_STATE_TOL * r_norm

    for t in range(n):
        if steady:
            v = w[t] - state[0]
            sum_sq += v * v
            for i in range(r - 1):
                state[i] = phi_full[i] * w[t] + state[i + 1] + rvec[i + 1] * v
            state[r - 1] = phi_full[r - 1] * w[t]
            continue

        f = pred_cov[0, 0]
        if not (f > 0.0) or f == np.inf:
            return 0.0, 0.0, state, False
        v = w[t] - state[0]
        sum_sq += v * v / f
        sum_log += np.log(f)

        # Measurement update: gain P[:, 0] / F, filtered covariance
        # P - gain * P[0, :] (its row and column 0 are zero and never read).
        for i in range(r):
            gain[i] = pred_cov[i, 0] / f
        largest = 0.0
        for i in range(1, r):
            for j in range(1, r):
                value = pred_cov[i, j] - gain[i] * pred_cov[0, j]
                filt_cov[i, j] = value
                if abs(value) > largest:
                    largest = abs(value)

        # Time update: alpha_{t+1} = T * alpha_t + R * e_{t+1}.
        for i in range(r - 1):
            state[i] = phi_full[i] * w[t] + state[i + 1] + gain[i + 1] * v
        state[r - 1] = phi_full[r - 1] * w[t]
        for i in range(r):
            for j in range(r):
                shifted = filt_cov[i + 1, j + 1] if (i + 1 < r and j + 1 < r) else 0.0
                pred_cov[i, j] = shifted + rvec[i] * rvec[j]

        if largest <= steady_tol:
            steady = True

    return sum_sq, sum_log, state, True


@numba.njit(cache=True)
def _exact_mle_objective_kernel(
    u: npt.NDArray[np.float64], w: npt.NDArray[np.float64], p: int, q: int
) -> float:
    """Negative concentrated exact log-likelihood per observation, without its constant.

    With sigma2 maximized out, `log L = -n/2 * (log(2*pi) + 1 +
    log(sigma2_hat)) - 1/2 * sum_t log(F_t)`; this is `-log L / n` less
    the constant `(log(2*pi) + 1) / 2`.

    Args:
        u: Unconstrained variables, shape (p + q,); see
            `_unconstrained_to_arma_kernel`.
        w: Observed (already differenced) series, shape (n,).
        p: AR order.
        q: MA order.

    Returns:
        `0.5 * log(sigma2_hat) + (1 / 2n) * sum_t log(F_t)`, or `inf`
        when the likelihood cannot be evaluated at `u` or the sum of
        squared prediction errors is not positive.
    """
    phi, theta = _unconstrained_to_arma_kernel(u, p, q)
    sum_sq, sum_log, _, ok = _arma_kalman_filter_kernel(phi, theta, w)
    n = w.shape[0]
    if not ok or not (sum_sq > 0.0):
        return np.inf
    return float(0.5 * np.log(sum_sq / n) + 0.5 * sum_log / n)


@numba.njit(cache=True)
def _exact_mle_objective_and_gradient_kernel(
    u: npt.NDArray[np.float64], w: npt.NDArray[np.float64], p: int, q: int, step: float
) -> tuple[float, npt.NDArray[np.float64]]:
    """`_exact_mle_objective_kernel` at `u` and its central-difference gradient.

    Args:
        u: Unconstrained variables, shape (p + q,).
        w: Observed (already differenced) series, shape (n,).
        p: AR order.
        q: MA order.
        step: Half-width of the central difference in each coordinate.

    Returns:
        `(value, gradient)`, the gradient of shape (p + q,).
    """
    value = _exact_mle_objective_kernel(u, w, p, q)
    k = u.shape[0]
    gradient = np.zeros(k, dtype=np.float64)
    shifted = u.copy()
    for i in range(k):
        shifted[i] = u[i] + step
        upper = _exact_mle_objective_kernel(shifted, w, p, q)
        shifted[i] = u[i] - step
        lower = _exact_mle_objective_kernel(shifted, w, p, q)
        shifted[i] = u[i]
        gradient[i] = (upper - lower) / (2.0 * step)
    return value, gradient


def _exact_mle_objective(
    u: npt.NDArray[np.float64], w: npt.NDArray[np.float64], p: int, q: int
) -> tuple[float, npt.NDArray[np.float64]]:
    # The kernels signal an unevaluable likelihood through their `ok` flag
    # and an infinite value rather than by raising: every scalar division
    # in them has a divisor checked to be non-zero first (a positive F_t, a
    # non-zero pivot, 1 - a^2 with |a| < 1). L-BFGS-B's line search cannot
    # recover from inf/NaN, so such a point, or one whose gradient is not
    # finite, is reported as a large finite value with a zero gradient,
    # which makes the line search back off.
    value, gradient = _exact_mle_objective_and_gradient_kernel(u, w, p, q, _GRADIENT_STEP)
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        return _EXACT_MLE_INFEASIBLE_PENALTY, np.zeros_like(u)
    return float(value), np.asarray(gradient, dtype=np.float64)


@numba.njit(cache=True)
def _yule_walker_kernel(
    w: npt.NDArray[np.float64], max_order: int, min_order: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit a Yule-Walker AR(h) with h chosen by AIC, and return its residuals.

    Sample autocovariances use divisor n and no mean correction (the model
    has no constant), which makes every AR(h) from the Levinson-Durbin
    recursion stationary. The order minimizes `n * log(sigma2_h) + 2*h`
    over `min_order <= h <= max_order`, where `sigma2_h` is the AR(h)
    innovation variance of the recursion.

    Args:
        w: Observed (already differenced) series, shape (n,), not
            identically zero.
        max_order: Largest order considered; must be below n.
        min_order: Smallest order considered.

    Returns:
        `(ar_coefs, residuals)`: the selected coefficients, shape (h,), and
        `w_t - sum_j ar_coefs[j] * w_{t-1-j}` for t >= h, shape (n - h,).
        h is below `min_order` (possibly 0) only when the recursion's
        innovation variance reaches zero first, i.e. a lower-order
        autoregression already reproduces the sample exactly.
    """
    n = w.shape[0]
    acov = np.zeros(max_order + 1, dtype=np.float64)
    for k in range(max_order + 1):
        total = 0.0
        for t in range(k, n):
            total += w[t] * w[t - k]
        acov[k] = total / n

    best = np.zeros(0, dtype=np.float64)
    best_aic = np.inf
    coefs = np.zeros(max_order, dtype=np.float64)
    prev = np.zeros(max_order, dtype=np.float64)
    variance = acov[0]
    if min_order == 0 and variance > 0.0:
        best_aic = n * np.log(variance)
    for h in range(1, max_order + 1):
        if not (variance > 0.0):
            break
        numerator = acov[h]
        for j in range(h - 1):
            numerator -= coefs[j] * acov[h - 1 - j]
        reflection = numerator / variance
        for j in range(h - 1):
            prev[j] = coefs[j]
        for j in range(h - 1):
            coefs[j] = prev[j] - reflection * prev[h - 2 - j]
        coefs[h - 1] = reflection
        variance *= 1.0 - reflection * reflection
        if h >= min_order and variance > 0.0:
            aic = n * np.log(variance) + 2.0 * h
            if aic < best_aic:
                best_aic = aic
                best = coefs[:h].copy()

    h = best.shape[0]
    residuals = np.empty(n - h, dtype=np.float64)
    for t in range(h, n):
        fitted = 0.0
        for j in range(h):
            fitted += best[j] * w[t - 1 - j]
        residuals[t - h] = w[t] - fitted
    return best, residuals


def _hannan_rissanen(
    w: npt.NDArray[np.float64], p: int, q: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Two-stage regression estimate of (phi, theta) (Hannan & Rissanen 1982).

    Stage 1 fits a long autoregression by Yule-Walker, its order chosen by
    AIC among 1..max((log n)^2, p + q), and takes its residuals as
    estimates of the innovations. Stage 2 regresses `w_t` on
    `w_{t-1}..w_{t-p}` and on the stage-1 residuals at lags 1..q by least
    squares. With q = 0 only the autoregression on `w`'s own lags remains.

    Args:
        w: Observed (already differenced) series, shape (n,), not
            identically zero.
        p: AR order.
        q: MA order.

    Returns:
        `(phi, theta)`, shapes (p,) and (q,), not necessarily stationary or
        invertible; or None when the sample is too short for the stage-2
        regression to have more rows than coefficients, or the stage-1
        residuals are not finite.
    """
    n = w.shape[0]
    if q == 0:
        lag = 0
        residuals = np.empty(0, dtype=np.float64)
    else:
        max_order = min(max(int(np.log(n) ** 2), p + q), (n - 1) // 2)
        if max_order < 1:
            return None
        ar_long, residuals = _yule_walker_kernel(w, max_order, 1)
        if not np.all(np.isfinite(residuals)):
            return None
        lag = ar_long.shape[0]

    first = max(p, lag + q)
    rows = n - first
    if rows <= p + q:
        return None
    design = np.empty((rows, p + q), dtype=np.float64)
    for j in range(p):
        design[:, j] = w[first - 1 - j : n - 1 - j]
    for j in range(q):
        # residuals[i] estimates the innovation at time lag + i.
        design[:, p + j] = residuals[first - 1 - j - lag : n - 1 - j - lag]
    solution = np.linalg.lstsq(design, w[first:], rcond=None)[0]
    return solution[:p].copy(), solution[p : p + q].copy()


def _starting_values(w: npt.NDArray[np.float64], p: int, q: int) -> npt.NDArray[np.float64]:
    """Starting point of the exact-likelihood search, in the optimizer's unconstrained variables.

    The Hannan-Rissanen estimate where it is usable. A non-stationary AR
    part is replaced by the Yule-Walker AR(p) estimate, which is always
    stationary; a non-invertible MA part, or a sample for which the
    Hannan-Rissanen estimate is unavailable, starts from zero. Partial
    autocorrelations are clipped to magnitude `_START_PACF_CLIP`.

    Args:
        w: Observed (already differenced) series, shape (n,), not
            identically zero.
        p: AR order.
        q: MA order.

    Returns:
        Starting vector, shape (p + q,).
    """
    phi = np.zeros(p, dtype=np.float64)
    theta = np.zeros(q, dtype=np.float64)
    estimate = _hannan_rissanen(w, p, q)
    if (
        estimate is not None
        and np.all(np.isfinite(estimate[0]))
        and np.all(np.isfinite(estimate[1]))
    ):
        phi, theta = estimate

    ar_pacf, stationary = _coefs_to_pacf_kernel(phi)
    if not stationary:
        ar_pacf = np.zeros(p, dtype=np.float64)
        if p < w.shape[0]:
            yule_walker, _ = _yule_walker_kernel(w, p, p)
            if yule_walker.shape[0] == p:
                ar_pacf, _ = _coefs_to_pacf_kernel(yule_walker)
    ma_pacf, invertible = _coefs_to_pacf_kernel(-theta)
    ma_pacf = -ma_pacf if invertible else np.zeros(q, dtype=np.float64)

    pacf = np.clip(np.concatenate([ar_pacf, ma_pacf]), -_START_PACF_CLIP, _START_PACF_CLIP)
    return np.asarray(np.arctanh(pacf), dtype=np.float64)


class _OptimizerRun(NamedTuple):
    """End point of one L-BFGS-B run on the exact-likelihood objective.

    Attributes:
        u: The end point, in the optimizer's unconstrained variables.
        objective: `_exact_mle_objective`'s value there.
        converged: Whether L-BFGS-B reported success or the projected
            gradient there is below `_CONVERGED_GRADIENT_NORM`.
        message: L-BFGS-B's termination message.
    """

    u: npt.NDArray[np.float64]
    objective: float
    converged: bool
    message: str


class _ExactMleFit(NamedTuple):
    """Exact maximum-likelihood fit of a zero-mean ARMA(p, q) to a differenced series.

    Attributes:
        ar_coefs: `phi_1..phi_p`, stationary.
        ma_coefs: `theta_1..theta_q`, invertible.
        sigma2: Maximum-likelihood innovation variance,
            `(1/n) * sum_t v_t^2 / F_t`.
        log_likelihood: Exact Gaussian log-likelihood of the n observations
            at these parameters; `inf` for an identically zero sample.
        converged: Whether the optimizer run these parameters come from
            met a convergence test and its likelihood is not below that of
            a fitted model nested in this one.
        optimizer_message: That run's termination message.
        unconstrained: The parameters in the optimizer's unconstrained
            variables, shape (p + q,).
    """

    ar_coefs: npt.NDArray[np.float64]
    ma_coefs: npt.NDArray[np.float64]
    sigma2: float
    log_likelihood: float
    converged: bool
    optimizer_message: str
    unconstrained: npt.NDArray[np.float64]


def _run_exact_mle_optimizer(
    w: npt.NDArray[np.float64], p: int, q: int, start: npt.NDArray[np.float64]
) -> _OptimizerRun | None:
    """Minimize the exact-likelihood objective by L-BFGS-B from one starting point.

    Args:
        w: Observed (already differenced) series, shape (n,).
        p: AR order.
        q: MA order.
        start: Starting point in the unconstrained variables, shape (p + q,).

    Returns:
        The run's end point, or None when it is unusable: the likelihood
        cannot be evaluated there, or the coefficients it maps to are not
        stationary and invertible once rounded to doubles (possible only
        where several partial autocorrelations sit on their bound at once).
    """
    result = minimize(
        _exact_mle_objective,
        start,
        args=(w, p, q),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-_MAX_ABS_UNCONSTRAINED, _MAX_ABS_UNCONSTRAINED)] * (p + q),
        options={
            "maxiter": _MAX_ITERATIONS,
            "maxfun": _MAX_EVALUATIONS,
            "ftol": _FTOL,
            "gtol": _GTOL,
        },
    )
    u = np.asarray(result.x, dtype=np.float64)
    objective, gradient = _exact_mle_objective(u, w, p, q)
    if objective >= _EXACT_MLE_INFEASIBLE_PENALTY:
        return None
    phi, theta = _unconstrained_to_arma_kernel(u, p, q)
    if not (_is_strictly_stationary(phi) and _is_strictly_stationary(-theta)):
        return None

    # A gradient component pushing against an active bound does not count:
    # the constrained optimum sits on that bound.
    pushing_out = ((u <= -_MAX_ABS_UNCONSTRAINED) & (gradient > 0.0)) | (
        (u >= _MAX_ABS_UNCONSTRAINED) & (gradient < 0.0)
    )
    projected_gradient_norm = float(np.max(np.abs(np.where(pushing_out, 0.0, gradient))))
    converged = bool(result.success) or projected_gradient_norm <= _CONVERGED_GRADIENT_NORM
    return _OptimizerRun(u, objective, converged, str(result.message))


# Slack, per observation, within which a fit's log-likelihood counts as not
# below that of a model nested in it: the two are evaluated with state
# vectors of different length and agree only to rounding.
_NESTED_LOGLIKE_SLACK = 1e-10


def _fit_exact_mle(
    w: npt.NDArray[np.float64],
    p: int,
    q: int,
    fitted: dict[tuple[int, int], _ExactMleFit | None] | None = None,
) -> _ExactMleFit | None:
    """Maximize the exact Gaussian likelihood of a zero-mean ARMA(p, q) over (phi, theta, sigma2).

    The likelihood of a mixed model can have several local maxima, so
    L-BFGS-B is run from up to four starting points: the Hannan-Rissanen
    estimate, the origin (white noise), and the fitted ARMA(p - 1, q) and
    ARMA(p, q - 1), each extended by a zero partial autocorrelation, which
    is the same model written as an ARMA(p, q). Those two fits are obtained
    by this same procedure, so fitting order (p, q) fits every order (i, j)
    with i <= p and j <= q once. A converged run is preferred to one that
    is not; among equals the higher likelihood wins. A run started from a
    nested fit cannot end below it, so a converged result is never below
    the fit of an order nested in it; a result that is (possible only when
    the run started from that nested fit did not converge) is reported as
    not converged.

    Args:
        w: Observed (already differenced) series as a C-contiguous float64
            array, shape (n,), with n > max(p, q) and p + q >= 1.
        p: AR order.
        q: MA order.
        fitted: Fits of this same `w` by order (i, j), reused instead of
            being recomputed and extended with every order fitted here.
            None starts from an empty record.

    Returns:
        The fit, whose `converged` flag the caller must check; or None
        when no run ended at a usable point (see
        `_run_exact_mle_optimizer`).
    """
    if fitted is None:
        fitted = {}
    if (p, q) in fitted:
        return fitted[(p, q)]

    n = w.shape[0]
    if not np.any(w):
        # The likelihood of an identically zero sample grows without bound
        # as sigma2 -> 0 whatever the coefficients are.
        return _ExactMleFit(
            np.zeros(p, dtype=np.float64),
            np.zeros(q, dtype=np.float64),
            0.0,
            np.inf,
            True,
            "identically zero sample: no optimization run",
            np.zeros(p + q, dtype=np.float64),
        )

    starts = [_starting_values(w, p, q)]
    if np.any(starts[0] != 0.0):
        starts.append(np.zeros(p + q, dtype=np.float64))
    nested_log_likelihood = -np.inf
    for nested_p, nested_q in ((p - 1, q), (p, q - 1)):
        if nested_p < 0 or nested_q < 0 or nested_p + nested_q == 0:
            continue
        nested = _fit_exact_mle(w, nested_p, nested_q, fitted)
        if nested is None or not nested.converged:
            continue
        # A zero inserted after the AR block, or appended to the MA block,
        # leaves the nested coefficients unchanged and adds a zero one.
        insert_at = p - 1 if nested_p < p else p + q - 1
        starts.append(np.insert(nested.unconstrained, insert_at, 0.0))
        nested_log_likelihood = max(nested_log_likelihood, nested.log_likelihood)

    best: _OptimizerRun | None = None
    for start in starts:
        run = _run_exact_mle_optimizer(w, p, q, start)
        if run is None:
            continue
        if (
            best is None
            or (run.converged and not best.converged)
            or (run.converged == best.converged and run.objective < best.objective)
        ):
            best = run
    if best is None:
        fitted[(p, q)] = None
        return None

    phi, theta = _unconstrained_to_arma_kernel(best.u, p, q)
    sum_sq, sum_log, _, _ = _arma_kalman_filter_kernel(phi, theta, w)
    sigma2 = float(sum_sq) / n
    log_likelihood = float(
        -0.5 * n * (np.log(2.0 * np.pi) + 1.0 + np.log(sigma2)) - 0.5 * float(sum_log)
    )
    fit = _ExactMleFit(
        np.asarray(phi, dtype=np.float64),
        np.asarray(theta, dtype=np.float64),
        sigma2,
        log_likelihood,
        best.converged and log_likelihood >= nested_log_likelihood - _NESTED_LOGLIKE_SLACK * n,
        best.message,
        best.u,
    )
    fitted[(p, q)] = fit
    return fit


def _differenced_for_exact_mle(series: npt.NDArray[np.float64], d: int) -> npt.NDArray[np.float64]:
    # Always a fresh, writable, C-contiguous float64 array, whatever the
    # caller's array is (integer, strided, read-only): the exact-likelihood
    # kernels are then compiled for that one array type only.
    return np.array(_difference(series, d), dtype=np.float64)


def _arima_fit_exact_mle(
    series: npt.NDArray[np.float64],
    p: int,
    d: int,
    q: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    fit = _fit_exact_mle(_differenced_for_exact_mle(series, d), p, q)

    if fit is None:
        raise ConvergenceError(
            f"arima_fit: ARIMA({p},{d},{q}) exact maximum-likelihood optimization ended at "
            "no usable point from any starting point: the likelihood could not be evaluated "
            "there, or the coefficients were not stationary and invertible in floating point",
            params=None,
            log_likelihood=None,
            optimizer_message=None,
        )
    if not fit.converged:
        raise ConvergenceError(
            f"arima_fit: ARIMA({p},{d},{q}) exact maximum-likelihood optimization failed "
            "to converge to a maximum from any starting point",
            params=(*fit.ar_coefs.tolist(), *fit.ma_coefs.tolist(), fit.sigma2),
            log_likelihood=fit.log_likelihood,
            optimizer_message=fit.optimizer_message,
        )

    return fit.ar_coefs, fit.ma_coefs, fit.sigma2


def _validate_arima_forecast_inputs(
    series: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    ma_coefs: npt.NDArray[np.float64],
    d: int,
    h: int,
) -> None:
    if series.size == 0:
        raise ValueError("series must be non-empty")
    if series.ndim != 1:
        raise ValueError("series must be a 1-D array")
    if ar_coefs.ndim != 1:
        raise ValueError("ar_coefs must be a 1-D array")
    if ma_coefs.ndim != 1:
        raise ValueError("ma_coefs must be a 1-D array")
    if d not in (0, 1, 2):
        raise ValueError("d must be 0, 1, or 2")
    if h <= 0:
        raise ValueError("h must be a positive integer")
    if series.size <= d + ar_coefs.size:
        raise ValueError("series is too short to seed the forecast for the given order")
    if not (
        np.all(np.isfinite(series))
        and np.all(np.isfinite(ar_coefs))
        and np.all(np.isfinite(ma_coefs))
    ):
        raise ValueError("series, ar_coefs and ma_coefs must contain only finite values")


def arima_forecast(
    series: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    ma_coefs: npt.NDArray[np.float64],
    d: int,
    h: int,
) -> npt.NDArray[np.float64]:
    """Produce h-step-ahead ARIMA point forecasts on the original scale.

    The forecast is the Gaussian conditional mean of the next `h` values
    given the whole series, `E[y_{T+k} | y_1..y_T]` for k = 1..h, at the
    supplied coefficients. On the differenced scale `w_t` it obeys the
    model's difference equation with every future innovation replaced by
    its mean, 0, and every innovation inside the sample by its
    conditional mean given the sample:

        w_hat_{T+k} = sum_i phi_i * w_hat_{T+k-i}
                      + sum_{j >= k} theta_j * e_hat_{T+k-j}

    with `w_hat_s = w_s` for s <= T. The last q in-sample innovations are
    not observed, so their conditional means come from the Kalman filter
    of the exact likelihood (see the module docstring), run over the
    differenced series from its stationary initial state: the state it
    predicts for T+1 carries them, and iterating the state transition
    from there gives `w_hat_{T+1}..w_hat_{T+h}`. These are then summed
    `d` times from the last observed values, which makes the result the
    conditional mean given the series when nothing is assumed about its
    first `d` values (a diffuse prior on the starting level).

    With no MA part (`ma_coefs` empty or all zero) the forecast depends
    only on the last p differenced values and is computed from the AR
    recursion alone, for any `ar_coefs`, stationary or not.

    Args:
        series: Original (undifferenced) series the model was fit on.
        ar_coefs: AR coefficients `phi_1..phi_p`, shape (p,), e.g.
            `arima_fit`'s first return value.
        ma_coefs: MA coefficients `theta_1..theta_q`, shape (q,), with the
            sign convention `theta(B) = 1 + theta_1*B + ...`, e.g.
            `arima_fit`'s second return value. They need not be
            invertible.
        d: Integration order used when fitting.
        h: Forecast horizon.

    Returns:
        Array of shape (h,) with point forecasts on the original scale.

    Raises:
        ValueError: If `series` is empty or has at most `d + p` values; if
            `series`, `ar_coefs` or `ma_coefs` is not 1-D; if `d` is not 0,
            1, or 2;
            if `h` is not positive; if any input value is not finite; or if
            `ma_coefs` has a non-zero entry and `ar_coefs` is not
            stationary, in which case the pre-sample state has no
            stationary distribution to start the filter from.

    References:
        Box, G.E.P., Jenkins, G.M., Reinsel, G.C., and Ljung, G.M. (2015).
        *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
        (Ch. 5: minimum mean square error forecasts from the difference
        equation.)

        Harvey, A.C. (1989). *Forecasting, Structural Time Series Models
        and the Kalman Filter*. Cambridge University Press. (Ch. 3:
        prediction from the Kalman filter's final state.) See
        docs/REFERENCES.md.
    """
    _validate_arima_forecast_inputs(series, ar_coefs, ma_coefs, d, h)
    return _arima_forecast(series, ar_coefs, ma_coefs, d, h)


def _arima_forecast(
    series: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    ma_coefs: npt.NDArray[np.float64],
    d: int,
    h: int,
) -> npt.NDArray[np.float64]:
    diffs = [series]
    for _ in range(d):
        diffs.append(np.diff(diffs[-1]))

    if np.any(ma_coefs):
        forecast = _arma_forecast_from_filter(diffs[-1], ar_coefs, ma_coefs, h)
    else:
        # A pure autoregression: the last p values determine the forecast,
        # whatever the pre-sample state was.
        p = ar_coefs.size
        history = list(diffs[-1])
        for _ in range(h):
            if p > 0:
                lags = np.array(history[-p:][::-1], dtype=np.float64)
                next_value = float(np.dot(ar_coefs, lags))
            else:
                next_value = 0.0
            history.append(next_value)

        forecast = np.array(history[-h:], dtype=np.float64)

    for k in range(d - 1, -1, -1):
        seed = diffs[k][-1]
        forecast = seed + np.cumsum(forecast)

    return forecast


def _arma_forecast_from_filter(
    w: npt.NDArray[np.float64],
    ar_coefs: npt.NDArray[np.float64],
    ma_coefs: npt.NDArray[np.float64],
    h: int,
) -> npt.NDArray[np.float64]:
    """Conditional-mean forecast of a zero-mean ARMA(p, q) from its Kalman-filtered state.

    Args:
        w: Observed (already differenced) series, shape (n,).
        ar_coefs: AR coefficients, shape (p,).
        ma_coefs: MA coefficients, shape (q,).
        h: Forecast horizon.

    Returns:
        `E[w_{n+k} | w_1..w_n]` for k = 1..h, shape (h,).

    Raises:
        ValueError: `ar_coefs` is not stationary, or so close to the edge
            of the stationary region that the stationary state covariance
            cannot be computed in floating point.
    """
    # Fresh C-contiguous float64 arrays, the one array type the filter
    # kernel is compiled for.
    phi = np.array(ar_coefs, dtype=np.float64)
    theta = np.array(ma_coefs, dtype=np.float64)

    if not _is_strictly_stationary(phi):
        raise ValueError(
            "ar_coefs must be stationary when ma_coefs has a non-zero entry: the forecast "
            "starts the Kalman filter from the stationary distribution of the pre-sample state"
        )
    _, _, next_state, ok = _arma_kalman_filter_kernel(phi, theta, np.array(w, dtype=np.float64))
    if not ok:
        raise ValueError(
            "the stationary state covariance could not be computed for ar_coefs: they are "
            "too close to the edge of the stationary region"
        )

    # next_state is the conditional mean of alpha_{n+1}; each further step
    # applies the transition (T*a)[i] = phi_i * a[0] + a[i+1].
    state = np.asarray(next_state, dtype=np.float64)
    phi_full = np.zeros(state.size, dtype=np.float64)
    phi_full[: phi.size] = phi
    forecast = np.empty(h, dtype=np.float64)
    for k in range(h):
        forecast[k] = state[0]
        state = phi_full * state[0] + np.append(state[1:], 0.0)
    return forecast
