# Changelog

All notable changes to quantcore are recorded here. Entries follow
[Keep a Changelog](https://keepachangelog.com/) conventions. Numerical
results that changed carry a before/after example, as the project's
ground rules require — other projects install this package and must be able
to see exactly what changed and why before upgrading.

## [0.3.0] — Unreleased

### Changed — default behaviour changed

**`arima_fit` estimates by exact Gaussian maximum likelihood by default.** It
previously minimised a conditional sum of squares (CSS), whose optimum
attains a lower exact likelihood than the maximum-likelihood estimate on
every finite sample. The exact likelihood of the differenced series is now
evaluated by a Kalman filter started from the stationary state distribution
(Gardner, Harvey and Phillips 1980; Harvey 1989), the AR part is kept
stationary and the MA part invertible through partial autocorrelations
(Jones 1980), and L-BFGS-B runs from the Hannan-Rissanen (1982) estimate,
from white noise and from the fits of orders (p - 1, q) and (p, q - 1)
(themselves fitted this way), each extended by a zero coefficient. `sigma2_hat` is the maximum-likelihood variance (divisor
`n - d`), not `SSE / (n - d - max(p, q))`. The positional signature and the
return shape are unchanged; the new keyword-only `method="css"` returns the
earlier estimator bit for bit.

```python
# AR(1), phi = 0.6, 300 observations: x[0] = e[0], x[t] = 0.6*x[t-1] + e[t],
# e = numpy.random.default_rng(7).normal(0, 1, 300)
arima_fit(x, p=1, d=0, q=0)
# before (now method="css"): ar 0.6365242614746099, sigma2_hat 0.8689611975116575, exact log-lik -404.3727735336
# after  (method="mle"):     ar 0.6344189870378044, sigma2_hat 0.8660711163423243, exact log-lik -404.3708119624

# ARMA(1,1), phi = 0.5, theta = 0.3, 300 observations, default_rng(7)
arima_fit(y, p=1, d=0, q=1)  # (ar, ma, sigma2_hat)
# before: (0.5513010759, 0.2676818335, 0.8693152182), log-lik -404.5281430698
# after:  (0.5495632753, 0.2666657917, 0.8664134957), log-lik -404.5260312419
```

The median coefficient shift is about 1e-3 (worst seen 0.064) and
`sigma2_hat` is about 0.1% lower. The default fit always returns a stationary
AR part and an invertible MA part; CSS does not (an MA coefficient of -1.074
occurred on one test case), and on a series with a unit root in levels the
estimate now stops at the edge of the region (0.999999) instead of at or
beyond 1. `arima_residuals` is unchanged and still returns conditional (CSS)
residuals, so with the default fit's coefficients
`sum(residuals**2) / residuals.size` is close to, but no longer equal to,
`sigma2_hat`.

The likelihood of a mixed model can have several local maxima, so a fit
returned by the default method is never below the fit of an order nested in
it. Started from the Hannan-Rissanen and white-noise points alone, L-BFGS-B
can converge to a lower local maximum:

```python
# n = 300 from phi = (0.3, 0.4), theta = (0.2, -0.6), default_rng(75)
arima_fit(w, 2, 0, 2)
# from the two starts alone: exact log-lik -410.3065, below ARMA(2,1)'s -409.9786
# as released:               exact log-lik -409.9638
```

Fitting order (p, q) fits every order (i, j) with i <= p and j <= q, so mixed
fits take longer: 2.3 / 5.4 / 15.3 / 27.1 ms warm for (1,1) / (2,1) / (2,2) /
(3,2) at n = 300 (one machine). Pure AR(1) and MA(1) fits are unaffected. The
starts do not guarantee the global maximum: against the best of 20 random
starts, 13 of 400 ARMA(2,2) and 3 of 400 ARMA(2,1) fits on the test seeds
were still lower, by up to 4.3 and 0.46 log-likelihood units.

**`select_arima_order` scores every candidate on its maximised exact
log-likelihood** over the `n - d` differenced observations, with
`k = p + q + 1`. It previously rebuilt a likelihood from the CSS variance on
`n - d - max(p, q)` observations, a sample size that differed between the
candidates being compared and favoured large orders. Candidates whose fit
does not converge are skipped, so the order returned can always be fit.
Signature and return shape are unchanged. Across different `d` the choice is
still a heuristic (under AIC it can over-difference). A `d` whose
differenced series has no variation (all values equal) is skipped for every
candidate; it used to score an infinite or near-infinite likelihood and always
won. Each fitted order is reused as a starting point for the larger ones.
The test is exact equality, so a series constant only up to rounding noise is
still fitted.

```python
# AR(1), phi = 0.7, 400 observations, default_rng(7)
select_arima_order(x, max_p=3, max_d=1, max_q=2, criterion="bic")
# before: (2, 0, 1)    after: (1, 0, 0)  (the true order)

select_arima_order(np.full(40, 2.0), 2, 1, 2)  # before: (0, 1, 1)  after: ValueError
select_arima_order(np.arange(40.0), 2, 2, 2)  # before: (0, 2, 1)  after: (2, 0, 2)
```

**`dynamic_hedge_ratio` starts from a data-independent prior mean of 0.** The
filter's state was seeded with the first observed ratio `y[0] / x[0]` (or
1.0 when `x[0] == 0`). That seed is unbounded: `x[0] = 1e-12` put the first
value at 2e12 and the filter needed dozens of observations to forget it, and
a subnormal `x[0]` overflowed the division and turned every returned value
into NaN. The filter now starts from the prior `beta ~ N(initial_beta, 1)`
with `initial_beta=0.0`, and runs the same predict/update step for every
observation including the first. The recursion itself is unchanged. The
keyword-only `initial_beta` sets the prior mean; `initial_beta=y[0] / x[0]`
reproduces the old values (the caller guards `x[0] == 0`).

```python
y = np.array([2.1, 3.9, 6.2, 7.9, 10.1, 12.2])
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
dynamic_hedge_ratio(y, x)[:4]
# before: [2.1,                1.979785520212058,  2.036637415302432, 2.001529957159298]
# after:  [2.0979021188599516, 1.9793689436713953, 2.036493431455965, 2.0014679836768012]
dynamic_hedge_ratio(y, x, initial_beta=y[0] / x[0])[:4]  # the old values
```

With `|x|` of order 5 and the default variances the change is about 4e-5
relative at t = 0 and below 1e-9 after 25 points. `beta[0]` is no longer
exactly `y[0] / x[0]`: it is that ratio shrunk towards 0 by
`obs_var / (x[0]**2 * (1 + proc_var) + obs_var)`. Where `|x|` is small
against `sqrt(obs_var)` (about 0.03 at the default), for example cumulative
log returns, the zero-mean prior keeps early values near 0 for longer. Edge
cases, 40-point series with a true ratio of 1.5 and `y[0] = 2.0`:

| `x[0]` | before: `beta[0]`, `beta[1]` | after: `beta[0]`, `beta[1]` |
|---|---|---|
| `0.0` | 1.0, 1.50119 | 0.0, 1.50115 |
| `1e-12` | 2e12, 7.1e7 | 2e-09, 1.50115 |
| `5e-324` | nan, nan (plus `RuntimeWarning`s) | 9.9e-321, 1.50115 |

Verified against `filterpy.kalman.KalmanFilter` started from the same prior
on every element from the first, to 1e-10.

### Fixed — numerical results changed

**`arima_forecast` ignored the in-sample innovations.** It dropped the MA
terms altogether, treating the innovations inside the sample as zero, so every
forecast from a model with an MA part was wrong. It now returns the exact
Gaussian conditional mean of the next `h` values given the whole series; the
conditional means of the last `q` innovations come from the same Kalman filter
the fit uses. Forecasts with no MA part (`ma_coefs` empty or all zero) are
unchanged bit for bit.

```python
# y: ARMA(1,1), phi = 0.5, theta = 0.3, 300 observations, default_rng(7)
arima_forecast(y, np.array([0.5]), np.array([0.3]), d=0, h=3)
# before: [-0.04154350411754, -0.02077175205877, -0.01038587602939]
# after:  [-0.02525242060343, -0.01262621030172, -0.00631310515086]
```

Verified against statsmodels' state-space ARIMA filtered at the same
coefficients (worst absolute error 6.7e-16 over 200 cases).

**`adf_test` clipped its p-value to [0.0001, 0.9999].** The p-value is
`Phi(polynomial(statistic))` on MacKinnon's (1994) response surface, valid
from the tabulated minimum statistic (-19.04, -18.83, -16.18 for trend `"n"`,
`"c"`, `"ct"`) to the tabulated maximum (none, 2.74, 0.70). Inside that range
the surface goes down to about 1e-30, but every p-value below 0.0001 was
reported as 0.0001 and every one above 0.9999 as 0.9999. `adf_test` now
returns the surface value as computed, exactly 0.0 below the tabulated
minimum and exactly 1.0 above the tabulated maximum (the convention
`engle_granger_test` already used and the p-value `statsmodels`'
`adfuller` reports). The statistic is unchanged, and so is every
reject/accept decision at a significance level of 0.0001 or more. A caller
that takes `log(p)` or divides by `p` must now handle 0.0.

```python
# stationary AR(1), phi = 0.5, 200 observations, default_rng(0)
adf_test(series, max_lags=1)  # before: (-6.2489, 0.0001)
adf_test(
    series, max_lags=1
)  # after:  (-6.2489, 4.5045e-08) == adfuller(series, maxlag=1, autolag=None)[:2]

# explosive: 1.05 ** np.arange(40) + default_rng(0).normal(0.0, 0.01, 40)
adf_test(series, max_lags=1)  # before: (9.3764, 0.9999)
adf_test(series, max_lags=1)  # after:  (9.3764, 1.0)
```

`engle_granger_test` changes only at a statistic exactly equal to a tabulated
bound (the surface value is used there instead of the out-of-range value).
Verified against `adfuller` for trends `"n"`, `"c"` and `"ct"` in both tails.

**`mean_variance_weights`, `l1_turnover_penalized_weights` and
`min_variance_weights` solve their quadratic programs exactly** (primal
active-set method; Nocedal and Wright 2006, Algorithm 16.3, with Bland's 1977
rule against cycling) instead of calling SLSQP. Results are the optimum to
round-off and no longer depend on the units of the inputs. SLSQP's default
`ftol=1e-6` is absolute and the objective was unscaled, so on
daily-return-scale covariances (variances around 1e-4) the solver stopped at
its equal-weight starting point and reported success. Assets excluded by
`w >= 0` are now exactly 0.0 (previously about 1e-15, sometimes slightly
negative). A zero-variance asset (cash) is solved exactly when shorting is
allowed:

```python
# one cash asset (zero row and column) plus [[.04,.01,0],[.01,.09,.02],[0,.02,.16]]
min_variance_weights(cov, allow_short=True)
# before: [1.00018, -0.00097, 0.00077, 0.00002]   after: [1, 0, 0, 0]
min_variance_weights(np.diag([0, 4, 0, 4.0]), allow_short=True)
# before: [0.5, -3.7e-9, 0.5, -3.7e-9]            after: [0.5, 0, 0.5, 0]
```

Rescaling the covariance by 1e-300 to 1e308 gives the unscaled weights to
2.2e-16.

```python
# daily-scale inputs, risk_aversion=3.0
mu = np.array([0.0004, 0.0006, 0.0003, 0.0008])
sd = np.array([0.010, 0.015, 0.008, 0.020])
corr = np.array([[1, 0.3, 0.1, 0.2], [0.3, 1, 0.4, 0.1], [0.1, 0.4, 1, 0.3], [0.2, 0.1, 0.3, 1]])
mean_variance_weights(mu, corr * np.outer(sd, sd), 3.0)
# before: [0.25, 0.25, 0.25, 0.25]                      utility 4.0997e-4 (the starting point)
# after:  [0.14809068, 0.43858288, 0.0, 0.41332644]     utility 4.4991e-4

min_variance_weights(np.diag([0.01, 0.02, 0.03]) ** 2)
# before: [1/3, 1/3, 1/3]   after: [0.7346939, 0.1836735, 0.0816327]

# well-scaled
cov = np.array([[0.04, 0.006, 0.002], [0.006, 0.09, 0.012], [0.002, 0.012, 0.16]])
min_variance_weights(cov)
# before: [0.62387083, 0.23557235, 0.14055682]  variance 0.02663310655
# after:  [0.62344281, 0.23556059, 0.14099660]  variance 0.02663306908

# annualised inputs (mu*252, cov*252), cost_bps=20, risk_aversion=3
l1_turnover_penalized_weights(
    mu * 252, (corr * np.outer(sd, sd)) * 252, np.array([0.5, 0.3, 0.1, 0.1]), 20, 3.0
)
# before: [0.17812164, 0.41731905, 0.0, 0.40455931]  objective 0.111630099967
# after:  [0.17901303, 0.41805920, 0.0, 0.40292777]  objective 0.111630530154
```

On well-scaled inputs the weights move by 1e-4 to 1e-3. With the daily-scale
inputs above, `l1_turnover_penalized_weights` returns `previous_weights` both
before and after (the cost outweighs any gain).

### Changed — behaviour on invalid input

- `min_variance_weights`, `mean_variance_weights`,
  `unconstrained_mean_variance_weights`, `l1_turnover_penalized_weights` and
  `risk_parity_weights` raise `ValueError` naming the argument when
  `cov_matrix`, `expected_returns`, `previous_weights`, `risk_aversion` or
  `cost_bps` contains a NaN or infinite value, before any solve. Before, the
  SLSQP-based optimisers and `risk_parity_weights` raised `RuntimeError`
  (non-convergence), `unconstrained_mean_variance_weights` returned NaN
  weights and `l1_turnover_penalized_weights` accepted a NaN in
  `previous_weights`.
- `min_variance_weights`, `mean_variance_weights` and
  `l1_turnover_penalized_weights` raise `ValueError: cov_matrix must be positive
  semi-definite (its smallest eigenvalue is ...)` for a covariance matrix that
  is not positive semi-definite, for either value of `allow_short`, before any
  solve. Before, `allow_short=False` returned a weight vector
  (`min_variance_weights([[1,2,2],[2,5,-10],[2,-10,5]])` gave `[0, 0.5, 0.5]`, a
  "variance" of -2.5) and `allow_short=True` returned the maximiser along the
  budget line (`min_variance_weights([[1,2],[2,1]], allow_short=True)` gave
  `[0.5, 0.5]`) or raised `RuntimeError: SLSQP optimization did not converge`.
  The test is on the symmetric part: an eigenvalue below -1e-12 times the
  largest absolute row sum counts as negative. Rank-deficient covariances (zero
  eigenvalues, an all-zero matrix, `factors @ factors.T`) are accepted.
  `unconstrained_mean_variance_weights` and `risk_parity_weights` do not run
  the check.
- `mean_variance_weights` with `risk_aversion=0` and `allow_short=True` used to
  raise `RuntimeError: SLSQP optimization did not converge`; it still raises
  `RuntimeError`, now saying the problem is unbounded. The same holds for
  `mean_variance_weights` and `l1_turnover_penalized_weights` whenever the
  curvature of the objective along a free direction is below 1e-12 times the
  largest absolute row sum of `risk_aversion * cov_matrix`. An unbounded
  `l1_turnover_penalized_weights` problem (zero covariance,
  `allow_short=True`, differing `expected_returns`) used to return
  meaningless weights (`[-1.47e8, 0.32, 1.47e8]`) and now raises
  `RuntimeError`.
- `min_variance_weights(np.eye(3) * 1e308)` and
  `mean_variance_weights([.1, .2, .3], np.eye(3) * 1e-320, 1.0, True)` raise
  `RuntimeError` naming a non-finite objective or solution ("the objective is
  not finite (its coefficients overflow double precision)", "the solution is
  not finite (...)"); both used to raise an SLSQP non-convergence
  `RuntimeError`.
- `kelly_fraction` raises `ValueError` ("expected_return must be finite" /
  "variance must be finite") for a NaN or infinite input. Before,
  `kelly_fraction(nan, 0.04)` returned `nan`, `kelly_fraction(inf, 0.04)`
  returned `1.0` (`inf` with `clip=False`) and `kelly_fraction(0.05, inf)`
  returned `0.0`.
- `arima_fit` (both methods) raises `ValueError` for a series that is not 1-D
  or contains NaN or infinity; the CSS estimator previously returned NaN for
  NaN input.
- `arima_fit` with the default method raises `quantcore.risk.ConvergenceError`
  when the exact-likelihood optimisation does not converge from either
  starting point, carrying the best attempt as `(*ar, *ma, sigma2)`, its exact
  log-likelihood and the optimiser message. No
  occurrence in 2,160 well-specified fits with n >= 100; it occurs on samples
  with about as many observations as coefficients, on series whose squares
  overflow or underflow (magnitudes near 1e200 or 1e-200), and where the
  likelihood peaks with three or more partial autocorrelations on their bound.
- `arima_fit(..., method="css")` raises `ConvergenceError` when Nelder-Mead
  stops on its iteration or evaluation limit, with the point it stopped at in
  `params` and the conditional Gaussian log-likelihood there; 0.2.0 returned
  that point as if it were the estimate. Converged CSS fits keep their numbers
  bit for bit. It did not occur at the true order (0 of 2,100 fits) but does on
  over-parameterised orders: on AR(1) samples of 200 observations, 2% of
  (2,0,1), 5% of (2,0,2), 10% of (2,1,1) and 25% of (3,0,3) fits exhaust the
  5,000 evaluations (for example `(2, 1, 1)` on a phi = 0.5 sample, whose
  stopping point has an MA coefficient of -1.074).
- `select_arima_order` raises `ValueError` naming the problem for an empty,
  non-finite or 2-D series, and for a constant series (no `d` leaves variation
  to fit) instead of returning `(0, 1, 1)`.
- `arima_forecast` raises `ValueError` if `ma_coefs` has a non-zero entry and
  `ar_coefs` is not stationary (the exact conditional mean needs the
  stationary initial state), and for non-finite inputs or non-1-D coefficient
  arrays; a `series` that is not 1-D also raises `ValueError`. Without an MA
  part any `ar_coefs` is accepted, as before.
- `monte_carlo_call_price` and `monte_carlo_put_price` with `antithetic=True`
  and `num_paths=2` (one mirrored pair) return a standard error of `0.0`; it was
  `nan` with a NumPy `RuntimeWarning`. The price is unchanged:
  `monte_carlo_call_price(100.0, 100.0, 0.05, 0.2, 1.0, 2, 10, seed=1)` returned
  `(8.94..., nan)` and returns `(8.94..., 0.0)`.
- Importing `quantcore.time_series` now imports `quantcore.risk` (for
  `ConvergenceError`). The first `arima_fit` call in a new environment
  compiles the exact-likelihood kernels; later processes load them from
  Numba's cache.
- `dynamic_hedge_ratio` raises `ValueError("initial_beta must be finite")`,
  and `ValueError` when `y` or `x` contains a NaN or infinite value or when
  `obs_var` or `proc_var` is NaN or infinite (a NaN variance passed the
  positivity check and gave all-NaN output; an infinite `obs_var` gave zeros).
  A tiny or subnormal `x[0]` no longer produces a huge or NaN output or an
  overflow `RuntimeWarning`.
- The six `simulate_*_price_paths` functions raise `ValueError` naming the
  parameter before the first draw (generator untouched) for a non-positive
  `n_paths`/`n_steps`, an invalid `dt`, `s0` or scale parameter, or a
  non-finite real-valued parameter; `simulate_jump_diffusion_price_paths` also
  raises `ValueError` when `exp(jump_mean_log + 0.5*jump_std_log**2)` exceeds
  float64. They raise `FloatingPointError` after all draws (so the generator has
  advanced) when a simulated log-price is not finite because a drift or
  variance term overflowed (for example `sigma=1e160` in
  `simulate_gbm_price_paths`), or when a price is non-finite because
  `exp` of the log-price overflowed. Before the log-price check such cases
  could return prices of exactly 0.0. A finite log-price that underflows
  returns 0.0.
- The nine ratios in `quantcore.performance.ratios` raise `ValueError` for
  input that is not 1-D, is empty, or contains NaN or infinity (the older
  functions in `metrics.py` reject only empty input).
- `jarque_bera_test`, `arch_lm_test` and `kpss_test` raise `ValueError` for
  degenerate input, listed under Added.

### Added

- `quantcore.performance.ratios` (exported from `quantcore.performance`), nine
  functions `(returns: npt.NDArray[np.float64]) -> float` on periodic simple
  returns:
  - `ulcer_index`: `sqrt(sum(drawdown**2) / (n - 1))`, drawdowns as fractions.
  - `ulcer_performance_index`: `(prod(1 + r) - 1) / ulcer_index`; `inf` when
    the Ulcer Index is 0.
  - `serenity_index`: `sum(r) / (ulcer_index * CDaR / std(r, ddof=1))`, `CDaR`
    the mean of the drawdowns above `mean(D) + 1.645 * std(D, ddof=1)`; `nan`
    for constant returns or a constant drawdown series.
  - `gain_to_pain_ratio`: `sum(r) / |sum of negative r|`; `inf` with no loss.
  - `common_sense_ratio`: `profit_factor * |Q(0.95) / Q(0.05)|`; `inf` with no
    loss and a positive lower quantile, `nan` when the lower quantile is
    exactly 0, `0.0` with no gain.
  - `cpc_index`: `profit_factor * win rate * payoff ratio`; `nan` with no
    winning or no losing period.
  - `outlier_win_ratio`: `Q(0.99) / mean(r[r >= 0])`; `nan` when that mean is
    missing or 0.
  - `outlier_loss_ratio`: `Q(0.01) / mean(r[r < 0])`; `nan` with no loss.
  - `risk_of_ruin`: `((1 - w) / (1 + w)) ** n`, `w` the winning share of the
    non-zero periods.

  All nine compute the definitions of quantstats 0.0.81 and are `verified`
  against it in the conformance suite. Where that definition differs from the
  statistic's source, the docstring says so (the Ulcer Index divides by `n - 1`;
  the Ulcer Performance Index and Serenity index are not annualised and subtract
  no risk-free rate; gain-to-pain is not resampled to months). A caller moving
  from quantstats sees: `inf` where quantstats gives `nan` for a zero risk
  denominator; `nan` for `serenity_index` of constant returns (quantstats
  returns rounding residue); and inputs are always read as returns, whereas
  quantstats reads some series as prices. Returns below -1 are accepted, as in
  `drawdown_series`. Finite returns whose compounded wealth, sum or squared
  deviations exceed float64 give `inf` or `nan` with NumPy's `RuntimeWarning`.
  `ulcer_index`, `ulcer_performance_index` and `serenity_index` also reject a
  single observation.
- `quantcore.statistics.diagnostics` (exported from `quantcore.statistics`).
  Inputs are 1-D float64 arrays of finite values; pandas objects are not
  converted.
  - `jarque_bera_test(series) -> (statistic, p_value)`: `n/6 * (S^2 + K^2/4)`
    from population moments, p-value `exp(-statistic/2)`; independent of the
    units of the input up to the ends of float64 (values of order 1e300 or
    1e-300 give the statistic of order one). Agrees with
    `scipy.stats.jarque_bera`, except that a constant series raises
    `ValueError` (scipy returns NaN).
  - `arch_lm_test(residuals, n_lags=None) -> (statistic, p_value)`: Engle's
    ARCH-LM test, `(n - n_lags) * R^2` of the squared series on its own lags;
    `n_lags=None` means `min(10, n // 5)`. Agrees with
    `statsmodels.stats.diagnostic.het_arch` and, unlike it, does not depend on
    the scale of the input: on a GARCH(1,1) series of 1,000 points with 5 lags
    the statistic is 16.3969 from both at unit scale, and after multiplying
    the series by 1e-6 `het_arch` returns 0.0 and quantcore 16.3969. Needs
    `n >= 2 * n_lags + 2`; raises `ValueError` when the squares are constant
    over the regression sample (`het_arch` returns `(-inf, 1.0)`); the
    statistic is clipped at 0.0.
  - `kpss_test(series, n_lags=None, trend="c") -> (statistic, p_value,
    p_value_at_bound)`: KPSS stationarity test for the level (`"c"`) or trend
    (`"ct"`) null; `n_lags=None` selects the lag count from the data; the
    statistic is independent of the units of the input in the same way and the
    long-run variance is computed directly from the Bartlett weights; the
    p-value is interpolated in the paper's four-point table and
    `p_value_at_bound` (a `bool`) says when it sits at an end of it. Agrees
    with `statsmodels.tsa.stattools.kpss(nlags="auto")`, including the lag
    count. Needs 3 observations for `"c"` and 4 for `"ct"`; raises `ValueError`
    when no residual exceeds rounding error (a constant series, or an exactly
    linear one under `"ct"`) or when the automatic lag count divides by zero.
- `quantcore.core.price_paths` (exported from `quantcore.core`), six functions
  taking the caller's `numpy.random.Generator` first and returning float64
  prices of shape `(n_paths, n_steps + 1)`:
  - `simulate_gbm_price_paths(rng, n_paths, n_steps, dt, *, s0, mu, sigma)`:
    geometric Brownian motion, exact log-normal step.
  - `simulate_exp_ou_price_paths(rng, n_paths, n_steps, dt, *, s0, kappa,
    theta, sigma)`: Ornstein-Uhlenbeck process on the log-price (exact Gaussian
    transition); `theta=None` uses `log(s0)`.
  - `simulate_heston_price_paths(rng, n_paths, n_steps, dt, *, s0, mu, v0,
    kappa, theta, xi, rho)`: Heston stochastic volatility, full-truncation
    Euler.
  - `simulate_garch_t_price_paths(rng, n_paths, n_steps, dt, *, s0, mu, omega,
    alpha, beta, nu)`: GARCH(1,1) with standardised Student-t innovations;
    `omega`, `alpha`, `beta` are per-step and `dt` scales the drift only.
  - `simulate_regime_switching_price_paths(rng, n_paths, n_steps, dt, *, s0,
    mu0, sigma0, mu1, sigma1, p01, p10)`: two-state Markov regime-switching
    GBM; regime probabilities are per step.
  - `simulate_jump_diffusion_price_paths(rng, n_paths, n_steps, dt, *, s0, mu,
    sigma, jump_intensity, jump_mean_log, jump_std_log)`: Merton jump diffusion
    with compensated drift, exact transition on the grid.

  Unlike the seed-taking `simulate_*_paths` in `sde_solver.py` and
  `stochastic.py`, these advance the caller's generator, so a second call
  continues the stream. The variate order and amount drawn, and the grouping of
  every floating-point product and sum, are part of the contract: for a given
  generator state the output and the final generator state are fixed (bit
  identical to a downstream implementation on 12,512 cases on macOS arm64 with
  NumPy 2.5.3; NumPy does not promise identical rounding on other platforms).
  They use plain NumPy rather than Numba because Numba's `standard_t`/`normal`
  and C `exp` do not match NumPy's bits.
- `arima_fit(..., *, method="mle")`: keyword-only `method`, `"mle"` (default)
  or `"css"`.
- `dynamic_hedge_ratio(y, x, obs_var=1e-3, proc_var=1e-5, *, initial_beta=0.0)`:
  keyword-only `initial_beta`, the prior mean of beta before the first
  observation; the prior variance stays fixed at 1.

### Performance

- `ledoit_wolf_shrinkage` computes the dispersion term in closed form
  (`sum_t ||x_t||^4 - T*||S||_F^2`) in place of a per-observation loop of
  outer products. Values are unchanged to round-off (largest difference to
  the previous implementation 3e-20 on 252x10, 2520x100 and 2520x500). Time
  per call for 252x10 / 2520x100 / 2520x500: before 1.59 ms / 94.5 ms /
  3823 ms; after 0.04 ms / 0.66 ms / 17.3 ms (scikit-learn's
  `LedoitWolf().fit`: 0.35 ms / 2.77 ms / 233 ms).

- `min_variance_weights`, `mean_variance_weights` and
  `l1_turnover_penalized_weights` at several hundred assets. The long-only
  `min_variance_weights` and `mean_variance_weights` start the exact solver
  from the budget-only solution with negative weights dropped, so the number
  of iterations does not grow with the number of assets the optimum holds.
  Milliseconds per long-only call on one machine (Apple Silicon, Accelerate;
  best of three), on a daily-scale 5-factor covariance / the sample
  covariance of 2k x k normal returns, with the number of assets held in
  brackets:

  | k | `min_variance_weights` | `mean_variance_weights` | `l1_turnover_penalized_weights` |
  |---|---|---|---|
  | 200 | 0.8 (200) / 4.0 (171) | 1.3 / 1.5 | 27 / 29 |
  | 600 | 8.1 (600) / 41 (504) | 8.7 / 9.0 | 107 / 110 |
  | 1000 | 27 (1000) / 139 (855) | 30 / 30 | 204 / 217 |

  Times depend on the machine and BLAS.

### Documentation only

- `ewma_covariance`: the docstring states that returns are not demeaned (each
  update uses the raw outer product, the RiskMetrics zero-mean definition), so
  the estimate differs from a mean-centred exponentially weighted covariance
  such as pandas' `ewm().cov()` by the contribution of the mean. No numerical
  change.
- `monte_carlo_call_price` and `monte_carlo_put_price`: the `Returns:` section
  says what the returned standard error is (a sample estimate from the same
  payoffs as the price; per-pair averages for antithetic variates with an even
  path count; the unpaired formula, which overstates, for an odd one; 0.0 for a
  single path), that it tends to understate the true standard error when few
  paths finish in the money, and that it is exactly 0.0, together with the
  price, when none do.
- `dynamic_hedge_ratio`: the docstring states the model (one state, no
  intercept, random-walk slope), the prior `N(initial_beta, 1)`, the recursion
  and its sources.
- `ConvergenceError`: the docstring lists `arima_fit` among the functions that
  raise it.

### Verification

The conformance suite lives in `conformance/`. For each covered quantity,
hypothesis draws inputs and compares quantcore with an independent oracle
(a reference library, a closed form or a hand recursion from the primary
source) under a written tolerance. `conformance/conformance_map.json` is the
committed record of every target's status, oracle and tolerance. The CI job
`conformance` fails when a fresh run's statuses differ from it, and a weekly
exploratory run draws fresh random inputs and opens an issue when a `verified`
target disagrees with its oracle. See `conformance/README.md` for how to run
and extend it; `docs/CONFORMANCE.md` is the generated report. The conformance
suite is not part of the installed package, and no library numbers change
because of it.

## [0.2.0] — Unreleased

### Fixed — numerical results changed (edgelab conformance hand-over, 2026-09-24)

Four functions edgelab's conformance suite classified `defect` against
independent oracles. Each is now `verified` against the oracle named in
its finding; no other public function's numbers changed. Per-function
one-liners for edgelab's re-pin are at the end of each item.

**`engle_granger_test`'s residual ADF regression carried a redundant
constant, and its p-value was floored at 0.0001.** Step 1 fits
`y = alpha + beta*x + u` with an intercept, so the residuals are mean-zero
by construction and the textbook Engle-Granger (1987) step 2 runs the ADF
regression on them *without* a constant — the MacKinnon (1994) N=2
cointegration surface is tabulated for the statistic computed that way,
and `statsmodels.tsa.stattools.coint` does exactly this
(`adfuller(resid, regression="n")`). quantcore's step 2 went through the
constant-including `_adf_test`, changing the statistic itself (by 0.013 on
average over simulated pairs, less negative in 93% of them). Separately,
the p-value went through `adf_test`'s [0.0001, 0.9999] clamp, so strongly
cointegrated pairs reported 0.0001 where the surface gives ~1e-12 and
`coint` reports 0.0. Fixed: step 2 now fits `[level, lagged_diff]` with no
deterministic terms, and its p-value comes from the N=2 "c" surface
unclamped (saturating at exactly 0.0 / 1.0 beyond the tabulated range —
`statsmodels.tsa.adfvalues.mackinnonp`'s convention). `adf_test` is
unchanged: it keeps its constant and its clamp.

```python
# cointegrated pair: x a random walk, y = 2x + AR(1) spread (phi=0.5),
# 180 observations, numpy.random.default_rng(169)
engle_granger_test(y, x)[1:]  # before: (-6.9361, 0.0001)
engle_granger_test(y, x)[
    1:
]  # after:  (-6.9570, 1.09e-08)  == statsmodels coint(y, x, maxlag=1, autolag=None)[:2]
```

Verified against `statsmodels.tsa.stattools.coint(y, x, trend="c",
maxlag=1, autolag=None)`: statistic and p-value now agree to 1e-10 on
cointegrated, independent-random-walk and near-unit-root pairs from 10 to
5,000 observations (`tests/oracle/statistics/test_cointegration_oracle.py`).
Monte Carlo size under the null remains ≈5% at the 5% level. *For edgelab:*
convention change — step-2 ADF has no constant; p-value unclamped,
saturating at 0.0 / 1.0.

**`maximum_drawdown` (and `drawdown_series`) ignored the starting
capital.** The running peak was seeded from the first cumulative value
rather than from the wealth of 1.0 in place *before* the first return, so
`drawdowns[0]` was always 0 and any losses before the path first exceeded
its starting value were understated or missed entirely (Bacon 2008;
`empyrical.stats.drawdown_series` prepends the start value for exactly this
reason). Fixed by seeding the running maximum with 1.0 in the shared
`_drawdown_series` helper. `time_under_water`, `max_drawdown_duration` and
`calmar_ratio` build on the same helper, so an early-loss path now counts
its first periods as underwater and Calmar's denominator reflects the true
maximum drawdown — the documented invariant
`maximum_drawdown(r) == drawdown_series(r).max()` is preserved. Paths that
make a new high on their first period are unaffected.

```python
maximum_drawdown(np.array([-0.5]))  # before: 0.0   after: 0.5
maximum_drawdown(np.array([-0.1, -0.1, 0.5]))  # before: 0.10  after: 0.19  (= 1 - 0.9*0.9)
drawdown_series(np.array([-0.1, -0.1, 0.5]))  # before: [0.0, 0.10, 0.0]  after: [0.10, 0.19, 0.0]
```

Verified against `empyrical.max_drawdown` (500 random paths of length 1 to
200, worst disagreement 1e-16) and `quantstats.stats.to_drawdown_series`
(200 paths, 1e-16; the previous oracle test had to flip a negative first
return to avoid this very edge case — it no longer does). *For edgelab:*
formula change — running peak seeded with the starting capital of 1.0;
`drawdown_series`, `time_under_water`, `max_drawdown_duration` and
`calmar_ratio` move with it.

**`risk_parity_weights` could return its equal-weight starting point
unchanged, with `success=True`, on a covariance with one much
higher-variance asset.** The SLSQP objective was the sum of squared
differences of *raw* contributions `w_i*(Sigma*w)_i`, whose scale is set by
the largest variance; with an absolute `ftol=1e-16` on an objective of
order 1e7, SLSQP declared convergence at `x0` after five internal
iterations. Replaced the SLSQP formulation entirely with Spinu (2013)'s
strictly convex reformulation `min_{w>0} 0.5*w'Sigma*w - (1/k)*sum(ln w)`,
solved by damped Newton iteration (Armijo backtracking, gradient-norm
acceptance once the objective is within round-off of its minimum) from the
inverse-volatility portfolio, on a unit-mean-variance rescaling of Sigma so
the input's scale never enters the tolerance. Its first-order condition
`w_i*(Sigma*w)_i = 1/k` *is* the equal-risk-contribution condition, and the
Hessian `Sigma + diag(1/(k*w_i^2))` is positive definite for any PSD Sigma,
so the minimiser is unique. Target tolerance is 1e-10 on the normalised
contributions; on covariances so ill-conditioned that round-off in
`Sigma*w` prevents that (rank-one-plus-1e-8, eigenvalues spanning 1e8), the
best iterate is returned once the iteration can no longer improve, provided
it is within 1e-6 — otherwise `RuntimeError`, as before. `cov_matrix` must
now have strictly positive diagonal variances (`ValueError` otherwise).

```python
cov = np.array([[4658.98781, -1.41048619], [-1.41048619, 1.72544360]])
w = risk_parity_weights(cov)
w * (cov @ w) / (w @ cov @ w)
# before: w = [0.5, 0.5],       contributions = [0.99993, 0.00007]
# after:  w = [0.0189, 0.9811], contributions = [0.5, 0.5]  (to 1e-10)
```

Verified: normalised contributions equal `1/k` within 1e-10 on
well-conditioned, equal-volatility and one-dominant-asset covariances and
within 5e-8 on near-singular and condition-number-1e8 ones, over 2,500
random matrices with 2 ≤ k ≤ 10 (`tests/unit/portfolio/test_optimization.py`);
weights still match the independent `cvxpy` solution of the same convex
problem (`tests/oracle/portfolio/test_optimization_oracle.py`). ~0.1 ms per
call. Well-conditioned inputs that SLSQP already solved correctly change
only at the 1e-4 level (SLSQP's own accuracy). *For edgelab:* formula
change — Spinu (2013) convex reformulation, Newton-solved; contributions
equal to 1e-10 (1e-6 worst case on near-singular input).

**`heston_cos_call` / `heston_cos_put` priced far-from-the-money and
short-dated options wildly wrong (one-week K=60 call on S=100 at 10.44
instead of 40.00; K=300 at 82.83 instead of ~0).** Three defects in
`_heston_cos_call`, fixed together:

1. *Truncation range not shifted by the log-moneyness.* Fang & Oosterlee
   (2008) expand in `y = ln(S_T/K)`, so the range `[a, b]` must be the
   log-return's cumulant range shifted by `x = ln(S0/K)`; the code used the
   unshifted range and, whenever `|x|` exceeded the half-width, the payoff
   kink fell outside it. Fixed: `[a, b] = x + c1 ∓ L*sqrt(c2 + sqrt(c4))`.
2. *Wrong second cumulant.* The `c2` transcribed from Fang & Oosterlee's
   Table 11 has a `xi^2 * theta * (6e^{-kT} - 7)` term where the variance of
   the log-return actually requires `theta * (4e^{-kT} - 5)`; derived from
   `Var[-I/2 + M]` with the CIR covariance function and confirmed against
   numerical derivatives of the characteristic function (the printed form
   understates the variance by up to 20% at xi=1). Fixed to the exact form;
   `c4` (from the cumulant-generating function by central differences) is
   now included in the range so it widens automatically for fat-tailed,
   Feller-violating parameters.
3. *Call expanded directly.* The call payoff grows like `e^y`, so any right
   -tail truncation error is amplified; Fang & Oosterlee recommend expanding
   the put and using parity. `heston_cos_put` is now the direct COS
   computation and `heston_cos_call` is `put + S*e^{-qT} - K*e^{-rT}`
   (previously the reverse).

Also in the same function: the `xi = 0` branch of the characteristic
function assumed a *constant* variance `v0`, but with zero vol-of-vol the
variance path is the deterministic mean-reverting curve; it now uses the
integrated variance `theta*T + (v0 - theta)*(1 - e^{-kT})/k`. Only
`xi = 0` with `v0 != theta` is affected.

The default `n_terms` rises from 128 to 2048 (~0.2 ms per price): the
tail-adaptive range is wider than the old fixed one, and 128 terms no
longer resolve it on short-dated or high-vol-of-vol cases. **Callers
pinning `n_terms=128` explicitly should drop the argument** — at 128 terms
the worst case on the grid below degrades to ~1e-2 relative, versus ~1e-8
at the default.

```python
heston_cos_call(100, 60, 0.0, 0.02, 0.04, 1.0, 0.04, 0.3, 0.0)  # before: 10.438430   after: 40.0
heston_cos_call(100, 300, 0.0, 0.02, 0.04, 1.0, 0.04, 0.3, 0.0)  # before: 82.832795   after: 0.0
heston_cos_put(100, 300, 0.0, 0.02, 0.04, 1.0, 0.04, 0.3, 0.0)  # before: 282.832795  after: 200.0
```

Verified against `QuantLib.AnalyticHestonEngine` (adaptive Gauss-Lobatto
at 1e-12) over K/S ∈ {0.5 … 3}, T ∈ {7d … 2y}, xi ∈ {0.1 … 1.0},
rho ∈ {0, -0.7}, at S=100, v0=theta=0.04, kappa=1, r=q=0: worst relative
error 1.3e-8 on prices above 1e-4, worst absolute error 4e-11 below that
(`tests/oracle/pricing/test_heston_cos_oracle.py`). At-the-money prices
at moderate vol-of-vol are unchanged to ~1e-11 relative. *For
edgelab:* formula change (range shifted by log-moneyness, exact `c2`, `c4`
term, put expanded and call by parity) and default change
(`n_terms` 128 → 2048).

### Added — additive (Workstream C, remaining items C2–C11)

All defaults unchanged; every addition below is a new optional parameter
or a new function. Oracle tests per item are listed in
`docs/VERIFICATION_PLAN.md`.

- **C2** — `sharpe_ratio`/`sortino_ratio` docstrings now state
  `risk_free_rate` is a per-period rate (confirmed to match
  `empyrical`'s own convention exactly; explicitly *not*
  `quantstats.stats.sharpe`'s annualized `rf`).
- **C3** — new `drawdown_series()`, `time_under_water()`,
  `max_drawdown_duration()`. `maximum_drawdown`'s sign convention
  (positive fraction) is unchanged.
- **C4** — `hit_rate` gained `zero_policy: Literal["loss", "win",
  "exclude"] = "loss"` (default matches the original behavior exactly).
- **C5** — `value_at_risk`/`conditional_value_at_risk` gained `method:
  Literal["gaussian", "historical", "cornish_fisher"] = "gaussian"`
  (default unchanged).
- **C6** — `newey_west_cov` gained `small_sample_correction: bool =
  False` (matches `statsmodels`' `use_correction=True`: scales by
  `T/(T-k)`).
- **C10** — `fama_macbeth_regression` gained `n_lags_nw: int | None =
  None` (Newey-West HAC standard errors of the mean risk premia when set;
  `None`, the default, keeps the original plain time-series standard
  errors).
- **C11** — `ljung_box_test` gained `model_df: int = 0` (matches
  `statsmodels.stats.diagnostic.acorr_ljungbox`'s `model_df` — subtracts
  fitted ARMA parameters from the chi-squared degrees of freedom).

### Fixed — bug fix (Workstream D priorities: VAR)

**`var_fit`'s `sigma_u` used the wrong degrees-of-freedom correction.**
`sigma_u` (the VAR residual covariance estimate) was computed as
`np.cov(residuals, rowvar=False, ddof=1)` — dividing by `T_eff - 1`, the
correction appropriate for a plain 1-parameter (mean-only) covariance
estimate. A VAR(p) equation with a constant estimates `k*p + 1`
parameters per equation, not 1; Lütkepohl (2005, Section 3.2.2)'s
unbiased estimator — and `statsmodels.tsa.api.VAR`'s — divides by
`T_eff - (k*p + 1)` instead. Fixed:

```python
n_eff, n_params = x.shape
dof = n_eff - n_params
sigma_u = (residuals.T @ residuals) / dof
```

For a 2-variable VAR(1) (`k=2, p=1`, so `k*p+1=3`) fit on 300
observations (`T_eff=299`), the old divisor was 298 vs. the correct 296 —
roughly a 0.7% understatement of `sigma_u`, which propagates into
`impulse_response`'s orthogonalized (Cholesky) impulse responses and any
downstream forecast-error variance calculation. Coefficients themselves
were already correct (OLS is unaffected by this). Verified against
`statsmodels.tsa.api.VAR`: `sigma_u` now matches to 1e-8 relative
(previously ~1% off). Docstring now cites Lütkepohl (2005).

### Fixed — bug fix (Workstream D remainder: VAR lag-order selection)

**`select_var_lag_order` compared information criteria across candidates
fit on different effective sample sizes.** Each candidate `n_lags` was fit
on `n_obs - n_lags` observations — a larger effective sample for smaller
lag orders. Information criteria are only valid for model comparison when
computed on the *same* effective sample size (Lütkepohl 2005, pp.
146-150); `statsmodels.tsa.api.VAR.select_order` enforces this by fitting
every candidate on the same trailing window of `n_obs - max_lags`
observations. On a 40-observation, 2-variable synthetic VAR(1) series with
`max_lags=10`, the old (varying-sample-size) AIC selected `n_lags=1`; the
fixed-sample-size answer — matching `statsmodels` exactly — is `n_lags=2`.
Fixed to hold the effective sample size fixed across all candidates and
to use the maximum-likelihood (not `var_fit`'s bias-corrected) residual
covariance in the criterion itself, matching `statsmodels`' formula
exactly (`ld + (2/n_eff)*free_params` for AIC, `ld +
(ln(n_eff)/n_eff)*free_params` for BIC). Verified: selected lag orders and
the underlying numeric AIC/BIC scores both now match
`statsmodels.tsa.api.VAR.select_order` exactly (to 1e-8 relative) when
restricted to the same `n_lags >= 1` candidate set (`select_var_lag_order`
only ever searches from 1, so it does not offer `statsmodels`' `n_lags=0`
candidate — an intentional scope difference, not a defect, since
`select_var_lag_order`'s stated contract is "search from 1").

### Verified — no code change (Workstream D remainder)

- **EWMA** (`ewma_variance` in `risk/egarch.py`, `ewma_covariance` in
  `portfolio/covariance.py`) — confirmed exact against the RiskMetrics
  (1996) one-step-ahead-forecast definition and its correct pandas
  equivalent (`.ewm(...).mean()` of squared/outer-product returns,
  shifted by one lag — *not* `.ewm().var()`, which is a same-period
  smoother and answers a different question). Oracle:
  `tests/oracle/risk/test_ewma_oracle.py`.
- **Portfolio optimizers** (`mean_variance_weights`,
  `min_variance_weights`, `risk_parity_weights`,
  `l1_turnover_penalized_weights`) — verified against independent
  `cvxpy` convex reformulations (Spinu (2013) for risk parity). Weights
  match to 1e-3/1e-4 absolute; `l1_turnover_penalized_weights`'s
  objective value matches to 1e-6 absolute (its weights are checked more
  loosely, since the L1 penalty makes the objective genuinely flat in
  some directions near the optimum — two correct solvers can land on
  visibly different points with near-identical objective values there).
  Oracle: `tests/oracle/portfolio/test_optimization_oracle.py`.
- **`black_litterman`, `implied_equilibrium_returns`** — verified against
  the He & Litterman (1999) / Idzorek (2005) eight-asset-class worked
  example (Π, posterior returns match the published tables to 1e-3
  absolute, the table's own rounding precision). Oracle:
  `tests/oracle/portfolio/test_black_litterman_worked_example_oracle.py`.
- **`impulse_response`, `granger_causality_test`** — verified exact
  against `statsmodels.tsa.api.VAR`/`grangercausalitytests`. Oracle:
  `tests/oracle/time_series/test_var_oracle.py`.
- **`arima_fit`** — confirmed close (not exact) to
  `statsmodels.tsa.arima.model.ARIMA`'s full Kalman-filter MLE on AR(1),
  ARMA(1,1), and AR(1)-with-differencing synthetic series. `arima_fit`
  uses conditional sum-of-squares (Box, Jenkins & Reinsel 2015); CSS and
  full ML are different, both valid, asymptotically-equivalent
  estimators for the same model — exact agreement isn't the right bar,
  and a residual numeric gap on its own isn't a defect. Oracle:
  `tests/oracle/time_series/test_arima_oracle.py`.

### Added — additive (Workstream B remainder)

**`adf_test`: new `trend: str = "c"` and `autolag: str | None = None`
parameters.** Previously `adf_test` always fit the "constant, no trend"
case with exactly `max_lags` lagged differences. `trend` additionally
accepts `"n"` (no constant, no trend) and `"ct"` (constant and linear
trend), each with its own MacKinnon (1994) p-value response surface (the
existing "c" table, table `_MACKINNON_*["c"]`, is unchanged). `autolag`
additionally accepts `"aic"`, `"bic"` (minimize the corresponding
information criterion over `0..max_lags`), and `"t-stat"` (Hall's (1994)
general-to-specific procedure, matching
`statsmodels.tsa.stattools.adfuller`'s `autolag="t-stat"` exactly,
including its `|t| >= norm.ppf(0.95)` stopping rule). Defaults
(`trend="c"`, `autolag=None`) reproduce the prior behavior exactly — every
existing call is unaffected. Verified against `statsmodels.tsa.stattools.adfuller`
for all nine (`trend`, `autolag`) combinations, matching to 1e-6 relative
on the statistic.

### Changed — BREAKING, owner-approved

**Owner follow-up 2d/decision — GARCH-family non-convergence unified on
raising.** `fit_garch_11` raised `RuntimeError` on non-convergence;
`egarch_fit`/`gjr_garch_fit` instead returned `EGARCHParams`/
`GJRGARCHParams` with `converged=False` — a real, if previously
intentional, asymmetry. Owner decision: unify on raising. All three now
raise `quantcore.risk.volatility.ConvergenceError` (new; a `RuntimeError`
subclass, so existing `except RuntimeError` callers are unaffected) on
non-convergence, carrying the best attempt's `params` (a plain tuple —
`(omega, alpha, beta)` for `fit_garch_11`, `(omega, alpha, gamma, beta)`
for `egarch_fit`/`gjr_garch_fit`), `log_likelihood`, and
`optimizer_message` for diagnostic use. The `converged` field is removed
from `EGARCHParams` and `GJRGARCHParams` (it always would have been
`True` now, since non-convergence raises instead).

```python
# before
result = egarch_fit(returns)
if not result.converged:
    ...  # fall back to a simpler model
# after
try:
    result = egarch_fit(returns)
except ConvergenceError as exc:
    ...  # exc.params, exc.log_likelihood, exc.optimizer_message available
```

**Downstream action required**: any code checking `EGARCHParams.converged`
or `GJRGARCHParams.converged` must switch to a `try`/`except
ConvergenceError` (or `RuntimeError`) block instead — that field no longer
exists.

**C8 — `ols`'s R² convention was centred even without an intercept, and
reported R²=1.0 (not undefined) for a constant `y`.** `ols` always used
`R² = 1 - SS_res / sum((y - mean(y))^2)`. Without an intercept column in
`x`, the standard convention (matching statsmodels) is the *uncentered*
R² = `1 - SS_res / sum(y^2)` — fitting against the origin, not the mean.
Separately, a constant `y` made `SS_tot` exactly 0, and the old code
returned `1.0` for that division rather than treating it as undefined.

Fixed: `ols` now detects whether `x` has an intercept column (any column
constant across all rows) and uses the matching convention; `R²` is `NaN`,
not `1.0`, whenever its denominator is exactly 0. Only affects calls with
no intercept column, or a constant `y` — both edge cases, not the typical
call with an intercept and varying `y`, which is unchanged.

**C9 — `factor_loadings` return type widened (standard errors, p-values,
R², residuals, and the resolved lag count), and its default `n_lags_nw`
changed from a fixed 4 to an automatic sample-size-based rule.**
`factor_loadings` returned only `(loadings, t_stats)` and always used 4
Newey-West lags regardless of sample size. It now returns a
`FactorLoadingsResult` dataclass (`loadings`, `standard_errors`, `t_stats`,
`p_values`, `r_squared`, `residuals`, `n_lags_nw`), and `n_lags_nw=None`
(the new default) selects `newey_west_optimal_lags(residuals)`
automatically; pass an explicit `n_lags_nw` to keep a fixed count.

```python
# before
loadings, t_stats = factor_loadings(returns, factors)  # tuple, always 4 lags
# after
result = factor_loadings(returns, factors)  # dataclass, auto lag count
result.loadings, result.t_stats  # same numbers as before, IF n_lags_nw=4 is passed explicitly
result.standard_errors, result.p_values, result.r_squared, result.residuals  # new
```

**Downstream action required**: any code unpacking `factor_loadings`'s
return as a 2-tuple must switch to the new `FactorLoadingsResult` fields;
any code relying on exactly 4 Newey-West lags by default must now pass
`n_lags_nw=4` explicitly, since the default is now automatic (and will
generally select a different lag count).

**C1 — `calmar_ratio`'s default annualized-return convention changed from
arithmetic to CAGR.** `calmar_ratio` annualized its numerator as
`mean(returns) * periods_per_year` (arithmetic), where empyrical,
QuantStats, and Young (1991) as commonly applied use the compound annual
growth rate (CAGR). Owner-approved (see `docs/VERIFICATION_PLAN.md`,
"Needs approval" — now resolved): the default is now CAGR, matching
empyrical/QuantStats. The prior arithmetic behavior remains available via
the new `return_method="arithmetic"` keyword.

```python
returns = ...  # 1000 daily returns, drift=0.001, vol=0.03
calmar_ratio(returns)  # now: CAGR-based, e.g. 1.675 (matches empyrical.calmar_ratio)
calmar_ratio(returns, return_method="arithmetic")  # prior default's formula, e.g. 1.466
```

Also added: public `cagr()` and `annualized_return(method="cagr"|"arithmetic")`.

**Downstream action required**: any code or cached result depending on
`calmar_ratio`'s previous (arithmetic) numbers must either be re-run
against the new default or pass `return_method="arithmetic"` explicitly to
keep the old numbers.

### Fixed — documentation only, no numerical or signature change

**C7 — `newey_west_optimal_lags`'s docstring/`docs/REFERENCES.md` entry
overclaimed what it computes.** It cites Newey & West (1994) and is
described (in `docs/REFERENCES.md`) as an "automatic bandwidth plug-in
rule" — read as the paper's fully data-driven, residual-adaptive
procedure. What it actually computes is the paper's separate closed-form
rule of thumb `L = floor(4*(n/100)^(2/9))`, a function of sample size
alone; `residuals`' values were never used, only its length. The
computation is correct for what it's documented to be and is unchanged;
only the documentation's precision changed — see its docstring and the
updated `docs/REFERENCES.md` entry.

### Added — additive

**C12/C13 — dividend yield / cost-of-carry (`dividend_yield`, q).**
Black-Scholes(-Merton) pricing, all five Greeks, `implied_volatility`, and
the Heston COS pricer had no way to price against an underlying that pays
a continuous dividend yield (or, for FX/futures, a cost-of-carry
adjustment) — every formula implicitly assumed q=0. Added
`dividend_yield: float = 0.0` to `black_scholes_call`, `black_scholes_put`,
`bs_delta`, `bs_gamma`, `bs_vega`, `bs_theta`, `bs_rho`,
`implied_volatility`, and `heston_cos_call`; new function `heston_cos_put`
(Heston put via put-call parity, mirroring `heston_cos_call`). `q=0.0` (the
default everywhere) reproduces every function's prior output exactly — no
existing call is affected. Verified against `QuantLib`
(`AnalyticEuropeanEngine`, `AnalyticHestonEngine`) and `py_vollib`
(`black_scholes_merton`) at q > 0, including Greeks units (vega/rho scaled
per 1.00, matching QuantLib and quantcore's existing per-1.00 convention;
`py_vollib` scales those two per 1%, confirmed via the /100 conversion in
the oracle test).

### Fixed — numerical results changed (narrow edge case)

**C14 — `implied_volatility` could silently return a search-boundary
value instead of raising, for deep ITM/OTM and/or very-short-expiry
options.** For parameters where vega is negligible (e.g. spot/strike
ratio of 100, or a strike far from spot with days to expiry), the option
price is essentially flat in volatility to double-precision across nearly
the whole `[1e-9, 10.0]` search domain. `scipy.optimize.brentq` would then
"converge" to a sigma near the domain boundary — a specific, plausible-
looking number with no relationship to the market's actual implied
volatility — rather than reporting that no reliable root exists.

Fixed: after `brentq` converges, `implied_volatility` now checks vega at
the candidate root; if it's below `1e-8 * spot`, it raises `ValueError`
instead of returning that candidate. Only affects the specific inputs
where this degeneracy occurs — a normal round trip (reasonable moneyness,
non-trivial time to expiry) is unaffected; see
`tests/unit/pricing/test_greeks.py::TestImpliedVolatility` for the
boundary between the two.

### Added — additive (continued)

**Owner follow-up 2c — `hmm_filtered_proba`, a causal alternative to
`hmm_predict_proba`.** `hmm_predict_proba` computes SMOOTHED state
posteriors (forward-backward: gamma_t(k) = P(s_t=k | y_1..y_T)) — it uses
future observations. Despite its `predict_proba`-style name, it was never
safe for a live signal or a walk-forward backtest; using it that way is a
lookahead-bias bug in the *caller*, not in quantcore, but the name invited
exactly that mistake. `hmm_predict_proba`'s signature and behavior are
unchanged (kept for API compatibility) — its docstring now states the
lookahead risk explicitly. New function `hmm_filtered_proba` computes the
causal alternative (Hamilton 1989: filtered_t(k) = P(s_t=k | y_1..y_t),
forward pass only) for code that needs a signal usable in real time.

### Fixed — documentation only, no numerical change (owner follow-up 2d)

**`value_at_risk` vs. `component_var` sign convention: verified NOT a
bug.** The follow-up asked to "make component VaR sum to portfolio VaR
under one sign convention," flagging that `value_at_risk` uses
`norm.ppf(1 - confidence_level)` where `component_var` uses
`norm.ppf(confidence_level)`. Checked directly: `component_var(...).sum()`
already equals `value_at_risk(..., mean=0.0, volatility=portfolio_std)`
to floating-point precision at every confidence level tested, because the
two functions' opposite-sign z_alpha cancels against their differing
formula structure (`value_at_risk` subtracts `mu + z_alpha*sigma`;
`component_var` has no mean term and multiplies `z_alpha` directly). No
numerical change made. Both docstrings now spell out why the sign
difference is not a bug, and
`tests/unit/risk/test_var.py::test_component_var_sums_to_portfolio_var_across_confidence_levels`
pins the identity down so it can't silently regress — or be "fixed" into
actually breaking by a future reader who reasonably suspects a mismatch
without checking the algebra.

**GARCH-family non-convergence behavior: initially verified to be a real,
already-documented, intentional asymmetry, then unified on owner
decision.** See the breaking-change entry above (`ConvergenceError`) — the
asymmetry identified here was resolved in the same pass once the owner
picked a direction, so it no longer stands as an open item.

**`ols`'s `NaN`-on-constant-`y` R² behavior: kept, now documented as a
value rather than implied by a bug-fix note.** Owner decision (following
up on the C8 fix above): the docstring's Returns section now states the
`NaN` behavior explicitly as a documented value, not just an incidental
consequence of the fix; pinned by a new test for both the centered
(constant `y`) and uncentered (all-zero `y`) cases.

### Fixed — numerical results changed (owner follow-up 2e)

**`min_variance_weights`, `mean_variance_weights`,
`l1_turnover_penalized_weights`, and `risk_parity_weights` never checked
`scipy.optimize.minimize`'s `result.success`.** All four ran SLSQP and
returned `result.x` unconditionally — a silent fallback to whatever
point SLSQP stopped at, even when it reported failure to converge (e.g.
an ill-conditioned or infeasible covariance matrix), violating
`CLAUDE.md`'s "no silent fallbacks" rule. Fixed: each now raises
`RuntimeError` if `result.success` is `False`. No test in the existing
suite triggers this path (SLSQP converges on every case already
covered), so no previously-returned number changes for any existing
passing call; only inputs that previously produced an unconverged,
possibly-wrong result are affected, and those now raise.

### Added — additive (owner follow-up 2e)

**`kelly_fraction`: new `clip: bool = True` parameter.** The default
[-1, 1] clamp is a leverage cap, not part of Kelly's (1956) original
formula, which has no such bound. `clip=False` returns the unclamped
`expected_return / variance`. Default behavior (and therefore every
existing call) is unchanged.

### Changed — default behavior changed, non-breaking to any call passing `seed` explicitly (owner follow-up 2f)

**`simulate_ou_paths`, `simulate_heston_paths`, `simulate_cir_paths`:
default `seed` changed from `0` to `None`.** These three defaulted to a
fixed seed (`seed=0`) — silently reproducible: two calls with no `seed`
argument returned identical "random" paths. `euler_maruyama` and
`simulate_gbm_paths` already defaulted to `seed=None` (fresh entropy each
call). Unified all five toward `seed=None`, the more conventional default
(matches `numpy.random.default_rng`'s own default) and the one two of the
five already used. **Only affects calls that don't pass `seed`
explicitly** — those now get independent draws on each call rather than
the same path every time; any call already passing `seed=<value>` is
completely unaffected.

### Added — additive (owner follow-up 2g)

**`quantcore.__version__`.** Reads the installed package version via
`importlib.metadata`, so it can never drift from `pyproject.toml`'s
`version` (falls back to `"0.0.0+unknown"` if the package metadata isn't
found, e.g. an uninstalled source checkout).

### Fixed — numerical results changed (Workstream D priority 1)

**`ledoit_wolf_shrinkage` mixed the unbiased (1/(T-1)) sample covariance
into a Theorem 1 computation whose derivation uses the biased (1/T)
one.** Ledoit & Wolf (2004)'s Theorem 1 — and `sklearn.covariance.LedoitWolf`'s
implementation of it — normalizes the sample covariance S, the shrinkage
target's scale mu = trace(S)/k, and the shrinkage-intensity estimate all
by 1/T (population/biased), consistently. `ledoit_wolf_shrinkage` used
`sample_covariance`'s 1/(T-1) (unbiased) convention for S instead — not
merely a different but equally valid normalization, since Theorem 1's
alpha is only the asymptotically-optimal shrinkage intensity *for* the
1/T estimator; substituting a differently-scaled S makes the result a
different, non-Theorem-1-optimal estimator.

On a 5-asset synthetic panel (200 observations), the shrunk covariance
differed from `sklearn.covariance.LedoitWolf`'s by up to ~1% of typical
variance magnitudes before the fix (not numerical noise) — verified
against `sklearn` in `tests/oracle/portfolio/test_covariance_oracle.py`,
which now matches to 1e-8 relative. Fixed by switching the shrinkage
computation's internal sample covariance to 1/T. `sample_covariance` (the
standalone public function) is unaffected and keeps its own 1/(T-1)
convention — it was never part of the Theorem 1 computation to begin
with, it's simply a separate function that happened to get reused there.

## [0.1.1] — 2026-09-21

### Fixed — numerical results changed

**B1 — `engle_granger_test` used the wrong null distribution for its
p-value (finding B1).** `engle_granger_test`'s residual-based ADF test
applied the N=1 (plain Dickey-Fuller) p-value distribution regardless of
how many series were in the cointegrating regression. Residuals from an
estimated cointegrating regression follow the Engle-Granger distribution,
which shifts left as the number of I(1) series (N) grows — using the N=1
distribution overstated significance, the expensive direction for
pairs-trading signals built on this test.

Fixed by parameterizing MacKinnon's (1994) response-surface p-value by N:
`adf_test` continues to use N=1 (a direct unit-root test on one series);
`engle_granger_test` now uses N=2 (its residuals come from a 2-variable
cointegrating regression).

`adf_test`'s own signature, and its p-value for a *raw* unit-root test
(N=1), are unchanged. Only `engle_granger_test`'s p-value changes:

| ADF statistic on residuals | p-value before (N=1, wrong) | p-value after (N=2, correct) |
|---|---|---|
| −3.00 | 0.035 | 0.110 |
| −2.86 | 0.050 | 0.147 |

A Monte Carlo size check (`tests/oracle/statistics/test_cointegration_oracle.py`)
confirms `engle_granger_test`'s rejection rate under a true null (independent
random walks) is now ≈5% at the 5% level; verified against
`statsmodels.tsa.stattools.adfuller` / `statsmodels.tsa.adfvalues.mackinnonp`.

**B2 — `johansen_trace_test` critical values were for the wrong model, and
shifted by one index (finding B2).** The regression fits an unrestricted
constant, but the table held the no-deterministic-term case's values,
additionally shifted by one index (the table's 12.3212 belonged at
k−r=2 in the no-deterministic-term case, not at k−r=1 where it was used).
Combined, the test needed a trace statistic above ~12.32 to reject at 95%
where the correct threshold for the model actually fitted is 3.84 — the
test almost never rejected.

Fixed by rebuilding the table from MacKinnon, Haug & Michelis (1999) for
all three deterministic-term cases ("n", "c", "ct") and all three
confidence levels (90/95/99%):

| k − r | 95% critical value before (wrong) | 95% critical value after (correct, "c" case) |
|---|---|---|
| 1 | 12.3212 | 3.8415 |
| 2 | 24.2761 | 15.4943 |
| 3 | 40.1749 | 29.7961 |
| 4 | 60.0627 | 47.8545 |
| 5 | 84.4500 | 69.8189 |
| 6 | 114.9020 | 95.7542 |

A Monte Carlo size check confirms the rank-0 rejection rate under a true
null (independent random walks) is now ≈5% at the 5% level; verified
against `statsmodels.tsa.vector_ar.vecm.coint_johansen`.

### Added — additive

- `johansen_trace_test`: new optional `deterministic: str = "c"` and
  `confidence: float = 0.95` keyword arguments. Defaults reproduce the
  previously-fitted model (unrestricted constant) at the 95% level, so
  existing calls are unaffected beyond the critical-value fix above.
- `johansen_max_eigenvalue_test`: new public function — the sharper
  max-eigenvalue alternative to the trace test (H0: rank = r vs.
  H1: rank = r + 1), returning the statistic, its critical values,
  the eigenvalues, and the eigenvectors from the same reduced-rank
  regression `johansen_trace_test` uses.
- `[project.optional-dependencies] oracle`: new extra
  (`statsmodels`, `arch`, `scikit-learn`, `linearmodels`,
  `empyrical-reloaded`, `quantstats`, `QuantLib`, `py_vollib`, `hmmlearn`,
  `filterpy`, `cvxpy`, `hypothesis`) for `tests/oracle/`, which
  differentially tests quantcore against independent reference
  implementations. Never imported from `src/quantcore`.

### Unchanged

- `adf_test`'s public signature and its p-value for a direct unit-root
  test (N=1) are unchanged.
- `johansen_trace_test`'s return shape (3-tuple:
  `trace_statistics, critical_values, eigenvectors`) is unchanged; it now
  returns numerically correct critical values (above) rather than a
  different shape. Eigenvalues are available from the new
  `johansen_max_eigenvalue_test` without changing this function's contract.

See `docs/VERIFICATION_PLAN.md` for the full verification plan and
`docs/VERIFICATION_REPORT.md` for the owner-facing summary of what was
checked, what changed, and what downstream projects should re-run.
