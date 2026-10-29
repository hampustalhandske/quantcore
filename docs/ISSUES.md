# Issues

Work items for trimming quantcore to what the consuming projects call, with the
consumer results each one can move. Consumers: edgelab (`src/edgelab/numeric/`) and
agentic-trading (`src/`).

Status values: `open`, `in progress`, `done`.

## Deletions (no consumer calls them; no consumer result moves)

### QC-1: Remove the COS Heston and Monte Carlo pricers — in progress
`heston_cos_call`, `heston_cos_put`, `monte_carlo_call_price`, `monte_carlo_put_price`,
their tests, conformance targets, tolerances and benchmark rows. QuantLib leaves the runtime
dependencies and the `oracle` extra; the dividend-yield oracle test uses py_vollib.

### QC-2: Remove unused performance and execution helpers — in progress
`brinson_fachler_attribution`, `rolling_sharpe`, `futures_roll_yield`, `time_under_water`,
`max_drawdown_duration`, and the public `drawdown_series`, `cagr` and `annualized_return`.
`calmar_ratio` and the Ulcer and Serenity ratios use the private versions of the last three.

### QC-3: Remove unused portfolio, filtering and core helpers — in progress
`mean_variance_weights`, `l1_turnover_penalized_weights`, `kelly_fraction`, `sample_covariance`,
`ewma_covariance`, `kalman_filter`, `kalman_smooth`, `euler_maruyama`,
`core.stochastic.simulate_heston_paths`. `dynamic_hedge_ratio`, `simulate_heston_price_paths` and
`simulate_exp_ou_price_paths` stay. The active-set quadratic-program solver in
`portfolio/optimization.py` was written for the mean-variance and L1-turnover programs; only
`min_variance_weights` still uses it. Replacing it with something smaller could move
`min_variance_weights` results, so it is left as it is.

### QC-3b: Remove unused option helpers — in progress
`implied_volatility`, `option_expiry_payoff`, `option_position_pnl`.

### QC-4: Remove Johansen — in progress
`johansen_trace_test`, `johansen_max_eigenvalue_test`.

## Delegation (a library does the job; quantcore keeps one wrapper per function)

Every delegated call passes every option explicitly.

### QC-5: Ljung-Box, KPSS, ARCH-LM to statsmodels — code done, gates pending
Moves: edgelab `stats/diagnostics`; agentic `arima_forecast_analyst` (Ljung-Box).
`statsmodels` is now a runtime dependency (`>=0.15,<0.16`). Every option is passed
explicitly; the conformance targets for the three are replaced by convention tests
(`TestStatsmodelsConventions`, `test_ljung_box_passes_every_statsmodels_option_explicitly`).
Unit tests that need running: the KPSS hand-computed values, the scale-invariance
cases down to 1e-300, and the constant-series errors.

### QC-6: ADF and Engle-Granger to statsmodels — code done, gates pending
Moves: ADF tail p-values (edgelab diagnostics; agentic pair admission, including
`scripts/screen_cointegrated_pairs.py`).
`adf_test` calls `adfuller` and `engle_granger_test` calls `coint` (beta from the step-1 OLS),
all options explicit; `statsmodels>=0.15` is required because `result_object=False` is
passed. The MacKinnon tables and the hand-written ADF regression are deleted, and with
them the oracle tests that compared them with statsmodels. The `adf_test` and
`engle_granger_test` conformance targets, repro configs and benchmark rows still compare
statsmodels with itself; they go in QC-11 (conformance trim).

### QC-7: OLS with HAC (`ols`, `newey_west_cov`, `newey_west_optimal_lags`) — kept
Moves nothing. Kept as closed forms (lstsq, a Bartlett sandwich, a one-line lag rule): they
take residuals and regressors directly, statsmodels' HAC needs a fitted results object, and
arch's Newey-West bandwidth is data-adaptive, which would move edgelab's HAC t-statistics
for no gain in correctness. The reason is in `statistics/regression.py`'s module docstring.

### QC-8: VAR, Granger and impulse response to statsmodels — code done, gates pending
Moves: agentic `macro_var_spillover_analyst`.
`var_fit` is `VAR.fit(maxlags=p, method="ols", ic=None, trend="c")`, `var_forecast` and `impulse_response` use
statsmodels' `forecast` and `ma_rep` (Cholesky applied here), `granger_causality_test` is
`grangercausalitytests(..., maxlag=[p], addconst=True)`'s `ssr_ftest`. Coefficients, sigma_u,
forecasts, IRFs, F and p agree with the previous implementation to 4e-13 on 30 fits. `select_var_lag_order` fits each candidate with `VAR.fit` on the shared trailing window and
scores it with Lutkepohl's criteria from `sigma_u_mle`; selections equal the previous rule's on
480 seeded cases, tiny samples included. `var_fit` and `select_var_lag_order` now raise for a
single series (statsmodels' `VAR` needs two); no caller passes one.

### QC-9: EGARCH and GJR to arch, `backend` removed from fit_garch_11 — code done, gates pending
Parameterisation pinned; `ConvergenceError` stays the exception raised on non-convergence.
Moves: agentic egarch and gjr analysts, and the GARCH overlay (together with the
`fit_garch_11` default already in this tree).
The fit options are explicit (`mean="Zero"`, normal innovations, `rescale=False`, residuals
divided by their standard deviation, `maxiter=1000`) and the constant is converted back to the
units of `returns`. The variance kernels take `presample_variance`, the fit returns the value it
used, and the kernel path equals arch's `conditional_volatility**2` to 1e-10 (pinned in
`tests/unit/risk/test_arch_fits.py`).

Decision: arch is kept as the only implementation. agentic fits on the last 750 returns
(`risk_window_garch`) and requires at least 250 (`min_garch_observations`), so every window is
250 observations or more. Non-convergence or an infeasible point (raised as `ConvergenceError`,
which agentic treats as "analyst unavailable that day"), 100 seeded windows each, previous
SLSQP fitters against arch:

| model | n=60 | n=120 | n=250 | n=500 | n=750 |
|---|---|---|---|---|---|
| EGARCH, previous | 0% | 0% | 0% | 0% | 0% |
| EGARCH, arch | 10% | 4% | 1% | 1% | 0% |
| GJR-GARCH, previous | 0% | 0% | 0% | 0% | 0% |
| GJR-GARCH, arch | 17% | 9% | 3% | 0% | 0% |

GARCH(1,1) already used arch: 2%, 0%, 0% at n=60, 120, 500. The two fitters maximise
different likelihoods (the first variance is the unconditional variance, or arch's backcast
for the pre-sample term), so each wins under its own seeding. Fitted parameters of both,
evaluated under each seeding, at n=750: under arch's seeding arch's parameters have the higher
log-likelihood by more than 1e-4 in 92 of 98 EGARCH and 99 of 100 GJR cases (median 6e-3 and
0.1); under the unconditional seeding the previous parameters do in 94 of 98 and 100 of 100
(median 8e-3 and 0.1). At n=250 the medians are 4e-3 (EGARCH) and 0.3 (GJR). Fitted parameters
therefore move by the seeding rule, not by a worse optimum. Previous EGARCH parameters could
not be evaluated under arch's recursion in 2 of 100 windows at n=750 and 18 at n=250, where the
recursion divides by an underflowed standard deviation.
Consumers that recompute the variance path from a fit pass `presample_variance` (the fitted
`EGARCHParams`/`GJRGARCHParams` field, or `variance_backcast(returns)` for `fit_garch_11`);
without it the path starts from the unconditional variance and differs from the fitted path.

### QC-10: Historical CVaR without empyrical — open
Closed form that reproduces `empyrical.conditional_value_at_risk`, ties included, bit-identical
on the conformance inputs. If it cannot be made identical, empyrical stays. Moves nothing.

## Conformance

### QC-11: Trim the conformance machinery — open
Delegated targets become convention tests (options passed, units, annualisation, sign);
hand-formula oracles stay; harness, classification and report code shrink to what those need.
`conformance_map.json` and `docs/CONFORMANCE.md` are regenerated in the same change.

## Dependencies

### QC-12: Compatible dependency ranges — open
`statsmodels` (and `scikit-learn` once QC-13 lands) declared as ranges in `pyproject.toml`;
`uv.lock` holds the exact versions for quantcore's CI. Ranges are checked against both
consumers' locks.

## Receive from edgelab

### QC-13: General calculations moved in from edgelab — open
Sharpe and its standard error, HAC t, bootstrap CI, PSR, DSR, CSCV/PBO, effective N, and the
resampling calls. Function list and bit-identity agreed with edgelab before work starts.
Ledoit-Wolf is delegated to scikit-learn in this step only if it is identical to the current
closed form.

## Evaluation

### QC-14: Replace the active-set QP solver? — open (after QC-12)
`portfolio/optimization.py` carries a general active-set solver (~700 lines) used only by
`min_variance_weights`. Measure a replacement (`scipy.optimize` or a small QP library) on
agentic's real covariance shapes for speed on the per-pod-per-day path and for the maximum
weight difference. Replace it only if it is not slower and the weights agree to 1e-8; otherwise
keep it and record the reason here.
