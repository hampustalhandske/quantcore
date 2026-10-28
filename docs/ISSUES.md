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
`statsmodels` is now a runtime dependency (`>=0.14,<0.16`). Every option is passed
explicitly; the conformance targets for the three are replaced by convention tests
(`TestStatsmodelsConventions`, `test_ljung_box_passes_every_statsmodels_option_explicitly`).
Unit tests that need running: the KPSS hand-computed values, the scale-invariance
cases down to 1e-300, and the constant-series errors.

### QC-6: ADF and Engle-Granger to statsmodels — open
Moves: ADF tail p-values (edgelab diagnostics; agentic pair admission, including
`scripts/screen_cointegrated_pairs.py`).

### QC-7: OLS with HAC (`ols`, `newey_west_cov`, `newey_west_optimal_lags`) — open
Moves: edgelab `stats/hac`, `attribution/regression`; agentic Newey-West.

### QC-8: VAR, Granger and impulse response to statsmodels — open
Moves: agentic `macro_var_spillover_analyst`.

### QC-9: EGARCH and GJR to arch — open
Parameterisation pinned; `ConvergenceError` stays the exception raised on non-convergence.
Moves: agentic egarch and gjr analysts, and the GARCH overlay (together with the
`fit_garch_11` default already in this tree).

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
