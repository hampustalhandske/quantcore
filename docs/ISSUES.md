# Issues

Work items for trimming quantcore to what the consuming projects call, with the
consumer results each one can move. Consumers: edgelab (`src/edgelab/numeric/`) and
agentic-trading (`src/`).

Status values: `open`, `in progress`, `done`.

## Deletions (no consumer calls them; no consumer result moves)

### QC-1: Remove the COS Heston and Monte Carlo pricers — in progress
`heston_cos_call`, `heston_cos_put`, `monte_carlo_call_price`, `monte_carlo_put_price`,
their tests, conformance targets, tolerances and benchmark rows. QuantLib leaves the runtime
dependencies and stays an oracle in the `oracle` extra.

### QC-2: Remove unused performance, risk and execution helpers — open
`brinson_fachler_attribution`, `rolling_sharpe`, `futures_roll_yield`, `drawdown_series`,
`time_under_water`, `max_drawdown_duration`, `cagr`, `annualized_return`.

### QC-3: Remove unused portfolio, filtering, core and options helpers — open
`mean_variance_weights`, `l1_turnover_penalized_weights`, `kelly_fraction`, `sample_covariance`,
`ewma_covariance`, `kalman_filter`, `kalman_smooth`, `euler_maruyama`,
`core.stochastic.simulate_heston_paths`, `implied_volatility`, `option_expiry_payoff`,
`option_position_pnl`. `dynamic_hedge_ratio`, `simulate_heston_price_paths` and
`simulate_exp_ou_price_paths` stay.

### QC-4: Remove Johansen — open
`johansen_trace_test`, `johansen_max_eigenvalue_test`.

## Delegation (a library does the job; quantcore keeps one wrapper per function)

Every delegated call passes every option explicitly.

### QC-5: Ljung-Box, KPSS, ARCH-LM to statsmodels — open
Moves: edgelab `stats/diagnostics`; agentic `arima_forecast_analyst` (Ljung-Box).

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
