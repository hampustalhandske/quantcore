# Undefined results

A public function that returns a single number (`float` or `int`) returns a finite number or
raises. When the number is undefined or infinite for valid input, it raises
`quantcore.UndefinedResultError`, a subclass of `ValueError` whose message names the condition. It
never returns NaN or infinity and never lets a `ZeroDivisionError` escape. Invalid input (a wrong
shape, a non-finite value, an unknown option, too few observations) raises a plain `ValueError`.

A zero spread or volatility means zero range, `np.ptp(x) == 0`. A constant such as
`np.full(n, 0.002)` therefore counts as zero even though its computed standard deviation is
rounding residue near 1e-19 rather than 0. For a count or sum of one category of returns (no
negative return, no loss, no drawdown), the empty or all-zero category is the zero. A result that
comes out non-finite anyway, for example when compounding the returns overflows float64, also
raises `UndefinedResultError`.

`tests/unit/test_undefined_results.py` discovers every public single-number function from the
packages' `__all__` and runs each one that takes a returns series on a battery of degenerate
series. It also discovers every public multi-value function and runs each one that takes a series
on that series held constant at 0.0, 0.01 and 3.7: each case raises `UndefinedResultError`,
returns NaN exactly at the positions in the NaN table below, or is in the list of results defined
for a constant input. A constant input that returns a finite number anywhere else fails the test.

## Single-number functions

| Function | Undefined case(s); each raises `UndefinedResultError` |
|---|---|
| `performance.sharpe_ratio` | `returns` has zero range (constant, or a single observation) |
| `performance.sortino_ratio` | downside deviation is 0: no return below `risk_free_rate` |
| `performance.calmar_ratio` | maximum drawdown is 0 (both `return_method`s) |
| `performance.information_ratio` | `returns - benchmark_returns` has zero range |
| `performance.hit_rate` | `zero_policy="exclude"` and every return is 0; always defined for `"loss"` and `"win"` |
| `performance.profit_factor` | no negative return (all-zero included) |
| `performance.maximum_drawdown` | always defined given valid input; raises only on float64 overflow |
| `performance.ulcer_index` | always defined given valid input; raises only on float64 overflow |
| `performance.ulcer_performance_index` | Ulcer Index is 0: the path never falls below its running peak |
| `performance.serenity_index` | `returns` has zero range; the drawdown series has zero range; Ulcer Index times pitfall is 0 (volatility overflows) |
| `performance.gain_to_pain_ratio` | no negative return (all-zero included) |
| `performance.common_sense_ratio` | 5% quantile of the returns is exactly 0; no negative return |
| `performance.cpc_index` | no positive return or no negative return |
| `performance.outlier_win_ratio` | no non-negative return; the non-negative returns are all 0 |
| `performance.outlier_loss_ratio` | no negative return |
| `performance.risk_of_ruin` | always defined given valid input |
| `performance.probabilistic_sharpe_ratio` | `returns` has zero range; a power of the standard deviation over- or underflows; variance factor not positive |
| `performance.minimum_track_record_length` | as `probabilistic_sharpe_ratio`, plus SR equal to `sr_star` |
| `performance.expected_max_sharpe` | `n_trials` above about 1e16, where `1 - 1/N` rounds to 1 |
| `risk.value_at_risk` | `method="cornish_fisher"`: `returns` has zero range, or a power of its standard deviation over- or underflows. `"gaussian"` and `"historical"`: always defined (a constant series has Gaussian VaR `-mu`) |
| `risk.conditional_value_at_risk` | as `value_at_risk` |
| `risk.variance_backcast` | always defined given valid input (0 for an all-zero series); raises only on float64 overflow |
| `statistics.effective_n_eigen` | always defined given valid input (1.0 when every eigenvalue is 0) |
| `statistics.newey_west_optimal_lags` | always defined given valid input (an integer from the length alone) |
| `statistics.optimal_block_length` | `x` has zero range |
| `statistics.ou_half_life` | fitted slope shows no mean reversion: `a >= 0`, `1 + a <= 0`, or `a` not negative beyond its rounding error (straight lines, constant series) |
| `time_series.select_var_lag_order` | a column of the system has zero range (see the multi-value table) |
| `pricing.black_scholes_call` | always defined given valid input; at `volatility = 0`, `T > 0` it is the limit `max(S e^{-qT} - K e^{-rT}, 0)`; raises only on float64 overflow |
| `pricing.black_scholes_put` | always defined given valid input; at `volatility = 0`, `T > 0` it is the limit `max(K e^{-rT} - S e^{-qT}, 0)`; raises only on float64 overflow |
| `pricing.bs_delta` | `volatility = 0`, `T > 0` and the forward exactly at the strike (`ln(S/K) + (r - q)T == 0`), where d1 is 0/0 |
| `pricing.bs_gamma` | `volatility = 0`, `T > 0` and the forward exactly at the strike, where the limit is +inf; 0 elsewhere at `volatility = 0` and at `T = 0` |
| `pricing.bs_vega` | as `bs_delta` |
| `pricing.bs_theta` | as `bs_delta` |
| `execution.short_borrow_cost` | always defined given valid input; raises only on float64 overflow |
| `execution.margin_financing_cost` | always defined given valid input; raises only on float64 overflow |
| `execution.square_root_market_impact` | always defined given valid input; raises only on float64 overflow |

Every function in the table also raises `UndefinedResultError` if its result is not finite for any
other reason.

## Multi-value functions that raise `UndefinedResultError`

| Function | Undefined case(s) |
|---|---|
| `performance.sharpe_se_mertens` | as `probabilistic_sharpe_ratio` |
| `performance.deflated_sharpe_ratio` | any undefined case of `probabilistic_sharpe_ratio` or `expected_max_sharpe` |
| `statistics.adf_test` | `series` has zero range |
| `statistics.engle_granger_test` | `y` or `x` has zero range (the message names which); the ADF statistic of the residuals is not finite |
| `statistics.hac_t_test` | `x` has zero range; the Newey-West variance is not positive |
| `statistics.jarque_bera_test` | `series` has zero range |
| `statistics.arch_lm_test` | `residuals` has zero range, or their squares do |
| `statistics.kpss_test` | `series` has zero range; with `trend="ct"`, `series` is an exact straight line; the automatic lag count or the statistic is undefined |
| `statistics.factor_loadings` | a column of `factors` has zero range: it is collinear with the intercept (the message names the column) |
| `statistics.fama_macbeth_regression` | a column of `betas` is all zero, so that factor's slope is not identified (the message names the column) |
| `statistics.spa_test` | a column of the loss differential `benchmark_losses - model_losses[:, j]` has zero range: a model identical to the benchmark, or a constant offset from it, better or worse (the message names the columns; `rng` is not advanced) |
| `statistics.cscv_pbo` | every in-sample score ties on every combination, so no trial is ever selected |
| `statistics.stationary_bootstrap_ci` | `block_length=None` and `x` has zero range (from `optimal_block_length`) |
| `time_series.ljung_box_test` | `residuals` has zero range |
| `time_series.granger_causality_test` | `y` or `x` has zero range, or one of their lags is constant over the rows the regressions use |
| `time_series.var_fit` | a column of `data` has zero range (the message names the columns) |
| `time_series.select_var_lag_order` | as `var_fit` |
| `time_series.arima_fit` | the `d`-times-differenced series has zero range (a constant series at any `d`, a straight line at `d = 1`): the likelihood has no maximum |
| `time_series.select_arima_order` | every candidate long enough for its order has a zero-range differenced series. When no candidate is left for another reason (too short, not converging), it raises a plain `ValueError` |
| `risk.fit_garch_11` | `returns` has zero range |
| `risk.egarch_fit` | `returns` has zero range |
| `risk.gjr_garch_fit` | `returns` has zero range |
| `risk.egarch_11_variance` | `presample_variance=None` and the backcast of `returns` is 0 (all-zero returns, or squares that underflow), so ln(sigma_0^2) is -infinity |

Zero range is an exact test. A loss differential or a difference that is constant in exact
arithmetic but carries float rounding (for example `(b + 0.01) - b` over a varying `b`) has a
small non-zero range and is computed, not raised on.

`core.cir_fit` is outside this rule for now: on a positive constant series it raises
`CIRNoMeanReversionError`, a `RuntimeError` (not a `ValueError`) that other packages catch by
name, and it rejects an all-zero series with a plain `ValueError` because the rates must be
strictly positive.

## Documented NaN in arrays and multi-value results

NaN is allowed only inside an array or a multi-value result, at these documented positions.

| Function | Where NaN appears |
|---|---|
| `statistics.spread_zscore` | the first `window - 1` entries (warm-up), and any window whose standard deviation is 0 (every entry for a constant spread) |
| `statistics.ols` | `r_squared` when `y` has zero range (centred, `x` has a constant column) or `y` is all zero (uncentred) |
| `statistics.factor_loadings` | `r_squared`, every entry of `t_stats` and every entry of `p_values` when `returns` has zero range. `loadings`, `standard_errors` and `residuals` stay as computed: the intercept is the constant, the factor betas and standard errors are 0 or rounding residue |
| `statistics.fama_macbeth_regression` | the t-statistic of each factor whose cross-sectional slopes have zero range across periods: every period has the same cross-section of returns (a constant panel included), or there is one period. The mean risk premia stay as computed |

## Results defined for a constant input

These take a constant series and return finite numbers; the constant battery pins each.

| Function | Result for a constant input, and why it is defined |
|---|---|
| `statistics.stationary_bootstrap_ci` with an explicit `block_length` | `(c, c)` with `estimate == ci_low == ci_high` for a statistic that maps a constant to `c`: every resample of a constant is that constant (both `percentile` and `bca`) |
| `statistics.newey_west_cov` | the HAC covariance of the given regressors and residuals; constant residuals give a finite matrix (0 for all-zero residuals) |
| `portfolio.ledoit_wolf_shrinkage` | a matrix of 0, or rounding residue near 0 (about 1e-35 at 0.01): the sample covariance of constant columns is 0 and so is its shrunk estimate |
| `filtering.dynamic_hedge_ratio` | the Kalman recursion stays finite for a constant `x` or `y`. With a constant non-zero `x` the ratio absorbs every move of `y` and can grow large (above 80 for `x = 0.01` and a unit-variance random walk `y`); with `x` all zero it stays at `initial_beta` |
| `regime.hmm_fit` | every state's mean equals the constant and every variance sits at the 1e-8 floor, so the states are indistinguishable |
| `regime.hmm_decode`, `regime.hmm_predict_proba` | the model parameters are given; the path and the filtered probabilities are finite |
| `time_series.arima_forecast`, `time_series.arima_residuals`, `time_series.var_forecast` | the coefficients are given; each is a finite linear recursion of the data |
| `risk.garch_11_variance`, `risk.gjr_garch_11_variance`, `risk.ewma_variance` | the recursion runs from given parameters and a non-negative start; `omega > 0` keeps the GARCH variances positive |
| `risk.egarch_11_variance` | from a given `presample_variance`, or from the backcast of a non-zero constant |

`statistics.effective_n_eigen` (1.0 for a constant panel) and the scalar-input functions are in the
single-number table above.
