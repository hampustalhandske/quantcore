# Issues

Work items for trimming quantcore to what the consuming projects call, with the
consumer results each one can move. Consumers: edgelab (`src/edgelab/numeric/`) and
agentic-trading (`src/`).

Status values: `open`, `in progress`, `done`.

## Deletions (no consumer calls them; no consumer result moves)

### QC-1: Remove the COS Heston and Monte Carlo pricers — done (fe8f1f9)
`heston_cos_call`, `heston_cos_put`, `monte_carlo_call_price`, `monte_carlo_put_price`,
their tests, conformance targets, tolerances and benchmark rows. QuantLib leaves the runtime
dependencies and the `oracle` extra; the dividend-yield oracle test uses py_vollib.

### QC-2: Remove unused performance and execution helpers — done (580e597)
`brinson_fachler_attribution`, `rolling_sharpe`, `futures_roll_yield`, `time_under_water`,
`max_drawdown_duration`, and the public `drawdown_series`, `cagr` and `annualized_return`.
`calmar_ratio` and the Ulcer and Serenity ratios use the private versions of the last three.

### QC-3: Remove unused portfolio, filtering and core helpers — done (372938f)
`mean_variance_weights`, `l1_turnover_penalized_weights`, `kelly_fraction`, `sample_covariance`,
`ewma_covariance`, `kalman_filter`, `kalman_smooth`, `euler_maruyama`,
`core.stochastic.simulate_heston_paths`. `dynamic_hedge_ratio` and `simulate_heston_price_paths` stay. The active-set quadratic-program
solver in `portfolio/optimization.py` is removed in QC-14.

### QC-3b: Remove unused option helpers — done (ca32f58)
`implied_volatility`, `option_expiry_payoff`, `option_position_pnl`.

### QC-4: Remove Johansen — done (215f209)
`johansen_trace_test`, `johansen_max_eigenvalue_test`.

## Delegation (a library does the job; quantcore keeps one wrapper per function)

Every delegated call passes every option explicitly.

### QC-5: Ljung-Box, KPSS, ARCH-LM to statsmodels — done (f6b5d32)
Moves: edgelab `stats/diagnostics`; agentic `arima_forecast_analyst` (Ljung-Box).
`statsmodels` is now a runtime dependency (`>=0.15,<0.16`). Every option is passed
explicitly; the conformance targets for the three are replaced by convention tests
(`TestStatsmodelsConventions`, `test_ljung_box_passes_every_statsmodels_option_explicitly`).
The full test suite runs on the final tree (see "Release gates" below).

### QC-6: ADF and Engle-Granger to statsmodels — done (b8d4853, e72de88)
Moves: ADF tail p-values (edgelab diagnostics; agentic pair admission, including
`scripts/screen_cointegrated_pairs.py`).
`adf_test` calls `adfuller` and `engle_granger_test` calls `coint` (beta from the step-1 OLS),
all options explicit; `statsmodels>=0.15` is required because `result_object=False` is
passed. The MacKinnon tables and the hand-written ADF regression are deleted, and with
them the oracle tests that compared them with statsmodels. The `adf_test` and
`engle_granger_test` conformance targets, repro configs and benchmark rows, which compared
statsmodels with itself, are removed (e72de88).

### QC-7: OLS with HAC (`ols`, `newey_west_cov`, `newey_west_optimal_lags`) — kept
Moves nothing. Kept as closed forms (lstsq, a Bartlett sandwich, a one-line lag rule): they
take residuals and regressors directly, statsmodels' HAC needs a fitted results object, and
arch's Newey-West bandwidth is data-adaptive, which would move edgelab's HAC t-statistics
for no gain in correctness. The reason is in `statistics/regression.py`'s module docstring.

### QC-8: VAR, Granger and impulse response to statsmodels — done (d2d57fa)
Moves: agentic `macro_var_spillover_analyst`.
`var_fit` is `VAR.fit(maxlags=p, method="ols", ic=None, trend="c")`, `var_forecast` and `impulse_response` use
statsmodels' `forecast` and `ma_rep` (Cholesky applied here), `granger_causality_test` is
`grangercausalitytests(..., maxlag=[p], addconst=True)`'s `ssr_ftest`. Coefficients, sigma_u,
forecasts, IRFs, F and p agree with the previous implementation to 4e-13 on 30 fits. `select_var_lag_order` fits each candidate with `VAR.fit` on the shared trailing window and
scores it with Lutkepohl's criteria from `sigma_u_mle`; selections equal the previous rule's on
480 seeded cases, tiny samples included. `var_fit` and `select_var_lag_order` now raise for a
single series (statsmodels' `VAR` needs two); no caller passes one.

### QC-9: EGARCH and GJR to arch, `backend` removed from fit_garch_11 — done (68e09a3, ede360c)
Parameterisation pinned; `ConvergenceError` stays the exception raised on non-convergence.
Moves: agentic egarch and gjr analysts, and the GARCH overlay (together with the
`fit_garch_11` default already in this tree).
The fit options are explicit (`mean="Zero"`, normal innovations, `rescale=False`, residuals
divided by their standard deviation, `maxiter=1000`) and the constant is converted back to the
units of `returns`. The variance kernels take `presample_variance` (default: the backcast of the residuals), the
fit returns the value it used, and the default kernel path equals arch's `conditional_volatility**2` to 1e-10 (pinned in
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
The kernels' default `presample_variance` is arch's backcast of the residuals passed in, so a
consumer that recomputes the variance path from a fit gets the fitted path without any change;
passing the unconditional variance explicitly restores the previous start.

On 40 simulated GARCH(1,1) series of 750 returns, scored under arch's own Gaussian likelihood,
the new `fit_garch_11` parameters are at least as good as the previous ones in all 40 (gains
between 3e-5 and 5.2 log-likelihood units; the three series whose parameters moved most gain
0.23, 5.25 and 0.65). The previous optimiser stopped at a weaker point under that likelihood, so
the moves in omega (up to 0.033) and in the last variance (up to 11%) are the fit improving
together with the seeding change, not arch landing on a local optimum.

### QC-10: Historical CVaR without empyrical — done
The mean of the `int((n - 1) * (1 - alpha)) + 1` smallest returns via `np.partition`, the
same expression empyrical evaluates, so the result is bit-identical (pinned by
`tests/oracle/risk/test_var_methods_oracle.py` on 120 inputs with heavy ties). `backend`,
`HistoricalCvarBackend` and the tie-inclusive variant are gone; empyrical-reloaded is only in the
oracle extra. The historical CVaR moves for returns that tie at the tail edge (0.2.0 included
all tied values).

## Conformance

### QC-11: Trim the conformance machinery — done, map regenerated at the release gates
Delegated targets become convention tests (options passed, units, annualisation, sign);
hand-formula oracles stay; harness, classification and report code shrink to what those need.
`conformance_map.json` and `docs/CONFORMANCE.md` are regenerated in the same change.
Done in e72de88: the ADF and Engle-Granger targets and their generators, repro configs and
benchmark rows, and the KPSS and ARCH-LM benchmark rows; the three variance targets compare with
arch from the first step. No target compares a library with itself. The machinery no target
uses is removed (about 850 lines: the edge-case recorder, the unit converters, the repro
summaries for non-scalar quantities, the alternative-definition branch of the classifier). The
findings and repro pipeline stays, because a live target can still be classified `defect` or
`non-standard`. The next regeneration adds the nine QC-13 targets and drops nothing.

## Dependencies

### QC-12: Compatible dependency ranges — done
`pyproject.toml` declares `numba>=0.67`, `numpy>=2.5`, `scipy>=1.18`, `arch>=8.0,<9` and
`statsmodels>=0.15,<0.16`: lower bounds at the versions the suite runs on, upper bounds at arch's
next major and statsmodels' next minor. scikit-learn stays out of the runtime dependencies
(Ledoit-Wolf is kept, 2026-11-05). Both consumers' locks fall inside the ranges: edgelab locks
numpy 2.5.3, scipy 1.18.1, numba 0.67.0, arch 8.0.0 and statsmodels 0.15.0; agentic-trading locks
numpy 2.5.2, scipy 1.18.1 and numba 0.67.0, and resolves arch and statsmodels at its re-pin.
`uv.lock` is gitignored, so CI resolves the ranges afresh on each run.

## Receive from edgelab

### QC-13: General calculations moved in from edgelab — done (ffc4e26, 12fe474, cf56cc6, 7d22c44)
Agreed list (edgelab audit and rulings): numpy in, frozen dataclass or tuple out, no
pandas and no edgelab types; edgelab keeps N logic, annual-to-per-period rf conversion and the
`StatOutcome` wrappers.
1. `hac_t_test` (`statistics/hac.py`): done (ffc4e26).
2. `sharpe_se_mertens`, `probabilistic_sharpe_ratio`, `minimum_track_record_length`,
   `expected_max_sharpe` (arithmetic unchanged, Bailey and Lopez de Prado 2014) and
   `deflated_sharpe_ratio` (`performance/sharpe_inference.py`): done (12fe474).
3. `statistics/bootstrap.py`: done (7d22c44).
   - `optimal_block_length(x) -> float`: arch's `optimal_block_length`, `"stationary"` column.
   - `stationary_bootstrap_ci(x, stat, *, rng, n_boot=2000, ci=0.95, method="percentile"|"bca",
     block_length=None) -> BootstrapInterval(estimate, ci_low, ci_high, ci_level, block_length,
     n_boot, n_dropped)`: arch `StationaryBootstrap` on a caller-supplied `Generator`; a resample
     on which `stat` raises `ValueError` or is non-finite is dropped and counted. BCa (Efron 1987)
     stays because edgelab's `validation/runs.py` calls it.
   - `spa_test(benchmark_losses, model_losses, *, rng, block_size, reps, studentize) ->
     SpaPValues(lower, consistent, upper)`: arch `SPA`, in losses. In arch 8.0.0 `studentize`
     changes no p-value: the statistic is the largest mean loss differential, unstudentised.
4. `cscv_pbo(returns, *, n_splits=16, evaluate, max_combinations=None, rng=None) ->
   CscvPbo(pbo, n_splits, n_combinations, n_rows_dropped, subsampled)` in
   `statistics/multiple_testing.py`: done (cf56cc6). The logits, per-combination records,
   `prob_oos_loss` and the degradation regression stay out: no consumer reads them.
5. `effective_n_eigen(returns) -> float` in `statistics/multiple_testing.py`: done (cf56cc6).
API agreed with edgelab on 2026-11-05. edgelab keeps the 20-observation floor, the
10%-dropped rule, the Ljung-Box warning, the 3-trial floor, the returns-to-losses conversion and
the SPA block-size resolution. Every move is bit-identical.
Not moved, deleted in edgelab: `sharpe_se_lo`, `sharpe_power`, `min_detectable_sharpe`,
`stationary_bootstrap_resamples`, `paired_stationary_bootstrap_replicates`,
`effective_n_clustered` (with it the k-means and silhouette wrappers). Ruled 2026-11-05:
`dsr_trial_count` stays in edgelab (N logic), and Ledoit-Wolf stays quantcore's own code.
The conformance targets are added per item; `conformance_map.json` and `docs/CONFORMANCE.md` are
regenerated once after the last item (one conformance run), so `build_report --check` fails on the
commits between 358a6cc and that regeneration. Measured: 920 lines added to `src/`
(7,881 at 358a6cc to 8,801), SPA included, which the earlier estimate of about 480 left out.
The trim plan's bound is the lines edgelab deletes, which edgelab measures when it removes its
copies (estimated about 750).

## Release gates

After QC-11 to QC-13: one conformance run with `conformance_map.json` and `docs/CONFORMANCE.md`
regenerated, then the full gates (pytest, ruff, mypy, `build_report --check`) on the final tree.
Both need the machine's heavy-job slot. The owner then pushes `trim`, merges it and tags v0.3.0.
Run on 2026-11-05: conformance 79 tests passed, 74 targets (73 verified, `select_arima_order`
unverifiable; the nine QC-13 targets added, no status changed); full suite 2470 passed; ruff,
mypy and `build_report --check` clean.

### QC-18: Tests that depend on the platform's last bits fail on Linux — fixed (test-only)
GitHub Actions on ubuntu-latest (run 42168519256, at 599b8bd) failed 9 tests in
`tests/unit/statistics/test_bootstrap.py`: the three `optimal_block_length` pins and six of
the seven `stationary_bootstrap_ci` pins. Each value differs from the macOS arm64 pin by one to
four units in the last place (at most about 5e-16 relative). The NumPy sums take a different
order on that build. The tests compared with `==`. They now compare at relative 1e-12, the
tolerance the effective-N pins already use. A resample that differs by one draw moves these
values by many orders of magnitude more, so the check still catches a change in the resamples
or in the statistic. The SPA p-values (counts over replicates) and the PBO pins (ratios of
small integers) stay exact; they passed on Linux. The conformance job passed. No library code
changed, so the v0.3.1 tag stays at 599b8bd.

The pull-request run at a09faa2 (42170073473) then failed
`test_units_of_the_returns_only_rescale_omega[10000.0]` in `tests/unit/risk/test_arch_fits.py`:
omega differed by 1.2e-6 relative, and the test allowed 1e-6. Measured locally: perturbing each
input by at most one unit in the last place moves the fitted omega by up to 1.03e-6 relative,
and alpha and beta by up to 1.2e-7 absolute. 1e-6 is therefore below the fit's own noise. The
same measurement on the GJR rescaling test (`TestArchConventions`) gives 1.5e-6 relative on
omega against its 1e-5. Both omegas are now compared at 1e-4 relative, and `fit_garch_11`'s
alpha and beta at 1e-5 absolute: about 50 to 100 times the measured noise. A wrong unit
convention would move omega by a factor of `scale**2`.

Sweep of the whole suite. A local pytest plugin flipped the last bit of the generated test
inputs. It recorded, for every `pytest.approx` and `assert_allclose`, the share of the tolerance
used. Under the flips, the only failures were the 18 bit-exact price-path pins. These are kept
exact: they are skipped off macOS arm64 already. No other comparison used more than 16% of its
tolerance where the tolerance is under 1e-9. The largest was the PACF step-up formula at rtol
1e-15: IEEE arithmetic on fixed inputs, kept.

A linux/arm64 container run of the test job (Linux OpenBLAS and libm) failed one more test.
`TestOuHalfLife::test_invalid_inputs` used `cumsum(ones(50))` as its diverging series. Its
level slope is exactly 0, so whether `ou_half_life` raises depends on the sign of lstsq's
rounding: macOS gives 0.0 and raises, while Linux arm64 gives about -1e-17 and returns a
half-life of about 1e16. The test now uses an explosive series (slope +0.05). The zero-slope
case itself is QC-19. A linux/amd64 run was not possible on this Mac: under emulation,
importing quantcore alone took over 10 minutes. The x86-64 evidence is GitHub's own runs.
`TestCirMoments` in `tests/oracle/core/test_sde_moments_oracle.py` simulates a
200,000 x 2,001 array (3.2 GB). It was left out of the container run, which was capped at
4 GB. It passes on GitHub's runners and on macOS.

### QC-19: `ou_half_life` on a series with no mean reversion up to rounding — done (v0.3.2)
For a straight line, such as `cumsum(ones(n))`, the AR(1) slope on the level is exactly 0. lstsq
returns rounding noise of either sign. A negative noise of about -1e-17 gave a half-life of about
1e16, or `-inf` once `log(1 + a)` rounds to 0, instead of the documented `ValueError`.
`ou_half_life` now also raises unless the centred slope `sum(c * delta) / sum(c**2)`, with
`c = level - mean(level)`, is below minus its worst-case rounding bound
`eps * (n * sum(|c * delta|) + 2 * max(|level|) * sum(|delta|)) / sum(c**2)`. The bound has no
tuned constant. Measured: on 474 exact straight lines (n from 5 to 1e6, steps 1e-8
to 1e5, offsets up to 1e12) the centred slope used at most 0.28 of the bound; on 2,144
mean-reverting AR(1) fits (n from 5 to 20,000, phi from -0.5 to 0.999, offsets up to 1e8) the
smallest margin was 1.45e5 times the bound, and none was refused. The returned half-life is
still the formula on lstsq's slope, so no returned number moves. Owner's word given 2026-11-06.

### QC-20: `ConvergenceError` does not unpickle — done (v0.3.2)
`pickle.loads(pickle.dumps(err))` raised `TypeError`: `__init__` takes four arguments and
`BaseException`'s default `__reduce__` replays only the message. agentic-trading checkpoints
fits that hold a `ConvergenceError` (GJR, EGARCH). `ConvergenceError.__reduce__` returns all
four constructor arguments and the instance `__dict__`, so notes survive too.
`CIRNoMeanReversionError` takes only a message and already round-trips; quantcore has no other
exception classes.

### QC-22: `cscv_pbo` returns 1.0 when no trial is ever selected — done (v0.3.2)
Reported by edgelab on its re-pin to v0.3.1: `cscv_pbo(np.zeros((80, 4)), n_splits=8)` gives PBO
1.0. When every in-sample score ties, the selected trial's out-of-sample mid-rank is `(N + 1) / 2`,
so omega is 1/2, the logit is exactly 0 and the combination counts as overfit. With no trial
ever selected PBO is undefined, so `cscv_pbo` now raises `ValueError` when every in-sample score
ties on every combination evaluated. Under the default Sharpe that is an all-constant panel
(QC-17's rule makes every constant column score 0) or a panel of identical trials. A custom
`evaluate` that separates the columns, such as the mean on distinct constants, still returns a
PBO, and a panel where any combination has a best trial is unchanged. The conformance
generator draws continuous panels, which never tie, so the conformance map does not move.
Agreed with edgelab (its one caller, `_pbo_outcome`, reports the case as not evaluated) and
accepted by the master under the owner's rule for producer-consumer fixes, 2026-11-06.

### QC-21: `ou_half_life` loses its slope when the level is far from 0 — done (batch 2)
lstsq fits the slope on the uncentred `[1, level]`, whose condition number grows with
`|mean(level)| / std(level)`. Measured against the centred slope on AR(1) spreads with unit
noise: relative error up to 9e-11 at an offset of 1e4 and 8e-9 at 1e6, and at 1e8 the slope is
lost (relative error 1, sign wrong on 345 of 720 fits). Since v0.3.2 a wrong-sign slope there
raises `ValueError`, but a wrong negative slope still returns a wrong half-life. Proposed fix:
compute the slope in centred form, which moves the last bits of every half-life, so it needs
a conformance run, a changelog example and the owner's word.
Done (owner yes, 2026-11-10): the centred slope, and `log1p` for `ln(1 + a)`. Against the exact
rational OLS half-life on 359 fits, the median error moves 3 → 0.5 ULPs at offset 0, 9 → 1 at
1e2 and 405 → 37 at 1e4; the shift from 0.3.2 is median 3, 9 and 453 ULPs. The tests bound the
error at 4, 8 and 64 ULPs on nine fixed cases, where the old formula fails seven.

## Evaluation

### QC-14: Replace the active-set QP solver — done (deleted)
`portfolio/optimization.py` carried a general active-set solver (~700 lines) used only by
`min_variance_weights`. No caller remained: agentic removed its `min_variance` method (AT-117),
and edgelab and pitch-edge never called it. The solver and `min_variance_weights` are deleted.
A replacement was measured before that and is recorded for anyone reintroducing the function:
with shorting allowed it is a Cholesky solve of `Sigma w = 1` (least squares on the Lagrange
system where `Sigma` is singular), which agreed with the removed solver to 5e-11 in the weights
on 192 random covariances (k = 2 to 100, scales 1e-8 to 1e6) and was faster at every size
(k = 300: 2.1 against 3.8 ms). A long-only SLSQP version differed by up to 1.2e-7 and was 26 times
slower at k = 300, so it would need a better solver.
Source lines in `portfolio/optimization.py`: 1034 to 234.

### QC-15: ARIMA by statsmodels — rejected, CSS path deleted
`arima_fit` stays quantcore's exact Gaussian maximum-likelihood fit (zero-mean ARMA on the
differenced series, parametrised through partial autocorrelations so every estimate is
stationary and invertible). The conditional-sum-of-squares fit (`method="css"`) had no caller
and is deleted; `arima_residuals` keeps the conditional-residual kernel.

statsmodels was compared on 250-point log-price windows (AR(1), MA(1), ARMA(1,1), AR(2), 12
seeds each, d = 1; 48 fits), against the tolerances 1e-6 relative on the log-likelihood and 1e-4
on coefficients and forecasts. To score both on one function, statsmodels was run as
`SARIMAX(order=(p,1,q), trend="n", simple_differencing=True)` with stationarity and invertibility
enforced, L-BFGS `maxiter=1000`, `pgtol=1e-10`, `factr=10`:
- The statsmodels log-likelihood evaluated at quantcore's estimates is at least as high as at
  statsmodels' own in all 48 fits (higher by more than 1e-9 relative in 47; largest gap
  4.3e-4 relative). quantcore's own likelihood agrees with statsmodels' at identical parameters
  to 8.4e-9, so both are the same function.
- No quantcore estimate is non-stationary or non-invertible (0 of 48); neither is statsmodels'.
- statsmodels met its convergence flag in 43 of 48 fits.
- Coefficient differences: maximum 0.10, median 1.9e-3; 6 of 48 within 1e-4. Forecast
  differences: maximum 1.6e-3 in log price; 34 of 48 within 1e-4. Log-likelihood within 1e-6
  relative in 24 of 48.
With the statsmodels defaults (`ARIMA`, d = 1 in the state space) the non-convergence rate over
800 fits per length was 16.1% at 60 points and 11.5% at 120 points, against 0% for quantcore.

### QC-16: A typed exception for a CIR fit with no mean reversion — done (`CIRNoMeanReversionError`)
Priority: should; not on the v0.3.0 path unless it folds into the trim at no cost. Raised by
agentic-trading (AT-119), 2026-11-05.

`cir_fit` (`core/stochastic.py`, `_cir_fit`) raises plain `RuntimeError` in three cases: kappa
<= 0 (the series shows no mean reversion), theta <= 0 and sigma^2 <= 0 (inadmissible fits).
agentic-trading's CIR analyst treats the first as the model declining to bet, because the data
has no mean reversion, and the other two as a failed fit. With one exception type it has to
match the message text "kappa <= 0". A test runs the real `cir_fit` on a rising series to pin
that wording.

Acceptance: `cir_fit` raises a named subclass of `RuntimeError` for kappa <= 0, for example
`CIRNoMeanReversionError` (the name is quantcore's choice), exported next to `cir_fit`. theta
<= 0 and sigma^2 <= 0 keep raising `RuntimeError`, or a separate subclass. Because it subclasses
`RuntimeError`, existing callers that catch `RuntimeError` are unaffected. A test covers each of
the three cases. agentic switches from the message match to the type at the AT-101 re-pin.

### QC-17: A constant column in CSCV and effective N is one with zero range — done
Raised by edgelab (EL-70), 2026-11-05. A constant column whose value is inexact in floating point
(0.002 repeated) has a computed standard deviation near 1e-19, not 0. Under the zero-deviation
test the default Sharpe ratio scored it about 4.5e15, so it won every in-sample half whenever its
mean was positive, and `effective_n_eigen` counted it as one uncorrelated trial. `cscv_pbo`'s
default `evaluate` and `effective_n_eigen` now treat any column with `np.ptp == 0` as constant:
Sharpe ratio 0, correlations 0 with the diagonal included. Results for columns with a nonzero range are
unchanged. This is a pin move against edgelab's copies on inputs with such a column; on the
seeded 41 x 4 panel with one column at 0.001 and `n_splits=4`, the PBO moves from 0.0 to 1/3,
the value an exactly representable constant gives. The conformance oracles use the same rule.

## Cleanup (audit 2026-11-10)

An audit against ten engineering rules: determinism tested, state outside checkpoints, contracts
vendored and tested on both sides, fail closed, and every check shown to fail on a planted
defect. Batch 1 (QC-23, 24, 27, 28, 29, 30, 33, 35) moves no number. Batch 2 changes the API or
behaviour and needs the owner's word.

### QC-23: No test that two processes give bit-identical results — done (batch 1)
Every reproducibility test repeats a call inside one process (`test_sde_solver.py`,
`test_stochastic.py`, `test_price_paths.py`, `test_bootstrap.py`). Nothing shows that a result is
the same under another BLAS thread count, or with a cold against a warm Numba cache.
Acceptance: a battery that calls each public function on fixed inputs runs in two subprocesses,
one with the default threads and an empty Numba cache, one with every BLAS thread count at 1 and
the cache warm, and the digests of all outputs agree. Negative control: a planted function that
draws from an unseeded generator makes the test fail.
Done: `tests/determinism_battery.py` calls all 87 public functions; the test requires every
digest to agree and the planted call to differ (about 20 s, 0.34 GB per process).
On Linux aarch64 (OpenBLAS) the first run failed: one BLAS thread rounds differently from
several (2 and 8 agree), which moved `newey_west_cov` by up to 3.6e-13 and the three `arch` fits
by up to 1.4e-6 relative. Ruling (b), 2026-11-10: `newey_west_cov` computes its products with
`np.einsum`, which does not call BLAS; the guarantee is bit identity for a fixed BLAS thread
configuration, with `fit_garch_11`, `egarch_fit` and `gjr_garch_fit` named as thread-sensitive.
The test runs three processes (cold and warm cache at the default threads must agree bit for
bit; one thread may move only the three fits, within 1e-3 relative, SLSQP's stopping
tolerance), and the README states the guarantee.

### QC-24: The bit-exact price-path pins run only on macOS arm64 — done for macOS (batch 1)
`tests/unit/core/test_price_paths.py` skips the whole-array digests on every other platform, so
CI (ubuntu-latest) never checks the bit-stability contract that edgelab's simulations rely on.
Acceptance: digests recorded per platform and checked wherever one is recorded; a platform with
none is reported, not passed silently.
Done: digests are keyed by platform and an unrecorded platform fails the digest test, naming
the digest to record. Only macOS arm64 is recorded; the Linux arm64 and x86_64 digests are
recorded from the first Linux runs, before the next push.

### QC-25: A quantcore number's provenance is only the commit — open (batch 2, consumers)
A number also depends on the resolved NumPy, SciPy, Numba, arch and statsmodels and on the
platform (QC-18). edgelab and agentic record only the quantcore commit. Their fix; quantcore
can supply the version list.

### QC-26: The path simulators draw fresh entropy when no seed is given — done (batch 2)
`simulate_gbm_paths`, `simulate_ou_paths` and `simulate_cir_paths` default to `seed=None`, so a
forgotten seed gives a non-reproducible result without a word. Both consumers pass seeds today.
Changing it is an API change.
Done (owner yes, 2026-11-10): `seed` is a required non-negative integer.

### QC-27: Nothing enforces that quantcore keeps no module-level mutable state — done (batch 1)
No module holds a cache, registry or global today. The only mutable module-level object is
`_KPSS_DETERMINISTIC_TERMS`, a dict that is only read. Acceptance: a census test fails on any
module-level mutable object not on an allowlist with a reason; negative control: a planted
module-level dict fails it.
Done: `tests/unit/test_module_state.py` also fails on a `global` or `nonlocal` statement; the
KPSS map is a read-only mapping.

### QC-28: Only `ConvergenceError` is tested to round-trip through pickle — done (batch 1)
Consumers checkpoint quantcore objects. Acceptance: every public result class and exception
round-trips with all fields equal; negative control: an exception with required keyword
arguments and no `__reduce__` fails.
Done: `tests/unit/test_public_types_pickle.py` walks the exported classes, so a new one is
covered without being listed.

### QC-29: The public API has no machine-readable manifest — done in quantcore (batch 1)
No quantcore test fails when a name or signature a consumer imports changes, and the consumer
notice is prose. Proposed: a generated manifest (names, signatures, result fields, exceptions,
version) checked by a test, vendored by each consumer.
Done: `src/quantcore/api_manifest.json`, built from the source with `ast` and shipped in the
package, with its SHA-256; `docs/API_MANIFEST.md` gives the consumer check. Both consumers'
current imports resolve against the 0.3.1 manifest. Left: edgelab and agentic vendor it.

### QC-30: `kpss_test` silences every warning — done (batch 1)
`_kpss_test` wraps statsmodels' `kpss` in `warnings.simplefilter("ignore")`, which hides any
warning, not only the out-of-table `InterpolationWarning` it reports as `p_value_at_bound`.
Acceptance: only `InterpolationWarning` is filtered; a warning of another class reaches the
caller. Done; the blanket filter fails the new test.

### QC-31: `__version__` falls back to "0.0.0+unknown" — open (later)
An import from an uninstalled tree reports a made-up version. No consumer reads it.

### QC-32: Undefined quantities give NaN in some functions and raise in others — done (batch 2)
Several ratios return NaN, and a zero risk denominator gives inf, while `cscv_pbo` and
`ou_half_life` raise `ValueError`. A NaN passes into a downstream result without a word. One
policy is the owner's decision.
Done (owner ruling (a), 2026-11-10; agentic-trading and edgelab agreed): an undefined or
infinite result raises `UndefinedResultError(ValueError)`; NaN only at documented positions of
multi-value results; bad input stays a plain `ValueError`. Zero spread is tested as zero range.
A sweep of every multi-value function on constant input found finite wrong answers in
`ljung_box_test`, `hac_t_test`, `engle_granger_test`, `spa_test` (p = 0 for a model that is
always worse), `ols`, `factor_loadings` and `fama_macbeth_regression`; all fixed. Black-Scholes
prices at zero volatility return their limit. `docs/UNDEFINED_RESULTS.md` lists every case;
`tests/unit/test_undefined_results.py` discovers the public functions and runs a degenerate
battery on the single-number functions and a constant battery on the multi-value ones, with
planted negative controls. `cir_fit`'s `CIRNoMeanReversionError` is a `RuntimeError` and stays
as it is.

### QC-33: The oracle tests skip when a reference library is missing — done (batch 1)
Twenty oracle modules and the conformance builder tests start with `pytest.importorskip`, so
without the `oracle` extra the oracle coverage turns into skips and the suite still passes.
Acceptance: a missing reference library fails the run; negative control: hiding one library
makes the suite fail.
Done: `tests/conftest.py` fails the run on any skip for a library that cannot be imported, and
lists those skips; with filterpy hidden the Kalman oracle run exits 1.

### QC-34: CI resolves the latest uv and dependencies, not the consumers' locked versions — open (batch 2)
A green run certifies whatever resolved that day. Proposed: pin uv, and a job that runs the
suite on the dependency versions in each consumer's lock.

### QC-35: Stale descriptions — done (batch 1)
`tests/oracle/README.md` still describes a `backend` argument. The package docstring and the
`pyproject.toml` description name only pricing, SDE simulation and risk.

### QC-36: v0.3.2 waits for GitHub Actions — open
The release PR cannot run CI while the owner's GitHub billing blocks Actions. The local gates
are green on the same tree.

### QC-37: `select_arima_order` does not say how many candidates it skipped — open (later)
Candidates that do not converge are skipped, as documented, but the caller cannot tell how many.
