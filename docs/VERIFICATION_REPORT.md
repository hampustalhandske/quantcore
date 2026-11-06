# Verification report

Owner-facing summary of this pass of `QUANTCORE_FIX_PROMPT.md`. Full detail
in `docs/VERIFICATION_PLAN.md` (plan), `CHANGELOG.md` (numbers that
changed), `docs/CONFORMANCE.md` (status per function).

## What was wrong

Both Workstream B findings were independently reproduced against
`statsmodels` before being fixed, per the brief's instruction not to take
the finding list on faith:

- **B1**: `engle_granger_test` reused the plain Dickey-Fuller (N=1)
  p-value distribution for a residual-based test whose residuals actually
  come from an N=2 cointegrating regression. Confirmed: at a statistic of
  −3.0, the old code reported p≈0.035 where the correct value (verified
  against `statsmodels.tsa.adfvalues.mackinnonp`) is p≈0.110. This
  overstates significance — the expensive direction for pairs-trading
  signals, since it reports cointegration that isn't there.
- **B2**: `johansen_trace_test`'s critical-value table was for the wrong
  model (no deterministic term, where the code fits an unrestricted
  constant) and additionally shifted by one index within that wrong table.
  Confirmed: the code required a trace statistic above ~12.32 to reject at
  95% where the correct threshold (verified against
  `statsmodels.tsa.vector_ar.vecm.coint_johansen`) is 3.84. This
  understates significance — real cointegration relationships were missed.

Both defects were caught by the review found in commit `450a5a2` reading
the source and checking published tables; this pass reproduced each one
with a failing oracle test before fixing it, and added a Monte Carlo size
test (does the test reject at its nominal rate under a null that's true by
construction) so the class of bug — right code shape, wrong constant table
— can't recur silently.

## What changed

See `CHANGELOG.md` for the full before/after tables. Summary:

- `engle_granger_test`'s p-value changed (numbers above). `adf_test`'s
  signature and its own p-value (N=1 case) are unchanged.
- `johansen_trace_test`'s critical values changed (numbers in
  `CHANGELOG.md`). Its return shape is unchanged. New optional
  `deterministic` and `confidence` keyword arguments were added
  (additive; defaults reproduce the previously-fitted model at 95%).
- New public function `johansen_max_eigenvalue_test` (additive).
- New `oracle` optional dependency extra and `tests/oracle/` — quantcore's
  runtime dependencies are still NumPy/SciPy/Numba only.
- No default changed. No breaking changes were made in this pass — see
  "Needs approval" in `docs/VERIFICATION_PLAN.md` for the one item
  (`calmar_ratio`'s definition) that will need your sign-off before it can
  move.

## What downstream projects must re-run

- **edgelab** (and any other consumer): any stored or cached result of
  `engle_granger_test` or `johansen_trace_test` should be re-run. Pairs
  or baskets that were flagged cointegrated by `engle_granger_test` under
  the old p-values should be re-screened — some will no longer clear a
  5% threshold. Pairs/baskets that `johansen_trace_test` rejected under
  the old (too-high) critical values should also be re-screened — some
  will now show genuine cointegration that was previously missed.
- No other function's numbers changed in this pass. `adf_test` on a raw
  series is unaffected.
- Bump the pinned quantcore version to `0.1.1` (or later) to pick this up;
  see `CHANGELOG.md` for the exact diff if you're diffing against cached
  output rather than just re-running.

## Pass 2 (owner follow-up)

Approved C1 (breaking) and directed Workstream C to continue, prioritized
by what edgelab calls. Everything below has its own oracle test; see
`docs/CONFORMANCE.md` for the per-function status and `CHANGELOG.md` for
exact before/after numbers.

**C1 — `calmar_ratio` default is now CAGR (breaking, approved).** Matches
`empyrical`/`quantstats`. `return_method="arithmetic"` keeps the old
formula. Version bumped to `0.2.0`.

**2a — statistics.** `ols`: uncentered R² without an intercept column,
`NaN` (not `1.0`) for a constant `y`. `factor_loadings`: now returns a
`FactorLoadingsResult` (standard errors, p-values, R², residuals, resolved
lag count) instead of a 2-tuple — **breaking**; `n_lags_nw=None` (new
default) auto-selects the lag count instead of a fixed 4.
`newey_west_optimal_lags`: documentation-only fix (it's the closed-form
rule of thumb, not the paper's separate data-driven procedure — the
*computation* was already correct, only the citation overclaimed it).

**2b — pricing.** `dividend_yield` (q) added to Black-Scholes(-Merton)
pricing, all five Greeks, `implied_volatility`, and `heston_cos_call`;
new `heston_cos_put`. All additive (q=0.0 default reproduces every prior
output exactly). Verified against QuantLib and py_vollib. **Found and
fixed a real defect along the way**: `implied_volatility` could silently
return a search-boundary value instead of raising, for deep ITM/OTM
options where vega is negligible — now raises `ValueError` in that case.

**2c — regime.** `hmm_predict_proba` was already correctly documented as
computing smoothed (non-causal) posteriors, but its `predict_proba`-style
name invites exactly the lookahead-bias mistake its behavior warns
against; strengthened its docstring rather than renaming it (a rename
would break the API contract). New function `hmm_filtered_proba` for
code that needs a genuinely causal signal.

**2d — risk.** Checked two claims from your message against the code and
found **neither is a real defect**: `component_var` already sums to
`value_at_risk` exactly (the differing `norm.ppf` sign conventions cancel
against the formulas' differing structure — pinned down with a test so a
future reader doesn't need to redo the algebra); the GARCH-family
non-convergence asymmetry (`fit_garch_11` raises, `egarch_fit`/
`gjr_garch_fit` return `converged=False`) is real but already
intentionally documented on the `egarch.py` side — cross-referenced both
directions. Unifying it either way is a breaking change and needs your
decision on which behavior should win (see `docs/VERIFICATION_PLAN.md`,
"Needs approval").

**2e — portfolio.** `min_variance_weights`, `mean_variance_weights`,
`l1_turnover_penalized_weights`, `risk_parity_weights` never checked
`scipy.optimize.minimize`'s `result.success` — a silent-fallback bug per
`CLAUDE.md`'s own rule. All four now raise `RuntimeError` on SLSQP
non-convergence. `kelly_fraction` gained `clip: bool = True` (the [-1, 1]
clamp isn't part of Kelly's 1956 formula).

**2f — core.** `simulate_ou_paths`/`simulate_heston_paths`/
`simulate_cir_paths`'s default `seed` changed from `0` to `None`, matching
`euler_maruyama`/`simulate_gbm_paths`. Only affects calls not passing
`seed` explicitly (those are now non-reproducible by default, like the
other two already were).

**2g.** `quantcore.__version__`, read from installed package metadata so
it can't drift from `pyproject.toml`.

`git add`/`git commit` were denied by this session's permission layer in
pass 2, and are still denied in pass 3 — none of this is committed yet;
you'll need to commit it (or grant git write permissions and ask me to).

## Pass 3 (owner decisions)

**Decision 1 — Johansen.** No code change: `johansen_max_eigenvalue_test`
stays a separate function from `johansen_trace_test`, as already
implemented. If unified later, it'll be via a `JohansenResult` dataclass.

**Decision 2 — GARCH-family non-convergence unified on raising
(breaking).** New `quantcore.risk.volatility.ConvergenceError` (a
`RuntimeError` subclass carrying the best attempt's params/log-likelihood/
optimizer message). `fit_garch_11`, `egarch_fit`, `gjr_garch_fit` all raise
it now; `EGARCHParams`/`GJRGARCHParams`'s `converged` field is removed
(any code checking it needs a `try`/`except` instead — see `CHANGELOG.md`).

**Decision 3 — `ols`'s `NaN`-on-constant-`y`.** Kept, now stated explicitly
in the docstring's Returns section as a documented value; pinned by tests
for both the centered and uncentered cases.

**Decision 4 — Workstream D, in priority order (what edgelab calls):**
- GARCH-family formulas (`garch_11_variance`, `gjr_garch_11_variance`,
  `egarch_11_variance`) vs `arch`: verified, including the EGARCH
  asymmetry-term parameterization specifically.
- `ledoit_wolf_shrinkage` vs `sklearn.covariance.LedoitWolf`: **found and
  fixed a real defect** — the shrinkage computation mixed the unbiased
  (1/(T-1)) sample covariance into a Theorem 1 derivation that requires
  the biased (1/T) one throughout, making the result a different,
  non-optimal estimator (not just a scale difference) — differed from
  `sklearn` by up to ~1% of typical variance magnitude before the fix.
  Now matches to 1e-8 relative.
- `kalman_filter`/`kalman_smooth`/`dynamic_hedge_ratio` vs `filterpy`:
  verified, exact match (~1e-10).
- `hmm_fit`/`hmm_decode` vs `hmmlearn`: verified — exact match at fixed
  parameters (Viterbi path, forward-algorithm log-likelihood), and a
  closely-matching MLE when both are fit end to end on the same data.
- `simulate_gbm_paths`/`simulate_ou_paths`/`simulate_cir_paths`/
  `simulate_heston_paths` vs their published closed-form transition
  moments: verified, including CIR's Feller-condition edge cases. One
  methodological finding worth knowing about: the CIR Milstein scheme's
  discretization bias at a coarse timestep can exceed the Monte Carlo
  standard error for fast-mean-reverting parameters — not a simulator
  defect, but a reminder that oracle tests using "k standard errors" need
  a small enough timestep that discretization bias is actually negligible,
  which isn't automatic.

`uv run pytest` (834 tests, unit + oracle), `ruff check`/
`ruff format --check`, and `mypy src/` are all clean against the current
working tree as of the end of pass 3.

### What downstream projects must re-run (pass 3 additions)

On top of pass 1's `engle_granger_test`/`johansen_trace_test` re-screening:

- Any cached `ledoit_wolf_shrinkage` output should be re-run — the fixed
  values differ from the old (buggy) ones by up to ~1% of typical
  variance magnitude, which can matter for a portfolio optimizer
  downstream of it.
- Any code catching `EGARCHParams.converged`/`GJRGARCHParams.converged`
  must switch to `try`/`except ConvergenceError` (or `RuntimeError`) —
  that field no longer exists.
- Nothing else in pass 3 changed a returned number for an existing call;
  the rest was verification (confirming correctness) plus additive
  features.

## Pass 4 & 5 (autonomous — "implement it all, I will go to sleep")

You approved continuing through the rest of the Workstream D backlog
without further check-ins. Everything below has its own oracle test; see
`docs/CONFORMANCE.md` for the per-function status and `CHANGELOG.md` for
exact before/after numbers.

**Verified, no code change**: EWMA (`ewma_variance`/`ewma_covariance`) vs
RiskMetrics/pandas; the portfolio optimizers' own formulas
(`mean_variance_weights`, `min_variance_weights`, `risk_parity_weights`,
`l1_turnover_penalized_weights`) vs `cvxpy`; `black_litterman`/
`implied_equilibrium_returns` vs the He & Litterman (1999) worked example;
`impulse_response`/`granger_causality_test` vs `statsmodels`; `arima_fit`
(CSS) confirmed close to `statsmodels`' full-MLE `ARIMA` (expected — two
different, both-valid estimators, not meant to match exactly);
`heston_cos_call`/`heston_cos_put` vs QuantLib's `AnalyticHestonEngine`
(the base COS pricing formula itself, already covered by an earlier
dividend-yield test); `brinson_fachler_attribution` and `component_var`'s
own formula against independent hand derivations (no specific Bacon table
could be reliably sourced for Brinson-Fachler, so a from-scratch worked
example was used instead, documented as such); the eight hand-derived
execution/options-position helpers (`square_root_market_impact`,
`margin_financing_cost`, `short_borrow_cost`, `futures_roll_yield`,
`option_expiry_payoff`, `option_position_pnl`, `option_roll_schedule`,
`protective_put_sizing`) confirmed already adequately pinned by existing
unit tests; `cir_fit` confirmed already adequately verified via its own
exact-parameter-recovery test; `euler_maruyama` (the generic SDE solver)
newly checked directly against OU's closed-form moments.

**Found and fixed two more real defects:**

- **`var_fit`'s `sigma_u`** used `T_eff - 1` (the correction for a
  1-parameter mean-only estimate) instead of `T_eff - (k*n_lags + 1)`
  (Lütkepohl 2005's unbiased estimator for a VAR(p) equation, which
  estimates `k*n_lags + 1` parameters). Understated `sigma_u` by roughly
  0.7% in a representative case; propagates into `impulse_response`'s
  orthogonalized responses and any forecast-error-variance calculation
  downstream. Fixed; now matches `statsmodels.tsa.api.VAR` to 1e-8
  relative.
- **`select_var_lag_order`** compared AIC/BIC across candidate lag orders
  fit on *different* effective sample sizes (`n_obs - n_lags`, varying per
  candidate) — invalid for a fair information-criterion comparison per
  Lütkepohl (2005, pp. 146-150); `statsmodels.tsa.api.VAR.select_order`
  holds the sample size fixed at `n_obs - max_lags` across all candidates.
  Confirmed this changes the actual selected lag order in small samples
  (a 40-observation synthetic series: old code selected lag 1, the
  correct/statsmodels answer is 2). Fixed to match `statsmodels` exactly
  (selected lags and the underlying numeric AIC/BIC scores both match to
  1e-8 relative, restricted to the function's documented `n_lags >= 1`
  search range).

`uv run pytest` (959 tests, unit + oracle), `ruff check`/
`ruff format --check`, and `mypy src/` are all clean against the current
working tree.

### What downstream projects must re-run (pass 4/5 additions)

On top of the pass 1 and pass 3 re-screening above:

- Any cached `var_fit` (or downstream `impulse_response`/forecast-error-
  variance) output should be re-run — `sigma_u` changed by roughly 0.7%
  of its typical magnitude in a representative case; coefficients
  themselves are unaffected (OLS was already correct).
- Any cached `select_var_lag_order` output for a small sample relative to
  `max_lags` should be re-run — the selected lag order can differ from
  before the fix. Large-sample results are largely unaffected (the two
  sample-size conventions converge as `n_obs` grows relative to
  `max_lags`).
- Nothing else in pass 4/5 changed a returned number for an existing
  call; the rest was verification (confirming correctness) plus one new
  oracle test for an already-existing code path (`euler_maruyama`).

`git add`/`git commit` remain denied by this session's permission layer —
none of pass 3, 4, or 5's work is committed yet; you'll need to commit it
yourself (or grant git write permissions and ask me to).

## What's still open

`docs/VERIFICATION_PLAN.md` has the full remaining backlog. As of the end
of pass 5, the only known open items are:

- **`select_arima_order`** has the same sample-size-comparability defect
  class as `select_var_lag_order` (its information criterion is computed
  on `n_used = series.size - d - max(p, q)`, which varies per candidate)
  — **found but deliberately not fixed**, for two reasons: there is no
  external oracle to validate a fix against (unlike `select_var_lag_order`,
  which could be checked bit-for-bit against `statsmodels`), and comparing
  AIC/BIC across different differencing orders `d` is itself an unresolved
  design question in the ARIMA literature, not just an implementation
  bug. Needs your decision on how to proceed before anyone touches it.
- `var_forecast`, `select_hmm_n_states`, and a handful of lower-priority
  functions listed in `docs/VERIFICATION_PLAN.md` are not yet
  independently oracle-tested (not on the Workstream D priority list you
  gave).
- Workstream E, blocked on edgelab's `UPSTREAM_CANDIDATES.md`, as before.

Nothing else open is a known defect — everything else in the original
Workstream C/D backlog you prioritized has been verified or fixed.
