# Oracle tests

Differential tests that compare quantcore's public functions against an
independent reference implementation: a reference library, a published
worked example or a closed form. Their purpose is to show that quantcore
agrees with the industry-standard result, not merely that it is
self-consistent.

These tests import reference libraries (`statsmodels`, `arch`,
`scikit-learn`, `linearmodels`, `empyrical`, `quantstats`, `QuantLib`,
`vollib`/`py_vollib`, `hmmlearn`, `filterpy`, `cvxpy`, `hypothesis`) from the
`oracle` extra. They are the only place in this repo those libraries are
imported — quantcore's runtime dependencies stay NumPy/SciPy/Numba (see
`CLAUDE.md`).

## Running

```bash
uv sync --extra dev --extra oracle
uv run pytest -m oracle
```

Oracle tests are marked `@pytest.mark.oracle` and run in the same `pytest`
invocation as the unit suite by default (`testpaths = ["tests"]`), but can be
selected or excluded with `-m oracle` / `-m "not oracle"`. CI runs both the
full suite (unit + oracle) on every push — see `.github/workflows/ci.yml`.

## Directory layout

Mirrors `src/quantcore/` 1:1, same as `tests/unit/`.

## Tolerance policy

Every oracle test's tolerance must follow the kind of calculation it checks,
and any tolerance looser than the defaults below carries a one-line comment
explaining why.

- **Closed-form vs. closed-form** (e.g. Black-Scholes price vs. QuantLib's
  analytic engine, a table lookup vs. the same published table): `1e-10`
  relative. Two exact formulas evaluated in double precision should agree to
  numerical noise.
- **Optimizer-based** (MLE fits — GARCH/EGARCH, ARIMA, HMM Baum-Welch —
  convex/QP solves — mean-variance, risk parity): `1e-6` on the objective
  (log-likelihood, sum of squares, QP objective) at the optimum. Parameter
  values themselves get a looser, explicitly justified bound, since a flat
  likelihood surface can leave the objective converged while individual
  parameters still differ more than that — e.g. GARCH alpha/beta on a short
  or low-persistence series.
- **Monte Carlo**: within `k` standard errors of the closed-form or
  reference value at a fixed seed, `k` stated per test (`k=3` unless
  otherwise justified — roughly a 1-in-370 false-fail rate per test, tight
  enough to catch a real bug, loose enough not to flake).
- **Table lookups / response-surface approximations** (MacKinnon p-values,
  Johansen critical values): the statistic to `1e-6` (both sides compute it
  the same closed-form way); the p-value or critical value to the tolerance
  the source paper itself claims for its approximation — `1e-3` unless a
  test states otherwise.
- **Monte Carlo size tests** (does a test reject at its nominal rate under a
  null that's true by construction — e.g. ADF/Engle-Granger on independent
  random walks): rejection rate within a binomial confidence interval around
  the nominal level at the given number of trials, not a point estimate.

## Contract tests

Every public function additionally gets, where applicable: NaN/inf input,
zero variance, single observation, shape mismatch, `float32` input,
non-contiguous arrays, and pandas `Series`/`DataFrame` input. The required
outcome is a clear exception or a documented value — never a silent NaN or a
plausible-looking wrong number. These live alongside the oracle comparison
in the same test file, under a `Test<Function>Contract` class, so a reader
sees the correctness check and the robustness check together.

## Relation to `conformance/`

This directory holds fixed-case oracle tests per module, mirroring
`src/quantcore/`: each compares a function with a reference on hand-chosen
inputs and published values. `conformance/` is the other half of the
verification: generated-input differential tests of registered targets
against independent oracles, whose recorded comparisons are classified into
the committed `conformance/conformance_map.json` and `docs/CONFORMANCE.md`
(see `conformance/README.md`).

## Functions verified only here

These public functions have no conformance target, so their oracle tests in
this directory are their only comparison with an independent reference:

- `statistics`: `johansen_trace_test`, `johansen_max_eigenvalue_test`
  (`statistics/test_johansen_oracle.py`).
- `performance`: `annualized_return`, `cagr` (`performance/test_metrics_oracle.py`;
  `cagr` is also the numerator of the `calmar_ratio` target),
  `drawdown_series` (`performance/test_drawdown_oracle.py`), `component_var`
  (`performance/test_component_var_oracle.py`) and
  `brinson_fachler_attribution` (`performance/test_brinson_fachler_oracle.py`).
- `core`: `simulate_gbm_paths`, `simulate_ou_paths`, `simulate_cir_paths`,
  `simulate_heston_paths` and `euler_maruyama`
  (`core/test_sde_moments_oracle.py`), and `simulate_gbm_price_paths`,
  `simulate_heston_price_paths`, `simulate_jump_diffusion_price_paths`,
  `simulate_regime_switching_price_paths`, `simulate_exp_ou_price_paths` and
  `simulate_garch_t_price_paths` (`core/test_price_paths_moments_oracle.py`),
  each compared with the analytic moments of its process.
