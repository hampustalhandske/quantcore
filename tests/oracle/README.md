# Oracle tests

Differential tests that compare quantcore's public functions against an
independent reference implementation, per `QUANTCORE_FIX_PROMPT.md`'s
mandate: quantcore is trusted for a calculation only once it is proven to
agree with the industry-standard result, not merely self-consistent.

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

## Status

This directory currently covers the Workstream B confirmed defects
(`statistics/cointegration.py`: B1 MacKinnon N-parameterized p-values). The
full coverage plan — which function, which oracle, and its classification —
is tracked in `docs/VERIFICATION_PLAN.md`; results feed `docs/CONFORMANCE.md`.
