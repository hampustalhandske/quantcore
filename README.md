# quantcore

Pip-installable numerical backend for derivative pricing, stochastic
differential equation (SDE) simulation, and risk/portfolio metrics.

Other projects (fraud detection, backtesting, agentic trading) depend on
this package for all numerically heavy calculations — correctness and
performance here matter more than in a typical app, since downstream
projects trust these numbers without re-deriving them.

This repo is pure math/compute: **no trading strategy logic, no agent
code, no live data fetching**. That belongs in downstream projects.

## What it computes

Every formula/algorithm is traceable to a published source, listed in
[`docs/REFERENCES.md`](docs/REFERENCES.md) (the master bibliography —
each module cites its entry in its own docstring). At a high level:

| Domain | Examples |
|---|---|
| Option pricing | Black-Scholes closed-form, Monte Carlo (antithetic/control variates), Heston stochastic-vol via COS method, analytical Greeks, Brent's-method implied vol |
| SDE simulation | Euler-Maruyama solver, GBM, Ornstein-Uhlenbeck / Vasicek, CIR, Heston paths; generator-driven price paths (GBM, exp-OU, Heston, GARCH-t, regime-switching, jump diffusion) |
| Risk | Value at Risk, Conditional VaR (Expected Shortfall), GARCH(1,1) (+ MLE fit), EGARCH(1,1), GJR-GARCH, EWMA volatility |
| Portfolio | Mean-variance / min-variance / risk-parity weights, Kelly fraction, Black-Litterman, Ledoit-Wolf shrinkage, EWMA covariance |
| Time series | ARIMA(p,d,q) by exact Gaussian maximum likelihood (conditional sum of squares optional) with exact conditional-mean forecasts and order selection, VAR(p) + impulse response + Granger causality, Ljung-Box test |
| Statistics | OLS, Fama-MacBeth, Newey-West HAC errors, factor loadings, ADF unit-root and KPSS stationarity tests, Jarque-Bera normality and ARCH-LM tests, Engle-Granger & Johansen cointegration, OU half-life/z-score |
| Regime detection | Gaussian HMM (Baum-Welch fit, Viterbi decode, forward-filter predict) |
| Filtering | Kalman filter/smoother (RTS), dynamic hedge ratio estimation |
| Performance | Sharpe, Sortino, Calmar, information ratio, max drawdown, hit rate, profit factor, component VaR, Ulcer Index / Ulcer Performance Index, Serenity index, gain-to-pain, common sense ratio, CPC index, outlier win/loss ratios, risk of ruin |

## Structure

```
src/quantcore/
├── core/          # Numba-optimized kernels: SDE solvers, path simulation; generator-driven price paths
├── pricing/       # Black-Scholes, Monte Carlo, Heston, Greeks, implied vol
├── risk/          # VaR/CVaR, GARCH/EGARCH/GJR-GARCH volatility models
├── portfolio/      # Mean-variance/risk-parity optimization, Black-Litterman, covariance estimation
├── time_series/   # ARIMA, VAR, causality/unit-root/autocorrelation tests
├── statistics/    # OLS, factor regressions, cointegration, unit-root and diagnostic tests
├── regime/        # Gaussian HMM regime detection
├── filtering/     # Kalman filter/smoother
└── performance/   # Risk-adjusted return metrics, drawdown/payoff/tail ratios, attribution
tests/             # mirrors src/ 1:1 (unit/ + integration/ + oracle/)
conformance/       # differential tests against independent oracles on generated inputs, and the builder of docs/CONFORMANCE.md
benchmarks/        # wall-clock benchmarks: Numba kernels and comparisons with external libraries
docs/REFERENCES.md # source-of-truth bibliography for every calculation
```

Each module's `__init__.py` is the public API contract for that
subpackage — see "Using it as a library" below.

## Verification status & changelog

Public functions are checked against an independent reference
implementation or a published closed-form/worked example, not only by
internal unit tests. Two layers do this:

- `tests/oracle/` holds fixed-case oracle tests: hand-picked inputs
  compared with a reference library (`statsmodels`, `arch`, `QuantLib`,
  ...) or a published worked example.
- `conformance/` holds the conformance suite: each covered quantity is
  registered once with an independent oracle and a written tolerance,
  hypothesis draws inputs, and the builder turns the recorded comparisons
  into one status per quantity. `conformance/conformance_map.json` is the
  committed record; CI fails when a fresh run's statuses differ from it, and
  a weekly run on fresh random inputs opens an issue when a `verified`
  function disagrees with its oracle. How to run and extend it:
  [`conformance/README.md`](conformance/README.md).

Documents:

- [`docs/CONFORMANCE.md`](docs/CONFORMANCE.md) — generated report of the
  conformance suite (do not edit by hand): every covered function's status
  (`verified` / `non-standard` / `defect` / `unverifiable`) against its
  oracle, with case counts, worst errors, tolerances and the oracle library
  versions.
- [`docs/VERIFICATION_REPORT.md`](docs/VERIFICATION_REPORT.md) —
  owner-facing summary of what's been found and fixed, pass by pass.
- [`CHANGELOG.md`](CHANGELOG.md) — every numerical result that changed,
  with a before/after example; also documents additive vs. breaking
  changes.

Before relying on a specific function's numbers, check its status in
`docs/CONFORMANCE.md`.

### Numba vs. plain NumPy

`core/` holds Numba-JIT-compiled kernels for tight numerical loops
(variance recursions, path simulation). Everywhere else uses plain
NumPy/SciPy — closed-form expressions and functions passed to
`scipy.optimize` (which can't call Numba JIT functions directly). There
is one implementation per function; no parallel "slow" reference copy is
kept once a Numba path exists. If a compiled path fails, it raises rather
than silently falling back to a slower implementation.

## Requirements

Python 3.12+. Dependencies: NumPy, SciPy, Numba.

## Using it as a library (downstream projects)

Install directly from GitHub:

```bash
pip install "git+https://github.com/hampustalhandske/quantcore.git"
# or
uv pip install "git+https://github.com/hampustalhandske/quantcore.git"
```

Then import from the subpackage you need — each subpackage's `__init__.py`
re-exports its public functions:

```python
from quantcore.pricing import black_scholes_call, monte_carlo_call_price
from quantcore.risk import value_at_risk, conditional_value_at_risk, garch_11_variance
from quantcore.core import simulate_gbm_paths, simulate_gbm_price_paths
from quantcore.portfolio import mean_variance_weights, black_litterman

price = black_scholes_call(spot=100, strike=105, rate=0.03, vol=0.2, maturity=1.0)
```

Treat these `__init__.py` exports as a stable contract: function names
and signatures shouldn't change without a corresponding version bump and
check of downstream usage.

## Developing locally

```bash
uv sync --extra dev        # install all dependencies (incl. dev tools)
uv run pytest               # run tests with coverage
uv run ruff check .         # lint
uv run ruff format .        # format
uv run mypy src/            # type check (strict outside core/)
```

`tests/unit/` and `tests/integration/` run by default. `tests/oracle/`
(fixed-case differential tests against reference libraries — `statsmodels`,
`arch`, `scikit-learn`, `QuantLib`, etc. — see "Verification status" above)
needs the `oracle` extra and is marked with `@pytest.mark.oracle`:

```bash
uv sync --extra dev --extra oracle
uv run pytest tests/oracle -m oracle
```

The conformance suite lives outside `tests/` and needs the same extra:

```bash
uv run pytest conformance -q                       # quantcore vs. independent oracles (ci profile, ~90 s)
uv run python -m conformance.build_report          # rebuild conformance_map.json, docs/CONFORMANCE.md and the findings
uv run python -m conformance.build_report --check  # fail if the committed map is stale (what CI runs)
```

A passing `pytest conformance` run is not a verdict: the status of each
function is what the builder writes. `conformance/conformance_map.json` and
`docs/CONFORMANCE.md` are generated and committed; never edit them by hand.
Profiles (`ci`, `full`, `explore`) and the rest are in
`conformance/README.md`. The Numba-versus-NumPy and external-library timings
are in `benchmarks/bench_quantcore.py` (`uv run python benchmarks/bench_quantcore.py`,
needs the `oracle` extra).

The oracle libraries are test-only and never imported from `src/quantcore` —
quantcore's runtime dependencies stay NumPy/SciPy/Numba.

### Conventions

- Use Numba for tight numerical loops (recursive variance recursions,
  path simulation, Viterbi decoding, Kalman predict/update cycles); plain
  NumPy/SciPy for closed-form formulas and `scipy.optimize` objectives.
- Every Monte Carlo method must have a closed-form or reference test to
  validate against, where one exists (see `tests/integration/`).
- Public functions are fully typed; mypy strict mode is enforced outside
  `core/` (Numba decorators aren't fully typeable yet, so that module has
  relaxed rules — see `pyproject.toml`).
- No silent fallbacks: a failing Numba-compiled path raises rather than
  transparently degrading to slow Python.
- New formulas/algorithms need a `References:` section in the module
  docstring citing the source paper/text, plus an entry in
  `docs/REFERENCES.md`.

## Status

- [x] Black-Scholes closed-form pricing + analytical Greeks + implied volatility
- [x] Monte Carlo option pricer (variance reduction)
- [x] Heston stochastic-volatility pricing (COS method)
- [x] Euler-Maruyama SDE solver; GBM, OU/Vasicek, CIR, Heston path simulation
- [x] Generator-driven price paths: GBM, exp-OU, Heston, GARCH-t, regime-switching, jump diffusion
- [x] VaR / CVaR, component VaR
- [x] GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1), EWMA volatility
- [x] Portfolio optimization (mean-variance, min-variance, risk parity, Kelly), Black-Litterman, covariance shrinkage
- [x] ARIMA (exact maximum likelihood), VAR models, Granger causality, Ljung-Box test
- [x] OLS, Fama-MacBeth, Newey-West, cointegration, ADF/KPSS/Jarque-Bera/ARCH-LM tests
- [x] Gaussian HMM regime detection
- [x] Kalman filter/smoother, dynamic hedge ratio
- [x] Performance metrics (Sharpe, Sortino, Calmar, drawdown, Ulcer/Serenity/gain-to-pain ratios, etc.)
- [x] Conformance suite against independent oracles
