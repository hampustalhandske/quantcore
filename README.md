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
| SDE simulation | Euler-Maruyama solver, GBM, Ornstein-Uhlenbeck / Vasicek, CIR, Heston paths |
| Risk | Value at Risk, Conditional VaR (Expected Shortfall), GARCH(1,1) (+ MLE fit), EGARCH(1,1), GJR-GARCH, EWMA volatility |
| Portfolio | Mean-variance / min-variance / risk-parity weights, Kelly fraction, Black-Litterman, Ledoit-Wolf shrinkage, EWMA covariance |
| Time series | ARIMA(p,d,q), VAR(p) + impulse response + Granger causality, ADF unit-root test, Ljung-Box test |
| Statistics | OLS, Fama-MacBeth, Newey-West HAC errors, factor loadings, Engle-Granger & Johansen cointegration, OU half-life/z-score |
| Regime detection | Gaussian HMM (Baum-Welch fit, Viterbi decode, forward-filter predict) |
| Filtering | Kalman filter/smoother (RTS), dynamic hedge ratio estimation |
| Performance | Sharpe, Sortino, Calmar, information ratio, max drawdown, hit rate, profit factor, component VaR |

## Structure

```
src/quantcore/
├── core/          # Numba-optimized kernels: SDE solvers, path simulation
├── pricing/       # Black-Scholes, Monte Carlo, Heston, Greeks, implied vol
├── risk/          # VaR/CVaR, GARCH/EGARCH/GJR-GARCH volatility models
├── portfolio/      # Mean-variance/risk-parity optimization, Black-Litterman, covariance estimation
├── time_series/   # ARIMA, VAR, causality/unit-root/autocorrelation tests
├── statistics/    # OLS, factor regressions, cointegration tests
├── regime/        # Gaussian HMM regime detection
├── filtering/     # Kalman filter/smoother
└── performance/   # Risk-adjusted return metrics
tests/             # mirrors src/ 1:1 (unit/ + integration/)
docs/REFERENCES.md # source-of-truth bibliography for every calculation
```

Each module's `__init__.py` is the public API contract for that
subpackage — see "Using it as a library" below.

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
from quantcore.core import simulate_gbm_paths
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
- [x] VaR / CVaR, component VaR
- [x] GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1), EWMA volatility
- [x] Portfolio optimization (mean-variance, min-variance, risk parity, Kelly), Black-Litterman, covariance shrinkage
- [x] ARIMA, VAR models, Granger causality, ADF/Ljung-Box tests
- [x] OLS, Fama-MacBeth, Newey-West, cointegration tests
- [x] Gaussian HMM regime detection
- [x] Kalman filter/smoother, dynamic hedge ratio
- [x] Performance metrics (Sharpe, Sortino, Calmar, drawdown, etc.)
