# quantcore

Optimized numerical backend for derivative pricing, stochastic differential
equation simulation, and risk metrics. Designed to be pip-installed as a
dependency by downstream projects.

## Setup

```bash
uv sync --extra dev
```

## Development

```bash
uv run pytest            # run tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy src/         # type check
```

## Design principle

Every numerically heavy function ships with a plain NumPy reference
implementation and, where performance matters, a Numba-accelerated version
tested against the reference for numerical agreement. See
`src/quantcore/pricing/black_scholes.py` and its test for the pattern.

## Status

- [x] Black-Scholes closed-form pricing
- [ ] Monte Carlo option pricer (with variance reduction)
- [ ] Euler-Maruyama SDE solver
- [ ] VaR / CVaR
- [ ] GARCH volatility model


## REFERENCES
`docs/REFERENCES.md` contains the source of truth for the mathematical model to use for calculations. Both the source and its corresponding file.
