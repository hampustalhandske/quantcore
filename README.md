# quantcore

Optimized numerical backend for derivative pricing, stochastic differential
equation simulation, and risk metrics. Designed to be pip-installed as a
dependency by downstream projects.

## Install from GitHub

Install the package directly from the GitHub repository using pip:

```bash
pip install "git+https://github.com/hampustalhandske/quantcore.git"
```

Or with uv:

```bash
uv pip install "git+https://github.com/hampustalhandske/quantcore.git"
```

## Local setup

For local development in this repository:

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
- [x] Monte Carlo option pricer (with variance reduction)
- [x] Euler-Maruyama SDE solver
- [x] VaR / CVaR
- [x] GARCH volatility model


## REFERENCES
`docs/REFERENCES.md` contains the source of truth for the mathematical model to use for calculations. Both the source and its corresponding file.
