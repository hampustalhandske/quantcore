# quantcore

Pip-installable numerical backend for derivative pricing, SDE simulation, and risk
metrics. Other projects (fraud detection, backtesting, agentic trading) depend on
this package for all numerically heavy calculations — correctness and performance
here matter more than in a typical app, since downstream projects trust these
numbers without re-deriving them.

## Stack

- Python 3.12, managed with `uv` (not pip/poetry directly)
- NumPy + SciPy for reference implementations
- ruff (lint + format), mypy (strict), pytest

## Commands

- `uv sync --extra dev` — install all dependencies
- `uv run pytest` — run tests with coverage
- `uv run ruff check .` — lint
- `uv run ruff format .` — format
- `uv run mypy src/` — type check

## Structure

- `src/quantcore/pricing/` — option pricing: closed-form (Black-Scholes) and
  Monte Carlo. Every Monte Carlo method must have a closed-form or reference
  test to validate against where one exists.
- `src/quantcore/risk/` — VaR/CVaR, volatility models (GARCH)
- `src/quantcore/core/` — Numba-optimized numerical kernels (SDE solvers,
  random path generation). Relaxed mypy rules here (see pyproject.toml)
  because Numba decorators aren't fully typeable yet.
- `tests/` — mirrors src/ structure 1:1

## Conventions

- **Every numerically heavy function needs two versions**: a plain NumPy
  reference implementation and a Numba-accelerated one, with a test asserting
  they agree within a numerical tolerance (`np.allclose`). Never ship an
  optimized version without the reference to test it against.
- Public functions must be fully typed (mypy strict mode is enforced outside
  `core/`).
- No silent fallbacks: if a Numba-compiled path fails, raise — don't
  transparently fall back to slow Python without telling the caller.
- This package will be `pip install`-ed by other projects (agentic-trading,
  fraud-detection). Treat the public API in each module's `__init__.py` as a
  contract — don't rename or change signatures without checking downstream use.

## REFERENCES
`REFERENCES.md` contains the source of truth for the mathematical model to use
for calculations. Both the source and its corresponding file. 

## Out of scope for this repo

- No trading strategy logic, no agent code, no live data fetching. Those live
  in downstream projects that depend on quantcore. This repo is pure math/compute.
