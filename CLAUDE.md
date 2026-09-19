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

- **Use Numba where the computation is a tight numerical loop** — recursive
  variance recursions, path simulation, Viterbi decoding, Kalman predict/update
  cycles. Use plain NumPy/SciPy for closed-form expressions (Greeks, yield curve
  formulas, OLS). Objective functions passed to `scipy.optimize` can and should
  call into Numba JIT kernels directly — scipy only requires the objective
  itself to be a plain callable, it doesn't care what it calls internally, and
  the Numba kernel is typically an order of magnitude faster than the NumPy
  loop it replaces. The one real constraint: Numba's njit-compiled scalar
  float division raises `ZeroDivisionError` on exact zero, where NumPy's array
  division silently returns `inf`. This only matters where an unguarded
  division sits inside the hot loop itself (e.g. dividing by a value that can
  underflow to exactly 0.0 at an interior step) — wrap the call in
  `try/except ZeroDivisionError` and return the objective's usual
  infeasible-point penalty in that case. One implementation per function — no
  parallel NumPy reference copy unless a specific need (like this
  ZeroDivisionError mismatch, or the NumPy/Numba agreement test itself)
  forces it.
- Public functions must be fully typed (mypy strict mode is enforced outside
  `core/`).
- No silent fallbacks: if a Numba-compiled path fails, raise — don't
  transparently fall back to slow Python without telling the caller.
- This package will be `pip install`-ed by other projects (agentic-trading,
  fraud-detection). Treat the public API in each module's `__init__.py` as a
  contract — don't rename or change signatures without checking downstream use.

## REFERENCES
`docs/REFERENCES.md` contains the source of truth for the mathematical model to use
for calculations. Both the source and its corresponding file. 

## Out of scope for this repo

- No trading strategy logic, no agent code, no live data fetching. Those live
  in downstream projects that depend on quantcore. This repo is pure math/compute.
