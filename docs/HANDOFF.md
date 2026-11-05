# quantcore handoff

What someone picking up quantcore needs and won't find elsewhere in the repo. The work items and
their status are in `docs/ISSUES.md`. What changes for consumers is in `CHANGELOG.md` and
`docs/REPIN_0.3.0.md`.

## Where things stand

- Work is on branch `trim`. Nothing is pushed: `origin` has only `main` and `dev`. `dev` is
  contained in `trim`.
- The re-pin target for agentic-trading is 358a6cc (`docs/REPIN_0.3.0.md`). Later commits add
  the functions moved in from edgelab (QC-13). edgelab re-pins once, at the v0.3.0 tag.
- `conformance/conformance_map.json` was last regenerated at 358a6cc. `build_report --check`
  fails on later commits until the one regeneration after QC-13 ("Release gates" in
  `docs/ISSUES.md`).
- Consumers install quantcore from GitHub at a commit SHA, so a consumer can re-pin only to a
  pushed commit. Pushing, merging and tagging v0.3.0 are the owner's.

## How work is run here

- The project's orchestrator session takes plans and heavy-job slots from the master session
  (`/Users/hampusstalhandske/dev/personal/master`, reachable by SendMessage; find it with
  ListAgents). Commit on `trim`, never push, never commit on `main`.
- One heavy job at a time on this 16 GB machine. Single test files are light. The whole test
  suite, a conformance run, `build_report` and benchmarks need the master's slot.
- Run outputs and work in progress go in the repo's gitignored folders (`temp/`,
  `conformance/results/`), never in a session scratchpad under `/private/tmp`.

## Known behaviour worth knowing

- `ols` treats a regression as having an intercept only when a column is constant across rows.
  An implicit constant, such as a full set of dummies, gets the uncentred R-squared
  (`statistics/regression.py`, docstring).
- `calmar_ratio` goes through `_cagr`, which raises `OverflowError` when
  `ending_value ** (1 / n_years)` overflows. It loses relative accuracy when the total return is
  about 1e-9. Against empyrical it differs by up to 1.5e-8 relative on such inputs, which is why
  its conformance tolerance is condition-aware (`conformance/tolerances.py`).
- The performance ratios return `inf` for a zero risk denominator, where quantstats returns
  `nan`. Every input is a return series. quantstats reads some non-negative series as prices.
- The bit-exact pins of the price-path simulators were recorded on macOS arm64 and are skipped on
  other platforms (`tests/unit/core/test_price_paths.py`). CI runs on ubuntu-latest, so it runs
  the skip, never the pins.
- `uv.lock` is gitignored, so CI resolves the dependency ranges fresh on every run (QC-12).

## Measured speed (pre-trim tree, re-run before quoting)

These figures are from `benchmarks/bench_quantcore.py` on an Apple M3, measured on revision
1c65d9f with uncommitted changes. The benchmark functions still exist.

| function | against | speed-up |
|---|---|---|
| `arima_fit` | statsmodels `ARIMA` | 6.8x to 36x (ARMA(1,1), 1,000 points: 4.6 against 31 ms) |
| `arima_forecast` | statsmodels | 5x to 9x |
| `ols` | statsmodels | about 2x; with `newey_west_cov` 1.4x to 1.7x |
| `ledoit_wolf_shrinkage` | scikit-learn | 4.5x to 14x |
| the nine performance ratios | quantstats | 7.7x to 35x |
| `dynamic_hedge_ratio` | a filterpy loop | about 26x |

The Gaussian VaR and CVaR rows predate the native historical CVaR (604e48e) and are not quoted.
The ADF and Engle-Granger rows are void, because both tests are now statsmodels itself.

## Waiting on the owner

- Pushing `trim` now as a backup, and tagging v0.3.0 at the step-13 tip.
- Deleting the local `temp/` (old notes; everything still true from them is in this file) and
  `dist/` (0.1.0 wheels).
- The CLAUDE.md edits below. CLAUDE.md is the owner's to change, so they are proposed here and
  applied on the owner's yes.

### Proposed CLAUDE.md edits

1. Stack, the runtime-dependency bullet. Replace it with:

   > - Runtime dependencies: NumPy, SciPy, Numba, `arch` (the GARCH-family fits, the stationary
   >   bootstrap and SPA) and `statsmodels` (ADF, Engle-Granger, KPSS, ARCH-LM, Ljung-Box and the
   >   VAR functions). Each library is imported inside the function that uses it, on first use.

2. Conventions, the `backend` bullet. Replace it with:

   > - A function that delegates to a library calls it directly. There is no `backend` argument,
   >   and the function passes every option explicitly, so a change in a library default cannot
   >   move a number. The library is imported inside the function, on first use. No library
   >   outside the runtime dependencies is imported from `src/quantcore`. The `oracle` extra is
   >   for tests, conformance and benchmarks only.

3. Conventions, after the conformance bullet. Add:

   > - A delegated function is checked by convention tests (options passed, units,
   >   annualisation, sign), not by a conformance target that compares the library with itself.

4. Opening paragraph. "derivative pricing, SDE simulation, and risk metrics" becomes
   "derivative pricing, SDE simulation, risk, statistics, portfolio construction and
   backtest-overfitting statistics".

5. Structure. Add the packages the list leaves out: `statistics/` (tests, regression, HAC, the
   stationary bootstrap and multiple-testing statistics), `performance/` (ratios and Sharpe
   inference), `portfolio/`, `time_series/`, `regime/`, `execution/`, `filtering/`.
