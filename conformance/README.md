# Conformance suite

Differential tests of quantcore against independent oracles. Every function the suite covers is
registered once as a `ConformanceTarget` (a *target*): a quantcore adapter, an oracle that
computes the same standard definition by other means (a reference library, a closed form or a
hand recursion from the primary source), a comparator and a written tolerance. Hypothesis draws
inputs, both sides are run on each input, and the comparisons are recorded. A separate builder
turns the recorded comparisons into one status per target.

Two generated files are committed:

- `conformance/conformance_map.json`: the record of every target's status, oracle, tolerance and
  finding. CI fails when a fresh run disagrees with it.
- `docs/CONFORMANCE.md`: the same record as a report for people.

Everything else the suite produces (`conformance/results/`, `conformance/findings/`) is
git-ignored.

## Running

```bash
uv sync --extra dev --extra oracle                 # the oracle libraries and hypothesis
uv run pytest conformance -q                       # run every comparison (ci profile, about 90 s)
uv run python -m conformance.build_report          # rebuild the map, the report and the findings
uv run python -m conformance.build_report --check  # write nothing; fail if the committed map is stale
```

A passing pytest run does not mean quantcore agrees with its oracles. `harness.check()` records
a mismatch, a quantcore exception or a non-finite quantcore result and carries on, so that one
wrong function cannot stop the rest from being measured; the verdict is the status the builder
writes. A test does fail when the *oracle* raises or returns a non-finite value, because that
invalidates the comparison itself.

The test run writes `raw.json` (one row per case and quantity), `details.json` (seeds and
values, for findings), `edge_cases.json` (what quantcore does on inputs outside its contract)
and `meta.json` (the profile) to `conformance/results/`. Set `QUANTCORE_CONFORMANCE_RESULTS_DIR`
to write them, and have the builder read them, somewhere else; this lets several runs proceed at
once. The builder needs the results of a whole run: after `uv run pytest
conformance/test_tier1_risk.py` alone, every other target has no recorded case and the build
fails its coverage gate.

The machinery itself (comparators, harness, classifier, builder) is unit-tested under
`tests/unit/conformance/` and runs with the ordinary `uv run pytest`.

## Profiles

`QUANTCORE_CONFORMANCE_PROFILE` selects a hypothesis profile (`conformance/conftest.py`); an
unknown value is an error at import time. The builder reads the profile a run was recorded under
from `meta.json` and refuses to build if the variable says otherwise when it runs.

| profile | examples per test | derandomized | builds the map | purpose |
|---|---|---|---|---|
| `ci` (default) | 200 | yes | yes | every push; fast and reproducible |
| `full` | 5000 | yes | yes | a thorough run before a release |
| `explore` | 5000 | no | no | the weekly job; fresh inputs the fixed profiles never draw |

A test that sets its own `@settings(max_examples=...)` (the slower optimiser, pricing and
filtering tests do) keeps that number under every profile.

Under a derandomized profile hypothesis seeds each test from the source text and signature of
the test function, so editing a test (decorators, signature, docstring or body) changes which
cases it draws and records. The derandomized profiles also ignore literals harvested from local
source: hypothesis normally collects the numeric and string literals of every imported module
outside site-packages and substitutes one of them for about one draw in twenty, which would make
the drawn cases depend on which modules happen to be imported and on every literal anywhere in
`src/quantcore` and `conformance/`. `conftest.py` switches that collection off for `ci` and
`full`, so the cases they draw depend only on each test function's own source text and the
hypothesis version. The switch replaces a function private to hypothesis, which is why the
`oracle` extra pins hypothesis exactly; if the installed hypothesis does not have that function,
the run stops at import with a message instead of running unreproducibly. `explore` keeps the
collection, because finding inputs the fixed profiles never draw is its job.

Under `explore` the builder writes nothing. It loads the committed map and prints every mismatch
recorded for a target that map calls `verified`, with the quantity, family, bucket and seed, and
exits 1 if there is one:

```bash
QUANTCORE_CONFORMANCE_PROFILE=explore uv run pytest conformance -q
QUANTCORE_CONFORMANCE_PROFILE=explore uv run python -m conformance.build_report
```

`numpy.random.default_rng(seed)` regenerates a reported input the way `generators.py` does for
that family. If a test itself fails under `explore`, hypothesis prints a blob; replay it with
`@reproduce_failure(version, blob)` above the failing test's `@given`.
`.github/workflows/conformance-explore.yml` runs this weekly and opens an issue on a mismatch.

## Statuses

| status | meaning |
|---|---|
| `verified` | Within the written tolerance of the oracle on every recorded case, and quantcore never raised on a generated input. |
| `non-standard` | Sound, but not the standard result: a documented alternative definition, or (iterative estimators) an attained objective that is systematically but only slightly worse than the oracle's. |
| `defect` | Outside tolerance, or raising, with no such explanation. |
| `unverifiable` | Nothing judged it: the target has no oracle (`reference=None`), or its comparator only records agreement (`ReportOnlyScalarComparator`, used where several answers are equally right, such as an information-criterion-optimal order). |

`classify.py` turns recorded cases into a status. It has no I/O and is unit-tested in
`tests/unit/conformance/test_classify.py`.

**Iterative estimators.** A fitter or optimiser without a unique numerical optimum
(`ITERATIVE_ESTIMATORS` in `build_report.py`: the GARCH-family fits, `arima_fit` and the
mean-variance optimisers) is not compared on its parameters. Both solutions are scored by one
shared objective function, and a one-sided sign test over the well-specified cases decides the
status (`classify_iterative_estimator`): `verified` when quantcore's attained objective is not
worse significantly more often than it is better and its median gap is not below the tolerance;
otherwise `non-standard` while the median shortfall stays small, and `defect` when it does not
or when quantcore fails to converge on more than 5% of the cases. Each estimator declares
whether its recorded objective is maximised (a log-likelihood, a mean-variance utility) or
minimised (a portfolio variance); a win is always the better objective.

**Condition-aware tolerances.** A `ConditionAwareTolerance` widens with a per-case condition
number. A case whose effective tolerance would exceed the cap in `tolerances.py` is recorded as
ill-conditioned and left out of the decision; it is still counted and reported, and the build
fails if fewer than 90% of a target's cases are well-conditioned.

**Second oracles.** A target may be cross-checked against a second library
(`harness.check_second_oracle`, for example `quantstats` for `sharpe_ratio`). The second oracle
is compared with the target's own oracle, never with quantcore, and never affects a status; the
report lists each cross-check and, where they disagree, an example and the cause.

## What the builder writes

- `conformance/conformance_map.json`. Top level: `schema_version`, `quantcore_version`,
  `oracle_versions` (every distribution of the `oracle` extra, plus numpy and scipy),
  `hypothesis_profile` and `targets`. Per target: `status`, `oracle`, `cases`, `max_abs_error`,
  `max_rel_error`, `by_quantity` (cases and worst errors per named quantity, such as `statistic`
  and `p_value`), `tolerance` (`rtol`, `atol`, `justification`; `null` for a target with no
  oracle), `reason` (empty for `verified`) and `finding`. Error figures are rounded to three
  significant digits.
- `docs/CONFORMANCE.md`: one row per target and quantity with oracle, cases, worst errors,
  tolerance and status, then the reasons for every status other than `verified`, coverage
  counts, the iterative-estimator sign tests, ill-conditioned counts and the second-oracle
  cross-checks.
- `conformance/findings/NNN-<target>.md`, one per `defect` or `non-standard` target.

## Findings

`findings_src/<target>.md` holds the hand-written part of a finding, with exactly these `## `
sections: `Title`, `Severity`, `Location`, `Root cause`, `Standard definition and citation`,
`Proposed patch`, `Promotion test` (parsed by `findings.py`). The builder merges it with the
run's evidence (case counts, worst errors, failing cases per family and bucket, the sign test
for an iterative estimator) into `conformance/findings/NNN-<target>.md`, and fails if a `defect`
or `non-standard` target has no source file. The title also becomes the target's `reason` in
the map.

Repros are executed, not trusted. For a target with an entry in `REPRO_CONFIGS` the builder
writes a standalone script for the shortest failing input, runs it in a subprocess, and puts the
values that run printed into the finding; if the script fails, or does not show a mismatch,
the build stops instead of writing a stale finding. A repro imports only numpy, quantcore and
the oracle library. A short input is inlined; a long one is regenerated from its recorded seed
by a recipe in `REGEN_RECIPES` that repeats the generator's arithmetic.

## `--check`

Rebuilds the map in memory and compares only the decisions with the committed map: the set of
targets and, per target, `status`, `tolerance` and `finding` (`build_report.decision_diff`).
Error magnitudes, case counts and library versions are ignored, since they can differ in the
last digits between platforms. It writes nothing and exits non-zero when

- any decision differs (each difference is printed as `<target>.<field>: <old> -> <new>`),
- the registry is misconfigured or a declared family or bucket recorded no case (see Coverage),
- fewer than 90% of a target's cases are well-conditioned, or
- a `defect` or `non-standard` target has no `findings_src/<target>.md`.

A quantcore defect alone does not fail the check once it is recorded: the map says `defect`,
and the check fails only if a later run stops agreeing with that.

## Coverage

Each tier test module declares, per target, `<NAME>_FAMILIES` (the generator families it draws
from), `<NAME>_BUCKETS` (the buckets its inputs fall into) and, where the input has a length,
`<NAME>_MIN_LEN` (the shortest input it draws). Series lengths fall into `short` (up to 10),
`medium` (11 to 500) and `long` (above 500); the shared length strategy in `generators.py`
picks a bucket first and a length within it second, so short inputs are not crowded out by the
much wider long range. Targets with scalar arguments use the single bucket `scalar`; the
portfolio targets bucket by number of assets (`dim_small`, `dim_medium`, `dim_large`).

`COVERAGE_SOURCES` in `build_report.py` maps each target to the module and constant prefix that
declare its coverage. The build fails, naming every problem at once, when

- a registered target has no quantcore callable,
- a registered target with an oracle has no line in `COVERAGE_SOURCES` or no entry in
  `TOLERANCES`,
- a declared length bucket lies entirely below the declared minimum length, or
- a declared family or bucket has no recorded case in the run.

`build_report.configuration_errors` and `build_report.missing_coverage` hold the complete rules.

## Adding a quantity

1. In `targets/<domain>.py`, write the quantcore adapter, the oracle and the `ConformanceTarget`
   that pairs them. The oracle must not share code with quantcore.
2. Add its tolerance to `tolerances.py` with a written justification, derived from the numerical
   analysis of the computation before looking at whether the cases pass (closed-form default:
   `1e-10` relative).
3. Add a generator family to `generators.py` if none covers the input shape.
4. In a `test_tier1_*.py` or `test_tier2_*.py` module, declare `<NAME>_FAMILIES`,
   `<NAME>_BUCKETS` and, for a series input, `<NAME>_MIN_LEN`, and add a `@given` test that calls
   `harness.check(...)` with the case's family, bucket and seed.
5. Add one line to `COVERAGE_SOURCES` in `build_report.py`. An estimator without a unique
   optimum also needs an entry in `ITERATIVE_ESTIMATORS`, stating the recorded objective and
   whether it is maximised or minimised.
6. Run the suite and the builder, and commit `conformance_map.json` and `docs/CONFORMANCE.md`
   with the code.

While a target is half-added (an oracle registered before quantcore implements the function,
or a test not yet written), `--allow-incomplete` lets the builder run: it lists every incomplete
target, leaves those out of the map and builds from the rest, and the report says which were
left out. CI runs without the flag, so such a map does not pass `--check` there.

## When a status changes

`--check` fails and prints the change. What to do depends on its direction.

- **A target stops being `verified`.** Reproduce it from the family, bucket and seed of a failing
  case in `results/raw.json` and `results/details.json`. If quantcore is wrong, fix it; the
  status returns to `verified` and nothing else changes. If the difference stays for now, write
  `findings_src/<target>.md`, rebuild, and commit the map and the report. Do not loosen the
  tolerance to make it pass: a tolerance changes only when its numerical justification does.
- **A target becomes `verified`.** Delete its `findings_src/<target>.md`, rebuild and commit.
- **A tolerance or its justification changes.** Rebuild and commit; `--check` treats the
  tolerance as part of the decision.
- **Only a `finding` path changes.** Findings are numbered in name order, so adding or removing
  one renumbers those after it. Rebuild and commit.
- **The weekly explore job reports a mismatch.** The committed map is unchanged, but a
  `verified` target disagreed with its oracle on a fresh input. Replay the reported seed and
  treat it as the first case above.
