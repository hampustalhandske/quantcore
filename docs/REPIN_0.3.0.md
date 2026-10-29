# Re-pinning from a041d0d to 0.3.0: which numbers move

For projects that pin quantcore at a041d0d. Everything not listed here agrees with a041d0d to
1e-12 or better on the seeded comparison inputs (ADF, KPSS, ARCH-LM, Ljung-Box and
Engle-Granger statistics and p-values, the chosen lag counts, the VAR functions).
Full detail is in `CHANGELOG.md` and `docs/ISSUES.md` (QC-9, QC-10).

## Numbers that change

**1. `conditional_value_at_risk(..., method="historical")` on tied or discrete returns.**
The tail is now the `int((n - 1) * (1 - alpha)) + 1` smallest returns (the rule
`empyrical.conditional_value_at_risk` uses, matched bit for bit). a041d0d averaged every return at
or below the interpolated VaR. The two differ only when returns tie at the tail edge: rounded or
discrete returns (betting, prices in ticks). On 80 seeded series rounded to three decimals, 117 of
240 cases at three confidence levels moved, by up to 9% relative; continuous returns
did not move. `backend=` is gone; the Gaussian and Cornish-Fisher methods are unchanged.

**2. `fit_garch_11`, `egarch_fit`, `gjr_garch_fit` parameters.**
All three are fitted by `arch`. On 40 simulated GARCH(1,1) series of 750 returns, omega moved
by up to 0.033, alpha by 0.010 and beta by 0.030. Scored under arch's own likelihood the new
parameters are at least as good in all 40, so the earlier optimiser stopped at a weaker point.
`egarch_fit` and `gjr_garch_fit` now raise `ConvergenceError` where the fit lands outside the
stationary region; on short windows this is more frequent than before (EGARCH 10% at n=60, 1% at
n=250, 0% at n=750; GJR 17%, 3%, 0%; a041d0d's fitters never raised on the same inputs).
`fit_garch_11` no longer has a `backend` argument.

**3. Variance paths: the backcast seed.**
`garch_11_variance`, `egarch_11_variance` and `gjr_garch_11_variance` start from `arch`'s
backcast of the residuals passed in (weights `0.94**j` on the first `min(75, n)` squares),
where a041d0d started at the unconditional variance. The default path now equals the fitted
path (arch's `conditional_volatility ** 2`, to 1e-10). The first values differ visibly (up to
68% on the seeded cases) and the last variance by up to 11% (the parameters and the seed both
changed). `presample_variance=` overrides the seed; pass `omega / (1 - alpha - beta)` for
GARCH to get the old start, `omega / (1 - alpha - gamma / 2 - beta)` for GJR, and
`exp(omega / (1 - beta))` for EGARCH.
`variance_backcast(returns)` returns the seed for a raw return series.

## Other behaviour changes

- `var_fit`, `select_var_lag_order` and the other VAR functions reject a single series
  (statsmodels needs at least two).
- `statsmodels>=0.15,<0.16` and `arch>=7.0` are runtime dependencies; `empyrical-reloaded` is not.

## What to state when re-pinning

For each pinned result that calls the functions above, say whether it moves and which of
items 1 to 3 explains it. A result that uses only the unchanged functions should reproduce to
1e-12.
