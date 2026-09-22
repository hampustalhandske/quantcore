1. Critical: var.py can't accept an external volatility — this is the "fixed sigma" issue

Look at value_at_risk and conditional_value_at_risk: both take only returns and confidence_level, and internally compute:

python
mu = float(np.mean(returns))
sigma = float(np.std(returns, ddof=1))

There's no way to inject a GARCH-forecasted sigma into these functions — they always derive sigma from the raw sample themselves. This isn't a bug exactly (it's a valid, correctly-implemented VaR calculation), but it's an architectural gap that directly blocks the project we scoped. Right now, "static VaR" is the only thing this module can produce — there's no seam to swap in GARCH's time-varying estimate instead.

Fix needed: separate "estimate the distribution parameters" from "compute VaR given parameters." Something like:

python
def value_at_risk(
    returns: npt.NDArray[np.float64],
    confidence_level: float,
    volatility: float | None = None,
) -> float:
    ...
    sigma = volatility if volatility is not None else float(np.std(returns, ddof=1))

Same for mu/mean, and mirrored in conditional_value_at_risk. This is a small, surgical change, but without it, you literally cannot do the static-vs-GARCH comparison as designed.

2. Critical: volatility.py has no way to actually fit GARCH to real data

garch_11_variance requires the caller to already know omega, alpha, beta — there's no function that estimates these parameters from a returns series. In practice, you never know these ahead of time; they come from maximum-likelihood estimation on historical data. Right now, if you fed this module real market returns, you'd have nothing to pass in for the three parameters except a guess.

What's missing: a fit_garch_11(returns) -> (omega, alpha, beta) function, using MLE (maximize the GARCH log-likelihood via scipy.optimize.minimize, typically with a constrained optimizer since you need omega > 0, alpha, beta >= 0, alpha + beta < 1). Without this, volatility.py can compute a variance series given parameters, but can't actually calibrate itself to real data — which makes it unusable for a real backtest as it stands, not just limited.

This is the bigger of the two gaps. Fixing #1 alone still leaves you with no real GARCH forecast to feed in.

3. Moderate: GARCH recursion has no Numba version (performance, not correctness)

_garch_11_variance is a plain Python for loop. For a single call this is fine, but a rolling-window backtest means re-running (and, once #2 is added, re-fitting) this potentially thousands of times across your history. Worth adding the Numba-accelerated version once #1 and #2 are settled — you already removed the broken stub, so this is a clean rebuild, not a patch.

4. Minor: the variance seed choice, worth knowing about rather than "fixing"

variance[0] = omega / (1 - alpha - beta) seeds the recursion with the model's theoretical long-run variance rather than anything derived from the actual window's data. This is a legitimate, defensible choice (it's what the stationary GARCH process converges to), but for short rolling windows in a backtest, it means the first several days of each window are influenced by a value that has nothing to do with that window's actual recent volatility — a warm-up distortion. Not wrong, just something to be aware of when you pick your rolling window length; a common alternative is seeding from the sample variance of a short burn-in period instead.

5. Worth double-checking, not confirmed broken: return convention consistency

sde_solver.py/black_scholes.py/monte_carlo.py all work in log-price/GBM terms (d log(S_t) = ...). var.py/volatility.py just take a generic returns array with no stated convention (simple returns vs. log returns). If your backtest eventually feeds the same return series into both the pricing side and the risk side, mixing simple and log returns between them would introduce a subtle, hard-to-notice inconsistency. Worth explicitly deciding (log returns is the standard, more consistent choice) and documenting it in both modules rather than leaving it implicit.

Bottom line: nothing here is "useless" as in wrong — the math that exists is correct and well-cited. But #1 and #2 together mean the module can't currently do the one thing your risk-trading project needs it to do (produce and compare a real GARCH-based VaR against a static one on real data). Want me to have your test-writer/implementer pair fix these two first, before touching risk-trading at all?