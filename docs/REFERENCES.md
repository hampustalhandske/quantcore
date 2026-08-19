# References

Every calculation implemented in quantcore is traceable to a specific
published source. This file is the master bibliography; individual modules
cite the relevant entry in their docstrings.

## Option pricing

- Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate
  Liabilities." *Journal of Political Economy*, 81(3), 637–654.
  → `pricing/black_scholes.py`

- Boyle, P. (1977). "Options: A Monte Carlo Approach." *Journal of
  Financial Economics*, 4(3), 323–338.
  → `pricing/monte_carlo.py` (base method)

- Boyle, P., Broadie, M., and Glasserman, P. (1997). "Monte Carlo Methods
  for Security Pricing." *Journal of Economic Dynamics and Control*,
  21(8), 1267–1321.
  → `pricing/monte_carlo.py` (variance reduction: antithetic variates,
  control variates)

## Stochastic differential equations

- Kloeden, P.E. and Platen, E. (1992). *Numerical Solution of Stochastic
  Differential Equations*. Springer.
  → `core/sde_solver.py` (Euler-Maruyama scheme and convergence properties)

## Risk metrics

- J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document* (4th ed.).
  → `risk/var.py` (Value at Risk methodology)

- Rockafellar, R.T. and Uryasev, S. (2000). "Optimization of Conditional
  Value-at-Risk." *Journal of Risk*, 3, 21–41.
  → `risk/var.py` (Conditional Value at Risk / Expected Shortfall)

## Volatility modeling

- Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with
  Estimates of the Variance of United Kingdom Inflation." *Econometrica*,
  50(4), 987–1007.
  → `risk/volatility.py` (foundational ARCH model)

- Bollerslev, T. (1986). "Generalized Autoregressive Conditional
  Heteroskedasticity." *Journal of Econometrics*, 31(3), 307–327.
  → `risk/volatility.py` (GARCH(1,1) implementation)

## Citation convention

Every implementation file must include a `References:` section in its
module-level docstring, citing the specific paper/text the formula or
algorithm is drawn from. See `pricing/black_scholes.py` for the pattern.
