# References

Every calculation implemented in quantcore is traceable to a specific
published source. This file is the master bibliography; individual modules
cite the relevant entry in their docstrings.

## Option pricing

- Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate
  Liabilities." *Journal of Political Economy*, 81(3), 637–654.
  → `pricing/black_scholes.py`, `core/price_paths.py` (geometric Brownian
  motion)

- Merton, R.C. (1973). "Theory of Rational Option Pricing." *Bell Journal
  of Economics and Management Science*, 4(1), 141–183.
  → `pricing/black_scholes.py`, `pricing/greeks.py`, `pricing/heston.py`
  (continuous dividend yield / cost-of-carry extension, `dividend_yield`)

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

- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
  Springer.
  → `core/price_paths.py` (ch. 3: exact simulation of geometric Brownian
  motion, Gaussian processes and jump diffusions on a fixed time grid)

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
  → `statistics/diagnostics.py` (ARCH-LM test, `arch_lm_test`)

- Bollerslev, T. (1986). "Generalized Autoregressive Conditional
  Heteroskedasticity." *Journal of Econometrics*, 31(3), 307–327.
  → `risk/volatility.py` (GARCH(1,1) implementation; also GARCH(1,1) MLE
  estimation, `fit_garch_11`)
  → `core/price_paths.py` (GARCH variance recursion,
  `simulate_garch_t_price_paths`)

- Bollerslev, T. (1987). "A Conditionally Heteroskedastic Time Series Model
  for Speculative Prices and Rates of Return." *Review of Economics and
  Statistics*, 69(3), 542–547.
  → `core/price_paths.py` (standardised Student-t innovations,
  `simulate_garch_t_price_paths`)

## Stochastic processes

- Uhlenbeck, G.E. and Ornstein, L.S. (1930). "On the Theory of the Brownian
  Motion." *Physical Review*, 36(5), 823–841.
  → `core/stochastic.py` (Ornstein-Uhlenbeck SDE)
  → `core/price_paths.py` (OU transition law, `simulate_exp_ou_price_paths`)

- Vasicek, O. (1977). "An Equilibrium Characterization of the Term Structure."
  *Journal of Financial Economics*, 5(2), 177–188.
  → `core/stochastic.py` (OU / Vasicek mean-reverting diffusion in finance)

- Cox, J.C., Ingersoll, J.E., and Ross, S.A. (1985). "A Theory of the Term
  Structure of Interest Rates." *Econometrica*, 53(2), 385–407.
  → `core/stochastic.py` (CIR square-root diffusion)

- Chan, K.C., Karolyi, G.A., Longstaff, F.A., and Sanders, A.B. (1992). "An
  Empirical Comparison of Alternative Models of the Short-Term Interest
  Rate." *Journal of Finance*, 47(3), 1209–1227.
  → `core/stochastic.py` (CKLS-style GLS moment estimator, `cir_fit`)

- Heston, S.L. (1993). "A Closed-Form Solution for Options with Stochastic
  Volatility with Applications to Bond and Currency Options." *Review of
  Financial Studies*, 6(2), 327–343.
  → `core/stochastic.py` (Heston stochastic volatility SDE)
  → `core/price_paths.py` (Heston model, `simulate_heston_price_paths`)
  → `pricing/heston.py` (Heston characteristic function for COS pricing)

- Lord, R., Koekkoek, R. and van Dijk, D. (2010). "A Comparison of Biased
  Simulation Schemes for Stochastic Volatility Models." *Quantitative
  Finance*, 10(2), 177–194.
  → `core/price_paths.py` (full-truncation Euler scheme,
  `simulate_heston_price_paths`)

- Schwartz, E.S. (1997). "The Stochastic Behavior of Commodity Prices:
  Implications for Valuation and Hedging." *Journal of Finance*, 52(3),
  923–973.
  → `core/price_paths.py` (OU process on the log price,
  `simulate_exp_ou_price_paths`)

- Merton, R.C. (1976). "Option Pricing When Underlying Stock Returns Are
  Discontinuous." *Journal of Financial Economics*, 3(1–2), 125–144.
  → `core/price_paths.py` (jump diffusion with compensated drift,
  `simulate_jump_diffusion_price_paths`)

- Fang, F. and Oosterlee, C.W. (2008). "A Novel Pricing Method for European
  Options Based on Fourier-Cosine Series Expansions." *SIAM Journal on
  Scientific Computing*, 31(2), 826–848.
  → `pricing/heston.py` (COS method for option pricing under Heston: the
  put is expanded in `y = ln(S_T/K)` on the range `x + c1 ∓ L*sqrt(c2 +
  sqrt(c4))`, eq. 49, and the call follows by put-call parity as the paper
  recommends in Section 3.3. The paper's Table 11 `c2` is not used as
  printed: its `xi^2*theta*(6e^{-kT} - 7)` term should read
  `theta*(4e^{-kT} - 5)` for the expression to equal the log-return's
  variance; `_heston_cumulants` carries the corrected closed form.)

## Option Greeks and implied volatility

- Black, F. and Scholes, M. (1973). Already cited under Option pricing.
  → `pricing/greeks.py` (analytical Greeks derived from BS formula)

- Brent, R.P. (1973). *Algorithms for Minimization Without Derivatives*.
  Prentice-Hall.
  → `pricing/greeks.py` (Brent's method for implied volatility inversion)

## Volatility modeling (asymmetric)

- Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New
  Approach." *Econometrica*, 59(2), 347–370.
  → `risk/egarch.py` (EGARCH(1,1) conditional variance)

- Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993). "On the Relation
  Between the Expected Value and the Volatility of the Nominal Excess Return on
  Stocks." *Journal of Finance*, 48(5), 1779–1801.
  → `risk/egarch.py` (GJR-GARCH(1,1) leverage effect)

## Kalman filtering

- Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction
  Problems." *Journal of Basic Engineering*, 82(1), 35–45.
  → `filtering/kalman.py` (linear Kalman filter predict/update equations,
  also used by `dynamic_hedge_ratio`)

- Rauch, H.E., Tung, F., and Striebel, C.T. (1965). "Maximum Likelihood
  Estimates of Linear Dynamic Systems." *AIAA Journal*, 3(8), 1445–1450.
  → `filtering/kalman.py` (RTS smoother backward pass)

- Chan, E.P. (2013). *Algorithmic Trading: Winning Strategies and Their
  Rationale*. Wiley. Ch. 3. Details not verified.
  → `filtering/kalman.py` (`dynamic_hedge_ratio`: a hedge ratio estimated as
  the state of a Kalman filter. The one-state model and its
  `N(initial_beta, 1)` prior are quantcore's own; no starting value is
  attributed to this source.)

- Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press.
  Ch. 13. Details not verified.
  → `filtering/kalman.py` (`dynamic_hedge_ratio`: regression with
  time-varying coefficients in state-space form)

## Regime detection

- Baum, L.E., Petrie, T., Soules, G., and Weiss, N. (1970). "A Maximization
  Technique Occurring in the Statistical Analysis of Probabilistic Functions of
  Markov Chains." *Annals of Mathematical Statistics*, 41(1), 164–171.
  → `regime/hmm.py` (Baum-Welch EM algorithm for Gaussian HMM)

- Viterbi, A. (1967). "Error Bounds for Convolutional Codes and an
  Asymptotically Optimum Decoding Algorithm." *IEEE Transactions on Information
  Theory*, 13(2), 260–269.
  → `regime/hmm.py` (Viterbi most-probable-state-sequence decoding)

- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2),
  357–384.
  → `regime/hmm.py` (Markov-switching model applied to financial returns;
  causal filtered-probability recursion, `hmm_filtered_proba`)
  → `core/price_paths.py` (Markov regime switching,
  `simulate_regime_switching_price_paths`)

## Cointegration and mean reversion

- Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and Error
  Correction: Representation, Estimation, and Testing." *Econometrica*, 55(2),
  251–276.
  → `statistics/cointegration.py` (Engle-Granger two-step cointegration
  test: step 1 OLS with intercept, step 2 ADF on the residuals with one
  lagged difference and no constant, p-value from MacKinnon's N=2 surface)

- Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors."
  *Journal of Economic Dynamics and Control*, 12(2–3), 231–254.
  → `statistics/cointegration.py` (Johansen trace and maximum-eigenvalue
  tests, eigenvectors)

- Avellaneda, M. and Lee, J.-H. (2010). "Statistical Arbitrage in the U.S.
  Equities Market." *Quantitative Finance*, 10(7), 761–782.
  → `statistics/cointegration.py` (OU half-life and z-score for spread trading)

- MacKinnon, J.G. (1994). "Approximate Asymptotic Distribution Functions for
  Unit-Root and Cointegration Tests." *Journal of Business & Economic
  Statistics*, 12(2), 167–176.
  → `statistics/cointegration.py` (ADF/Engle-Granger p-values:
  `Phi(polynomial(tau))` for the no-constant, constant and
  constant-plus-trend cases, parameterized by N, the number of I(1) series in
  the regression the tested residuals came from; exactly 0.0 or 1.0 outside
  the tabulated range of tau; `_mackinnon_pvalue`. The small-p coefficients
  were checked against the working-paper version of the article; the
  large-p coefficients and the no-constant tables equal those of
  `statsmodels.tsa.adfvalues` and were not checked against a primary source.)

- MacKinnon, J.G., Haug, A.A., and Michelis, L. (1999). "Numerical
  Distribution Functions of Likelihood Ratio Tests for Cointegration."
  *Journal of Applied Econometrics*, 14(5), 563–577.
  → `statistics/cointegration.py` (Johansen trace and maximum-eigenvalue
  test critical values, by deterministic-term case and confidence level;
  `_JOHANSEN_TRACE_CRIT`, `_JOHANSEN_MAX_EIG_CRIT`)

## Time series

- Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for
  Autoregressive Time Series with a Unit Root." *Journal of the American
  Statistical Association*, 74(366), 427–431.
  → `statistics/cointegration.py` (ADF unit-root test, `adf_test`)

- Ljung, G.M. and Box, G.E.P. (1978). "On a Measure of Lack of Fit in Time
  Series Models." *Biometrika*, 65(2), 297–303.
  → `time_series/arima.py` (Ljung-Box Q-test for autocorrelation in residuals)

- Box, G.E.P., Jenkins, G.M., Reinsel, G.C., and Ljung, G.M. (2015). *Time
  Series Analysis: Forecasting and Control* (5th ed.). Wiley.
  → `time_series/arima.py` (ARIMA(p,d,q) model; ch. 5, minimum mean square
  error forecast, `arima_forecast`; ch. 7, conditional sum-of-squares
  estimation, `arima_fit(method="css")` and `arima_residuals`)

- Gardner, G., Harvey, A.C., and Phillips, G.D.A. (1980). "Algorithm AS 154:
  An Algorithm for Exact Maximum Likelihood Estimation of
  Autoregressive-Moving Average Models by Means of Kalman Filtering."
  *Applied Statistics*, 29(3), 311–322.
  → `time_series/arima.py` (state-space form of an ARMA model, stationary
  initial state covariance, exact likelihood)

- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter*. Cambridge University Press.
  → `time_series/arima.py` (ch. 3: Kalman filter, prediction-error
  decomposition of the likelihood, concentrated likelihood, prediction from
  the filtered state)

- Jones, R.H. (1980). "Maximum Likelihood Fitting of ARMA Models to Time
  Series with Missing Observations." *Technometrics*, 22(3), 389–395.
  → `time_series/arima.py` (reparameterisation by partial autocorrelations
  that keeps the AR part stationary and the MA part invertible)

- Barndorff-Nielsen, O. and Schou, G. (1973). "On the Parametrization of
  Autoregressive Models by Partial Autocorrelations." *Journal of
  Multivariate Analysis*, 3(4), 408–419.
  → `time_series/arima.py` (one-to-one map between partial autocorrelations
  in (-1, 1) and the stationary region)

- Hannan, E.J. and Rissanen, J. (1982). "Recursive Estimation of Mixed
  Autoregressive-Moving Average Order." *Biometrika*, 69(1), 81–94.
  → `time_series/arima.py` (two-stage regression starting values for the
  exact-likelihood fit)

- Akaike, H. (1974). "A New Look at the Statistical Model Identification."
  *IEEE Transactions on Automatic Control*, 19(6), 716–723.
  → `time_series/arima.py` (AIC in `select_arima_order`)

- Schwarz, G. (1978). "Estimating the Dimension of a Model." *The Annals of
  Statistics*, 6(2), 461–464.
  → `time_series/arima.py` (BIC in `select_arima_order`)

- Sims, C.A. (1980). "Macroeconomics and Reality." *Econometrica*, 48(1), 1–48.
  → `time_series/var_model.py` (Vector Autoregression — VAR(p) model)

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
  Springer.
  → `time_series/var_model.py` (VAR OLS estimation, impulse response functions)

- Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models
  and Cross-Spectral Methods." *Econometrica*, 37(3), 424–438.
  → `time_series/var_model.py` (Granger causality F-test)

## Diagnostic tests

- Jarque, C.M. and Bera, A.K. (1980). "Efficient Tests for Normality,
  Homoscedasticity and Serial Independence of Regression Residuals."
  *Economics Letters*, 6(3), 255–259.
  → `statistics/diagnostics.py` (Jarque-Bera statistic, `jarque_bera_test`)

- Jarque, C.M. and Bera, A.K. (1987). "A Test for Normality of Observations
  and Regression Residuals." *International Statistical Review*, 55(2),
  163–172.
  → `statistics/diagnostics.py` (`jarque_bera_test`)

- Engle, R.F. (1982). Already cited under Volatility modeling.
  → `statistics/diagnostics.py` (ARCH-LM test, `arch_lm_test`)

- Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. and Shin, Y. (1992).
  "Testing the Null Hypothesis of Stationarity Against the Alternative of a
  Unit Root." *Journal of Econometrics*, 54(1–3), 159–178.
  → `statistics/diagnostics.py` (KPSS statistic, Bartlett-kernel long-run
  variance and the Table 1 critical values `_KPSS_CRITICAL_VALUES`,
  `kpss_test`; checked against the Cowles Foundation working-paper version)

- Hobijn, B., Franses, P.H. and Ooms, M. (2004). "Generalizations of the
  KPSS-test for Stationarity." *Statistica Neerlandica*, 58(4), 483–502.
  → `statistics/diagnostics.py` (automatic lag count of `kpss_test`: the
  Newey-West (1994) plug-in bandwidth for the Bartlett kernel, computed from
  the first `int(n^(2/9))` autocovariances as `statsmodels`'
  `kpss(nlags="auto")` does. The paper itself was not obtained, so that this
  is the paper's choice of autocovariance count is not confirmed.)

## Cross-sectional regression

- Fama, E.F. and French, K.R. (1993). "Common Risk Factors in the Returns on
  Stocks and Bonds." *Journal of Financial Economics*, 33(1), 3–56.
  → `statistics/regression.py` (3-factor OLS loading estimation)

- Fama, E.F. and French, K.R. (2015). "A Five-Factor Asset Pricing Model."
  *Journal of Financial Economics*, 116(1), 1–22.
  → `statistics/regression.py` (5-factor extension)

- Fama, E.F. and MacBeth, J.D. (1973). "Risk, Return, and Equilibrium:
  Empirical Tests." *Journal of Political Economy*, 81(3), 607–636.
  → `statistics/regression.py` (Fama-MacBeth two-pass cross-sectional regression)

- Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
  *Econometrica*, 55(3), 703–708.
  → `statistics/regression.py` (Newey-West HAC standard errors)
  → `statistics/diagnostics.py` (Bartlett-kernel long-run variance in
  `kpss_test`, through `_newey_west_cov`)

- Newey, W.K. and West, K.D. (1994). "Automatic Lag Selection in Covariance
  Matrix Estimation." *Review of Economic Studies*, 61(4), 631–653.
  → `statistics/regression.py` (closed-form, sample-size-only bandwidth
  rule of thumb, `newey_west_optimal_lags`; not this paper's separate
  fully data-driven AR(1)-approximation procedure)
  → `statistics/diagnostics.py` (constant 1.1447 and `n^(1/3)` rate of the
  automatic lag count in `kpss_test`)

## Portfolio optimization

- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*, 7(1),
  77–91.
  → `portfolio/optimization.py` (mean-variance efficient frontier)

- Kelly, J.L. (1956). "A New Interpretation of Information Rate." *Bell System
  Technical Journal*, 35(4), 917–926.
  → `portfolio/optimization.py` (Kelly fraction for position sizing;
  `kelly_fraction`'s default [-1, 1] clamp is a leverage cap, not part of
  Kelly's original formula — `clip=False` returns the unclamped value)

- Maillard, S., Roncalli, T., and Teïletche, J. (2010). "The Properties of
  Equally Weighted Risk Contribution Portfolios." *Journal of Portfolio
  Management*, 36(4), 60–70.
  → `portfolio/optimization.py` (risk parity / equal risk contribution:
  the target condition `w_i*(Sigma*w)_i / (w'Sigma*w) = 1/k`)

- Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights."
  SSRN Working Paper 2297383.
  → `portfolio/optimization.py` (`risk_parity_weights` solves Spinu's
  strictly convex reformulation `min_{w>0} 0.5*w'Sigma*w - sum(b_i ln w_i)`
  by damped Newton iteration; its first-order condition is the
  equal-risk-contribution condition and its minimiser is unique)

- Lobo, M.S., Fazel, M., and Boyd, S. (2007). "Portfolio Optimization with
  Linear and Fixed Transaction Costs." *Annals of Operations Research*,
  152(1), 341–365.
  → `portfolio/optimization.py` (L1 turnover-penalized mean-variance,
  `l1_turnover_penalized_weights`)

- Nocedal, J. and Wright, S.J. (2006). *Numerical Optimization* (2nd ed.).
  Springer.
  → `portfolio/optimization.py` (ch. 16, active-set method for convex
  quadratic programming; `_solve_convex_qp`, used by `min_variance_weights`,
  `mean_variance_weights` and `l1_turnover_penalized_weights`)

- Bland, R.G. (1977). "New Finite Pivoting Rules for the Simplex Method."
  *Mathematics of Operations Research*, 2(2), 103–107.
  → `portfolio/optimization.py` (`_active_set_iterations`: lowest-index
  leaving/entering rule after a zero-length step, to prevent cycling)

- Black, F. and Litterman, R. (1992). "Global Portfolio Optimization."
  *Financial Analysts Journal*, 48(5), 28–43.
  → `portfolio/black_litterman.py` (Black-Litterman posterior returns)

- He, G. and Litterman, R. (1999). "The Intuition Behind Black-Litterman Model
  Portfolios." Goldman Sachs Investment Management Research.
  → `portfolio/black_litterman.py` (reverse-optimization for implied returns)

## Covariance estimation

- Ledoit, O. and Wolf, M. (2004). "A Well-Conditioned Estimator for
  Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*,
  88(2), 365–411.
  → `portfolio/covariance.py` (Ledoit-Wolf analytical shrinkage)

- J.P. Morgan/Reuters (1996). Already cited under Risk metrics.
  → `portfolio/covariance.py` (EWMA covariance / RiskMetrics)

## Performance metrics

- Sharpe, W.F. (1966). "Mutual Fund Performance." *Journal of Business*,
  39(1), 119–138.
  → `performance/metrics.py` (Sharpe ratio)

- Sortino, F.A. and van der Meer, R. (1991). "Downside Risk." *Journal of
  Portfolio Management*, 17(4), 27–31.
  → `performance/metrics.py` (Sortino ratio, downside deviation)

- Young, T.W. (1991). "Calmar Ratio: A Smoother Tool." *Futures Magazine*
  (January 1991).
  → `performance/metrics.py` (Calmar ratio)

- Bacon, C.R. (2008). *Practical Portfolio Performance Measurement and
  Attribution*, 2nd ed. Wiley.
  → `performance/metrics.py` (drawdown measured against the running maximum
  of the wealth path *including* its starting capital, so a loss before the
  first new high is a drawdown; `maximum_drawdown`, `drawdown_series`)

- Martin, P.G. and McCann, B.B. (1989). *The Investor's Guide to Fidelity
  Funds: Winning Strategies for Mutual Fund Investors*. Wiley. Details not
  verified.
  → `performance/ratios.py` (`ulcer_index`, `ulcer_performance_index`;
  quantcore follows QuantStats in dividing by `n - 1` and in not annualising
  the numerator, see the function docstrings)

- Chekhlov, A., Uryasev, S., and Zabarankin, M. (2005). "Drawdown Measure in
  Portfolio Optimization." *International Journal of Theoretical and Applied
  Finance*, 8(1), 13–58.
  → `performance/ratios.py` (conditional drawdown at risk, the tail measure
  in `serenity_index`; quantcore follows QuantStats in taking the cut-off
  from a normal distribution fitted to the drawdowns)

- Schwager, J.D. (2012). *Hedge Fund Market Wizards*. Wiley. Appendix A.
  Details not verified.
  → `performance/ratios.py` (`gain_to_pain_ratio`)

- Feller, W. (1968). *An Introduction to Probability Theory and Its
  Applications*, vol. 1, 3rd ed. Wiley. Ch. XIV.
  → `performance/ratios.py` (gambler's ruin, the model behind
  `risk_of_ruin`)

- Cited by attribution only; details not verified, and no primary text is
  cited in the code: a KeyQuant white paper that QuantStats names as its
  source for the Serenity index (`serenity_index`; title and year not
  confirmed); Bernut, L. (2021), *Algorithmic Short Selling with Python*,
  Packt (`common_sense_ratio`); Harris, S.J. (`cpc_index`); Kaufman, P.J.,
  *Trading Systems and Methods*, Wiley (the fixed-stake form of
  `risk_of_ruin`). The outlier win and loss ratios and the tail ratio have
  no published source.
  → `performance/ratios.py`

## Performance attribution

- Brinson, G.P., Hood, L.R., and Beebower, G.L. (1986). "Determinants of
  Portfolio Performance." *Financial Analysts Journal*, 42(4), 39–44.
  → `performance/attribution.py` (allocation/selection/interaction
  decomposition of active return)

- Brinson, G.P. and Fachler, N. (1985). "Measuring Non-U.S. Equity Portfolio
  Performance." *Journal of Portfolio Management*, 11(3), 73–76.
  → `performance/attribution.py` (Brinson-Fachler refinement: allocation
  effect measured against benchmark total return, `brinson_fachler_attribution`)

## Market impact and execution

- Almgren, R. and Chriss, N. (2000). "Optimal Execution of Portfolio
  Transactions." *Journal of Risk*, 3, 5–40.
  → `execution/market_impact.py` (square-root market impact law)

- Gatheral, J. (2010). "No-Dynamic-Arbitrage and Market Impact."
  *Quantitative Finance*, 10(7), 749–759.
  → `execution/market_impact.py` (square-root functional form)

## Citation convention

Every implementation file must include a `References:` section in its
module-level docstring, citing the specific paper/text the formula or
algorithm is drawn from. See `pricing/black_scholes.py` for the pattern.
