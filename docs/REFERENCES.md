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
  → `risk/volatility.py` (GARCH(1,1) implementation; also GARCH(1,1) MLE
  estimation, `fit_garch_11`)

## Stochastic processes

- Uhlenbeck, G.E. and Ornstein, L.S. (1930). "On the Theory of the Brownian
  Motion." *Physical Review*, 36(5), 823–841.
  → `core/stochastic.py` (Ornstein-Uhlenbeck SDE)

- Vasicek, O. (1977). "An Equilibrium Characterization of the Term Structure."
  *Journal of Financial Economics*, 5(2), 177–188.
  → `core/stochastic.py` (OU / Vasicek mean-reverting diffusion in finance)

- Cox, J.C., Ingersoll, J.E., and Ross, S.A. (1985). "A Theory of the Term
  Structure of Interest Rates." *Econometrica*, 53(2), 385–407.
  → `core/stochastic.py` (CIR square-root diffusion)

- Heston, S.L. (1993). "A Closed-Form Solution for Options with Stochastic
  Volatility with Applications to Bond and Currency Options." *Review of
  Financial Studies*, 6(2), 327–343.
  → `core/stochastic.py` (Heston stochastic volatility SDE)
  → `pricing/heston.py` (Heston characteristic function for COS pricing)

- Fang, F. and Oosterlee, C.W. (2008). "A Novel Pricing Method for European
  Options Based on Fourier-Cosine Series Expansions." *SIAM Journal on
  Scientific Computing*, 31(2), 826–848.
  → `pricing/heston.py` (COS method for option pricing under Heston)

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
  → `filtering/kalman.py` (linear Kalman filter predict/update equations)

- Rauch, H.E., Tung, F., and Striebel, C.T. (1965). "Maximum Likelihood
  Estimates of Linear Dynamic Systems." *AIAA Journal*, 3(8), 1445–1450.
  → `filtering/kalman.py` (RTS smoother backward pass)

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
  → `regime/hmm.py` (Markov-switching model applied to financial returns)

## Cointegration and mean reversion

- Engle, R.F. and Granger, C.W.J. (1987). "Co-Integration and Error
  Correction: Representation, Estimation, and Testing." *Econometrica*, 55(2),
  251–276.
  → `statistics/cointegration.py` (Engle-Granger two-step cointegration test)

- Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors."
  *Journal of Economic Dynamics and Control*, 12(2–3), 231–254.
  → `statistics/cointegration.py` (Johansen trace test, eigenvectors)

- Avellaneda, M. and Lee, J.-H. (2010). "Statistical Arbitrage in the U.S.
  Equities Market." *Quantitative Finance*, 10(7), 761–782.
  → `statistics/cointegration.py` (OU half-life and z-score for spread trading)

## Time series

- Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for
  Autoregressive Time Series with a Unit Root." *Journal of the American
  Statistical Association*, 74(366), 427–431.
  → `time_series/arima.py` (ADF unit-root test)

- Ljung, G.M. and Box, G.E.P. (1978). "On a Measure of Lack of Fit in Time
  Series Models." *Biometrika*, 65(2), 297–303.
  → `time_series/arima.py` (Ljung-Box Q-test for autocorrelation in residuals)

- Box, G.E.P., Jenkins, G.M., and Reinsel, G.C. (2015). *Time Series Analysis:
  Forecasting and Control* (5th ed.). Wiley.
  → `time_series/arima.py` (ARIMA(p,d,q) model, conditional sum-of-squares MLE)

- Sims, C.A. (1980). "Macroeconomics and Reality." *Econometrica*, 48(1), 1–48.
  → `time_series/var_model.py` (Vector Autoregression — VAR(p) model)

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
  Springer.
  → `time_series/var_model.py` (VAR OLS estimation, impulse response functions)

- Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models
  and Cross-Spectral Methods." *Econometrica*, 37(3), 424–438.
  → `time_series/var_model.py` (Granger causality F-test)

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

## Portfolio optimization

- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*, 7(1),
  77–91.
  → `portfolio/optimization.py` (mean-variance efficient frontier)

- Kelly, J.L. (1956). "A New Interpretation of Information Rate." *Bell System
  Technical Journal*, 35(4), 917–926.
  → `portfolio/optimization.py` (Kelly fraction for position sizing)

- Maillard, S., Roncalli, T., and Teïletche, J. (2010). "The Properties of
  Equally Weighted Risk Contribution Portfolios." *Journal of Portfolio
  Management*, 36(4), 60–70.
  → `portfolio/optimization.py` (risk parity / equal risk contribution)

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

## Citation convention

Every implementation file must include a `References:` section in its
module-level docstring, citing the specific paper/text the formula or
algorithm is drawn from. See `pricing/black_scholes.py` for the pattern.
