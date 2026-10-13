"""Black-Scholes-Merton option price, delta and vega.

`black_scholes_merton_price`, `bsm_delta` and `bsm_vega` take a continuous yield `q`, with
`d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt(T))` and `d2 = d1 - sigma sqrt(T)`. Each is
registered as a `ConformanceTarget` pairing a quantcore adapter (a thin call into
`quantcore.pricing`) with an independent closed-form oracle built on `math.erf`. Importing this
module registers all three.

Both callables of every target are meant for `t > 0`: at `t = 0` the closed form divides by
`sigma * sqrt(t) = 0`, so the oracles raise `ZeroDivisionError` there.
"""

import math
from typing import Literal

import quantcore.pricing as _qc_pricing
from conformance.compare import ConformanceTarget, ScalarComparator

__all__ = ["OptionRight"]

type OptionRight = Literal["call", "put"]

_SQRT_2 = math.sqrt(2.0)
_SQRT_2_PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function, via `math.erf`."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / _SQRT_2_PI


def _d1_d2(
    s: float, k: float, t: float, *, r: float, q: float, sigma: float
) -> tuple[float, float]:
    total_vol = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / total_vol
    d2 = d1 - total_vol
    return d1, d2


def _black_scholes_merton_price_reference(
    s: float, k: float, t: float, *, r: float, q: float, sigma: float, right: OptionRight
) -> float:
    """BSM closed-form price for `t > 0`; the oracle for `black_scholes_merton_price`."""
    d1, d2 = _d1_d2(s, k, t, r=r, q=q, sigma=sigma)
    if right == "call":
        return s * math.exp(-q * t) * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)
    return k * math.exp(-r * t) * _norm_cdf(-d2) - s * math.exp(-q * t) * _norm_cdf(-d1)


def _black_scholes_merton_price_quantcore(
    s: float, k: float, t: float, *, r: float, q: float, sigma: float, right: OptionRight
) -> float:
    """`quantcore.pricing.black_scholes_call`/`black_scholes_put` adapter, `t > 0` only."""
    fn = _qc_pricing.black_scholes_call if right == "call" else _qc_pricing.black_scholes_put
    return float(fn(s, k, r, sigma, t, q))


_BLACK_SCHOLES_MERTON_PRICE = ConformanceTarget(
    name="black_scholes_merton_price",
    tier=2,
    oracle="Black-Scholes-Merton closed form on math.erf, independent of quantcore",
    citation="Black, F. and Scholes, M. (1973); Merton, R.C. (1973).",
    compare=ScalarComparator(),
    quantcore=_black_scholes_merton_price_quantcore,
    reference=_black_scholes_merton_price_reference,
)


def _bsm_delta_reference(
    s: float, k: float, t: float, *, r: float, q: float, sigma: float, right: OptionRight
) -> float:
    """BSM closed-form delta for `t > 0`; the oracle for `bsm_delta`."""
    d1, _ = _d1_d2(s, k, t, r=r, q=q, sigma=sigma)
    discount = math.exp(-q * t)
    n_d1 = _norm_cdf(d1)
    return discount * n_d1 if right == "call" else discount * (n_d1 - 1.0)


def _bsm_delta_quantcore(
    s: float, k: float, t: float, *, r: float, q: float, sigma: float, right: OptionRight
) -> float:
    """`quantcore.pricing.bs_delta` adapter, `t > 0` only."""
    return float(_qc_pricing.bs_delta(s, k, r, sigma, t, right, q))


_BSM_DELTA = ConformanceTarget(
    name="bsm_delta",
    tier=2,
    oracle="Black-Scholes-Merton closed-form delta on math.erf, independent of quantcore",
    citation="Black, F. and Scholes, M. (1973); Merton, R.C. (1973).",
    compare=ScalarComparator(),
    quantcore=_bsm_delta_quantcore,
    reference=_bsm_delta_reference,
)


def _bsm_vega_reference(s: float, k: float, t: float, *, r: float, q: float, sigma: float) -> float:
    """BSM closed-form vega for `t > 0`; the oracle for `bsm_vega`."""
    d1, _ = _d1_d2(s, k, t, r=r, q=q, sigma=sigma)
    return s * math.exp(-q * t) * _norm_pdf(d1) * math.sqrt(t)


def _bsm_vega_quantcore(s: float, k: float, t: float, *, r: float, q: float, sigma: float) -> float:
    """`quantcore.pricing.bs_vega` adapter, `t > 0` only."""
    return float(_qc_pricing.bs_vega(s, k, r, sigma, t, q))


_BSM_VEGA = ConformanceTarget(
    name="bsm_vega",
    tier=2,
    oracle="Black-Scholes-Merton closed-form vega on math.erf, independent of quantcore",
    citation="Black, F. and Scholes, M. (1973); Merton, R.C. (1973).",
    compare=ScalarComparator(),
    quantcore=_bsm_vega_quantcore,
    reference=_bsm_vega_reference,
)
