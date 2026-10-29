"""Oracle tests for the Sharpe-ratio inference functions.

Independent oracles:

* Published values. Bailey and Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier",
  section 3: a monthly track record with SR = 0.458 (per period), skewness -2.448 and kurtosis
  10.164 has PSR(0) = 0.913 at n = 24 and 0.953 at n = 36. Bailey and Lopez de Prado (2014), "The
  Deflated Sharpe Ratio", numerical example: T = 1250, annualised SR 2.5 at 250 periods per year,
  skewness -3, kurtosis 10, V[{SR_n}] = 1/2 (annualised): SR0 = 0.1132 and DSR = 0.9004 at N = 100;
  DSR = 0.9505 at N = 46 and, for Normal moments, at N = 88. The papers print summary statistics
  only, so the return series here are constructed to have exactly those moments: the two
  24- and 36-observation series are edgelab's constructions (reproduced from its fixture file),
  the DSR series are a two-point and a three-point distribution built in `_standardised`.
  The comparison tolerance is the paper's printed precision.
* `scipy.stats.skew`/`kurtosis` (bias=True) and `scipy.stats.norm` for the building blocks of the
  variance factor, PSR, minimum track record length and `SR0`. These re-derive the formula from
  scipy primitives, so they check the moments and quantiles, not the formula itself.
* Numerical integration of `N x phi(x) Phi(x)^(N-1)` for the exact expected maximum of N
  standard Normal draws, against which `expected_max_sharpe` is an approximation.
"""

import math

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import kurtosis, norm, skew

from quantcore.performance import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_se_mertens,
)

pytestmark = pytest.mark.oracle

_PSR_24 = np.array(
    [
        0.9354873629748602,
        1.2834024874503691,
        0.9200231361107245,
        -3.461749348494797,
        1.3230960490059116,
        1.031171379721753,
        -0.5604326947543705,
        1.139063168878795,
        0.9544823118236367,
        0.8818709039568992,
        0.5630378552532208,
        1.1136359406996927,
        -0.08566629431707412,
        0.27954394654533465,
        -0.4715468701262337,
        1.1516297225418135,
        0.5779173584601067,
        0.018082842101133645,
        0.13779950916810263,
        0.09775630480852332,
        0.535983677205627,
        0.05702499931866656,
        1.4413986762950433,
        1.3654017186924623,
    ]
)

_PSR_36 = np.array(
    [
        1.3782559234275713,
        0.3228230680100733,
        0.47960400675154397,
        -3.686725019604676,
        0.6533073186306816,
        0.47416409343764365,
        0.6598338199475781,
        0.5709007179668325,
        1.1292475839579104,
        0.2865496138761685,
        0.502140368710595,
        0.6974842694101112,
        0.6515456735176237,
        0.04586744001652754,
        0.8189302776148397,
        1.4743504637146043,
        0.7205523313091649,
        0.22897148953569543,
        1.5586959646031981,
        -0.05664204759829761,
        0.5431568115729469,
        1.3812476236105615,
        1.0204511451962657,
        0.8805363056309451,
        -0.18793014066835356,
        0.5667169301826829,
        1.280004050305115,
        -0.46677323719972763,
        -2.32161902378627,
        -0.4972382062020374,
        0.5431579339971706,
        1.5534978589915835,
        0.4824810731829043,
        0.5960042298542243,
        1.315824866981723,
        1.122508443471418,
    ]
)


def _standardised(values: np.ndarray, sr: float) -> np.ndarray:
    """Affine map of `values` to sample mean `sr` and sample standard deviation 1."""
    return (values - values.mean()) / values.std(ddof=1) + sr


def _dsr_series(three_point: bool) -> np.ndarray:
    sr = 2.5 / math.sqrt(250)
    if three_point:
        return _standardised(np.r_[np.full(208, -1.0), np.zeros(834), np.full(208, 1.0)], sr)
    return _standardised(np.r_[np.zeros(105), np.ones(1145)], sr)


# V[{SR_n}] = 1/2 annualised = 0.002 per period; two values with that sample variance.
_FAMILY = np.array([0.0, math.sqrt(0.004)])


def test_psr_published_values() -> None:
    assert probabilistic_sharpe_ratio(_PSR_24) == pytest.approx(0.913, abs=1e-3)
    assert probabilistic_sharpe_ratio(_PSR_36) == pytest.approx(0.953, abs=1e-3)
    for x, n in ((_PSR_24, 24), (_PSR_36, 36)):
        assert x.size == n
        assert x.mean() / x.std(ddof=1) == pytest.approx(0.458, abs=1e-12)
        assert skew(x) == pytest.approx(-2.448, abs=1e-6)
        assert kurtosis(x, fisher=False) == pytest.approx(10.164, abs=1e-6)


def test_dsr_published_values() -> None:
    best = deflated_sharpe_ratio(_dsr_series(False), _FAMILY, n_effective=100)
    assert best.var_sharpe == pytest.approx(0.002, rel=1e-12)
    assert best.sr_zero == pytest.approx(0.1132, abs=5e-5)
    assert best.dsr == pytest.approx(0.9004, abs=5e-5)
    assert deflated_sharpe_ratio(_dsr_series(False), _FAMILY, n_effective=46).dsr == pytest.approx(
        0.9505, abs=5e-5
    )
    normal = deflated_sharpe_ratio(_dsr_series(True), _FAMILY, n_effective=88)
    assert normal.dsr == pytest.approx(0.9505, abs=5e-4)


def test_dsr_series_have_the_published_moments() -> None:
    x = _dsr_series(False)
    assert skew(x) == pytest.approx(-3.0, abs=1e-3)
    assert kurtosis(x, fisher=False) == pytest.approx(10.0, abs=5e-3)
    y = _dsr_series(True)
    assert skew(y) == pytest.approx(0.0, abs=1e-12)
    assert kurtosis(y, fisher=False) == pytest.approx(3.0, abs=1e-2)


@pytest.mark.parametrize("seed", range(5))
def test_building_blocks_match_scipy(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.standard_t(4, 300) * 0.01 + 0.001 * (seed + 1)
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = skew(x, bias=True), kurtosis(x, fisher=False, bias=True)
    factor = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    ppy = 12

    se = sharpe_se_mertens(x, periods_per_year=ppy)
    assert se.sharpe == pytest.approx(sr * math.sqrt(ppy), rel=1e-12)
    assert se.se == pytest.approx(math.sqrt(factor / (x.size - 1) * ppy), rel=1e-12)
    for sr_star in (-0.05, 0.0, 0.05):
        expected = norm.cdf((sr - sr_star) / math.sqrt(factor / (x.size - 1)))
        assert probabilistic_sharpe_ratio(x, sr_star) == pytest.approx(expected, rel=1e-12)
    trl = 1.0 + factor * (norm.ppf(0.95) / (sr - 0.02)) ** 2
    assert minimum_track_record_length(x, 0.02, 0.05) == pytest.approx(trl, rel=1e-12)


@pytest.mark.parametrize("n", [2, 5, 10, 46, 100, 1000])
def test_expected_max_sharpe_against_exact_expectation(n: int) -> None:
    exact, _ = integrate.quad(
        lambda x: n * x * norm.pdf(x) * norm.cdf(x) ** (n - 1), -12, 12, limit=200
    )
    relative_error = expected_max_sharpe(n, 1.0) / exact - 1.0
    bound = {2: 0.079, 5: 0.026, 10: 0.024, 46: 0.013, 100: 0.0095, 1000: 0.0045}[n]
    assert abs(relative_error) < bound
    assert abs(relative_error) > bound - 0.002


@pytest.mark.parametrize("n", [2, 7, 50, 3000])
def test_expected_max_sharpe_matches_scipy_quantiles(n: int) -> None:
    g = 0.5772156649015329
    expected = (1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * math.e))
    assert expected_max_sharpe(n, 1.0) == pytest.approx(expected, rel=1e-14)
    assert expected_max_sharpe(n, 0.09) == pytest.approx(0.3 * expected, rel=1e-14)
