"""Unit tests for `hac_t_test`."""

import numpy as np
import pytest
from scipy.stats import ttest_1samp

from quantcore.statistics import HacTResult, hac_t_test, newey_west_optimal_lags


def _ar1(seed: int, n: int) -> np.ndarray:
    """AR(1) series with coefficient 0.5, scaled to 1% volatility around a 0.2% mean."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + e[i]
    return x * 0.01 + 0.002


# (seed, n, kwargs, (t, p_value, lags, n, mean, se)) computed by edgelab.stats.hac.hac_t_test.
EDGELAB_CASES = [
    (
        1,
        50,
        {},
        (
            0.5625638455094097,
            0.5762968014191493,
            3,
            50,
            0.0010687916657592853,
            0.0018998584325864007,
        ),
    ),
    (
        2,
        200,
        {},
        (
            2.083446396537187,
            0.038487916844963006,
            4,
            200,
            0.0019085215641432335,
            0.0009160406369539004,
        ),
    ),
    (
        3,
        30,
        {"lags": 3},
        (0.9940053673509192, 0.3284422811459864, 3, 30, 0.0028009462163392416, 0.00281783812073764),
    ),
    (
        4,
        120,
        {"lags": 0, "alternative": "greater", "null": 0.001},
        (
            0.5030197257303307,
            0.30794007483711683,
            0,
            120,
            0.0014884612453977962,
            0.000971057834140804,
        ),
    ),
    (
        5,
        80,
        {"alternative": "less", "null": 0.003},
        (
            -2.407992177013237,
            0.009185562550849968,
            3,
            80,
            -0.0013152458671957033,
            0.0017920514478365688,
        ),
    ),
    (
        6,
        12,
        {},
        (2.7033130628254676, 0.02053481452978873, 2, 12, 0.010227583462562367, 0.00378335147460599),
    ),
]


@pytest.mark.parametrize(("seed", "n", "kwargs", "expected"), EDGELAB_CASES)
def test_matches_edgelab_values(seed: int, n: int, kwargs: dict, expected: tuple) -> None:
    result = hac_t_test(_ar1(seed, n), **kwargs)
    assert result.lags == expected[2]
    assert result.n == expected[3]
    actual = (result.t, result.p_value, result.mean, result.se)
    pinned = (expected[0], expected[1], expected[4], expected[5])
    np.testing.assert_allclose(actual, pinned, rtol=1e-12, atol=0.0)


def test_known_value_at_zero_lags_is_one_sample_t_test() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0])
    result = hac_t_test(x, lags=0)
    reference = ttest_1samp(x, 0.0)
    assert isinstance(result, HacTResult)
    assert result.mean == 5.6
    assert result.n == 10
    assert result.t == pytest.approx(reference.statistic, rel=1e-12)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-12)


def test_one_sided_p_values_sum_to_one_and_double_to_two_sided() -> None:
    x = _ar1(7, 100)
    two = hac_t_test(x, null=0.001).p_value
    greater = hac_t_test(x, alternative="greater", null=0.001).p_value
    less = hac_t_test(x, alternative="less", null=0.001).p_value
    assert greater + less == pytest.approx(1.0, abs=1e-12)
    assert two == pytest.approx(2.0 * min(greater, less), rel=1e-12)


def test_lags_none_uses_rule_of_thumb_and_override_changes_se() -> None:
    x = _ar1(8, 300)
    auto = hac_t_test(x)
    assert auto.lags == newey_west_optimal_lags(x)
    assert hac_t_test(x, lags=auto.lags).se == auto.se
    assert hac_t_test(x, lags=0).se != auto.se
    assert hac_t_test(x, lags=0).lags == 0


def test_null_shifts_t_by_mean_difference_over_se() -> None:
    x = _ar1(9, 60)
    base = hac_t_test(x)
    shifted = hac_t_test(x, null=0.004)
    assert shifted.se == base.se
    assert shifted.t == pytest.approx(base.t - 0.004 / base.se, rel=1e-12)


@pytest.mark.parametrize(
    ("x", "kwargs", "match"),
    [
        (np.zeros((10, 2)), {}, "1-D"),
        (np.arange(9.0), {}, "at least 10"),
        (np.append(np.arange(9.0), np.nan), {}, "finite"),
        (np.append(np.arange(9.0), np.inf), {}, "finite"),
        (np.arange(20.0), {"lags": -1}, "lags"),
        (np.arange(20.0), {"lags": 20}, "lags"),
        (np.arange(20.0), {"alternative": "both"}, "alternative"),
        (np.arange(20.0), {"null": float("nan")}, "null"),
        (np.ones(20), {}, "not positive"),
    ],
)
def test_invalid_inputs_raise(x: np.ndarray, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        hac_t_test(x, **kwargs)
