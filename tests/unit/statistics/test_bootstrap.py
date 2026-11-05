"""Unit tests for `optimal_block_length`, `stationary_bootstrap_ci` and `spa_test`.

The pinned values were computed by edgelab's `block_length`, `stationary_bootstrap_ci` and
`spa_p_values` on the inputs the helpers below build, with generators seeded as here. Both sides
draw the same `arch` resamples from the same generator state, so the values are compared exactly.

`optimal_block_length` and `spa_test` are computed by `arch`; their tests check the conventions
quantcore passes to it.
"""

import inspect

import numpy as np
import pytest
from arch.bootstrap import SPA, StationaryBootstrap
from arch.bootstrap import optimal_block_length as arch_optimal_block_length

from quantcore.statistics import (
    BootstrapInterval,
    SpaPValues,
    optimal_block_length,
    spa_test,
    stationary_bootstrap_ci,
)


def _series(seed: int, n: int, kind: str) -> np.ndarray:
    """`n` normal draws (mean 0.0004, sd 0.01), or the AR(1) with coefficient 0.5 they drive."""
    e = np.random.default_rng(seed).normal(0.0004, 0.01, n)
    if kind == "iid":
        return e
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + e[t]
    return x


def _mean(v: np.ndarray) -> float:
    return float(v.mean())


def _sharpe(v: np.ndarray) -> float:
    return float(v.mean() / v.std(ddof=1))


def _raising(x: np.ndarray):
    """Sharpe ratio; raises ValueError on a resample whose first value is above x's 92nd pct."""
    q = float(np.quantile(x, 0.92))
    n = x.shape[0]

    def stat(v: np.ndarray) -> float:
        if v.shape[0] == n and v[0] > q and not np.array_equal(v, x):
            raise ValueError("undefined")
        return _sharpe(v)

    return stat


def _nonfinite(x: np.ndarray):
    """Mean; NaN on a resample whose second value is above x's 95th percentile."""
    q = float(np.quantile(x, 0.95))
    n = x.shape[0]

    def stat(v: np.ndarray) -> float:
        if v.shape[0] == n and v[1] > q and not np.array_equal(v, x):
            return float("nan")
        return _mean(v)

    return stat


_STATS = {
    "mean": lambda x: _mean,
    "sharpe": lambda x: _sharpe,
    "raising": _raising,
    "nonfinite": _nonfinite,
}


# --- values pinned from edgelab -------------------------------------------------------------------

# (data seed, n, kind, stat, method, ci, rng seed, (estimate, ci_low, ci_high), n_dropped)
EDGELAB_CI_CASES = [
    (
        1001, 20, "iid", "mean", "percentile", 0.9, 1002,
        (0.0022398716146369343, -0.0010721830406252735, 0.005534000286893197), 0,
    ),
    (
        1001, 20, "iid", "sharpe", "bca", 0.95, 1007,
        (0.23863089193765852, -0.2262194083694271, 0.6764588626487213), 0,
    ),
    (
        1037, 100, "iid", "raising", "percentile", 0.95, 1040,
        (0.2374082620435669, 0.07494936058095182, 0.38171206566836247), 35,
    ),
    (
        1037, 100, "iid", "nonfinite", "bca", 0.95, 1045,
        (0.0022516044009525853, 0.0007358533531339567, 0.003778997933572082), 24,
    ),
    (
        1064, 250, "ar1", "sharpe", "percentile", 0.95, 1066,
        (-0.03240292930557224, -0.21509174114491816, 0.15820998416017729), 0,
    ),
    (
        1064, 250, "ar1", "mean", "bca", 0.95, 1069,
        (-0.00040117897672327314, -0.0026461361025886853, 0.0018577624664818826), 0,
    ),
    (
        1064, 250, "ar1", "raising", "bca", 0.9, 1071,
        (-0.03240292930557224, -0.1676583229830898, 0.168552286562054), 27,
    ),
]  # fmt: skip


@pytest.mark.parametrize(
    ("data_seed", "n", "kind", "stat", "method", "ci", "rng_seed", "expected", "n_dropped"),
    EDGELAB_CI_CASES,
)
def test_stationary_bootstrap_ci_matches_edgelab_values(
    data_seed: int,
    n: int,
    kind: str,
    stat: str,
    method: str,
    ci: float,
    rng_seed: int,
    expected: tuple[float, float, float],
    n_dropped: int,
) -> None:
    x = _series(data_seed, n, kind)
    result = stationary_bootstrap_ci(
        x, _STATS[stat](x), rng=np.random.default_rng(rng_seed), n_boot=400, ci=ci, method=method
    )
    assert isinstance(result, BootstrapInterval)
    assert (result.estimate, result.ci_low, result.ci_high) == expected
    assert result.n_dropped == n_dropped
    assert result.n_boot == 400
    assert result.ci_level == ci
    assert result.block_length == optimal_block_length(x)


@pytest.mark.parametrize(
    ("data_seed", "n", "kind", "expected"),
    [
        (1001, 20, "iid", 1.3338634189181722),
        (1037, 100, "iid", 3.9677894159448948),
        (1064, 250, "ar1", 6.0726127725763455),
    ],
)
def test_optimal_block_length_matches_edgelab_values(
    data_seed: int, n: int, kind: str, expected: float
) -> None:
    assert optimal_block_length(_series(data_seed, n, kind)) == expected


def _spa_inputs(seed: int, t: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed)
    benchmark = r.normal(0.0, 1.0, t)
    models = r.normal(-0.05, 1.0, (t, k)) + 0.3 * benchmark[:, None]
    return benchmark, models


@pytest.mark.parametrize(
    ("data_seed", "t", "k", "rng_seed", "block_size", "studentize", "expected"),
    [
        (1108, 60, 5, 1115, 3, False, (0.58, 0.76, 0.83)),
        (1113, 500, 5, 1120, 10, True, (0.19666666666666666, 0.22, 0.22)),
        (1106, 60, 1, 1113, 3, False, (0.5033333333333333, 0.62, 0.62)),
    ],
)
def test_spa_test_matches_edgelab_values(
    data_seed: int,
    t: int,
    k: int,
    rng_seed: int,
    block_size: int,
    studentize: bool,
    expected: tuple[float, float, float],
) -> None:
    benchmark, models = _spa_inputs(data_seed, t, k)
    result = spa_test(
        benchmark,
        models,
        rng=np.random.default_rng(rng_seed),
        block_size=block_size,
        reps=300,
        studentize=studentize,
    )
    assert isinstance(result, SpaPValues)
    assert (result.lower, result.consistent, result.upper) == expected


# --- optimal_block_length -------------------------------------------------------------------------


def test_optimal_block_length_is_arch_stationary_column_not_circular() -> None:
    x = _series(7, 400, "ar1")
    table = arch_optimal_block_length(x)
    assert optimal_block_length(x) == float(table["stationary"].iloc[0])
    assert optimal_block_length(x) != float(table["circular"].iloc[0])


@pytest.mark.parametrize(
    "x",
    [np.zeros((10, 2)), np.array([1.0]), np.array([0.1, np.nan, 0.2]), np.array([0.1, np.inf])],
)
def test_optimal_block_length_rejects_bad_input(x: np.ndarray) -> None:
    with pytest.raises(ValueError):
        optimal_block_length(x)


# --- stationary_bootstrap_ci ----------------------------------------------------------------------


def test_same_generator_state_gives_same_interval() -> None:
    x = _series(3, 120, "ar1")
    a = stationary_bootstrap_ci(x, _mean, rng=np.random.default_rng(5), n_boot=200, method="bca")
    b = stationary_bootstrap_ci(x, _mean, rng=np.random.default_rng(5), n_boot=200, method="bca")
    assert a == b


def test_resampling_is_arch_stationary_bootstrap_with_rounded_block_size() -> None:
    """The percentile interval equals np.percentile of arch's replicates at block size round(b)."""
    x = _series(4, 150, "iid")
    bootstrap = StationaryBootstrap(3, x, seed=np.random.default_rng(11))
    replicates = np.array([_mean(data[0]) for data, _kw in bootstrap.bootstrap(300)])
    alpha = 1.0 - 0.9
    expected = (
        float(np.percentile(replicates, 100.0 * alpha / 2.0)),
        float(np.percentile(replicates, 100.0 * (1.0 - alpha / 2.0))),
    )
    for block_length in (2.6, 3.0, 3.4):
        result = stationary_bootstrap_ci(
            x, _mean, rng=np.random.default_rng(11), n_boot=300, ci=0.9, block_length=block_length
        )
        assert (result.ci_low, result.ci_high) == expected
        assert result.block_length == block_length


def test_block_length_below_one_half_uses_block_size_one() -> None:
    x = _series(4, 80, "iid")
    small = stationary_bootstrap_ci(
        x, _mean, rng=np.random.default_rng(2), n_boot=100, block_length=0.3
    )
    one = stationary_bootstrap_ci(
        x, _mean, rng=np.random.default_rng(2), n_boot=100, block_length=1.0
    )
    assert (small.ci_low, small.ci_high) == (one.ci_low, one.ci_high)
    assert small.block_length == 0.3


def test_default_block_length_is_optimal_block_length() -> None:
    x = _series(9, 300, "ar1")
    default = stationary_bootstrap_ci(x, _mean, rng=np.random.default_rng(1), n_boot=100)
    explicit = stationary_bootstrap_ci(
        x, _mean, rng=np.random.default_rng(1), n_boot=100, block_length=optimal_block_length(x)
    )
    assert default == explicit


@pytest.mark.parametrize("method", ["percentile", "bca"])
def test_interval_is_ordered_and_contains_estimate_for_the_mean(method: str) -> None:
    x = _series(12, 200, "iid")
    result = stationary_bootstrap_ci(
        x, _mean, rng=np.random.default_rng(3), n_boot=300, method=method
    )
    assert result.ci_low < result.estimate < result.ci_high
    assert result.estimate == _mean(x)


def test_dropped_resamples_are_left_out_of_the_interval() -> None:
    """A dropped resample does not enter the percentiles: they equal those of the kept ones."""
    x = _series(21, 60, "iid")
    stat = _nonfinite(x)
    bootstrap = StationaryBootstrap(2, x, seed=np.random.default_rng(8))
    raw = np.array([stat(data[0]) for data, _kw in bootstrap.bootstrap(250)])
    kept = raw[np.isfinite(raw)]
    result = stationary_bootstrap_ci(
        x, stat, rng=np.random.default_rng(8), n_boot=250, block_length=2.0
    )
    assert result.n_dropped == 250 - kept.size > 0
    assert result.ci_low == float(np.percentile(kept, 2.5))
    assert result.ci_high == float(np.percentile(kept, 97.5))


def test_every_resample_dropped_raises() -> None:
    x = _series(2, 30, "iid")

    def stat(v: np.ndarray) -> float:
        if not np.array_equal(v, x):
            raise ValueError("undefined")
        return 0.0

    with pytest.raises(ValueError, match="every resample"):
        stationary_bootstrap_ci(x, stat, rng=np.random.default_rng(0), n_boot=20)


def test_exception_other_than_value_error_propagates() -> None:
    def stat(v: np.ndarray) -> float:
        raise ZeroDivisionError

    with pytest.raises(ZeroDivisionError):
        stationary_bootstrap_ci(_series(1, 30, "iid"), stat, rng=np.random.default_rng(0))


def test_bca_jackknife_value_error_propagates() -> None:
    x = _series(2, 30, "iid")

    def stat(v: np.ndarray) -> float:
        if v.shape[0] < x.shape[0]:
            raise ValueError("jackknife sample")
        return _mean(v)

    pct = stationary_bootstrap_ci(x, stat, rng=np.random.default_rng(0), n_boot=50)
    assert pct.n_dropped == 0
    with pytest.raises(ValueError, match="jackknife sample"):
        stationary_bootstrap_ci(x, stat, rng=np.random.default_rng(0), n_boot=50, method="bca")


def test_bca_with_constant_jackknife_uses_zero_acceleration() -> None:
    """Equal jackknife values give a = 0: the levels are Phi(2 * z0 + z_alpha)."""
    from scipy.stats import norm

    x = _series(6, 50, "iid")

    def stat(v: np.ndarray) -> float:
        return _mean(v) if v.shape[0] == x.shape[0] else 0.0

    bootstrap = StationaryBootstrap(2, x, seed=np.random.default_rng(4))
    replicates = np.array([_mean(data[0]) for data, _kw in bootstrap.bootstrap(400)])
    z0 = float(norm.ppf(np.mean(replicates < _mean(x))))
    expected = [
        float(np.percentile(replicates, 100.0 * float(norm.cdf(2.0 * z0 + z))))
        for z in (float(norm.ppf(0.025)), float(norm.ppf(0.975)))
    ]
    result = stationary_bootstrap_ci(
        x, stat, rng=np.random.default_rng(4), n_boot=400, method="bca", block_length=2.0
    )
    assert result.ci_low == pytest.approx(expected[0], rel=1e-14)
    assert result.ci_high == pytest.approx(expected[1], rel=1e-14)


@pytest.mark.parametrize(
    ("x", "kwargs"),
    [
        (np.array([0.01]), {}),
        (np.zeros((20, 2)), {}),
        (np.array([0.01, np.nan, 0.02]), {}),
        (np.linspace(-0.01, 0.01, 30), {"n_boot": 0}),
        (np.linspace(-0.01, 0.01, 30), {"ci": 0.0}),
        (np.linspace(-0.01, 0.01, 30), {"ci": 1.0}),
        (np.linspace(-0.01, 0.01, 30), {"method": "basic"}),
        (np.linspace(-0.01, 0.01, 30), {"block_length": 0.0}),
        (np.linspace(-0.01, 0.01, 30), {"block_length": -2.0}),
        (np.linspace(-0.01, 0.01, 30), {"block_length": float("nan")}),
        (np.linspace(-0.01, 0.01, 30), {"block_length": float("inf")}),
    ],
)
def test_stationary_bootstrap_ci_rejects_bad_input(x: np.ndarray, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        stationary_bootstrap_ci(x, _mean, rng=np.random.default_rng(0), **kwargs)


def test_caller_array_is_not_modified() -> None:
    x = _series(5, 40, "iid")
    before = x.copy()

    def stat(v: np.ndarray) -> float:
        value = _mean(v)
        v[:] = 0.0
        return value

    stationary_bootstrap_ci(x, stat, rng=np.random.default_rng(0), n_boot=20, method="bca")
    np.testing.assert_array_equal(x, before)


# --- spa_test: conventions passed to arch ---------------------------------------------------------


def test_spa_test_passes_every_arch_option_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every SPA parameter is given, the defaulted ones at arch 8.0.0's default values."""
    import arch.bootstrap

    captured: dict = {}
    real_spa = arch.bootstrap.SPA

    def recording_spa(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return real_spa(*args, **kwargs)

    monkeypatch.setattr(arch.bootstrap, "SPA", recording_spa)
    benchmark, models = _spa_inputs(1, 50, 2)
    rng = np.random.default_rng(0)
    spa_test(benchmark, models, rng=rng, block_size=4, reps=20, studentize=False)

    parameters = inspect.signature(real_spa).parameters
    assert len(captured["args"]) == 2
    assert set(captured["kwargs"]) == set(parameters) - {"benchmark", "models"}
    assert captured["kwargs"]["block_size"] == 4
    assert captured["kwargs"]["reps"] == 20
    assert captured["kwargs"]["studentize"] is False
    assert captured["kwargs"]["seed"] is rng
    for name in ("bootstrap", "nested"):
        assert captured["kwargs"][name] == parameters[name].default


def test_spa_test_equals_arch_spa_on_the_same_generator_state() -> None:
    benchmark, models = _spa_inputs(2, 200, 5)
    for studentize in (True, False):
        spa = SPA(
            benchmark,
            models,
            block_size=5,
            reps=200,
            studentize=studentize,
            seed=np.random.default_rng(3),
        )
        spa.compute()
        result = spa_test(
            benchmark,
            models,
            rng=np.random.default_rng(3),
            block_size=5,
            reps=200,
            studentize=studentize,
        )
        assert (result.lower, result.consistent, result.upper) == tuple(
            float(spa.pvalues[name]) for name in ("lower", "consistent", "upper")
        )


def test_spa_test_p_values_do_not_depend_on_studentize_in_arch_8() -> None:
    """arch 8.0.0's SPA statistic is never divided by its standard deviation (see `spa_test`)."""
    benchmark, models = _spa_inputs(4, 300, 5)
    models = models * np.array([1.0, 3.0, 0.5, 2.0, 1.0])
    results = [
        spa_test(
            benchmark, models, rng=np.random.default_rng(9), block_size=5, reps=200, studentize=s
        )
        for s in (True, False)
    ]
    assert results[0] == results[1]


def test_spa_test_takes_losses_not_returns() -> None:
    """A model with clearly lower losses rejects the null; the same series as returns does not."""
    r = np.random.default_rng(10)
    benchmark = r.normal(0.0, 1.0, 400)
    better = benchmark[:, None] - 0.5 + r.normal(0.0, 0.2, (400, 1))
    as_losses = spa_test(
        benchmark, better, rng=np.random.default_rng(1), block_size=5, reps=300, studentize=True
    )
    as_returns = spa_test(
        -benchmark, -better, rng=np.random.default_rng(1), block_size=5, reps=300, studentize=True
    )
    assert as_losses.consistent < 0.01
    assert as_returns.consistent > 0.5


@pytest.mark.parametrize("k", [1, 5, 20])
@pytest.mark.parametrize("studentize", [True, False])
def test_spa_p_values_are_ordered(k: int, studentize: bool) -> None:
    benchmark, models = _spa_inputs(30 + k, 150, k)
    result = spa_test(
        benchmark,
        models,
        rng=np.random.default_rng(k),
        block_size=4,
        reps=200,
        studentize=studentize,
    )
    assert 0.0 <= result.lower <= result.consistent <= result.upper <= 1.0


@pytest.mark.parametrize(
    ("benchmark", "models", "kwargs"),
    [
        (np.zeros(20), np.zeros((20, 2)), {"block_size": 0}),
        (np.zeros(20), np.zeros((20, 2)), {"reps": 0}),
        (np.zeros((20, 1)), np.zeros((20, 2)), {}),
        (np.zeros(1), np.zeros((1, 2)), {}),
        (np.zeros(20), np.zeros(20), {}),
        (np.zeros(20), np.zeros((20, 2, 1)), {}),
        (np.zeros(20), np.zeros((20, 0)), {}),
        (np.zeros(20), np.zeros((19, 2)), {}),
        (np.r_[np.zeros(19), np.nan], np.zeros((20, 2)), {}),
        (np.zeros(20), np.full((20, 2), np.inf), {}),
    ],
)
def test_spa_test_rejects_bad_input(
    benchmark: np.ndarray, models: np.ndarray, kwargs: dict
) -> None:
    options = {"block_size": 3, "reps": 10, "studentize": True} | kwargs
    with pytest.raises(ValueError):
        spa_test(benchmark, models, rng=np.random.default_rng(0), **options)
