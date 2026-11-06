"""Unit tests for the Sharpe-ratio inference functions.

The literal tables were computed once by an independent implementation of the same formulas
(per-period Sharpe, population skewness and non-excess kurtosis, `scipy.stats.norm`) on the
series built by `_series`, and are pinned at relative 1e-12.
"""

import math

import numpy as np
import pytest

from quantcore.performance import (
    DeflatedSharpeResult,
    SharpeSE,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_se_mertens,
)

PPY = 252
REL = 1e-12


def _series() -> dict[str, np.ndarray]:
    r = np.random.default_rng(11)
    out = {"normal": r.normal(0.001, 0.01, 500), "student3": r.standard_t(3, 800) * 0.01 + 0.002}
    out["skew_pos"] = (r.gamma(2.0, 1.0, 300) - 1.5) * 0.01
    out["skew_neg"] = -(r.gamma(1.5, 1.0, 400) - 1.2) * 0.01
    out["short"] = r.normal(0.002, 0.02, 12)
    crashy = r.normal(0.0005, 0.008, 1000)
    crashy[::97] -= 0.06
    out["crashy"] = crashy
    out["negmean"] = r.normal(-0.001, 0.01, 250)
    return out


_SE_PSR_TRL = {
    "normal": dict(
        sharpe=2.1469954290516653,
        se=0.7159208844643546,
        psr0=0.9986453448882378,
        psr1=0.9706377882944344,
        psrn=0.9997117108082908,
        trl=151.11458251855782,
        trl2=151.47853037585756,
    ),
    "student3": dict(
        sharpe=1.9490082207488766,
        se=0.7139899052607834,
        psr0=0.9968308044030952,
        psr1=0.9471758182205443,
        psrn=0.9992492988745437,
        trl=291.10692391106426,
        trl2=309.4121471620282,
    ),
    "skew_pos": dict(
        sharpe=4.893549560192916,
        se=0.7415265614553752,
        psr0=0.9999999999793435,
        psr1=0.9999999838874997,
        psrn=0.9999999999989483,
        trl=19.575099290516434,
        trl2=14.838218264167908,
    ),
    "skew_neg": dict(
        sharpe=-4.467659795391583,
        se=0.632848429268626,
        psr0=8.348841565435058e-13,
        psr1=4.633652976363602e-17,
        psrn=2.7281655474175093e-11,
        trl=22.660407981678222,
        trl2=11.737569731113956,
    ),
    "short": dict(
        sharpe=1.1273866895113436,
        se=4.963996005265117,
        psr0=0.5898319576119346,
        psr1=0.5267952336190174,
        psrn=0.6145016112155717,
        trl=577.9846233749684,
        trl2=1050.93804804804,
    ),
    "crashy": dict(
        sharpe=0.3160049747867651,
        se=0.5136288522621979,
        psr0=0.7308018565002108,
        psr1=0.17616256685540443,
        psrn=0.8912813832204383,
        trl=7141.538836161121,
        trl2=16860.605214286155,
    ),
    "negmean": dict(
        sharpe=-0.3796404880742017,
        se=1.0063872090891128,
        psr0=0.3530009593910728,
        psr1=0.12182361561810162,
        psrn=0.4753786183240088,
        trl=4735.117483125268,
        trl2=566.4319153588043,
    ),
}
_EMS = {
    (2, 0.0): 0.0,
    (2, 1.0): 0.5197553442805939,
    (2, 0.002): 0.02324416562960429,
    (2.5, 0.0): 0.0,
    (2.5, 1.0): 0.7124531286751234,
    (2.5, 0.002): 0.03186187252999961,
    (10, 0.0): 0.0,
    (10, 1.0): 1.57459830134575,
    (10, 0.002): 0.07041817678129592,
    (46, 0.0): 0.0,
    (46, 1.0): 2.2441708308410093,
    (46, 0.002): 0.10036237061765357,
    (100, 0.0): 0.0,
    (100, 1.0): 2.5306028932016846,
    (100, 0.002): 0.11317200186513214,
    (1000, 0.0): 0.0,
    (1000, 1.0): 3.255121513652723,
    (1000, 0.002): 0.14557345959098997,
}
_DSR = {
    "student3": dict(
        n=30,
        over=None,
        tv="family",
        nf=40,
        dsr=0.7900387240285054,
        sr_zero=0.08649941544232205,
        var=0.0017404573247502054,
        naive=0.9968308044030952,
    ),
    "skew_pos": dict(
        n=5.5,
        over=None,
        tv="family",
        nf=12,
        dsr=0.9999998850018075,
        sr_zero=0.06661062309938467,
        var=0.002841714190897336,
        naive=0.9999999999793435,
    ),
    "crashy": dict(
        n=100,
        over=None,
        tv="no_skill",
        nf=100,
        dsr=0.0314932618212104,
        sr_zero=0.08006473233855954,
        var=0.001001001001001001,
        naive=0.7308018565002108,
    ),
    "normal": dict(
        n=1,
        over=None,
        tv="family",
        nf=10,
        dsr=0.9986453448882378,
        sr_zero=0.0,
        var=0.0016328140673143802,
        naive=0.9986453448882378,
    ),
    "negmean": dict(
        n=20,
        over=0.03,
        tv="family",
        nf=20,
        dsr=0.19753920727689778,
        sr_zero=0.03,
        var=0.0032247308304971484,
        naive=0.3530009593910728,
    ),
    "short": dict(
        n=2,
        over=None,
        tv="family",
        nf=1,
        dsr=0.5898319576119346,
        sr_zero=0.0,
        var=0.0,
        naive=0.5898319576119346,
    ),
}

SERIES = _series()


@pytest.mark.parametrize("name", list(_SE_PSR_TRL))
def test_sharpe_se_mertens_pinned(name: str) -> None:
    result = sharpe_se_mertens(SERIES[name], periods_per_year=PPY)
    assert isinstance(result, SharpeSE)
    assert result.n == SERIES[name].size
    assert result.sharpe == pytest.approx(_SE_PSR_TRL[name]["sharpe"], rel=REL)
    assert result.se == pytest.approx(_SE_PSR_TRL[name]["se"], rel=REL)


@pytest.mark.parametrize("name", list(_SE_PSR_TRL))
def test_psr_pinned(name: str) -> None:
    x, ref = SERIES[name], _SE_PSR_TRL[name]
    assert probabilistic_sharpe_ratio(x) == pytest.approx(ref["psr0"], rel=REL, abs=1e-15)
    assert probabilistic_sharpe_ratio(x, 0.05) == pytest.approx(ref["psr1"], rel=REL, abs=1e-15)
    assert probabilistic_sharpe_ratio(x, -0.02) == pytest.approx(ref["psrn"], rel=REL, abs=1e-15)


@pytest.mark.parametrize("name", list(_SE_PSR_TRL))
def test_minimum_track_record_length_pinned(name: str) -> None:
    x, ref = SERIES[name], _SE_PSR_TRL[name]
    assert minimum_track_record_length(x, 0.0, 0.05) == pytest.approx(ref["trl"], rel=REL)
    assert minimum_track_record_length(x, 0.03, 0.1) == pytest.approx(ref["trl2"], rel=REL)


@pytest.mark.parametrize(("n", "v"), list(_EMS))
def test_expected_max_sharpe_pinned(n: float, v: float) -> None:
    assert expected_max_sharpe(n, v) == pytest.approx(_EMS[(n, v)], rel=REL, abs=0.0)


def test_expected_max_sharpe_hand_value() -> None:
    g = 0.5772156649015329
    # Phi^-1(0.9) = 1.2815515655446004, Phi^-1(1 - 1/(10 e)) evaluated by scipy.
    z2 = 1.7892417645816283
    assert expected_max_sharpe(10, 4.0) == pytest.approx(
        2.0 * ((1 - g) * 1.2815515655446004 + g * z2), rel=1e-12
    )


def test_expected_max_sharpe_euler_gamma_zero_is_first_quantile() -> None:
    assert expected_max_sharpe(10, 1.0, euler_gamma=0.0) == pytest.approx(
        1.2815515655446004, rel=1e-14
    )


def test_expected_max_sharpe_bias_against_exact_expectation() -> None:
    """The published approximation is high by 2.3% at N=10 and 0.9% at N=100, low at N=2."""
    from scipy import integrate
    from scipy.stats import norm

    def exact(n: int) -> float:
        value, _ = integrate.quad(
            lambda x: n * x * norm.pdf(x) * norm.cdf(x) ** (n - 1), -12, 12, limit=200
        )
        return float(value)

    assert expected_max_sharpe(10, 1.0) / exact(10) - 1 == pytest.approx(0.0233, abs=5e-4)
    assert expected_max_sharpe(100, 1.0) / exact(100) - 1 == pytest.approx(0.0092, abs=5e-4)
    assert expected_max_sharpe(2, 1.0) / exact(2) - 1 == pytest.approx(-0.0788, abs=5e-4)


def test_expected_max_sharpe_monotone() -> None:
    values = [expected_max_sharpe(n, 1.0) for n in (2, 3, 10, 100, 10_000)]
    assert values == sorted(values)
    assert expected_max_sharpe(10, 2.0) > expected_max_sharpe(10, 1.0)
    assert expected_max_sharpe(10, 0.0) == 0.0


@pytest.mark.parametrize(
    ("n", "v"),
    [(1.99, 1.0), (math.nan, 1.0), (math.inf, 1.0), (10, -1.0), (10, math.nan), (10, math.inf)],
)
def test_expected_max_sharpe_invalid(n: float, v: float) -> None:
    with pytest.raises(ValueError):
        expected_max_sharpe(n, v)


def test_dsr_pinned() -> None:
    rng = np.random.default_rng(5)
    for name, ref in _DSR.items():
        family = rng.normal(0.02, 0.05, ref["nf"])
        result = deflated_sharpe_ratio(
            SERIES[name],
            family,
            n_effective=ref["n"],
            sr_star_override=ref["over"],
            trial_variance=ref["tv"],
        )
        assert isinstance(result, DeflatedSharpeResult)
        assert result.n_effective == ref["n"]
        assert result.dsr == pytest.approx(ref["dsr"], rel=REL, abs=1e-15)
        assert result.sr_zero == pytest.approx(ref["sr_zero"], rel=REL, abs=1e-15)
        assert result.var_sharpe == pytest.approx(ref["var"], rel=REL, abs=1e-15)
        assert result.psr_naive == pytest.approx(ref["naive"], rel=REL, abs=1e-15)


def test_dsr_relations() -> None:
    x = SERIES["student3"]
    family = np.random.default_rng(3).normal(0.02, 0.05, 30)
    small = deflated_sharpe_ratio(x, family, n_effective=2)
    large = deflated_sharpe_ratio(x, family, n_effective=90)
    assert large.dsr <= small.dsr <= small.psr_naive
    single = deflated_sharpe_ratio(x, family, n_effective=1)
    assert single.sr_zero == 0.0
    assert single.dsr == single.psr_naive
    assert single.psr_naive == probabilistic_sharpe_ratio(x)


def test_dsr_no_skill_variance_is_one_over_t_minus_one() -> None:
    x = SERIES["normal"]
    result = deflated_sharpe_ratio(x, [0.1], n_effective=5, trial_variance="no_skill")
    assert result.var_sharpe == pytest.approx(1.0 / (x.size - 1), rel=1e-15)
    assert result.sr_zero == pytest.approx(expected_max_sharpe(5, 1.0 / (x.size - 1)), rel=1e-15)


def test_dsr_single_trial_family_has_zero_variance() -> None:
    result = deflated_sharpe_ratio(SERIES["normal"], [0.1], n_effective=5)
    assert result.var_sharpe == 0.0
    assert result.sr_zero == 0.0


def test_psr_properties() -> None:
    x = SERIES["normal"]
    sr = float(x.mean() / x.std(ddof=1))
    assert probabilistic_sharpe_ratio(x, sr) == pytest.approx(0.5, abs=1e-15)
    levels = [probabilistic_sharpe_ratio(x, s) for s in (-0.1, 0.0, 0.05, 0.1, 0.2)]
    assert levels == sorted(levels, reverse=True)
    # PSR is invariant to a positive rescaling of the returns.
    assert probabilistic_sharpe_ratio(x * 7.0) == pytest.approx(
        probabilistic_sharpe_ratio(x), rel=1e-12
    )


def test_minimum_track_record_decreases_with_gap() -> None:
    x = SERIES["student3"]
    lengths = [minimum_track_record_length(x, s, 0.05) for s in (0.0, -0.05, -0.1)]
    assert lengths == sorted(lengths, reverse=True)
    assert minimum_track_record_length(x, 0.0, 0.01) > minimum_track_record_length(x, 0.0, 0.1)


def test_psr_reaches_target_at_minimum_track_record_length() -> None:
    """With moments held fixed, n = MinTRL gives Phi^-1(PSR) = z_{1-alpha}."""
    from scipy.stats import norm

    x = SERIES["normal"]
    sr = float(x.mean() / x.std(ddof=1))
    d = x - x.mean()
    s = d.std()
    factor = 1 - np.mean(d**3) / s**3 * sr + (np.mean(d**4) / s**4 - 1) / 4 * sr**2
    n_min = minimum_track_record_length(x, 0.0, 0.05)
    z = sr * math.sqrt(n_min - 1) / math.sqrt(factor)
    assert z == pytest.approx(norm.ppf(0.95), rel=1e-12)


def test_se_scales_with_sqrt_periods_per_year() -> None:
    x = SERIES["skew_pos"]
    a = sharpe_se_mertens(x, periods_per_year=1)
    b = sharpe_se_mertens(x, periods_per_year=12)
    assert b.sharpe == pytest.approx(a.sharpe * math.sqrt(12), rel=1e-14)
    assert b.se == pytest.approx(a.se * math.sqrt(12), rel=1e-14)


def test_se_matches_iid_normal_limit() -> None:
    """Gaussian returns give sqrt((1 + SR^2 / 2) / (n - 1)) up to moment sampling error."""
    x = np.random.default_rng(1).normal(0.0, 1.0, 200_000)
    result = sharpe_se_mertens(x, periods_per_year=1)
    assert result.se == pytest.approx(
        math.sqrt((1 + result.sharpe**2 / 2) / (x.size - 1)), rel=5e-2
    )


def test_input_is_not_modified() -> None:
    x = SERIES["normal"].copy()
    probabilistic_sharpe_ratio(x)
    sharpe_se_mertens(x, periods_per_year=12)
    np.testing.assert_array_equal(x, SERIES["normal"])


def test_minimum_observations_boundary() -> None:
    x = np.random.default_rng(2).normal(0.01, 0.02, 10)
    assert 0.0 <= probabilistic_sharpe_ratio(x) <= 1.0
    with pytest.raises(ValueError, match="at least 10"):
        probabilistic_sharpe_ratio(x[:9])


@pytest.mark.parametrize(
    "bad",
    [np.full(20, 0.01), np.array([]), np.ones((5, 4)), np.r_[np.arange(19) * 0.01, np.nan]],
)
def test_invalid_returns(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(bad)
    with pytest.raises(ValueError):
        sharpe_se_mertens(bad, periods_per_year=12)
    with pytest.raises(ValueError):
        minimum_track_record_length(bad, 0.0, 0.05)


def test_mertens_and_trl_argument_validation() -> None:
    x = SERIES["normal"]
    for ppy in (0, -1):
        with pytest.raises(ValueError, match="periods_per_year"):
            sharpe_se_mertens(x, periods_per_year=ppy)
    for alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            minimum_track_record_length(x, 0.0, alpha)
    sr = float(x.mean() / x.std(ddof=1))
    with pytest.raises(ValueError, match="equals sr_star"):
        minimum_track_record_length(x, sr, 0.05)


def test_dsr_invalid_arguments() -> None:
    x = SERIES["normal"]
    with pytest.raises(ValueError, match="trial_variance"):
        deflated_sharpe_ratio(x, [0.1, 0.2], n_effective=3, trial_variance="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 1"):
        deflated_sharpe_ratio(x, [], n_effective=3)
    with pytest.raises(ValueError, match="finite"):
        deflated_sharpe_ratio(x, [0.1, math.nan], n_effective=3)
    with pytest.raises(ValueError, match="n_effective"):
        deflated_sharpe_ratio(x, [0.1, 0.2], n_effective=math.inf)
    with pytest.raises(ValueError, match="at least 10"):
        deflated_sharpe_ratio(x[:5], [0.1, 0.2], n_effective=3)
    with pytest.raises(ValueError, match="at least 2"):
        deflated_sharpe_ratio(x[:1], [0.1], n_effective=3, trial_variance="no_skill")
