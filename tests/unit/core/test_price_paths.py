"""Tests for the generator-driven price-path simulators in core/price_paths.py.

Every simulator has the signature `(rng, n_paths, n_steps, dt, **params)` and returns prices of
shape `(n_paths, n_steps + 1)` starting at `s0`. Four things are checked here:

- shape, dtype, positivity and the first column;
- argument validation, including that a rejected call draws nothing from the generator;
- the reproducibility contract: for a fixed generator state, output cells and the generator's
  next draw are pinned as exact hexadecimal floats and the whole array by its SHA-256
  (`TestPinnedOutput`);
- the analytical moments of each process at 2,000 paths. The full-size versions, with the bias of
  each estimator worked out, are in `tests/oracle/core/test_price_paths_moments_oracle.py`.
"""

from __future__ import annotations

import hashlib
import math
import platform
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats

from quantcore.core import (
    simulate_exp_ou_price_paths,
    simulate_garch_t_price_paths,
    simulate_gbm_price_paths,
    simulate_heston_price_paths,
    simulate_jump_diffusion_price_paths,
    simulate_regime_switching_price_paths,
)

Simulator = Callable[..., npt.NDArray[np.float64]]

_ALL_SIMULATORS: dict[str, Simulator] = {
    "gbm": simulate_gbm_price_paths,
    "exp_ou": simulate_exp_ou_price_paths,
    "heston": simulate_heston_price_paths,
    "garch_t": simulate_garch_t_price_paths,
    "regime_switching": simulate_regime_switching_price_paths,
    "jump_diffusion": simulate_jump_diffusion_price_paths,
}
_ALL_NAMES = sorted(_ALL_SIMULATORS)

# Every simulator except the exponential OU one builds its output as `s0 * exp(x)` with `x = 0`
# in the first column, which is `s0` exactly. The exponential OU one returns `exp(log(s0))`.
_FIRST_COLUMN_IS_EXACTLY_S0 = sorted(set(_ALL_SIMULATORS) - {"exp_ou"})

_DT = 1.0 / 252.0
_DT_WEEKLY = 1.0 / 52.0


# ---------------------------------------------------------------------------
# Shape, domain and use of the caller's generator
# ---------------------------------------------------------------------------


class TestOutputShapeAndDomain:
    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_shape_dtype_and_layout(self, name: str) -> None:
        paths = _ALL_SIMULATORS[name](np.random.default_rng(0), 7, 20, _DT, s0=123.0)
        assert paths.shape == (7, 21)
        assert paths.dtype == np.float64
        assert paths.flags["C_CONTIGUOUS"]

    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_first_column_is_the_starting_price(self, name: str) -> None:
        paths = _ALL_SIMULATORS[name](np.random.default_rng(0), 7, 20, _DT, s0=123.0)
        assert np.allclose(paths[:, 0], 123.0, rtol=1e-14, atol=0.0)

    @pytest.mark.parametrize("name", _FIRST_COLUMN_IS_EXACTLY_S0)
    @pytest.mark.parametrize("s0", [123.0, 57.3, 0.0123, 1e8])
    def test_first_column_is_exactly_s0(self, name: str, s0: float) -> None:
        paths = _ALL_SIMULATORS[name](np.random.default_rng(0), 4, 3, _DT, s0=s0)
        np.testing.assert_array_equal(paths[:, 0], np.full(4, s0))

    @pytest.mark.parametrize("s0", [100.0, 123.0, 57.3, 0.0123, 1e-8, 1e8])
    def test_exp_ou_first_column_is_exp_of_log_s0(self, s0: float) -> None:
        # The log-price starts at log(s0) and the whole array is exponentiated, so the first
        # column is exp(log(s0)): the same number in every row, and within the documented
        # relative distance (|ln s0| + 2) * 2.2e-16 of s0 rather than equal to it.
        paths = simulate_exp_ou_price_paths(np.random.default_rng(0), 5, 3, _DT, s0=s0)
        first = paths[:, 0]
        assert np.all(first == first[0])
        bound = (abs(math.log(s0)) + 2.0) * np.finfo(np.float64).eps
        assert abs(float(first[0]) - s0) <= bound * s0

    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_output_is_finite_and_strictly_positive(self, name: str) -> None:
        paths = _ALL_SIMULATORS[name](np.random.default_rng(1), 50, 100, _DT)
        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0.0)

    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_one_path_one_step(self, name: str) -> None:
        paths = _ALL_SIMULATORS[name](np.random.default_rng(2), 1, 1, _DT)
        assert paths.shape == (1, 2)
        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0.0)


class TestGeneratorUse:
    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_same_generator_state_gives_identical_paths(self, name: str) -> None:
        simulate = _ALL_SIMULATORS[name]
        first = simulate(np.random.default_rng(42), 10, 30, _DT)
        second = simulate(np.random.default_rng(42), 10, 30, _DT)
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_different_seeds_give_different_paths(self, name: str) -> None:
        simulate = _ALL_SIMULATORS[name]
        first = simulate(np.random.default_rng(1), 10, 30, _DT)
        second = simulate(np.random.default_rng(2), 10, 30, _DT)
        assert not np.array_equal(first, second)

    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_generator_is_left_in_the_same_state_by_identical_calls(self, name: str) -> None:
        simulate = _ALL_SIMULATORS[name]
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        simulate(rng_a, 10, 30, _DT)
        simulate(rng_b, 10, 30, _DT)
        assert rng_a.bit_generator.state == rng_b.bit_generator.state

    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_second_call_continues_the_callers_stream(self, name: str) -> None:
        # The simulator consumes the generator it is given instead of building its own, so a
        # second call on the same generator sees later variates and returns different paths.
        simulate = _ALL_SIMULATORS[name]
        rng = np.random.default_rng(42)
        state_before = rng.bit_generator.state
        first = simulate(rng, 10, 30, _DT)
        assert rng.bit_generator.state != state_before
        second = simulate(rng, 10, 30, _DT)
        assert not np.array_equal(first, second)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_NAN = float("nan")
_INF = float("inf")

# (simulator, parameter named in the error, keyword arguments that must be rejected)
_INVALID_PARAMETERS: list[tuple[str, str, dict[str, float]]] = [
    *[("gbm", "sigma", {"sigma": bad}) for bad in (0.0, -0.2, _NAN, _INF)],
    *[("exp_ou", "kappa", {"kappa": bad}) for bad in (0.0, -1.0, _NAN, _INF)],
    *[("exp_ou", "sigma", {"sigma": bad}) for bad in (0.0, -0.1, _NAN, _INF)],
    *[("heston", "v0", {"v0": bad}) for bad in (-0.01, _NAN, _INF)],
    *[("heston", "kappa", {"kappa": bad}) for bad in (0.0, -2.0, _NAN, _INF)],
    *[("heston", "theta", {"theta": bad}) for bad in (0.0, -0.04, _NAN, _INF)],
    *[("heston", "xi", {"xi": bad}) for bad in (0.0, -0.3, _NAN, _INF)],
    *[("heston", "rho", {"rho": bad}) for bad in (-1.0000001, 1.0000001, _NAN, _INF)],
    *[("garch_t", "omega", {"omega": bad}) for bad in (0.0, -1e-5, _NAN, _INF)],
    *[("garch_t", "alpha", {"alpha": bad}) for bad in (-0.01, _NAN, _INF)],
    *[("garch_t", "beta", {"beta": bad}) for bad in (-0.01, _NAN, _INF)],
    ("garch_t", r"alpha \+ beta", {"alpha": 0.1, "beta": 0.9}),
    ("garch_t", r"alpha \+ beta", {"alpha": 0.6, "beta": 0.7}),
    *[("garch_t", "nu", {"nu": bad}) for bad in (2.0, 1.0, -5.0, _NAN, _INF)],
    *[("regime_switching", "sigma0", {"sigma0": bad}) for bad in (0.0, -0.15, _NAN, _INF)],
    *[("regime_switching", "sigma1", {"sigma1": bad}) for bad in (0.0, -0.4, _NAN, _INF)],
    *[("regime_switching", "p01", {"p01": bad}) for bad in (0.0, 1.0, -0.1, 1.1, _NAN)],
    *[("regime_switching", "p10", {"p10": bad}) for bad in (0.0, 1.0, -0.1, 1.1, _NAN)],
    *[("jump_diffusion", "sigma", {"sigma": bad}) for bad in (0.0, -0.2, _NAN, _INF)],
    *[("jump_diffusion", "jump_intensity", {"jump_intensity": bad}) for bad in (-1.0, _NAN, _INF)],
    *[("jump_diffusion", "jump_std_log", {"jump_std_log": bad}) for bad in (-0.1, _NAN, _INF)],
]

# (simulator, drift or level parameter with no range restriction other than being finite)
_REAL_VALUED_LEVEL_PARAMETERS: list[tuple[str, str]] = [
    ("gbm", "mu"),
    ("exp_ou", "theta"),
    ("heston", "mu"),
    ("garch_t", "mu"),
    ("regime_switching", "mu0"),
    ("regime_switching", "mu1"),
    ("jump_diffusion", "mu"),
    ("jump_diffusion", "jump_mean_log"),
]

# (simulator, finite keyword arguments whose prices overflow float64)
_OVERFLOWING_CALLS: list[tuple[str, dict[str, float]]] = [
    ("gbm", {"mu": 1e6}),
    ("exp_ou", {"theta": 800.0, "kappa": 500.0}),
    ("heston", {"mu": 1e6}),
    ("garch_t", {"mu": 1e6}),
    ("regime_switching", {"mu0": 1e6}),
    ("jump_diffusion", {"mu": 1e6}),
]


class TestValidation:
    @pytest.mark.parametrize("name", _ALL_NAMES)
    @pytest.mark.parametrize(
        ("size_name", "n_paths", "n_steps"),
        [("n_paths", 0, 10), ("n_paths", -1, 10), ("n_steps", 5, 0), ("n_steps", 5, -1)],
    )
    def test_non_positive_size_raises_naming_it(
        self, name: str, size_name: str, n_paths: int, n_steps: int
    ) -> None:
        with pytest.raises(ValueError, match=size_name):
            _ALL_SIMULATORS[name](np.random.default_rng(0), n_paths, n_steps, _DT)

    @pytest.mark.parametrize("name", _ALL_NAMES)
    @pytest.mark.parametrize("dt", [0.0, -1.0, _NAN, _INF])
    def test_invalid_dt_raises_naming_it(self, name: str, dt: float) -> None:
        with pytest.raises(ValueError, match="dt"):
            _ALL_SIMULATORS[name](np.random.default_rng(0), 5, 10, dt)

    @pytest.mark.parametrize("name", _ALL_NAMES)
    @pytest.mark.parametrize("s0", [0.0, -1.0, _NAN, _INF])
    def test_invalid_s0_raises_naming_it(self, name: str, s0: float) -> None:
        with pytest.raises(ValueError, match="s0"):
            _ALL_SIMULATORS[name](np.random.default_rng(0), 5, 10, _DT, s0=s0)

    @pytest.mark.parametrize(("name", "parameter", "bad_kwargs"), _INVALID_PARAMETERS)
    def test_invalid_process_parameter_raises_naming_it(
        self, name: str, parameter: str, bad_kwargs: dict[str, float]
    ) -> None:
        with pytest.raises(ValueError, match=parameter):
            _ALL_SIMULATORS[name](np.random.default_rng(0), 5, 10, _DT, **bad_kwargs)

    @pytest.mark.parametrize(("name", "parameter", "bad_kwargs"), _INVALID_PARAMETERS)
    def test_rejected_call_draws_nothing_from_the_generator(
        self, name: str, parameter: str, bad_kwargs: dict[str, float]
    ) -> None:
        rng = np.random.default_rng(0)
        state_before = rng.bit_generator.state
        with pytest.raises(ValueError):
            _ALL_SIMULATORS[name](rng, 5, 10, _DT, **bad_kwargs)
        assert rng.bit_generator.state == state_before

    @pytest.mark.parametrize(("name", "parameter"), _REAL_VALUED_LEVEL_PARAMETERS)
    @pytest.mark.parametrize("bad", [_NAN, _INF, -_INF])
    def test_non_finite_level_parameter_raises_before_any_draw(
        self, name: str, parameter: str, bad: float
    ) -> None:
        rng = np.random.default_rng(0)
        state_before = rng.bit_generator.state
        with pytest.raises(ValueError, match=parameter):
            _ALL_SIMULATORS[name](rng, 5, 10, _DT, **{parameter: bad})
        assert rng.bit_generator.state == state_before

    @pytest.mark.parametrize(
        ("name", "boundary_kwargs"),
        [
            ("heston", {"v0": 0.0}),
            ("heston", {"rho": 1.0}),
            ("heston", {"rho": -1.0}),
            ("heston", {"kappa": 0.5, "theta": 0.01, "xi": 1.5}),  # 2*kappa*theta < xi^2
            ("garch_t", {"alpha": 0.0}),
            ("garch_t", {"beta": 0.0}),
            ("garch_t", {"alpha": 0.0, "beta": 0.0}),
            ("jump_diffusion", {"jump_intensity": 0.0}),
            ("jump_diffusion", {"jump_std_log": 0.0, "jump_intensity": 50.0}),
        ],
    )
    def test_boundary_values_are_accepted(
        self, name: str, boundary_kwargs: dict[str, float]
    ) -> None:
        paths = _ALL_SIMULATORS[name](np.random.default_rng(3), 6, 40, _DT, **boundary_kwargs)
        assert paths.shape == (6, 41)
        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0.0)


class TestOverflow:
    @pytest.mark.parametrize(("name", "kwargs"), _OVERFLOWING_CALLS)
    def test_overflowing_prices_raise_floating_point_error(
        self, name: str, kwargs: dict[str, float]
    ) -> None:
        with pytest.raises(FloatingPointError, match="overflow"):
            _ALL_SIMULATORS[name](np.random.default_rng(0), 5, 10, _DT, **kwargs)

    @pytest.mark.parametrize(("name", "kwargs"), _OVERFLOWING_CALLS)
    def test_generator_has_been_advanced_when_overflow_is_raised(
        self, name: str, kwargs: dict[str, float]
    ) -> None:
        rng = np.random.default_rng(0)
        state_before = rng.bit_generator.state
        with pytest.raises(FloatingPointError):
            _ALL_SIMULATORS[name](rng, 5, 10, _DT, **kwargs)
        assert rng.bit_generator.state != state_before

    @pytest.mark.parametrize("name", _ALL_NAMES)
    def test_prices_underflowing_to_zero_are_returned(self, name: str) -> None:
        drift = {"mu0": -1e6, "mu1": -1e6} if name == "regime_switching" else {"mu": -1e6}
        if name == "exp_ou":
            drift = {"theta": -800.0, "kappa": 500.0}
        paths = _ALL_SIMULATORS[name](np.random.default_rng(0), 3, 10, _DT, **drift)
        assert np.all(np.isfinite(paths))
        assert np.all(paths[:, -1] == 0.0)


# ---------------------------------------------------------------------------
# Conventions that follow from the definitions and hold exactly
# ---------------------------------------------------------------------------


class TestConventions:
    def test_exp_ou_default_theta_is_log_s0(self) -> None:
        default = simulate_exp_ou_price_paths(np.random.default_rng(5), 4, 50, _DT, s0=57.3)
        explicit = simulate_exp_ou_price_paths(
            np.random.default_rng(5), 4, 50, _DT, s0=57.3, theta=math.log(57.3)
        )
        np.testing.assert_array_equal(default, explicit)

    def test_garch_t_dt_scales_the_drift_only(self) -> None:
        # omega, alpha and beta are per-step quantities: with zero drift the step length does
        # not enter the recursion at all, so the paths are the same for any dt.
        daily = simulate_garch_t_price_paths(np.random.default_rng(6), 4, 50, _DT, mu=0.0)
        weekly = simulate_garch_t_price_paths(np.random.default_rng(6), 4, 50, 1.0 / 52.0, mu=0.0)
        np.testing.assert_array_equal(daily, weekly)

        drifted = simulate_garch_t_price_paths(np.random.default_rng(6), 4, 50, _DT, mu=0.5)
        steps = np.arange(51)
        np.testing.assert_allclose(drifted, daily * np.exp(0.5 * _DT * steps), rtol=1e-12)

    def test_gbm_block_of_normals_is_laid_out_one_row_per_path(self) -> None:
        # One (n_paths, n_steps) block is drawn, so path 0 uses the first n_steps normals
        # whatever the number of paths: it equals the single path of a one-path call.
        one = simulate_gbm_price_paths(np.random.default_rng(7), 1, 40, _DT)
        several = simulate_gbm_price_paths(np.random.default_rng(7), 3, 40, _DT)
        np.testing.assert_array_equal(several[0], one[0])

    @pytest.mark.parametrize("name", sorted(set(_ALL_SIMULATORS) - {"gbm"}))
    def test_other_simulators_draw_across_paths_step_by_step(self, name: str) -> None:
        # Each step draws one vector across paths, so with several paths the first path no
        # longer sees the variates a one-path call gives it.
        simulate = _ALL_SIMULATORS[name]
        one = simulate(np.random.default_rng(7), 1, 40, _DT)
        several = simulate(np.random.default_rng(7), 3, 40, _DT)
        assert not np.array_equal(several[0], one[0])

    def test_regime_switching_with_identical_regimes_ignores_the_switch_probabilities(
        self,
    ) -> None:
        # The uniform that decides the switch is drawn every step whatever its outcome, so two
        # calls that differ only in p01 and p10 see the same normals; with identical regimes
        # the state then has no effect on the price.
        common = {"mu0": 0.07, "sigma0": 0.2, "mu1": 0.07, "sigma1": 0.2}
        rare = simulate_regime_switching_price_paths(
            np.random.default_rng(8), 3, 60, _DT, p01=0.01, p10=0.02, **common
        )
        frequent = simulate_regime_switching_price_paths(
            np.random.default_rng(8), 3, 60, _DT, p01=0.6, p10=0.7, **common
        )
        np.testing.assert_array_equal(rare, frequent)

    def test_jump_diffusion_draws_the_jump_normal_whether_or_not_a_jump_occurs(self) -> None:
        # With zero intensity no path ever jumps, yet the generator must end where it ends when
        # every step draws a diffusion normal and a jump-size normal for each path (a Poisson
        # variate with zero mean consumes nothing).
        n_paths, n_steps = 3, 25
        rng = np.random.default_rng(9)
        simulate_jump_diffusion_price_paths(rng, n_paths, n_steps, _DT, jump_intensity=0.0)
        expected = np.random.default_rng(9)
        expected.standard_normal(2 * n_paths * n_steps)
        assert rng.bit_generator.state == expected.bit_generator.state


# ---------------------------------------------------------------------------
# Pinned output
# ---------------------------------------------------------------------------

_PIN_SEED = 12345
_PIN_N_PATHS = 3
_PIN_N_STEPS = 250
_PIN_CELLS = ((0, 0), (0, 1), (1, 1), (0, 125), (1, 250), (2, 250))

# The pinned values were recorded with NumPy 2.5.3 on macOS arm64. NumPy does not promise
# bit-identical floating-point results across platforms, and at least one sampler used here
# depends on the build: on arm64 `Generator.normal(loc, scale)` evaluates `loc + scale * z` with a
# fused multiply-add (one rounding), which a build without that instruction cannot reproduce. The
# arrays are therefore required to match bit for bit only on the platform they were recorded on.
# Elsewhere the cells are held to 1e-12 relative, which last-bit differences stay far inside and
# which any change in the draw order, the assignment of variates or the scheme exceeds by many
# orders of magnitude; the whole-array digest cannot be compared there and is skipped.
_PINS_ARE_BIT_EXACT_HERE = platform.system() == "Darwin" and platform.machine() == "arm64"
_PIN_RTOL_ELSEWHERE = 1e-12


@dataclass(frozen=True)
class _Pin:
    """One pinned call: the simulator, its step length and keyword arguments, and its output.

    Attributes:
        simulate: The simulator under test.
        dt: Step length passed to it.
        kwargs: Keyword arguments passed to it.
        cells: The output at `_PIN_CELLS`, as `float.hex()` strings.
        sha256: SHA-256 of the whole output array (row-major, little-endian float64 bytes).
        next_uniform: The generator's next `rng.random()` after the call, as a `float.hex()`
            string.
    """

    simulate: Simulator
    dt: float
    kwargs: dict[str, float]
    cells: tuple[str, ...]
    sha256: str
    next_uniform: str


_PINS: dict[str, _Pin] = {
    "gbm-daily": _Pin(
        simulate_gbm_price_paths,
        _DT,
        {"s0": 57.3, "mu": 0.13, "sigma": 0.37},
        (
            "0x1.ca66666666666p+5",
            "0x1.bb8b9f7bf06e1p+5",
            "0x1.d69b05a17e3efp+5",
            "0x1.f12904e06b93bp+5",
            "0x1.7b4547b657be9p+5",
            "0x1.02202e356b26cp+6",
        ),
        "374de5eb5e9ebed09d24df3ccc474950d0bf5e11ea46f4af2dd83be16a3b3782",
        "0x1.e9dfc5f1669b7p-1",
    ),
    "gbm-weekly": _Pin(
        simulate_gbm_price_paths,
        _DT_WEEKLY,
        {"mu": -0.4},
        (
            "0x1.9000000000000p+6",
            "0x1.7d6b32718232bp+6",
            "0x1.9944384b854a6p+6",
            "0x1.35977d1399c50p+5",
            "0x1.3b4d421a49cd9p+3",
            "0x1.c710e5d8ef0d4p+3",
        ),
        "d0b3345c5a6396096a4855d9a7d67a1cca0d445a054e312841b763d56dc5c55e",
        "0x1.e9dfc5f1669b7p-1",
    ),
    "exp_ou-default-theta-daily": _Pin(
        simulate_exp_ou_price_paths,
        _DT,
        {"s0": 57.3, "kappa": 5.0, "sigma": 0.3},
        (
            "0x1.ca66666666667p+5",
            "0x1.be591d5759235p+5",
            "0x1.d55e7bfd81cddp+5",
            "0x1.c8e44fbe113c9p+5",
            "0x1.cedfa7c64b532p+5",
            "0x1.ebbc416518da1p+5",
        ),
        "a54c4c29e9dcf9bbd7cfc041746e24200b76fd08732e2c98ab7642de019a7c43",
        "0x1.e9dfc5f1669b7p-1",
    ),
    "exp_ou-given-theta-weekly": _Pin(
        simulate_exp_ou_price_paths,
        _DT_WEEKLY,
        {"s0": 57.3, "kappa": 0.37, "theta": 4.2, "sigma": 0.9},
        (
            "0x1.ca66666666667p+5",
            "0x1.806c55714c06fp+5",
            "0x1.0c7ed521c855dp+6",
            "0x1.85c5b9c34c837p+5",
            "0x1.a9a36431510cdp+6",
            "0x1.2b09e6d160e3cp+5",
        ),
        "5cfdc71637be6518c85d9901626405e399179872d62382881e5d9158d317780b",
        "0x1.e9dfc5f1669b7p-1",
    ),
    "heston-feller-violated-daily": _Pin(
        simulate_heston_price_paths,
        _DT,
        {"s0": 80.0, "mu": 0.11, "v0": 0.09, "kappa": 1.3, "theta": 0.05, "xi": 0.9, "rho": -0.8},
        (
            "0x1.4000000000000p+6",
            "0x1.3795ae051f519p+6",
            "0x1.47d19fdae5e4bp+6",
            "0x1.632a3bff5e8c2p+6",
            "0x1.a253f1e5d69e8p+6",
            "0x1.588aa9af1759dp+6",
        ),
        "b1c8bdd0cf21dc7d00111861f7fe03ce67ee2fb8018ab3fcb1e8cae3a290b3d2",
        "0x1.f4c99a86c4f28p-2",
    ),
    "heston-zero-initial-variance-weekly": _Pin(
        simulate_heston_price_paths,
        _DT_WEEKLY,
        {"v0": 0.0},
        (
            "0x1.9000000000000p+6",
            "0x1.90628246accf4p+6",
            "0x1.90628246accf4p+6",
            "0x1.afd0710649ca4p+6",
            "0x1.0393d2793b02ap+7",
            "0x1.d869e9279901bp+6",
        ),
        "657d09d04d4609bfe5b77bc03ffcc6d594a6a4500081c5f743261537b3a18651",
        "0x1.f4c99a86c4f28p-2",
    ),
    "garch_t-daily": _Pin(
        simulate_garch_t_price_paths,
        _DT,
        {"s0": 31.0, "mu": 0.2, "omega": 5e-05, "alpha": 0.12, "beta": 0.85, "nu": 3.5},
        (
            "0x1.f000000000000p+4",
            "0x1.e33e9c343e542p+4",
            "0x1.ec76b46355dc5p+4",
            "0x1.2d2fe28109f2dp+5",
            "0x1.c67ae8374f994p+4",
            "0x1.0df3fe81dc8ffp+4",
        ),
        "8ce4f4b9b3976b046f634eafa8e5adf1d9be20331a70b058a5b47a56902724fd",
        "0x1.6baf10418e6d0p-2",
    ),
    "garch_t-weekly": _Pin(
        simulate_garch_t_price_paths,
        _DT_WEEKLY,
        {"mu": -0.1, "alpha": 0.1, "beta": 0.2},
        (
            "0x1.9000000000000p+6",
            "0x1.8e0201c6c4829p+6",
            "0x1.8ee5d2511c79bp+6",
            "0x1.2ec27d15636edp+6",
            "0x1.0cb630522ff9ep+6",
            "0x1.e032208719f5fp+5",
        ),
        "053ed984e75d9dcd8edfce8261c3b178eac2401e351cceb6d1645fac33071dbc",
        "0x1.0589e3137f5bep-1",
    ),
    "regime_switching-daily": _Pin(
        simulate_regime_switching_price_paths,
        _DT,
        {
            "s0": 250.0,
            "mu0": 0.3,
            "sigma0": 0.1,
            "mu1": -0.6,
            "sigma1": 0.8,
            "p01": 0.2,
            "p10": 0.4,
        },
        (
            "0x1.f400000000000p+7",
            "0x1.f3c4e09512665p+7",
            "0x1.f4591f039a1c5p+7",
            "0x1.502ac1fd3108ap+8",
            "0x1.a3b8636feef6ap+8",
            "0x1.906603b60b72ap+7",
        ),
        "82a34c008d9aef145f89b40e3c8ec33641db5984504fa2b5f90c1b8bf269fb69",
        "0x1.7cc02add02accp-1",
    ),
    "regime_switching-defaults-weekly": _Pin(
        simulate_regime_switching_price_paths,
        _DT_WEEKLY,
        {},
        (
            "0x1.9000000000000p+6",
            "0x1.8e87696a7f711p+6",
            "0x1.900e490a1251cp+6",
            "0x1.0cce4a680ce9ep+6",
            "0x1.0c82d80ac2fe0p+8",
            "0x1.a8c479239b145p+6",
        ),
        "82ee1ec9ebde7a1f095fc949172ccbe68ecec3be4b064640039c8f3cea60f889",
        "0x1.7cc02add02accp-1",
    ),
    "jump_diffusion-daily": _Pin(
        simulate_jump_diffusion_price_paths,
        _DT,
        {
            "s0": 12.0,
            "mu": 0.07,
            "sigma": 0.25,
            "jump_intensity": 40.0,
            "jump_mean_log": -0.02,
            "jump_std_log": 0.07,
        },
        (
            "0x1.8000000000000p+3",
            "0x1.7894dfd01b60ep+3",
            "0x1.88dcca75ac4a6p+3",
            "0x1.184492c29a9f5p+2",
            "0x1.d7d7124782b3ap+2",
            "0x1.049195b681e4ap+4",
        ),
        "991b5239aad03c3f1892c74e1338addea998066558126d3a75aeb0e9f341ab09",
        "0x1.9e34141a4d637p-1",
    ),
    "jump_diffusion-many-jumps-weekly": _Pin(
        simulate_jump_diffusion_price_paths,
        _DT_WEEKLY,
        {"jump_intensity": 600.0, "jump_mean_log": -0.001, "jump_std_log": 0.004},
        (
            "0x1.9000000000000p+6",
            "0x1.75775a1e4e37fp+6",
            "0x1.ab9d88a24d390p+6",
            "0x1.b7a7e20dd4ba1p+6",
            "0x1.18c674d4eb826p+6",
            "0x1.c4a3bbdc4297cp+7",
        ),
        "e785f8dc097f56e272beaab7fdea0f14af3f56b0ccda308e2ee81710d5e634ce",
        "0x1.697c63c5a1fbcp-2",
    ),
}


def _simulate_pinned(pin: _Pin) -> tuple[npt.NDArray[np.float64], np.random.Generator]:
    rng = np.random.default_rng(_PIN_SEED)
    return pin.simulate(rng, _PIN_N_PATHS, _PIN_N_STEPS, pin.dt, **pin.kwargs), rng


class TestPinnedOutput:
    """Output and generator position for `np.random.default_rng(12345)`, 3 paths of 250 steps.

    A downstream pipeline hashes bars built from these arrays and keeps drawing from the same
    generator afterwards, so both the array and the amount of the stream consumed are part of
    each simulator's contract. Three things are pinned per case. The digest covers every
    element, so a regrouped product or sum, which moves scattered elements by one unit in the
    last place, cannot hide between the cells. The six cells (start, middle and end of the
    paths) show a reader what moved and by how much. The next uniform draw fixes how much of
    the stream the call consumed.
    """

    @pytest.mark.skipif(
        not _PINS_ARE_BIT_EXACT_HERE,
        reason="bits were recorded on macOS arm64; NumPy does not round identically elsewhere",
    )
    @pytest.mark.parametrize("case", sorted(_PINS))
    def test_whole_array_matches_the_pinned_digest(self, case: str) -> None:
        pin = _PINS[case]
        paths, _ = _simulate_pinned(pin)
        assert hashlib.sha256(paths.astype("<f8").tobytes()).hexdigest() == pin.sha256

    @pytest.mark.parametrize("case", sorted(_PINS))
    def test_cells_match_the_pinned_values(self, case: str) -> None:
        pin = _PINS[case]
        paths, _ = _simulate_pinned(pin)
        actual = [float(paths[path, step]) for path, step in _PIN_CELLS]
        if _PINS_ARE_BIT_EXACT_HERE:
            assert [value.hex() for value in actual] == list(pin.cells)
        else:
            expected = [float.fromhex(cell) for cell in pin.cells]
            np.testing.assert_allclose(actual, expected, rtol=_PIN_RTOL_ELSEWHERE, atol=0.0)

    @pytest.mark.parametrize("case", sorted(_PINS))
    def test_generator_next_draw_matches_the_pinned_value(self, case: str) -> None:
        # A uniform draw is an integer scaled by a power of two, so this pin is exact on every
        # platform.
        pin = _PINS[case]
        _, rng = _simulate_pinned(pin)
        assert float(rng.random()).hex() == pin.next_uniform


# ---------------------------------------------------------------------------
# Analytical moments, k = 4 Monte Carlo standard errors, 2,000 paths
# ---------------------------------------------------------------------------

_K = 4.0
_N_PATHS = 2_000
_N_BATCHES = 20


def _batched_mean_se(
    per_path_values: npt.NDArray[np.float64], n_batches: int
) -> tuple[float, float]:
    """Grand mean of one statistic per path and its standard error by batch means.

    The paths are split into `n_batches` equal batches and each batch is averaged; the standard
    error is `std(batch_means, ddof=1) / sqrt(n_batches)`. Batch means are averages of
    independent paths, so no assumption about the distribution of the per-path statistic is
    needed.
    """
    batch_size = per_path_values.size // n_batches
    batches = per_path_values[: batch_size * n_batches].reshape(n_batches, batch_size)
    batch_means = batches.mean(axis=1)
    return float(per_path_values.mean()), float(batch_means.std(ddof=1) / np.sqrt(n_batches))


def _batched_ratio_se(
    per_path_numerator: npt.NDArray[np.float64],
    per_path_denominator: npt.NDArray[np.float64],
    n_batches: int,
) -> tuple[float, float]:
    """Ratio of pooled sums and its standard error by batch means.

    The estimate is `sum(numerator) / sum(denominator)` over all paths (averaging per-path
    ratios would be biased); the standard error is that of the same ratio computed per batch.
    """
    batch_size = per_path_numerator.size // n_batches
    used = batch_size * n_batches
    numerators = per_path_numerator[:used].reshape(n_batches, batch_size).sum(axis=1)
    denominators = per_path_denominator[:used].reshape(n_batches, batch_size).sum(axis=1)
    batch_ratios = numerators / denominators
    estimate = float(per_path_numerator.sum() / per_path_denominator.sum())
    return estimate, float(batch_ratios.std(ddof=1) / np.sqrt(n_batches))


class TestAnalyticalMoments:
    def test_gbm_log_return_mean_and_variance(self) -> None:
        """Per-step log-return: mean (mu - sigma^2/2)*dt, variance sigma^2*dt."""
        mu, sigma = 0.08, 0.25
        paths = simulate_gbm_price_paths(
            np.random.default_rng(7), _N_PATHS, 252, _DT, mu=mu, sigma=sigma
        )
        log_returns = np.diff(np.log(paths), axis=1)

        mean_hat, mean_se = _batched_mean_se(log_returns.mean(axis=1), _N_BATCHES)
        assert abs(mean_hat - (mu - 0.5 * sigma**2) * _DT) <= _K * mean_se

        var_hat, var_se = _batched_mean_se(log_returns.var(axis=1, ddof=1), _N_BATCHES)
        assert abs(var_hat - sigma**2 * _DT) <= _K * var_se

    def test_exp_ou_stationary_mean_variance_and_lag_one_autocorrelation(self) -> None:
        """log(price): stationary mean theta, variance sigma^2/(2*kappa), lag-1 exp(-kappa*dt).

        The first 200 of 2,000 steps are dropped. Variance and autocorrelation are measured as
        deviations from the known theta rather than from each path's own sample mean, which for
        a strongly autocorrelated series follows local excursions and biases both.
        """
        kappa, sigma, s0 = 5.0, 0.3, 100.0
        theta = math.log(s0)
        paths = simulate_exp_ou_price_paths(
            np.random.default_rng(11),
            _N_PATHS,
            2_000,
            _DT,
            s0=s0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
        )
        after_burn_in = np.log(paths)[:, 200:]
        deviation = after_burn_in - theta

        mean_hat, mean_se = _batched_mean_se(after_burn_in.mean(axis=1), _N_BATCHES)
        assert abs(mean_hat - theta) <= _K * mean_se

        var_hat, var_se = _batched_mean_se((deviation**2).mean(axis=1), _N_BATCHES)
        assert abs(var_hat - sigma**2 / (2.0 * kappa)) <= _K * var_se

        lag_one_hat, lag_one_se = _batched_ratio_se(
            (deviation[:, :-1] * deviation[:, 1:]).sum(axis=1),
            (deviation[:, :-1] ** 2).sum(axis=1),
            _N_BATCHES,
        )
        assert abs(lag_one_hat - math.exp(-kappa * _DT)) <= _K * lag_one_se

    def test_heston_average_variance_and_mean_log_return(self) -> None:
        """Expected average variance A = theta + (v0 - theta)*(1 - exp(-kappa*T))/(kappa*T);
        E[log(S_T/s0)] = mu*T - A*T/2. A is estimated by the realised variance per unit time.
        """
        mu, v0, kappa, theta, xi = 0.05, 0.08, 2.0, 0.04, 0.3
        n_steps = 252
        horizon = n_steps * _DT
        paths = simulate_heston_price_paths(
            np.random.default_rng(13),
            _N_PATHS,
            n_steps,
            _DT,
            mu=mu,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
        )
        log_returns = np.diff(np.log(paths), axis=1)
        expected_average_variance = theta + (v0 - theta) * (1.0 - math.exp(-kappa * horizon)) / (
            kappa * horizon
        )

        var_hat, var_se = _batched_mean_se((log_returns**2).sum(axis=1) / horizon, _N_BATCHES)
        assert abs(var_hat - expected_average_variance) <= _K * var_se

        mean_hat, mean_se = _batched_mean_se(np.log(paths[:, -1] / paths[:, 0]), _N_BATCHES)
        expected_mean = mu * horizon - 0.5 * expected_average_variance * horizon
        assert abs(mean_hat - expected_mean) <= _K * mean_se

    def test_garch_t_unconditional_variance_and_heavy_tails(self) -> None:
        """Log-return variance omega/(1 - alpha - beta); excess kurtosis above zero."""
        omega, alpha, beta, nu = 5e-5, 0.05, 0.90, 6.0
        paths = simulate_garch_t_price_paths(
            np.random.default_rng(17),
            _N_PATHS,
            252,
            _DT,
            omega=omega,
            alpha=alpha,
            beta=beta,
            nu=nu,
        )
        log_returns = np.diff(np.log(paths), axis=1)[:, 20:]

        var_hat, var_se = _batched_mean_se(log_returns.var(axis=1, ddof=1), _N_BATCHES)
        assert abs(var_hat - omega / (1.0 - alpha - beta)) <= _K * var_se

        assert stats.kurtosis(log_returns.ravel(), fisher=True, bias=False) > 0.0

    def test_regime_switching_stationary_variance(self) -> None:
        """Stationary per-step log-return variance of the two-regime mixture.

        With pi0 = p10/(p01 + p10), pi1 = p01/(p01 + p10) and regime means
        m_s = (mu_s - sigma_s^2/2)*dt, the variance is
        pi0*sigma0^2*dt + pi1*sigma1^2*dt + pi0*pi1*(m0 - m1)^2. The first 50 of 500 steps are
        dropped (the chain mixes in about 1/(p01 + p10) steps).
        """
        mu0, sigma0, mu1, sigma1, p01, p10 = 0.10, 0.15, -0.20, 0.40, 0.02, 0.10
        paths = simulate_regime_switching_price_paths(
            np.random.default_rng(19),
            _N_PATHS,
            500,
            _DT,
            mu0=mu0,
            sigma0=sigma0,
            mu1=mu1,
            sigma1=sigma1,
            p01=p01,
            p10=p10,
        )
        log_returns = np.diff(np.log(paths), axis=1)[:, 50:]
        var_hat, var_se = _batched_mean_se(log_returns.var(axis=1, ddof=1), _N_BATCHES)

        pi0 = p10 / (p01 + p10)
        pi1 = p01 / (p01 + p10)
        m0 = (mu0 - 0.5 * sigma0**2) * _DT
        m1 = (mu1 - 0.5 * sigma1**2) * _DT
        expected_var = pi0 * sigma0**2 * _DT + pi1 * sigma1**2 * _DT + pi0 * pi1 * (m0 - m1) ** 2
        assert abs(var_hat - expected_var) <= _K * var_se

    def test_jump_diffusion_terminal_mean_and_log_return_variance(self) -> None:
        """E[S_T] = s0*exp(mu*T); Var[log(S_T/s0)] = (sigma^2 + lambda*(m^2 + s^2))*T."""
        s0, mu, sigma = 100.0, 0.06, 0.2
        jump_intensity, jump_mean_log, jump_std_log = 1.5, -0.02, 0.1
        n_steps = 252
        horizon = n_steps * _DT
        paths = simulate_jump_diffusion_price_paths(
            np.random.default_rng(23),
            _N_PATHS,
            n_steps,
            _DT,
            s0=s0,
            mu=mu,
            sigma=sigma,
            jump_intensity=jump_intensity,
            jump_mean_log=jump_mean_log,
            jump_std_log=jump_std_log,
        )
        terminal = paths[:, -1]

        mean_hat, mean_se = _batched_mean_se(terminal, _N_BATCHES)
        assert abs(mean_hat - s0 * math.exp(mu * horizon)) <= _K * mean_se

        log_return = np.log(terminal / s0)
        var_hat, var_se = _batched_mean_se(log_return**2 - log_return.mean() ** 2, _N_BATCHES)
        expected_var = (sigma**2 + jump_intensity * (jump_mean_log**2 + jump_std_log**2)) * horizon
        assert abs(var_hat - expected_var) <= _K * var_se


class TestJumpSizeBeyondFloat64:
    """A mean jump size too large for float64 is rejected before any draw."""

    @pytest.mark.parametrize(
        ("jump_mean_log", "jump_std_log"), [(800.0, 0.0), (0.0, 40.0), (700.0, 5.0)]
    )
    def test_raises_value_error_and_leaves_generator_untouched(
        self, jump_mean_log: float, jump_std_log: float
    ) -> None:
        rng = np.random.default_rng(7)
        state_before = rng.bit_generator.state
        with pytest.raises(ValueError, match="expected jump size"):
            simulate_jump_diffusion_price_paths(
                rng,
                3,
                4,
                1.0 / 252.0,
                jump_intensity=1.0,
                jump_mean_log=jump_mean_log,
                jump_std_log=jump_std_log,
            )
        assert rng.bit_generator.state == state_before


class TestOverflowedLogPrice:
    """A drift or variance term that overflows leaves an infinite log-price, which is an error.

    A finite log-price too negative for float64 is an ordinary underflow and still gives 0.0.
    """

    def test_gbm_with_a_volatility_whose_square_overflows(self) -> None:
        with pytest.raises(FloatingPointError, match="log-price"):
            simulate_gbm_price_paths(np.random.default_rng(1), 3, 4, 1.0 / 252.0, sigma=1e160)

    def test_jump_diffusion_with_an_overflowing_jump_compensation(self) -> None:
        with pytest.raises(FloatingPointError, match="log-price"):
            simulate_jump_diffusion_price_paths(
                np.random.default_rng(1),
                3,
                4,
                1.0 / 252.0,
                jump_intensity=1e10,
                jump_mean_log=700.0,
            )

    def test_finite_log_price_that_underflows_is_returned_as_zero(self) -> None:
        paths = simulate_gbm_price_paths(np.random.default_rng(1), 3, 4, 1.0, mu=-1e6)
        assert np.all(paths[:, 1:] == 0.0)
