"""Every public result class and exception survives a pickle round trip with all its fields.

Consumers checkpoint quantcore objects, and a fit that did not converge is kept as its
`ConvergenceError`. The test walks the classes each package exports, so a class added later is
covered without being listed here. A new exception class must be given a sample instance in
`_EXCEPTIONS`, or the walk fails.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pickle
import pkgutil
from typing import Any

import numpy as np
import pytest

import quantcore
from quantcore.core import CIRNoMeanReversionError
from quantcore.risk import ConvergenceError


def _public_classes() -> dict[str, type]:
    classes: dict[str, type] = {}
    for info in pkgutil.walk_packages(quantcore.__path__, prefix="quantcore."):
        module = importlib.import_module(info.name)
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            if inspect.isclass(value):
                classes[f"{value.__module__}.{value.__qualname__}"] = value
    return classes


def _sample_field(annotation: Any, position: int) -> Any:
    text = str(annotation)
    if "NDArray" in text:
        return np.arange(3.0) / 7.0 + position
    if text in ("int", "<class 'int'>"):
        return 3 + position
    if text in ("bool", "<class 'bool'>"):
        return True
    if "float" in text:
        return 0.1 * (position + 1) / 3.0
    raise AssertionError(f"no sample value for a field annotated {annotation!r}")


def _sample_dataclass(cls: type) -> Any:
    fields = dataclasses.fields(cls)
    return cls(**{f.name: _sample_field(f.type, i) for i, f in enumerate(fields)})


def _with_note(error: BaseException) -> BaseException:
    error.add_note("a note added by the caller")
    return error


_EXCEPTIONS: dict[type, BaseException] = {
    ConvergenceError: _with_note(
        ConvergenceError(
            "fit_garch_11: did not converge",
            params=(1e-6, 0.05, 0.9),
            log_likelihood=-1234.5,
            optimizer_message="Iteration limit reached",
        )
    ),
    CIRNoMeanReversionError: _with_note(CIRNoMeanReversionError("kappa <= 0")),
}


def _state(value: Any) -> Any:
    """Everything that defines `value`, as nested plain data for an equality check."""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, BaseException):
        return (
            type(value),
            value.args,
            {k: _state(v) for k, v in vars(value).items()},
            getattr(value, "__notes__", None),
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return (type(value), {f.name: _state(getattr(value, f.name)) for f in fields})
    if isinstance(value, (tuple, list)):
        return tuple(_state(item) for item in value)
    return (type(value), value)


def round_trip_differences(value: Any) -> str | None:
    """None when `value` comes back from pickle equal in every field, else what went wrong."""
    try:
        restored = pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    if _state(restored) != _state(value):
        return f"restored {_state(restored)!r} differs from {_state(value)!r}"
    return None


def test_the_walk_finds_the_known_classes() -> None:
    names = {cls.__name__ for cls in _public_classes().values()}
    assert {"ConvergenceError", "CIRParams", "FactorLoadingsResult", "CscvPbo"} <= names
    assert len(names) >= 13


@pytest.mark.parametrize("qualname", sorted(_public_classes()))
def test_public_class_round_trips_through_pickle(qualname: str) -> None:
    cls = _public_classes()[qualname]
    if issubclass(cls, BaseException):
        assert cls in _EXCEPTIONS, f"add a sample {cls.__name__} to _EXCEPTIONS"
        value: Any = _EXCEPTIONS[cls]
    else:
        assert dataclasses.is_dataclass(cls), f"{qualname} is neither a dataclass nor an exception"
        value = _sample_dataclass(cls)
    assert round_trip_differences(value) is None


class _RequiredKeywordsError(RuntimeError):
    """The shape of an exception that does not unpickle: required arguments, no `__reduce__`."""

    def __init__(self, message: str, *, params: tuple[float, ...]) -> None:
        super().__init__(message)
        self.params = params


class _DroppedAttributeError(RuntimeError):
    def __init__(self, message: str, extra: float = 0.0) -> None:
        super().__init__(message)
        self.extra = extra

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), self.args)


@pytest.mark.parametrize(
    "broken",
    [
        _RequiredKeywordsError("no fit", params=(1.0, 2.0)),
        _DroppedAttributeError("no fit", extra=2.5),
    ],
    ids=["required keywords", "attribute dropped"],
)
def test_round_trip_check_reports_a_broken_class(broken: BaseException) -> None:
    assert round_trip_differences(broken) is not None
