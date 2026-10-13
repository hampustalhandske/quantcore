"""Conformance targets: one module per quantcore domain.

Each module holds, for every quantity the suite verifies in that domain, three things: the
quantcore adapter (a thin wrapper that calls the quantcore function under test), the oracle
callable (an independent implementation of the same standard definition, usually built on a
reference library), and the `ConformanceTarget` registration that pairs the two under a name, a
tier, an oracle label, a citation and a comparator.

This package imports every target module, so importing `conformance.targets` registers every
target and `conformance.compare.registry()` then lists the whole verified surface.
"""

from conformance.targets import (
    execution,
    filtering,
    performance,
    performance_extra,
    portfolio,
    pricing,
    pricing_extra,
    regime,
    risk,
    risk_models,
    statistics,
    statistics_extra,
    time_series,
)

__all__ = [
    "execution",
    "filtering",
    "performance",
    "performance_extra",
    "portfolio",
    "pricing",
    "pricing_extra",
    "regime",
    "risk",
    "risk_models",
    "statistics",
    "statistics_extra",
    "time_series",
]
