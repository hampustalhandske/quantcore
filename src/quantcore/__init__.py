"""quantcore: numerical backend for pricing, SDE simulation, risk, statistics and portfolios."""

from importlib.metadata import PackageNotFoundError, version

from quantcore.errors import UndefinedResultError

try:
    __version__ = version("quantcore")
except PackageNotFoundError:
    # Package is being imported from a source checkout that hasn't been
    # installed (e.g. `pip install -e .` not yet run) -- there is no
    # installed distribution metadata to read the version from.
    __version__ = "0.0.0+unknown"

__all__ = ["UndefinedResultError", "__version__"]
