from __future__ import annotations

import numpy as np
import pytest

from quantcore.risk.var import conditional_value_at_risk, value_at_risk


def test_var_single_return_raises() -> None:
    returns = np.array([0.01])
    with pytest.raises(ValueError):
        value_at_risk(returns, 0.95)


def test_cvar_single_return_raises() -> None:
    returns = np.array([0.01])
    with pytest.raises(ValueError):
        conditional_value_at_risk(returns, 0.95)
