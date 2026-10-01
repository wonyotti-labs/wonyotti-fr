import numpy as np
import pytest

from wonyotti_fr.robustness import block_interval


def test_constant_returns_have_zero_sampling_uncertainty():
    result = block_interval(np.full(365, 0.001))
    expected = 1.001 ** 365.25 - 1
    assert result["annual_return_p025"] == pytest.approx(expected)
    assert result["annual_return_p975"] == pytest.approx(expected)
    assert result["fraction_above_cash"] == 1


def test_resampling_is_deterministic_and_rejects_bankruptcy():
    returns = np.random.default_rng(1).normal(0, 0.01, 365)
    assert block_interval(returns) == block_interval(returns)
    returns[0] = -1
    with pytest.raises(ValueError):
        block_interval(returns)
