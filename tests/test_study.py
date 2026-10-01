import pandas as pd
import pytest

from wonyotti_fr.study import enrich_actions


def test_scale_in_counts_orders_and_uses_prior_inverse_basis():
    times = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
    actions = pd.DataFrame({"time": times, "order_key": ["first", "first", "add"],
                            "action": ["open", "increase", "increase"], "before_qty": [0, 10, 20],
                            "after_qty": [10, 20, 25], "price": [100, 100, 80]})
    events = pd.DataFrame({"time": times, "order_key": actions.order_key, "exectype": "Trade",
                          "quantity": [10, 10, 5], "cost_satoshi": [-10000000, -10000000, -6250000]})
    result = enrich_actions(actions, events)
    assert result.new_increase_order.to_list() == [False, False, True]
    assert result.favorable_move_before.iloc[-1] == pytest.approx(-0.2)
    with pytest.raises(ValueError, match="순서"):
        enrich_actions(actions, events.iloc[::-1].reset_index(drop=True))
