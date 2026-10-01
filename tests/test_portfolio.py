import pandas as pd
import pytest

from wonyotti_fr.portfolio import original_cost_convention


def test_settlement_currency_and_cost_direction_are_not_guessed_from_ticker():
    frame = pd.DataFrame({"symbol": ["ANYUSDT", "ANYUSDT"], "exectype": ["Trade", "Trade"],
                          "side": ["Buy", "Sell"], "execcost": [100, -120], "settlcurrency": ["XBt", "XBt"]})
    assert not original_cost_convention(frame)
    frame["execcost"] *= -1
    assert original_cost_convention(frame)
    frame["settlcurrency"] = "USDt"
    with pytest.raises(ValueError, match="별도 통화"):
        original_cost_convention(frame)
