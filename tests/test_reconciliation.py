import pandas as pd
import pytest

from wonyotti_fr.reconciliation import reconcile_wallet


def test_noon_funding_belongs_to_next_wallet_posting_date():
    actions = pd.DataFrame({"time": pd.to_datetime(["2020-01-02T11:59:59Z", "2020-01-02T12:00:00Z"], utc=True),
                            "action": ["close", "funding"], "realized_btc": [0.2, 0], "fee_btc": [0.01, -0.03]})
    wallet = pd.DataFrame({"address": ["XBTUSD"], "transacttype": ["RealisedPNL"], "transactstatus": ["Completed"],
                           "currency": ["XBt"], "date": ["2020-01-02"], "amount": [19000000]})
    _, result = reconcile_wallet(actions, wallet)
    assert result["full_history_difference_btc"] == pytest.approx(0.03)
    assert result["outside_wallet_window_btc"] == pytest.approx(0.03)
    assert result["aligned_residual_satoshi"] == pytest.approx(0, abs=1e-8)
    assert result["outside_wallet_window_events"][0]["posting_date"] == "2020-01-03"
    wallet["currency"] = "USDt"
    with pytest.raises(ValueError, match="사토시"):
        reconcile_wallet(actions, wallet)
