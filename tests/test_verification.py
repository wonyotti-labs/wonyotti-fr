import pandas as pd

from wonyotti_fr.verification import compare_trade, select_samples


def test_public_direction_is_aggressor_not_maker():
    private = {"execid": "private", "trdmatchid": "match", "time": pd.Timestamp("2020-01-01T00:00:00.000937Z"),
               "lastliquidityind": "AddedLiquidity", "side": "Buy", "symbol": "XBTUSD", "lastqty": 3,
               "lastpx": 10000, "execcost": -30000}
    public = [{"trdMatchID": "match", "timestamp": "2020-01-01T00:00:00Z", "symbol": "XBTUSD", "size": 3,
               "price": 10000, "side": "Sell", "grossValue": 30000}]
    assert compare_trade(private, public)["matched"]
    public[0]["size"] = 4
    assert not compare_trade(private, public)["matched"]
    assert not compare_trade(private, [])["matched"]


def test_sample_rule_does_not_select_for_favorable_outcomes():
    rows = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=7, freq="1D", tz="UTC"),
                         "exectype": "Trade", "symbol": "XBTUSD", "index": range(7)})
    assert select_samples(rows)["index"].to_list() == [0, 3, 6]
