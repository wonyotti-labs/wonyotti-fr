import numpy as np
import pandas as pd

from wonyotti_fr.features import FEATURES, build_features, position_labels


def test_features_do_not_use_future_prices():
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 600)))
    frame = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=600, freq="15min", tz="UTC"),
                          "open": close, "high": close * 1.01, "low": close * 0.99,
                          "close": close, "volume": 100., "taker_buy_volume": 50.})
    frame["end"] = frame.time + pd.Timedelta(minutes=15)
    left = build_features(frame)
    frame.loc[450:, ["open", "high", "low", "close"]] *= 5
    right = build_features(frame)
    pd.testing.assert_frame_equal(left.loc[:449, FEATURES], right.loc[:449, FEATURES])


def test_label_does_not_forward_fill_beyond_source_coverage():
    times = pd.date_range("2020-01-01", periods=6, freq="15min", tz="UTC")
    frame = pd.DataFrame({"time": times, "end": times + pd.Timedelta(minutes=15)})
    actions = pd.DataFrame({"time": [times[1], times[3]], "after_qty": [100, -100]})
    result = position_labels(frame, actions)
    assert pd.isna(result.iloc[-1].label)
    assert result.iloc[0].label == 1
