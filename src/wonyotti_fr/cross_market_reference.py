from __future__ import annotations

from bisect import bisect_right
from math import ceil, floor

import numpy as np
import pandas as pd

from .position_boost_verification import compare_statistics
from .quiet_management_reference import independent_groups


def verify_cross_market(frame, btc, eth, expanded, links):
    names = ["5m", "15m", "1h", "4h", "1d"]
    offsets = [1, 3, 12, 48, 288]
    btc_times = btc.end.astype("datetime64[ns, UTC]").array.asi8
    eth_times = eth.end.astype("datetime64[ns, UTC]").array.asi8
    np.testing.assert_array_equal(btc_times, eth_times)
    ends = frame.end.astype("datetime64[ns, UTC]").array.asi8
    selected = np.array([bisect_right(btc_times, int(end)) - 1 for end in ends])
    assert (selected >= 288).all()
    age = ends - btc_times[selected]
    assert ((age >= 0) & (age < 300 * 10**9)).all()
    np.testing.assert_array_equal(links.source_row, selected)
    np.testing.assert_array_equal(links.market_end.astype("datetime64[ns, UTC]").array.asi8, btc_times[selected])
    np.testing.assert_array_equal(links.market_age_seconds, age / 10**9)
    pd.testing.assert_frame_equal(links[["end", "episode_id"]], frame[["end", "episode_id"]], check_exact=True)
    extras = []
    b, e = btc.close.to_numpy(float), eth.close.to_numpy(float)
    for name, offset in zip(names, offsets, strict=True):
        prior = selected - offset
        np.testing.assert_array_equal(links["lag_row_" + name], prior)
        np.testing.assert_array_equal(btc_times[selected] - btc_times[prior], np.full(len(frame), offset * 300 * 10**9))
        value = np.log(e[selected] / e[prior]) - np.log(b[selected] / b[prior])
        column = "eth_btc_log_return_" + name
        extras.extend([column, "directional_" + column])
        np.testing.assert_allclose(expanded[column], value, atol=1e-12, rtol=0)
        np.testing.assert_allclose(expanded["directional_" + column], value * frame.direction.to_numpy(), atol=1e-12, rtol=0)
    pd.testing.assert_frame_equal(expanded.drop(columns=extras), frame, check_exact=True)
    return {"rows": len(frame), "input_values_checked": len(frame) * 10,
        "all_original_columns_exact": True, "all_prices_windows_and_latest_causal_times_verified": True}


def verify_cross_support(frame, saved):
    features = ["eth_btc_log_return_" + n for n in ["5m", "15m", "1h", "4h", "1d"]]
    features += ["directional_" + n for n in features.copy()]
    expected = {}
    for action in ["reduce", "increase"]:
        expected[action] = {}
        for state, mask in independent_groups(frame, action).items():
            positions = np.flatnonzero(mask)
            part = frame.iloc[positions]
            y = part["y_" + action].to_numpy(int)
            distribution = {}
            for name in features:
                values = np.sort(part[name].to_numpy())
                result = None
                if len(values):
                    result = {}
                    for key, q in [("minimum", 0.), ("q25", .25), ("median", .5), ("q75", .75), ("maximum", 1.)]:
                        index = (len(values) - 1) * q
                        lo, hi = floor(index), ceil(index)
                        result[key] = float(values[lo] + (index - lo) * (values[hi] - values[lo]))
                distribution[name] = result
            expected[action][state] = {"rows": len(y), "positive": int(sum(y)),
                "negative": int(np.count_nonzero(y == 0)), "features": distribution}
    compare_statistics(saved, expected)
