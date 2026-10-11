from __future__ import annotations

import numpy as np
import pandas as pd

from .management_flow import EXTRA_FEATURES
from .minute_management import ACTIONS


def direct_flow_table(minute, five):
    result = {}
    volume, buy = minute.volume.to_numpy(float), minute.taker_buy_volume.to_numpy(float)
    raw = np.zeros(len(minute))
    active = volume != 0
    raw[active] = 2*(buy[active]/volume[active])-1
    result["minute"] = raw[:, None]
    volume, buy = five.volume.to_numpy(float), five.taker_buy_volume.to_numpy(float)
    values = np.full((len(five), 5), np.nan)
    for j, size in enumerate([1, 3, 12, 48, 288]):
        if len(five) < size:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(volume, size)
        bought = np.lib.stride_tricks.sliding_window_view(buy, size)
        for start in range(0, len(windows), 8192):
            stop = min(start+8192, len(windows))
            total = windows[start:stop].sum(axis=1)
            take = bought[start:stop].sum(axis=1)
            value = np.zeros(len(total))
            active = total != 0
            value[active] = 2*(take[active]/total[active])-1
            values[start+size-1:stop+size-1, j] = value
    result["five"] = values
    return result


def verify_flow_connection(frame, expanded, link, minute, five, direct):
    pd.testing.assert_frame_equal(expanded.drop(columns=EXTRA_FEATURES), frame.reset_index(drop=True), check_exact=True)
    times = frame.end.astype("datetime64[ns, UTC]").array.asi8
    step = 60*10**9
    expected = np.full((len(frame), 6), np.nan)
    found = []
    for source, requested, columns, key in [(minute, times, [5], "minute"),
                                           (five, times//(5*step)*(5*step), list(range(5)), "five")]:
        ends = source.end.astype("datetime64[ns, UTC]").array.asi8
        index = np.searchsorted(ends, requested)
        valid = (index < len(ends)) & (ends[np.minimum(index, len(ends)-1)] == requested)
        expected[np.ix_(valid, columns)] = direct[key][index[valid]]
        found.append(valid)
        actual = link["minute_source_end" if key == "minute" else "five_source_end"]
        assert actual.isna().equals(pd.Series(~valid))
        np.testing.assert_array_equal(actual[valid].astype("datetime64[ns, UTC]").array.asi8, requested[valid])
    raw_names = ["taker_imbalance_5m", "taker_imbalance_15m", "taker_imbalance_1h",
                 "taker_imbalance_4h", "taker_imbalance_1d", "minute_taker_imbalance_1m"]
    np.testing.assert_allclose(expanded[raw_names], expected, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(expanded[["directional_"+name for name in raw_names]],
        expected*frame.direction.to_numpy()[:, None], rtol=0, atol=1e-12, equal_nan=True)
    supported = found[0] & found[1] & np.isfinite(expected).all(axis=1)
    np.testing.assert_array_equal(link.supported, supported)
    np.testing.assert_array_equal(link.source_row, np.arange(len(frame)))
    pd.testing.assert_series_equal(link.end, frame.end.reset_index(drop=True), check_exact=True)
    reasons = ["missing_minute" if not a else "missing_five" if not b else
               "supported" if c else "unready_flow" for a, b, c in zip(*found, supported, strict=True)]
    assert link.reason.tolist() == reasons
    return {"rows": len(frame), "all_twelve_values_independently_verified": True,
        "source_times_and_membership_verified": True, "original_columns_and_labels_exact": True}


def verify_flow_support(frame, link, support):
    mask = link.supported.to_numpy()
    for name, selected in [("all", np.ones(len(frame), bool)), ("included", mask), ("excluded", ~mask)]:
        group = support["groups"][name]
        assert group["rows"] == int(np.count_nonzero(selected))
        for action in ACTIONS:
            assert group["positive"][action] == int(np.count_nonzero(frame["y_"+action].to_numpy()[selected]))
    assert support["all_original_rows_supported"] == bool(np.all(mask))
    assert support["action_support_sufficient"] == all(
        min(np.count_nonzero(frame["y_"+a].to_numpy()[mask]),
            np.count_nonzero(1-frame["y_"+a].to_numpy()[mask])) >= 20 for a in ACTIONS)
    assert sum(support["reasons"].values()) == len(frame)
    for reason, count in support["reasons"].items():
        assert count == int(np.count_nonzero(link.reason.to_numpy() == reason))
