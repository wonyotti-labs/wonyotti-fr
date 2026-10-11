from __future__ import annotations

import numpy as np
import pandas as pd

from .event_features import MARKET_FEATURES
from .management_market import FEATURES, REPLACED
from .minute_inputs import MINUTE_FEATURES
from .minute_management import ACTIONS


def verify_market_connection(frame, candidate, link, raw_minute, minute, five):
    times = frame.end.astype("datetime64[ns, UTC]").array.asi8
    source_minute = minute.end.astype("datetime64[ns, UTC]").array.asi8
    source_five = five.end.astype("datetime64[ns, UTC]").array.asi8
    raw_times = raw_minute.end.astype("datetime64[ns, UTC]").array.asi8
    np.testing.assert_array_equal(source_minute, raw_times)
    step = 60 * 10**9
    desired = times // (5 * step) * (5 * step)
    first = np.searchsorted(source_minute, times)
    second = np.searchsorted(source_five, desired)
    found_minute = (first < len(source_minute)) & (source_minute[np.minimum(first, len(source_minute)-1)] == times)
    found_five = (second < len(source_five)) & (source_five[np.minimum(second, len(source_five)-1)] == desired)
    np.testing.assert_array_equal(link.minute_found, found_minute)
    np.testing.assert_array_equal(link.five_found, found_five)
    np.testing.assert_array_equal(link.expected_five_end.astype("datetime64[ns, UTC]").array.asi8, desired)
    for names, table, indexes, valid in [(MARKET_FEATURES, five, second, found_five),
                                        (MINUTE_FEATURES, minute, first, found_minute)]:
        actual = candidate[names].to_numpy()
        expected = np.full_like(actual, np.nan)
        expected[valid] = table.iloc[indexes[valid]][names].to_numpy()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(candidate[["directional_" + n for n in names]],
                                      expected * frame.direction.to_numpy()[:, None])
    close, high, low, volume = [raw_minute[n].to_numpy(float) for n in ["close", "high", "low", "volume"]]
    windows = np.lib.stride_tricks.sliding_window_view(volume, 60)
    for start in range(0, len(frame), 8192):
        slots = np.arange(start, min(start + 8192, len(frame)))
        slots = slots[found_minute[slots]]
        pos = first[slots]
        direct = np.full((len(slots), 4), np.nan)
        for column, delay in [(0, 1), (1, 3)]:
            valid = pos >= delay
            direct[valid, column] = close[pos[valid]] / close[pos[valid] - delay] - 1
        direct[:, 2] = (high[pos] - low[pos]) / close[pos]
        valid = pos >= 60
        denominator = windows[pos[valid] - 60].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            direct[valid, 3] = volume[pos[valid]] / denominator
        direct[~np.isfinite(direct)] = np.nan
        np.testing.assert_allclose(candidate.iloc[slots][MINUTE_FEATURES], direct,
                                   rtol=0, atol=1e-10, equal_nan=True)
    pd.testing.assert_frame_equal(frame.drop(columns=REPLACED), candidate.drop(columns=REPLACED),
                                  check_exact=True)
    expected_support = found_minute & found_five & np.isfinite(candidate[FEATURES]).all(axis=1)
    np.testing.assert_array_equal(link.supported, expected_support)
    reasons = ["missing_minute" if not a else "missing_five" if not b else
               "supported" if c else "nonfinite_market" for a, b, c in
               zip(found_minute, found_five, expected_support, strict=True)]
    assert link.reason.tolist() == reasons
    return {"rows": len(frame), "all_connections_and_directional_values_verified": True,
            "direct_minute_formulas_verified": True, "original_state_and_labels_unchanged": True}


def verify_market_support(frame, candidate, link, summary):
    mask = link.supported.to_numpy()
    for name, selected in [("all", np.ones(len(frame), bool)), ("included", mask), ("excluded", ~mask)]:
        group = summary["groups"][name]
        assert group["rows"] == int(selected.sum())
        for action in ACTIONS:
            assert group["positive"][action] == int(np.count_nonzero(frame[f"y_{action}"].to_numpy()[selected]))
    assert sum(summary["reasons"].values()) == len(frame)
    for name, value in summary["reasons"].items():
        assert value == int(np.count_nonzero(link.reason.to_numpy() == name))
    for name in REPLACED:
        delta = abs(frame[name].to_numpy()[mask] - candidate[name].to_numpy()[mask])
        values = summary["changes"][name]
        assert values["different"] == int((delta != 0).sum())
        if len(delta):
            np.testing.assert_allclose([values["mean_absolute_difference"], values["maximum_absolute_difference"]],
                                       [np.mean(delta), np.max(delta)], rtol=0, atol=1e-12)
        else:
            assert values["mean_absolute_difference"] is None and values["maximum_absolute_difference"] is None
    included = summary["groups"]["included"]
    assert summary["action_support_sufficient"] == all(
        min(n, included["rows"] - n) >= 20 for n in included["positive"].values())
