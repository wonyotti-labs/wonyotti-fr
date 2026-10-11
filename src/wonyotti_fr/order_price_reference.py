from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .order_price_history import PRICE_FEATURES
from .position_boost_verification import compare_statistics
from .quiet_management_reference import independent_groups


def verify_execution_prices(executions, events, actions):
    keys = ["time", "order_key", "exectype", "side", "ordtype"]
    raw = executions.loc[executions.symbol.eq("XBTUSD")].copy()
    if (raw.empty or events.empty or raw.execid.isna().any() or raw.execid.duplicated().any()
        or not raw.exectype.isin(["Trade", "Funding"]).all()
        or raw[keys].isna().any().any() or events[keys].isna().any().any()
        or not np.isfinite(raw[["lastqty", "lastpx", "source_row"]]).all().all()
        or raw.lastqty.lt(0).any() or raw.lastqty.mod(1).ne(0).any()
        or float(raw.lastqty.sum()) >= 2**53
        or raw.source_row.lt(2).any() or raw.source_row.mod(1).ne(0).any()):
        raise ValueError("가격 원천 체결의 식별·수량·숫자 오류")
    traded = raw.exectype.eq("Trade")
    if (raw.loc[traded, "lastqty"].le(0).any() or raw.loc[traded, "lastpx"].le(0).any()
        or not raw.loc[traded, "side"].isin(["Buy", "Sell"]).all()):
        raise ValueError("가격 원천의 거래 방향·가격 오류")
    event_keys = pd.MultiIndex.from_frame(events[keys])
    if not event_keys.is_unique:
        raise ValueError("가격 원천 사건의 중복 키")
    positions = event_keys.get_indexer(pd.MultiIndex.from_frame(raw[keys]))
    if (positions < 0).any():
        raise ValueError("가격 원천의 체결·사건 연결 누락")
    count = np.bincount(positions, minlength=len(events))
    quantity = np.bincount(positions, weights=raw.lastqty.to_numpy(), minlength=len(events))
    numerator = np.bincount(positions, weights=(raw.lastqty * raw.lastpx).to_numpy(), minlength=len(events))
    first_row = np.full(len(events), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_row, positions, raw.source_row.to_numpy(np.int64))
    expected_price = np.divide(numerator, quantity, out=np.full(len(events), np.nan), where=quantity > 0)
    assert (count > 0).all()
    np.testing.assert_array_equal(count, events.fills)
    np.testing.assert_array_equal(quantity, events.quantity)
    np.testing.assert_array_equal(first_row, events.source_row)
    np.testing.assert_allclose(expected_price, events.price, rtol=1e-12, atol=1e-12, equal_nan=True)
    pd.testing.assert_frame_equal(events[["time", "source_row"]].reset_index(drop=True),
        events.sort_values(["time", "source_row"], kind="stable")[["time", "source_row"]].reset_index(drop=True), check_exact=True)
    assert len(events) == len(actions)
    for name in ["time", "order_key"]:
        pd.testing.assert_series_equal(events[name].reset_index(drop=True), actions[name].reset_index(drop=True), check_names=False, check_exact=True)
    np.testing.assert_allclose(actions.price, events.price, rtol=0, atol=0, equal_nan=True)
    trade = events.exectype.eq("Trade").to_numpy()
    expected_change = np.where(events.side.eq("Buy"), quantity, -quantity)
    np.testing.assert_array_equal((actions.after_qty - actions.before_qty).to_numpy()[trade], expected_change[trade])
    np.testing.assert_array_equal(actions.action.eq("funding"), ~trade)
    np.testing.assert_array_equal((actions.after_qty - actions.before_qty).to_numpy()[~trade], 0)
    return {"source_executions": len(raw), "events": len(events), "trade_events": int(trade.sum()),
            "all_prices_reaggregated": True, "actions_and_event_order_exact": True}


def verify_order_prices(source, actions, attached, links, ledger):
    pd.testing.assert_frame_equal(attached[source.columns], source, check_exact=True)
    pd.testing.assert_frame_equal(links[["end", "episode_id"]], source[["end", "episode_id"]], check_exact=True)
    seen, events = set(), []
    columns = ["time", "order_key", "action", "episode_id", "price", "before_qty", "after_qty"]
    for position, event in enumerate(actions[columns].itertuples(index=False)):
        if event.action == "funding" or event.order_key in seen:
            continue
        seen.add(event.order_key)
        if event.action in {"increase", "reduce"}:
            events.append((position, event))
    assert ledger.action_row.tolist() == [p for p, _ in events]
    for row, (_, event) in zip(ledger.itertuples(index=False), events, strict=True):
        for name in columns:
            assert getattr(row, name) == getattr(event, name)
    references = np.full((len(source), 2), -1, dtype=np.int64)
    times = np.full((len(source), 2), np.iinfo(np.int64).min, dtype=np.int64)
    prices, values = np.zeros((len(source), 2)), np.zeros((len(source), 4))
    cursor, state = 0, {}
    for i, row in enumerate(source[["end", "episode_id", "close", "direction"]].itertuples(index=False)):
        while cursor < len(events) and events[cursor][1].time < row.end:
            position, event = events[cursor]
            state[(event.episode_id, event.action)] = (position, event)
            cursor += 1
        for j, action in enumerate(["increase", "reduce"]):
            chosen = state.get((row.episode_id, action))
            if chosen is not None:
                position, event = chosen
                assert event.time < row.end and np.sign(event.before_qty) == row.direction
                references[i, j], times[i, j], prices[i, j] = position, event.time.value, event.price
                delta = math.log(row.close) - math.log(event.price)
                values[i, 2*j:2*j+2] = [delta, delta * row.direction]
    for j, action in enumerate(["increase", "reduce"]):
        np.testing.assert_array_equal(links[action + "_action_row"], references[:, j])
        np.testing.assert_array_equal(links[action + "_time"].astype("datetime64[ns, UTC]").array.asi8, times[:, j])
        np.testing.assert_array_equal(links[action + "_price"], prices[:, j])
        np.testing.assert_array_equal(source[f"past_{action}_exists"], (references[:, j] >= 0).astype(float))
    np.testing.assert_allclose(attached[PRICE_FEATURES], values, atol=1e-12, rtol=0)
    return {"rows": len(source), "features_checked": int(values.size), "first_observed_orders": len(events),
            "streaming_references_exact": True, "strict_past_only": True, "original_columns_exact": True}


def verify_price_support(frame, support):
    expected = {}
    for action in ["increase", "reduce"]:
        expected[action] = {}
        for name, mask in independent_groups(frame, action).items():
            part = frame.loc[mask]
            y = part["y_" + action].to_numpy(int)
            result = {"rows": len(y), "positive": int(np.count_nonzero(y)), "references": {}, "features": {}}
            for reference in ["increase", "reduce"]:
                observed = part[f"past_{reference}_exists"].to_numpy() == 1
                result["references"][reference] = {"present": int(np.count_nonzero(observed)),
                    "absent": int(np.count_nonzero(~observed)), "positive_present": int(np.count_nonzero(y[observed])),
                    "positive_absent": int(np.count_nonzero(y[~observed]))}
            for feature in PRICE_FEATURES:
                values = np.sort(part[feature].to_numpy())
                if not len(values):
                    result["features"][feature] = None
                    continue
                percentiles = {}
                for key, q in [("minimum", 0), ("q25", .25), ("median", .5), ("q75", .75), ("maximum", 1)]:
                    index = q * (len(values)-1)
                    lower, upper = int(math.floor(index)), int(math.ceil(index))
                    percentiles[key] = float(values[lower] + (index-lower)*(values[upper]-values[lower]))
                percentiles["zero"] = int(np.count_nonzero(values == 0))
                result["features"][feature] = percentiles
            expected[action][name] = result
    compare_statistics(support, expected)
