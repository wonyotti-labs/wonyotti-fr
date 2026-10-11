from __future__ import annotations

import numpy as np
import pandas as pd

from .event_features import teacher_states
from .first_management import BETAS
from .first_state_models import metrics
from .inventory_study import inventory_states
from .management_rank_research import CONTEXT
from .quiet_management import ACTIONS, groups

BUCKETS = ("0", "1", "2", "3", "4", "5", ">5")
MODELS = ("v127", "v133")


def validate_count_events(actions, events):
    required = {"time", "order_key", "exectype", "side", "quantity", "cost_satoshi"}
    if not required.issubset(events) or actions.empty or events.empty:
        raise ValueError("누적 추가 횟수의 원천 사건 누락")
    inventory_states(actions)
    for frame in [actions, events]:
        if (not isinstance(frame.time.dtype, pd.DatetimeTZDtype) or str(frame.time.dt.tz) != "UTC"
            or frame.time.isna().any() or not frame.time.is_monotonic_increasing):
            raise ValueError("누적 추가 횟수의 사건 시각 오류")
    if (not actions.action.isin(["open", "reverse", "increase", "reduce", "close", "funding"]).all()
        or not events.exectype.isin(["Trade", "Funding"]).all()
        or not np.isfinite(events[["quantity", "cost_satoshi"]]).all().all()
        or events.quantity.mod(1).ne(0).any() or events.quantity.abs().ge(2**53).any()
        or not np.array_equal(actions.action.eq("funding"), events.exectype.eq("Funding"))):
        raise ValueError("누적 추가 횟수의 행동·수량 오류")
    trade = events.exectype.eq("Trade")
    if (events.loc[trade, "quantity"].le(0).any() or not events.loc[trade, "side"].isin(["Buy", "Sell"]).all()
        or events.loc[trade, "order_key"].isna().any() or events.loc[trade, "order_key"].astype(str).str.len().eq(0).any()):
        raise ValueError("누적 추가 횟수의 체결 방향·키 오류")


def attach_add_counts(frame, actions, events):
    validate_count_events(actions, events)
    if (frame.empty or frame.end.isna().any() or frame.end.duplicated().any() or not frame.end.is_monotonic_increasing
        or not isinstance(frame.end.dtype, pd.DatetimeTZDtype) or str(frame.end.dt.tz) != "UTC"
        or not frame.index.is_unique):
        raise ValueError("누적 추가 횟수의 원본 분 순서 오류")
    states = teacher_states(actions, events)
    states["event_row"] = states.index.to_numpy(np.int64)
    ends = frame.end.astype("datetime64[ns, UTC]").array.asi8
    times = states.state_time.astype("datetime64[ns, UTC]").array.asi8
    positions = np.searchsorted(times, ends, side="left") - 1
    if (positions < 0).any():
        raise ValueError("누적 추가 횟수의 이전 상태 누락")
    selected = states.iloc[positions].reset_index(drop=True)
    for key in ["state_time", "entry_time"]:
        if not np.array_equal(frame[key].astype("datetime64[ns, UTC]").array.asi8,
                              selected[key].astype("datetime64[ns, UTC]").array.asi8):
            raise ValueError("누적 추가 횟수의 상태·진입 시각 불일치")
    for key in ["adds", "episode_id", "direction"]:
        if not np.array_equal(frame[key], selected[key]):
            raise ValueError("누적 추가 횟수의 저장 상태 불일치")
    count_masks(frame)
    if not np.array_equal(frame.adds_capped, np.minimum(selected.adds, 5)):
        raise ValueError("누적 추가 횟수의 기존 상한 입력 불일치")
    links = frame[["end", "episode_id", "state_time", "entry_time", "direction", "adds", "adds_capped"]].copy()
    links["event_row"] = selected.event_row.to_numpy()
    links["bucket"] = np.where(frame.adds.to_numpy() > 5, ">5", frame.adds.astype(np.int64).astype(str))
    return links


def count_masks(frame):
    counts = frame.adds.to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts >= 2**31).any() or (counts % 1 != 0).any():
        raise ValueError("누적 추가 횟수의 비정수·범위 오류")
    return {"all": np.ones(len(frame), dtype=bool), **{str(i): counts == i for i in range(6)}, ">5": counts > 5}


def validate_count_predictions(frame, predictions):
    if set(predictions) != set(MODELS):
        raise ValueError("누적 횟수 진단의 고정 대조 누락")
    count_masks(frame)
    for name in MODELS:
        saved = predictions[name]
        pd.testing.assert_frame_equal(frame[CONTEXT], saved[CONTEXT], check_exact=True)
        for action in ACTIONS:
            y = frame["y_" + action].to_numpy()
            p = saved[action + "_tree_score"].to_numpy()
            r = saved[action + "_tree_request"].to_numpy()
            if (not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
                or r.dtype.kind != "b"):
                raise ValueError("누적 횟수 진단의 정답·점수·요청 오류")


def audit_count_strata(frame, predictions):
    validate_count_predictions(frame, predictions)
    counts = count_masks(frame)
    result = {}
    for action in ACTIONS:
        result[action] = {}
        for state, state_mask in groups(frame, action).items():
            result[action][state] = {}
            for bucket, count_mask in counts.items():
                mask = state_mask & count_mask
                part = frame.loc[mask]
                values = part.adds.to_numpy()
                y = part["y_" + action].to_numpy()
                result[action][state][bucket] = {
                    "support": {"rows": len(part), "positive": int(y.sum()), "negative": int(len(y)-y.sum()),
                        "episodes": int(part.episode_id.nunique()), "distinct_counts": int(part.adds.nunique()),
                        "minimum_adds": int(values.min()) if len(values) else None,
                        "maximum_adds": int(values.max()) if len(values) else None},
                    "models": {name: metrics(y, saved.loc[mask, action + "_tree_score"].to_numpy(),
                        saved.loc[mask, action + "_tree_request"].to_numpy(), BETAS[action]) for name, saved in predictions.items()}}
    return result
