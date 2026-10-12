from __future__ import annotations

import numpy as np
import pandas as pd

MINUTE_NS = pd.Timedelta(minutes=1).value
HORIZONS = (1, 15)


def validate_temporal_inputs(frame, requests, targets, horizon):
    if type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError("관리 시간 대응의 고정 범위 오류")
    if not {"end", "episode_id"}.issubset(frame):
        raise ValueError("관리 시간 대응의 문맥 누락")
    if (
        not isinstance(frame.end.dtype, pd.DatetimeTZDtype)
        or str(frame.end.dt.tz) != "UTC"
        or frame.end.isna().any()
        or frame.end.duplicated().any()
        or not frame.end.is_monotonic_increasing
        or not frame.end.eq(frame.end.dt.floor("min")).all()
        or frame.episode_id.dtype.kind not in "ifu"
        or not np.isfinite(frame.episode_id).all()
        or frame.episode_id.le(0).any()
        or frame.episode_id.ge(2**53).any()
        or frame.episode_id.mod(1).ne(0).any()
    ):
        raise ValueError("관리 시간 대응의 시각·포지션 오류")
    masks = [np.asarray(v) for v in (requests, targets)]
    if any(v.dtype.kind != "b" or v.shape != (len(frame),) for v in masks):
        raise ValueError("관리 시간 대응의 요청·목표 마스크 오류")
    return masks


def match_temporal(frame, requests, targets, horizon):
    requests, targets = validate_temporal_inputs(frame, requests, targets, horizon)
    minutes = frame.end.astype("datetime64[ns, UTC]").array.asi8 // MINUTE_NS
    ids = frame.episode_id.to_numpy(np.int64)
    ri, ei = np.flatnonzero(requests), np.flatnonzero(targets)
    rt, et = minutes[ri], minutes[ei]
    re, ee = ids[ri], ids[ei]
    partner = np.full(len(ri), -1, dtype=np.int64)
    reverse = np.full(len(ei), -1, dtype=np.int64)
    available = np.zeros(len(ri), dtype=bool)
    covered = np.zeros(len(ei), dtype=bool)
    for episode in np.union1d(re, ee):
        r = np.flatnonzero(re == episode)
        e = np.flatnonzero(ee == episode)
        if not len(r) or not len(e):
            continue
        nxt = np.searchsorted(et[e], rt[r], side="left")
        valid = nxt < len(e)
        available[r[valid]] = et[e[nxt[valid]]] < rt[r[valid]] + horizon
        previous = np.searchsorted(rt[r], et[e], side="right") - 1
        valid = previous >= 0
        covered[e[valid]] = rt[r[previous[valid]]] > et[e[valid]] - horizon
        cursor = 0
        # 동일 길이 창에서는 만료 순서가 요청 순서와 같아 가장 이른 요청부터 배정한다.
        for event in e:
            while cursor < len(r) and rt[r[cursor]] <= et[event] - horizon:
                cursor += 1
            if cursor < len(r) and rt[r[cursor]] <= et[event]:
                request = r[cursor]
                partner[request], reverse[event] = event, request
                cursor += 1
    linked = partner >= 0
    captured = reverse >= 0
    request_table = frame.iloc[ri][["end", "episode_id"]].reset_index(drop=True)
    request_table = request_table.rename(columns={"end": "request_end"})
    target_table = frame.iloc[ei][["end", "episode_id"]].reset_index(drop=True)
    target_table = target_table.rename(columns={"end": "event_end"})
    request_table["original_positive"] = targets[ri]
    request_table["event_end"] = pd.Series(
        pd.NaT, index=request_table.index, dtype="datetime64[ns, UTC]"
    )
    target_table["request_end"] = pd.Series(
        pd.NaT, index=target_table.index, dtype="datetime64[ns, UTC]"
    )
    request_table.loc[linked, "event_end"] = frame.end.iloc[ei[partner[linked]]].to_numpy()
    target_table.loc[captured, "request_end"] = frame.end.iloc[ri[reverse[captured]]].to_numpy()
    leads = pd.array(np.zeros(len(ri), dtype=np.int64), dtype="Int64")
    leads[linked] = et[partner[linked]] - rt[linked]
    leads[~linked] = pd.NA
    request_table["lead_minutes"] = leads
    request_table["has_target"] = available
    target_table["has_request"] = covered
    # 관측 표의 빈 경계를 임의의 음성 정답으로 취급하지 않는다.
    request_table["window_grid_complete"] = np.isin(rt[:, None] + np.arange(horizon), minutes).all(
        axis=1
    )
    request_table["window_beyond_phase"] = (
        rt + horizon > minutes[-1] + 1 if len(minutes) else np.zeros(0, dtype=bool)
    )
    return request_table, target_table


def temporal_summary(requests, targets, horizon):
    matched = int(requests.event_end.notna().sum())
    captured = int(targets.request_end.notna().sum())
    return {
        "requests": len(requests),
        "original_true_positive": int(requests.original_positive.sum()),
        "matched_requests": matched,
        "unmatched_requests": len(requests) - matched,
        "request_match_rate": matched / len(requests) if len(requests) else None,
        "requests_with_any_target": int(requests.has_target.sum()),
        "complete_observation_windows": int(requests.window_grid_complete.sum()),
        "windows_beyond_phase": int(requests.window_beyond_phase.sum()),
        "lead_minutes_counts": {
            str(i): int(requests.lead_minutes.eq(i).sum()) for i in range(horizon)
        },
        "targets": len(targets),
        "matched_targets": captured,
        "unmatched_targets": len(targets) - captured,
        "target_capture_rate": captured / len(targets) if len(targets) else None,
        "targets_with_any_request": int(targets.has_request.sum()),
    }


def temporal_months(requests, targets, horizon, months):
    rm = requests.request_end.dt.strftime("%Y-%m")
    em = targets.event_end.dt.strftime("%Y-%m")
    return {
        month: temporal_summary(requests.loc[rm.eq(month)], targets.loc[em.eq(month)], horizon)
        for month in months
    }
