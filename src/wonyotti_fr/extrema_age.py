from __future__ import annotations

import numpy as np
import pandas as pd

from .first_state_models import validate_first_rows
from .path_management import PATH_FEATURES

MINUTE_NS = 60_000_000_000
CONTEXT = ["end", "close", "average_entry", "entry_time", "direction", "episode_id"]
SOURCE_COLUMNS = CONTEXT + PATH_FEATURES
EXTRA_FEATURES = ["path_log_minutes_since_best", "path_log_minutes_since_worst"]


def validate_extrema_source(frame):
    if (
        not len(frame)
        or not frame.index.is_unique
        or str(frame.end.dt.tz) != "UTC"
        or str(frame.entry_time.dt.tz) != "UTC"
        or frame.end.isna().any()
        or frame.end.duplicated().any()
        or not frame.end.is_monotonic_increasing
        or not frame.end.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all()
    ):
        raise ValueError("극값 시계의 연속 분 경계·UTC 오류")
    ends = frame.end.astype("datetime64[ns, UTC]").array.asi8
    if (
        (ends % MINUTE_NS != 0).any()
        or not np.isfinite(frame.close).all()
        or frame.close.le(0).any()
        or not frame.direction.isin([-1, 0, 1]).all()
    ):
        raise ValueError("극값 시계의 가격·방향 오류")
    held = frame.direction.ne(0)
    part = frame.loc[held]
    if (
        not len(part)
        or part.entry_time.isna().any()
        or not part.entry_time.lt(part.end).all()
        or not np.isfinite(part[["episode_id", "average_entry"]]).all().all()
        or part.episode_id.le(0).any()
        or part.episode_id.mod(1).ne(0).any()
        or part.average_entry.le(0).any()
    ):
        raise ValueError("극값 시계의 보유 상태 오류")
    positions = pd.Series(np.flatnonzero(held), index=part.index)
    for _, group in part.groupby("episode_id", sort=False):
        locations = positions.loc[group.index].to_numpy()
        if (
            len(group.direction.unique()) != 1
            or len(group.entry_time.unique()) != 1
            or locations[-1] - locations[0] + 1 != len(locations)
        ):
            raise ValueError("극값 시계의 포지션 재등장·방향·진입 변경")


def extrema_clock_table(frame):
    validate_extrema_source(frame)
    held = frame.loc[frame.direction.ne(0), CONTEXT].copy()
    groups = held.groupby("episode_id", sort=False)
    low, high = groups.close.cummin(), groups.close.cummax()
    previous_low = low.groupby(held.episode_id).shift(1)
    previous_high = high.groupby(held.episode_id).shift(1)
    # 같은 가격은 갱신으로 세지 않아 평탄 구간에서도 경과 시간이 누적된다.
    low_record = previous_low.isna() | held.close.lt(previous_low)
    high_record = previous_high.isna() | held.close.gt(previous_high)
    low_end = held.end.where(low_record).groupby(held.episode_id).ffill()
    high_end = held.end.where(high_record).groupby(held.episode_id).ffill()
    result = held.copy()
    result["path_first_end"] = groups.end.transform("first")
    result["path_low"], result["path_high"] = low, high
    result["best_record_end"] = high_end.where(held.direction.eq(1), low_end)
    result["worst_record_end"] = low_end.where(held.direction.eq(1), high_end)
    for label in ["best", "worst"]:
        delta = (
            held.end.astype("datetime64[ns, UTC]").array.asi8
            - result[label + "_record_end"].astype("datetime64[ns, UTC]").array.asi8
        )
        if (delta < 0).any() or (delta % MINUTE_NS != 0).any():
            raise ValueError("극값 시계의 경과 시간 오류")
        result["minutes_since_" + label] = delta // MINUTE_NS
        result["path_log_minutes_since_" + label] = np.log1p(delta / MINUTE_NS)
    return result.reset_index(drop=True)


def attach_extrema_age(rows, table):
    validate_first_rows(rows)
    if table.end.duplicated().any() or not table.end.is_monotonic_increasing:
        raise ValueError("극값 시계 연결의 중복·순서 오류")
    index = pd.Index(table.end)
    expanded, links = {}, {}
    for phase, frame in rows.items():
        if set(EXTRA_FEATURES) & set(frame):
            raise ValueError("극값 시계 입력의 중복 추가")
        positions = index.get_indexer(frame.end)
        if (positions < 0).any():
            raise ValueError("극값 시계 연결의 원본 행 누락")
        part = table.iloc[positions].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            frame[CONTEXT].reset_index(drop=True), part[CONTEXT], check_exact=True
        )
        expected_first = part.entry_time.dt.floor("min") + pd.Timedelta(minutes=1)
        if not part.path_first_end.eq(expected_first).all():
            raise ValueError("극값 시계의 진입 이후 초기 경로 누락")
        long = part.direction.eq(1).to_numpy()
        for name, price in [
            ("best_close_move", np.where(long, part.path_high, part.path_low)),
            ("worst_close_move", np.where(long, part.path_low, part.path_high)),
        ]:
            expected = part.direction.to_numpy() * (price / part.average_entry.to_numpy() - 1)
            np.testing.assert_allclose(frame[name], expected, atol=1e-12, rtol=0)
        expected = frame.best_close_move - frame.favorable_move
        np.testing.assert_allclose(frame.giveback_from_best, expected, atol=1e-12, rtol=0)
        for label in ["best", "worst"]:
            age = part["minutes_since_" + label].to_numpy()
            hold = (
                part.end.astype("datetime64[ns, UTC]").array.asi8
                - part.entry_time.astype("datetime64[ns, UTC]").array.asi8
            ) / MINUTE_NS
            if (age < 0).any() or (age > hold).any():
                raise ValueError("극값 시계의 보유 시간 초과")
        added = frame.copy()
        added[EXTRA_FEATURES] = part[EXTRA_FEATURES].to_numpy(float)
        pd.testing.assert_frame_equal(added.drop(columns=EXTRA_FEATURES), frame, check_exact=True)
        expanded[phase], links[phase] = added, part
    return expanded, links


def age_groups(link, label):
    if label not in {"best", "worst"}:
        raise ValueError("극값 시계 구간의 종류 오류")
    values = link["minutes_since_" + label].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError("극값 시계 구간의 경과 시간 오류")
    return np.searchsorted([1, 15, 60, 1440], values, side="right")
