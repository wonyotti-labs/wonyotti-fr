import copy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.management_flow import (
    EXPANDED_FEATURES,
    EXTRA_FEATURES,
    connect_flow,
    flow_support,
    flow_tables,
)
from wonyotti_fr.management_flow_reference import (
    direct_flow_table,
    verify_flow_connection,
    verify_flow_support,
)
from wonyotti_fr.management_flow_research import checked_flow_reference
from wonyotti_fr.management_market import FEATURES
from wonyotti_fr.minute_management import ACTIONS


def bars(count=2000):
    times = pd.date_range("2020-01-01", periods=count, freq="min", tz="UTC")
    close = 100+np.sin(np.arange(count)/17)
    volume = 5+np.arange(count)%19
    take = volume*(.5+.4*np.sin(np.arange(count)/5))
    minute = pd.DataFrame({"time": times, "end": times+pd.Timedelta(minutes=1),
        "open": close, "close": close, "high": close+.5, "low": close-.5,
        "volume": volume.astype(float), "count": 2, "taker_buy_volume": take})
    five = minute.groupby(minute.end.dt.ceil("5min")).agg(
        open=("open", "first"), close=("close", "last"), high=("high", "max"), low=("low", "min"),
        volume=("volume", "sum"), count=("count", "sum"), taker_buy_volume=("taker_buy_volume", "sum")).reset_index()
    five["time"] = five.end-pd.Timedelta(minutes=5)
    return minute, five


def frame_at(times):
    frame = pd.DataFrame(0., columns=FEATURES, index=range(len(times)))
    frame["end"] = times.to_numpy()
    frame["direction"] = np.where(np.arange(len(times))%2, 1, -1)
    frame["episode_id"] = 1
    frame["original_extra"] = np.arange(len(times))
    for action in ACTIONS:
        frame["y_"+action] = np.arange(len(times))%2
    return frame


def fixture():
    minute, five = bars()
    frame = frame_at(minute.end.iloc[1500:1560].reset_index(drop=True))
    return frame, minute, five


def test_all_values_timing_and_volume_weighting():
    frame, minute, five = fixture()
    small, large, proof = flow_tables(minute, five, frame.end.max())
    expanded, link = connect_flow(frame, small, large)
    assert len(EXPANDED_FEATURES) == 62 and len(EXTRA_FEATURES) == 12
    assert link.supported.all() and proof["maximum_buy_volume_error"] == 0
    assert (link.end-link.five_source_end).dt.total_seconds().tolist()[:5] == [60, 120, 180, 240, 0]
    end = link.five_source_end.iloc[0]
    selected = five[five.end.le(end)].tail(3)
    expected = 2*selected.taker_buy_volume.sum()/selected.volume.sum()-1
    assert expanded.taker_imbalance_15m.iloc[0] == pytest.approx(expected, abs=1e-12)
    assert abs(expected-(2*selected.taker_buy_volume/selected.volume-1).mean()) > 1e-5
    verify_flow_connection(frame, expanded, link, minute, five, direct_flow_table(minute, five))
    support = flow_support(frame, link)
    verify_flow_support(frame, link, support)
    assert support["all_original_rows_supported"] and support["action_support_sufficient"]


def test_future_change_and_extension_leave_past_exact():
    frame, minute, five = fixture()
    old = connect_flow(frame, *flow_tables(minute, five, frame.end.max())[:2])
    extended = bars(2500)
    for table in extended:
        table.loc[table.end.gt(frame.end.max()), "taker_buy_volume"] = -100.
    new = connect_flow(frame, *flow_tables(*extended, frame.end.max())[:2])
    for a, b in zip(old, new, strict=True):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_zero_volume_is_zero_but_warmup_stays_unsupported():
    minute, five = bars()
    for table in [minute, five]:
        table[["open", "high", "low", "close"]] = 100.
        table[["volume", "taker_buy_volume", "count"]] = 0
    frame = frame_at(minute.end.iloc[1400:1460].reset_index(drop=True))
    small, large, _ = flow_tables(minute, five, frame.end.max())
    expanded, link = connect_flow(frame, small, large)
    assert (~link.supported).sum() == 39
    assert expanded.loc[link.supported, EXTRA_FEATURES].eq(0).all().all()
    assert expanded.loc[~link.supported, "taker_imbalance_1d"].isna().all()
    support = flow_support(frame, link)
    verify_flow_support(frame, link, support)
    verify_flow_connection(frame, expanded, link, minute, five, direct_flow_table(minute, five))
    assert not support["all_original_rows_supported"]
    assert len(expanded) == len(frame)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "timezone", "off_grid", "negative_buy", "large_buy", "nonfinite_buy", "buy_disagreement", "invalid_price"])
def test_invalid_source_fails_without_imputation(kind):
    frame, minute, five = fixture()
    if kind == "missing":
        minute = minute.drop(index=1500)
    elif kind == "duplicate":
        minute.loc[1500, "time"] = minute.time.iloc[1499]
    elif kind == "timezone":
        minute["end"] = minute.end.dt.tz_convert("Asia/Seoul")
    elif kind == "off_grid":
        minute.loc[1500, "end"] += pd.Timedelta(seconds=1)
    elif kind == "negative_buy":
        minute.loc[1500, "taker_buy_volume"] = -1
    elif kind == "large_buy":
        minute.loc[1500, "taker_buy_volume"] = minute.volume.iloc[1500]+1
    elif kind == "nonfinite_buy":
        minute.loc[1500, "taker_buy_volume"] = np.nan
    elif kind == "buy_disagreement":
        five.loc[300, "taker_buy_volume"] += .1
    else:
        five.loc[300, "high"] = five.close.iloc[300]-1
    with pytest.raises(ValueError):
        flow_tables(minute, five, frame.end.max())


def test_missing_link_keeps_original_and_is_independent_of_labels():
    frame, minute, five = fixture()
    small, large, _ = flow_tables(minute, five, frame.end.max())
    small = small[small.end.ne(frame.end.iloc[0])]
    large = large[large.end.ne(frame.end.iloc[10].floor("5min"))]
    expanded, link = connect_flow(frame, small, large)
    assert link.reason.iloc[0] == "missing_minute" and link.reason.iloc[10] == "missing_five"
    changed = frame.copy()
    changed["y_exit"] = 1-changed.y_exit
    changed_expanded, other = connect_flow(changed, small, large)
    pd.testing.assert_frame_equal(link, other)
    pd.testing.assert_frame_equal(expanded[EXPANDED_FEATURES], changed_expanded[EXPANDED_FEATURES])
    support = flow_support(frame, link)
    assert support["groups"]["all"]["rows"] == len(frame)
    verify_flow_support(frame, link, support)


@pytest.mark.parametrize("kind", ["feature", "state", "direction", "source_time", "membership", "counts"])
def test_independent_verification_rejects_tampering(kind):
    frame, minute, five = fixture()
    expanded, link = connect_flow(frame, *flow_tables(minute, five, frame.end.max())[:2])
    support = flow_support(frame, link)
    if kind == "feature":
        expanded.loc[0, "taker_imbalance_1h"] += .01
    elif kind == "state":
        expanded.loc[0, "original_extra"] += 1
    elif kind == "direction":
        expanded.loc[0, "directional_taker_imbalance_1d"] += .01
    elif kind == "source_time":
        link.loc[0, "minute_source_end"] += pd.Timedelta(minutes=1)
    elif kind == "membership":
        link.loc[0, "source_row"] += 1
    else:
        support["groups"]["included"]["positive"]["exit"] += 1
    with pytest.raises(AssertionError):
        verify_flow_connection(frame, expanded, link, minute, five, direct_flow_table(minute, five))
        verify_flow_support(frame, link, support)


@pytest.mark.parametrize("kind", ["hash", "symlink"])
def test_reference_rejected_before_loading(tmp_path, kind):
    path = tmp_path/"reference.json"
    save_json(path, {})
    digest = sha256(path)
    if kind == "hash":
        digest = "0"*64
    else:
        alias = tmp_path/"alias.json"
        alias.symlink_to(path)
        path = alias
    with pytest.raises(ValueError, match="지문"):
        checked_flow_reference(path, digest)


def test_public_runner_full_rows_and_failure_record(tmp_path, monkeypatch):
    import wonyotti_fr.management_flow_research as research
    from wonyotti_fr.holding_stress import read_json
    from wonyotti_fr.retained_weight_close_reference import checked_files

    minute, five = bars(12960)
    teacher = tmp_path/"teacher"
    teacher.mkdir()
    periods = {"training": ("2020-01-01", "2020-01-04"),
        "calibration": ("2020-01-04", "2020-01-07"), "diagnosis": ("2020-01-07", "2020-01-10")}
    expected_rows = {}
    for i, (phase, (start, _)) in enumerate(periods.items()):
        date = pd.Timestamp(start, tz="UTC")+pd.Timedelta(days=1)
        count = 1200 if phase == "training" else 200
        frame = frame_at(pd.Series(pd.date_range(date, periods=count, freq="min")))
        frame["label_end"] = frame.end+pd.Timedelta(minutes=1)
        frame["entry_time"] = date-pd.Timedelta(hours=1)
        frame["episode_id"] = i+1
        frame.to_parquet(teacher/f"{phase}_used.parquet", index=False)
        expected_rows[phase] = frame
    monkeypatch.setattr(research, "PERIODS", periods)
    monkeypatch.setattr(research, "checked_flow_reference", lambda *args: {
        "teacher": str(teacher), "target_minute": "minute", "target_five": "five"})
    monkeypatch.setattr(research, "load_market", lambda path, symbol, interval: (
        minute.copy() if interval == "1m" else five.copy(), pd.DataFrame()))
    result = research.run_flow_support(tmp_path/"reference", "synthetic", tmp_path/"output")
    checked_files(result)
    summary = read_json(result/"summary.json")
    assert summary["paired_model_research_eligible"] and not summary["new_models_fitted"]
    assert summary["input_values_independently_verified"] == 1600*12
    for phase, frame in expected_rows.items():
        link = pd.read_parquet(result/f"{phase}_linkage.parquet")
        assert len(link) == len(frame) and link.supported.all()
    five.loc[300, "taker_buy_volume"] += .1
    with pytest.raises(ValueError, match="매수량"):
        research.run_flow_support(tmp_path/"reference", "synthetic", tmp_path/"failed")
    failure = list((tmp_path/"failed").glob("*/failure.json"))
    assert len(failure) == 1 and read_json(failure[0])["type"] == "ValueError"


def test_support_rejects_modified_summary():
    frame, minute, five = fixture()
    _, link = connect_flow(frame, *flow_tables(minute, five, frame.end.max())[:2])
    support = flow_support(frame, link)
    altered = copy.deepcopy(support)
    altered["all_original_rows_supported"] = False
    with pytest.raises(AssertionError):
        verify_flow_support(frame, link, altered)
