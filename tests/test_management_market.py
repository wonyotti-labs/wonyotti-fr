import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.management_market import (
    FEATURES,
    OBSERVED,
    REPLACED,
    check_original_market,
    connect_market,
    market_support,
    observed_tables,
)
from wonyotti_fr.management_market_reference import verify_market_connection, verify_market_support
from wonyotti_fr.management_market_research import checked_market_reference


def bars(count=1800):
    time = pd.date_range("2020-01-01", periods=count, freq="min", tz="UTC")
    close = 100 + np.sin(np.arange(count) / 19) + np.arange(count) / 100
    minute = pd.DataFrame({"time": time, "end": time + pd.Timedelta(minutes=1),
                           "open": close, "high": close + .2, "low": close - .2,
                           "close": close, "volume": 10 + np.arange(count) % 9})
    five = minute.groupby(minute.end.dt.ceil("5min")).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).reset_index()
    five["time"] = five.end - pd.Timedelta(minutes=5)
    return minute, five


def teacher():
    raw = bars()
    minute, five = observed_tables(*raw)
    frame = pd.DataFrame(0., index=range(60), columns=FEATURES)
    frame["end"] = raw[0].end.iloc[1500:1560].to_numpy()
    frame["direction"] = np.where(np.arange(60) % 2, 1, -1)
    frame["episode_id"] = 1
    for a in ["exit", "reduce", "increase"]:
        frame[f"y_{a}"] = np.arange(60) % 2
    frame["private_state"] = np.arange(60)
    frame, link = connect_market(frame, minute, five)
    assert link.supported.all()
    return frame, minute, five, raw


def test_original_reconstruction_and_exact_cutoffs():
    frame, minute, five, raw = teacher()
    candidate, link = connect_market(frame, minute, five)
    check_original_market(frame, candidate, link)
    assert len(REPLACED) == 36 and len(set(FEATURES) - set(REPLACED)) == 14
    assert (link.end - link.expected_five_end).dt.total_seconds().tolist()[:5] == [60, 120, 180, 240, 0]
    for name in OBSERVED:
        np.testing.assert_array_equal(candidate[f"directional_{name}"], candidate[name] * frame.direction)
    support = market_support(frame, candidate, link)
    verify_market_connection(frame, candidate, link, raw[0], minute, five)
    verify_market_support(frame, candidate, link, support)
    assert support["action_support_sufficient"]
    assert all(c["different"] == 0 for c in support["changes"].values())


def test_future_bars_do_not_change_past_features():
    frame, minute, five, (raw_minute, raw_five) = teacher()
    old, _ = connect_market(frame, minute, five)
    cutoff = frame.end.max()
    for raw in [raw_minute, raw_five]:
        selected = raw.end.gt(cutoff)
        raw.loc[selected, ["open", "high", "low", "close"]] *= 3
        raw.loc[selected, "volume"] *= 11
    new, _ = connect_market(frame, *observed_tables(raw_minute, raw_five))
    pd.testing.assert_frame_equal(old, new, check_exact=True)


def test_current_minute_not_in_volume_denominator():
    minute, five = bars()
    features, _ = observed_tables(minute, five)
    index = 1500
    assert features.minute_volume_ratio_1h.iloc[index] == minute.volume.iloc[index] / minute.volume.iloc[index-60:index].mean()
    original = features.minute_volume_ratio_1h.iloc[index]
    minute.loc[index, "volume"] *= 4
    changed, _ = observed_tables(minute, five)
    assert changed.minute_volume_ratio_1h.iloc[index] == original * 4


def test_missing_and_nonfinite_rows_preserve_labels_and_states():
    frame, minute, five, _ = teacher()
    end = frame.end.iloc[0]
    minute = minute[minute.end.ne(end)].reset_index(drop=True)
    minute.loc[minute.end.eq(frame.end.iloc[1]), "minute_ret_1m"] = np.nan
    five = five[five.end.ne(frame.end.iloc[5].floor("5min"))].reset_index(drop=True)
    candidate, link = connect_market(frame, minute, five)
    assert link.reason.iloc[0] == "missing_minute"
    assert link.reason.iloc[1] == "nonfinite_market"
    assert link.reason.iloc[5] == "missing_five"
    assert len(candidate) == len(frame)
    pd.testing.assert_frame_equal(candidate.drop(columns=REPLACED), frame.drop(columns=REPLACED))
    support = market_support(frame, candidate, link)
    for a in ["exit", "reduce", "increase"]:
        groups = support["groups"]
        assert groups["all"]["positive"][a] == groups["included"]["positive"][a] + groups["excluded"]["positive"][a]
    with pytest.raises(ValueError, match="재현"):
        check_original_market(frame, candidate, link)


@pytest.mark.parametrize("kind", ["duplicate", "naive", "off_minute", "direction", "label", "feature"])
def test_malformed_teacher_rejected(kind):
    frame, minute, five, _ = teacher()
    if kind == "duplicate":
        frame.loc[1, "end"] = frame.end.iloc[0]
    elif kind == "naive":
        frame["end"] = frame.end.dt.tz_localize(None)
    elif kind == "off_minute":
        frame.loc[0, "end"] += pd.Timedelta(seconds=1)
    elif kind == "direction":
        frame.loc[0, "direction"] = 0
    elif kind == "label":
        frame.loc[0, "y_exit"] = 2
    else:
        frame.loc[0, "favorable_move"] = np.nan
    with pytest.raises(ValueError):
        connect_market(frame, minute, five)


@pytest.mark.parametrize("kind", ["missing", "negative_volume", "invalid_price", "timezone"])
def test_invalid_observation_source_rejected(kind):
    minute, five = bars()
    if kind == "missing":
        minute = minute.drop(index=1500)
    elif kind == "negative_volume":
        minute.loc[1500, "volume"] = -1
    elif kind == "invalid_price":
        five.loc[300, "high"] = five.close.iloc[300] - 1
    else:
        five["end"] = five.end.dt.tz_convert("Asia/Seoul")
    with pytest.raises(ValueError):
        observed_tables(minute, five)


def test_support_not_selected_from_labels_and_tampering_rejected():
    frame, minute, five, _ = teacher()
    candidate, link = connect_market(frame, minute, five)
    altered = frame.copy()
    altered["y_exit"] = 1 - altered.y_exit
    other, other_link = connect_market(altered, minute, five)
    pd.testing.assert_frame_equal(link, other_link)
    pd.testing.assert_frame_equal(candidate[FEATURES], other[FEATURES])
    candidate.loc[0, "private_state"] += 1
    with pytest.raises(AssertionError):
        market_support(frame, candidate, link)


def test_independent_reference_detects_market_and_support_changes():
    frame, minute, five, raw = teacher()
    candidate, link = connect_market(frame, minute, five)
    candidate.loc[0, "ret_5m"] += .001
    with pytest.raises(AssertionError):
        verify_market_connection(frame, candidate, link, raw[0], minute, five)
    candidate, link = connect_market(frame, minute, five)
    support = market_support(frame, candidate, link)
    support["groups"]["included"]["positive"]["exit"] += 1
    with pytest.raises(AssertionError):
        verify_market_support(frame, candidate, link, support)


def test_empty_common_support_is_preserved_and_fails_gate():
    frame, minute, five, _ = teacher()
    five["ret_5m"] = np.nan
    candidate, link = connect_market(frame, minute, five)
    support = market_support(frame, candidate, link)
    assert support["groups"]["included"]["rows"] == 0
    assert support["groups"]["excluded"]["rows"] == len(frame)
    assert not support["action_support_sufficient"]
    verify_market_support(frame, candidate, link, support)


@pytest.mark.parametrize("kind", ["hash", "symlink"])
def test_untrusted_reference_rejected_before_loading(tmp_path, kind):
    path = tmp_path / "reference.json"
    path.write_text("{}")
    digest = sha256(path)
    if kind == "hash":
        digest = "0" * 64
    else:
        other = tmp_path / "link.json"
        other.symlink_to(path)
        path = other
    with pytest.raises(ValueError, match="출처 지문"):
        checked_market_reference(path, digest)


def test_public_runner_preserves_all_splits_and_records_failure(tmp_path, monkeypatch):
    import wonyotti_fr.management_market_research as research
    from wonyotti_fr.retained_weight_close_reference import checked_files

    raw_minute, raw_five = bars(12960)
    raw_minute["count"], raw_five["count"] = 1, 5
    minute, five = observed_tables(raw_minute, raw_five)
    source = tmp_path / "native"
    source.mkdir()
    raw_five.to_parquet(source / "five.parquet")
    save_json(source / "manifest.json", {"file": "five.parquet", "sha256": sha256(source / "five.parquet")})
    labels = tmp_path / "labels"
    labels.mkdir()
    source_hashes = {"minute_manifest_sha256": "a" * 64, "minute_history_sha256": "b" * 64}
    save_json(labels / "manifest.json", {"settings": source_hashes})
    save_json(labels / "files.json", {"manifest.json": sha256(labels / "manifest.json")})
    folder = tmp_path / "teacher"
    folder.mkdir()
    save_json(folder / "manifest.json", {"settings": {"labels": str(labels),
              "labels_files_sha256": sha256(labels / "files.json")}})
    periods = {"training": ["2020-01-01", "2020-01-04"],
               "calibration": ["2020-01-04", "2020-01-07"],
               "diagnosis": ["2020-01-07", "2020-01-10"]}
    frames = {}
    for index, (phase, bounds) in enumerate(periods.items()):
        frame = teacher()[0]
        start = 1500 + 4320 * index
        frame["end"] = raw_minute.end.iloc[start:start+60].to_numpy()
        frame["entry_time"] = pd.Timestamp(bounds[0], tz="UTC")
        frame["label_end"] = frame.end + pd.Timedelta(minutes=1)
        frame["episode_id"] = index + 1
        frame, _ = connect_market(frame, minute, five)
        frame.to_parquet(folder / f"{phase}_used.parquet", index=False)
        frames[phase] = frame
    spec = {"native_minute": str(source), "native_five": str(source),
             "target_minute": str(source), "target_five": str(source)}
    monkeypatch.setattr(research, "PERIODS", periods)
    monkeypatch.setattr(research, "checked_market_reference", lambda *args: (spec, {"teacher": str(folder)}))
    monkeypatch.setattr(research, "read_minute_history", lambda *args: (raw_minute.copy(), source_hashes))
    monkeypatch.setattr(research, "load_market", lambda root, symbol, interval:
                        ((raw_minute if interval == "1m" else raw_five).copy(), pd.DataFrame()))
    out = research.run_market_support(tmp_path / "unused.json", "c" * 64, tmp_path / "outputs")
    checked_files(out)
    summary = research.read(out / "summary.json")
    assert summary["complete"] and summary["paired_model_research_eligible"]
    assert not summary["new_models_fitted"] and not summary["profitability_accepted"]
    for phase in periods:
        pd.testing.assert_frame_equal(pd.read_parquet(out / f"{phase}_market.parquet"),
                                      frames[phase][["end", *REPLACED]], check_exact=True)
        assert pd.read_parquet(out / f"{phase}_linkage.parquet").supported.all()
    changed = frames["diagnosis"].copy()
    changed.loc[0, "ret_5m"] += .01
    changed.to_parquet(folder / "diagnosis_used.parquet", index=False)
    with pytest.raises(AssertionError):
        research.run_market_support(tmp_path / "unused.json", "c" * 64, tmp_path / "failures")
    assert len(list((tmp_path / "failures").glob("*/failure.json"))) == 1
