from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.request_timing import WINDOWS, audit_action, match_minutes
from wonyotti_fr.request_timing_reference import graph_match_count, verify_action
from wonyotti_fr.request_timing_research import MODELS, verify_saved_requests


def frame(requests=(), targets=(), length=80):
    return pd.DataFrame({"end": pd.date_range("2020-01-01", periods=length, freq="min", tz="UTC"),
                         "episode_id": 1, "request": np.isin(np.arange(length), requests),
                         "label": np.isin(np.arange(length), targets).astype(int)})


def audit(value):
    result, ledger = audit_action(value, "request", "label")
    verify_action(value, "request", "label", result, ledger)
    return result, ledger


def test_exact_and_inclusive_time_boundaries():
    result, _ = audit(frame([15, 16, 29, 30, 31, 44, 45], [30]))
    assert result["tp"] == 1
    assert result["false_positive_timing"] == {
        "future_only": 2, "past_only": 2, "both": 0, "far": 2, "no_positive": 0}
    assert all(v["additional_pairs"] == 0 for v in result["matching"].values())


def test_one_request_cannot_capture_two_targets():
    result, _ = audit(frame([10], [11, 12]))
    assert result["matching"]["early"] == {
        "additional_pairs": 1, "loose_remaining_targets": 2, "total_pairs_with_exact": 1}


def test_two_requests_cannot_reuse_target():
    result, _ = audit(frame([10, 11], [12]))
    assert result["matching"]["early"]["additional_pairs"] == 1


def test_both_directions_and_position_isolation():
    value = frame([10, 30, 60], [9, 11, 32])
    value.loc[50:, "episode_id"] = 2
    result, _ = audit(value)
    assert result["false_positive_timing"]["both"] == 1
    assert result["false_positive_timing"]["no_positive"] == 1
    assert result["matching"]["early"]["additional_pairs"] == 2
    assert result["matching"]["late"]["additional_pairs"] == 1


def test_transitive_clusters_do_not_simulate_cooldown():
    result, _ = audit(frame([1, 4, 7, 11], [4]))
    assert result["clusters"] == {"count": 2, "singletons": 1, "max_requests": 3,
        "max_duration_minutes": 6, "all_fp_count": 1, "all_fp_requests": 1}


@pytest.mark.parametrize("requests,targets", [([], []), ([], [1]), ([1], []), ([1], [1])])
def test_empty_requests_or_labels(requests, targets):
    audit(frame(requests, targets))


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_timestamp_units(unit):
    value = frame([1, 7], [3, 9])
    value["end"] = value.end.dt.as_unit(unit)
    assert audit(value)[0] == audit(frame([1, 7], [3, 9]))[0]


def test_integral_float_episode_ids_preserved():
    value = frame([1, 7], [3, 9])
    value["episode_id"] = value.episode_id.astype(float)
    assert audit(value) == audit(frame([1, 7], [3, 9]))


def test_saved_first_and_full_thresholds_are_not_reselected():
    value = frame(length=2)
    value["past_reduce_exists"] = [0, 1]
    value["past_increase_exists"] = [1, 0]
    thresholds = {"full": {}, "first": {}}
    for model in MODELS:
        thresholds["full"][model] = dict.fromkeys(["exit", "reduce", "increase"], 0.5)
        thresholds["first"][model] = dict.fromkeys(["reduce", "increase"], 0.25)
        for action, requests in [("exit", [False, False]), ("reduce", [True, False]),
                                 ("increase", [False, True])]:
            value[action + "_" + model] = 0.25
            value[action + "_" + model + "_request"] = requests
    verify_saved_requests(value, thresholds)
    value.loc[0, "reduce_native_calibrated_request"] = False
    with pytest.raises(AssertionError):
        verify_saved_requests(value, thresholds)


@pytest.mark.parametrize("kind", ["unsorted", "duplicate", "nat", "naive", "timezone", "seconds",
    "episode_null", "episode_float", "episode_zero", "episode_return", "label", "request", "empty"])
def test_reject_bad_source(kind):
    value = frame([1], [2])
    if kind == "unsorted":
        value = value.iloc[::-1]
    elif kind == "duplicate":
        value.loc[1, "end"] = value.loc[0, "end"]
    elif kind == "nat":
        value.loc[0, "end"] = pd.NaT
    elif kind == "naive":
        value["end"] = value.end.dt.tz_localize(None)
    elif kind == "timezone":
        value["end"] = value.end.dt.tz_convert("Asia/Seoul")
    elif kind == "seconds":
        value["end"] += pd.Timedelta(seconds=1)
    elif kind == "episode_null":
        value["episode_id"] = pd.Series([pd.NA] * len(value), dtype="Int64")
    elif kind == "episode_float":
        value["episode_id"] = 1.5
    elif kind == "episode_zero":
        value["episode_id"] = 0
    elif kind == "episode_return":
        value.loc[1, "episode_id"] = 2
    elif kind == "label":
        value.loc[0, "label"] = 2
    elif kind == "request":
        value["request"] = 1
    else:
        value = value.iloc[:0]
    with pytest.raises(ValueError):
        audit_action(value, "request", "label")


def test_match_randomized_against_augmenting_paths():
    rng = np.random.default_rng(125)
    for _ in range(250):
        r = np.sort(rng.choice(100, size=int(rng.integers(0, 25)), replace=False))
        y = np.sort(rng.choice(100, size=int(rng.integers(0, 25)), replace=False))
        common = np.intersect1d(r, y)
        r, y = np.setdiff1d(r, common), np.setdiff1d(y, common)
        for lower, upper in WINDOWS.values():
            pairs = match_minutes(r, y, lower, upper)
            assert len(pairs) == graph_match_count(r, y, lower, upper)
            assert len({a for a, _ in pairs}) == len(pairs)
            assert len({b for _, b in pairs}) == len(pairs)


def test_corrupt_matching_and_aggregate_rejected():
    value = frame([1, 2], [3, 4])
    result, ledger = audit(value)
    changed = deepcopy(ledger)
    changed["pairs"][0]["target_minute"] += 100
    with pytest.raises(AssertionError):
        verify_action(value, "request", "label", result, changed)
    result["fp"] += 1
    with pytest.raises(AssertionError):
        verify_action(value, "request", "label", result, ledger)
