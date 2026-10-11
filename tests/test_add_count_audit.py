import copy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.add_count_audit import attach_add_counts, audit_count_strata, count_masks
from wonyotti_fr.add_count_reference import verify_add_count_links, verify_count_strata
from wonyotti_fr.add_count_research import checked_count_reference, verify_original_metrics
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.holding_stress import seal
from wonyotti_fr.management_rank_research import CONTEXT


def source():
    start = pd.Timestamp("2020-01-01", tz="UTC")
    times = start + pd.to_timedelta([0, 1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], unit="min")
    signed = np.array([100, 20, 10, 5, -20, 3, 0, 1, 1, 1, 1, 1, -133, -5, -2, 17, 0, 10, 2, 1])
    quantities = np.cumsum(signed)
    events = pd.DataFrame({"time": times, "order_key": ["open", "open", "a", "a", "r", "r", "funding", "b", "c", "d", "e", "f", "reverse", "reverse", "g", "close", "funding", "a", "h", "a"],
        "exectype": np.where(signed == 0, "Funding", "Trade"), "side": np.where(signed >= 0, "Buy", "Sell"),
        "quantity": abs(signed), "cost_satoshi": abs(signed) * 10000})
    actions = events[["time", "order_key"]].copy()
    actions["action"] = ["open", "increase", "increase", "increase", "reduce", "increase", "funding", "increase", "increase", "increase", "increase", "increase", "reverse", "increase", "increase", "close", "funding", "open", "increase", "increase"]
    actions["before_qty"], actions["after_qty"] = np.r_[0, quantities[:-1]], quantities
    actions["episode_id"] = [1] * 12 + [2] * 5 + [3] * 3
    counts = np.array([0, 0, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 0, 0, 1, 0, 0, 0, 1, 1])
    positions = [0, 1, 1, 3, *range(4, 20)]
    ends = [times[0] + pd.Timedelta(nanoseconds=1), times[1] + pd.Timedelta(nanoseconds=1), times[2], times[3] + pd.Timedelta(nanoseconds=1), *[t + pd.Timedelta(nanoseconds=1) for t in times[4:]]]
    entries = [times[0]] * 12 + [times[12]] * 3 + [pd.NaT] * 2 + [times[17]] * 3
    frame = pd.DataFrame({"end": ends, "state_time": times[positions], "entry_time": pd.to_datetime([entries[i] for i in positions], utc=True),
        "direction": np.sign(quantities[positions]), "episode_id": np.where(quantities[positions] != 0, actions.episode_id.to_numpy()[positions], 0),
        "adds": counts[positions], "adds_capped": np.minimum(counts[positions], 5)}, index=np.arange(20) * 10)
    return frame, actions, events


def test_original_count_semantics_and_strict_same_time_boundary():
    frame, actions, events = source()
    links = attach_add_counts(frame, actions, events)
    verify_add_count_links(frame, actions, events, links)
    assert links.event_row.tolist() == [0, 1, 1, 3, *range(4, 20)]
    assert links.adds.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 3, 4, 5, 6, 0, 0, 1, 0, 0, 0, 1, 1]
    assert links.bucket.iloc[11] == ">5"
    for data in [actions, events]:
        data["time"] = data.time.astype("datetime64[us, UTC]")
    pd.testing.assert_frame_equal(attach_add_counts(frame, actions, events), links, check_exact=True)


def test_future_event_change_preserves_past_counts():
    frame, actions, events = source()
    old = attach_add_counts(frame, actions, events)
    actions.loc[19, "order_key"] = events.loc[19, "order_key"] = "new-future"
    frame.loc[190, "adds"] = frame.loc[190, "adds_capped"] = 2
    new = attach_add_counts(frame, actions, events)
    verify_add_count_links(frame, actions, events, new)
    pd.testing.assert_frame_equal(old.iloc[:-1], new.iloc[:-1], check_exact=True)
    assert old.adds.iloc[-1] != new.adds.iloc[-1]


@pytest.mark.parametrize("damage", ["count", "capped", "episode", "direction", "state_time", "entry_time", "end", "naive", "quantity", "cost", "side", "key", "funding", "event_time", "action_state", "action_label"])
def test_source_damage_rejected(damage):
    frame, actions, events = source()
    if damage in {"count", "capped", "episode", "direction"}:
        frame.loc[30, {"count": "adds", "capped": "adds_capped", "episode": "episode_id", "direction": "direction"}[damage]] += 1
    elif damage in {"state_time", "entry_time", "end"}:
        frame.loc[30, damage] = pd.NaT
    elif damage == "naive":
        frame["end"] = frame.end.dt.tz_localize(None)
    elif damage == "quantity":
        events.loc[3, "quantity"] += 1
    elif damage == "cost":
        events["cost_satoshi"] = events.cost_satoshi.astype(float)
        events.loc[3, "cost_satoshi"] = np.inf
    elif damage == "side":
        events.loc[3, "side"] = "Sell"
    elif damage == "key":
        events.loc[3, "order_key"] = "other"
    elif damage == "funding":
        events.loc[3, "exectype"] = "Funding"
    elif damage == "event_time":
        events.loc[3, "time"] += pd.Timedelta(seconds=1)
    elif damage == "action_state":
        actions.loc[3, "after_qty"] += 1
    else:
        actions.loc[3, "action"] = "reduce"
    with pytest.raises((ValueError, AssertionError)):
        links = attach_add_counts(frame, actions, events)
        verify_add_count_links(frame, actions, events, links)


@pytest.mark.parametrize("damage", ["event_row", "bucket", "adds", "state_time"])
def test_independent_link_damage_rejected(damage):
    frame, actions, events = source()
    links = attach_add_counts(frame, actions, events)
    if damage == "bucket":
        links.loc[30, damage] = "0"
    elif damage == "state_time":
        links.loc[30, damage] += pd.Timedelta(microseconds=1)
    else:
        links.loc[30, damage] += 1
    with pytest.raises(AssertionError):
        verify_add_count_links(frame, actions, events, links)


def population():
    frame = pd.DataFrame({"end": pd.date_range("2020-01-01", periods=28, freq="min", tz="UTC"),
        "episode_id": np.repeat([1, 2, 3, 4], 7), "adds": np.tile([0, 1, 2, 3, 4, 5, 19], 4), "y_exit": 0})
    for action in ["reduce", "increase"]:
        frame["past_" + action + "_exists"] = np.repeat([0, 1, 1, 1], 7)
        frame["past_" + action + "_log_count_15m"] = np.repeat([0, 0, np.log(2), np.log(3)], 7)
        frame["y_" + action] = np.tile([1, 0, 1, 0, 1, 0, 1], 4)
    predictions = {name: frame[CONTEXT].copy() for name in ["v127", "v133"]}
    for i, saved in enumerate(predictions.values()):
        for action in ["reduce", "increase"]:
            saved[action + "_tree_score"] = np.tile([0, .3, .3, .4, 1, .2, .1], 4)
            saved[action + "_tree_request"] = np.arange(len(frame)) % (i + 2) == 0
    return frame, predictions


def test_all_buckets_metrics_sums_ties_zero_positive_and_empty():
    frame, predictions = population()
    result = audit_count_strata(frame, predictions)
    verify_count_strata(frame, predictions, result)
    assert result["increase"]["all"]["5"]["models"]["v127"]["average_precision"] is None
    assert result["increase"]["all"][">5"]["support"]["maximum_adds"] == 19
    mask = frame.adds.ne(4)
    frame, predictions = frame.loc[mask], {k: v.loc[mask] for k, v in predictions.items()}
    result = audit_count_strata(frame, predictions)
    verify_count_strata(frame, predictions, result)
    assert result["increase"]["all"]["4"]["models"]["v127"] is None
    assert result["increase"]["all"]["4"]["support"]["rows"] == 0
    original = {name: {a: {s: {"tree": buckets["all"]["models"][name]} for s, buckets in states.items()} for a, states in result.items()} for name in predictions}
    verify_original_metrics(result, original)
    original["v127"]["increase"]["all"]["tree"] = None
    with pytest.raises((ValueError, AssertionError)):
        verify_original_metrics(result, original)


@pytest.mark.parametrize("damage", ["nan", "negative", "fraction", "large", "score", "request", "context", "missing_model", "saved_metric", "saved_support"])
def test_count_or_prediction_corruption_rejected(damage):
    frame, predictions = population()
    if damage in {"nan", "negative", "fraction", "large"}:
        frame["adds"] = frame.adds.astype(float)
        frame.loc[0, "adds"] = {"nan": np.nan, "negative": -1, "fraction": 1.5, "large": 2**31}[damage]
        with pytest.raises(ValueError):
            count_masks(frame)
        return
    if damage.startswith("saved"):
        saved = copy.deepcopy(audit_count_strata(frame, predictions))
        if damage == "saved_metric":
            saved["increase"]["all"]["0"]["models"]["v133"]["log_loss"] += .1
        else:
            saved["increase"]["all"]["0"]["support"]["episodes"] += 1
        with pytest.raises((ValueError, AssertionError)):
            verify_count_strata(frame, predictions, saved)
        return
    if damage == "score":
        predictions["v127"].loc[0, "increase_tree_score"] = np.inf
    elif damage == "request":
        predictions["v127"]["increase_tree_request"] = 1
    elif damage == "context":
        predictions["v127"].loc[0, "end"] += pd.Timedelta(seconds=1)
    else:
        predictions.pop("v127")
    with pytest.raises((ValueError, AssertionError)):
        audit_count_strata(frame, predictions)


@pytest.mark.parametrize("damage", ["digest", "symlink", "complete", "accepted", "accounts", "sources", "files"])
def test_prior_proof_damage_rejected(tmp_path, monkeypatch, damage):
    run = tmp_path / "run"
    run.mkdir()
    summary = {"complete": True, "independent_verification_complete": True, "all_inputs_and_code_unchanged": True,
        "order_price_admitted": False, "new_accounts_executed": False, "profitability_accepted": False}
    if damage == "complete":
        summary["complete"] = False
    elif damage == "accepted":
        summary["order_price_admitted"] = True
    elif damage == "accounts":
        summary["new_accounts_executed"] = True
    save_json(run / "summary.json", summary)
    save_json(run / "manifest.json", {"settings": {"reference": "unused", "reference_sha256": "unused", "sources": {"teacher": "a"}}})
    seal(run)
    proof = tmp_path / "proof.json"
    save_json(proof, {"run": str(run), "complete": True, "files_sha256": sha256(run / "files.json"),
        "summary_sha256": sha256(run / "summary.json"), "new_accounts_executed": False, "profitability_accepted": False})
    monkeypatch.setattr("wonyotti_fr.add_count_research.checked_price_learning_reference", lambda *args: {"teacher": "b" if damage == "sources" else "a"})
    digest = sha256(proof)
    if damage == "digest":
        digest = "0" * 64
    elif damage == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(proof)
        proof = link
    elif damage == "files":
        save_json(run / "summary.json", {})
    with pytest.raises(ValueError):
        checked_count_reference(proof, digest)
