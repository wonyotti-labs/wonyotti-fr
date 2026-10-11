from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.management_rank_audit import audit_scores, tail_cutoffs, validate_cutoffs
from wonyotti_fr.management_rank_reference import reference_cutoffs, verify_score_audit
from wonyotti_fr.management_rank_research import (
    checked_score_reference,
    verified_context,
    verify_original,
    views,
)


def sample(scores=None, labels=None, threshold=.8):
    scores = [.9, .9, .8, .4, .4, .1] if scores is None else scores
    labels = [0, 1, 1, 0, 1, 0] if labels is None else labels
    return pd.DataFrame({"end": pd.date_range("2021-07-02", periods=len(scores), freq="min", tz="UTC"),
                         "episode_id": np.arange(len(scores)) // 2, "label": labels, "score": scores,
                         "request": np.asarray(scores) >= threshold})


def test_tied_positive_ranks_and_threshold_boundary():
    frame = sample()
    cutoffs = tail_cutoffs(frame.score)
    result, ledger = audit_scores(frame, .8, cutoffs)
    verify_score_audit(frame, .8, frame.score, result, ledger)
    assert [(v["rank_start"], v["rank_end"], v["negative_above"], v["negative_equal"])
            for v in ledger["positives"]] == [(1, 2, 0, 1), (3, 3, 1, 0), (4, 5, 1, 1)]
    assert [result["original"][k] for k in ["tp", "fp", "fn"]] == [2, 1, 1]
    assert cutoffs["5%"]["tie_excess"] == 1
    assert result["concentration"]["positive_top1_fraction"] == 1 / 3


@pytest.mark.parametrize("labels", [[0] * 6, [1] * 6, []])
def test_empty_or_single_class_preserved(labels):
    frame = sample([.4] * len(labels), labels)
    calibration = [.1, .2, .3]
    result, ledger = audit_scores(frame, .8, tail_cutoffs(calibration))
    verify_score_audit(frame, .8, calibration, result, ledger)
    assert result["original"]["positive"] == sum(labels)
    if not sum(labels):
        assert result["original"]["recall"] is None
        assert result["scores"]["positive"]["max"] is None
    if not labels:
        assert result["tails"]["contemporaneous_hindsight"]["1%"]["source"] is None


@pytest.mark.parametrize("seed", range(5))
def test_random_ties_and_independent_computation(seed):
    rng = np.random.default_rng(seed)
    p, y = rng.integers(0, 100, 1057) / 100, rng.integers(0, 2, 1057)
    calibration = rng.integers(0, 100, 2011) / 100
    frame = sample(p, y, threshold=.96)
    result, ledger = audit_scores(frame, .96, tail_cutoffs(calibration))
    verify_score_audit(frame, .96, calibration, result, ledger)
    assert tail_cutoffs(calibration) == reference_cutoffs(calibration)


@pytest.mark.parametrize("kind", ["nan", "negative", "above_one", "label", "time", "duplicate_time",
                                  "naive_time", "fractional_episode", "negative_episode", "request", "request_type"])
def test_corrupt_inputs_rejected(kind):
    frame = sample()
    if kind in {"nan", "negative", "above_one"}:
        frame.loc[0, "score"] = {"nan": np.nan, "negative": -.1, "above_one": 1.1}[kind]
    elif kind == "label":
        frame.loc[0, "label"] = 2
    elif kind == "time":
        frame.loc[0, "end"] += pd.Timedelta(seconds=1)
    elif kind == "duplicate_time":
        frame.loc[0, "end"] = frame.end.iloc[1]
    elif kind == "naive_time":
        frame["end"] = frame.end.dt.tz_localize(None)
    elif kind in {"fractional_episode", "negative_episode"}:
        frame["episode_id"] = frame.episode_id.astype(float)
        frame.loc[0, "episode_id"] = .5 if kind == "fractional_episode" else -1
    elif kind == "request_type":
        frame["request"] = frame.request.astype(int)
    else:
        frame.loc[0, "request"] = False
    with pytest.raises(ValueError):
        audit_scores(frame, .8, tail_cutoffs([.1, .2]))


@pytest.mark.parametrize("kind", ["rank", "negative_tie", "episode", "confusion", "quantile", "cutoff", "concentration"])
def test_corrupt_outputs_rejected(kind):
    frame = sample()
    result, ledger = audit_scores(frame, .8, tail_cutoffs(frame.score))
    if kind == "rank":
        ledger["positives"][0]["rank_end"] += 1
    elif kind == "negative_tie":
        ledger["positives"][0]["negative_equal"] += 1
    elif kind == "episode":
        ledger["episodes"][0]["fp"] += 1
    elif kind == "confusion":
        result["tails"]["calibration_fixed"]["1%"]["metrics"]["added_tp"] += 1
    elif kind == "quantile":
        result["scores"]["positive"]["q90"] += .001
    elif kind == "cutoff":
        result["tails"]["contemporaneous_hindsight"]["1%"]["source"]["tie_excess"] += 1
    else:
        result["concentration"]["fp_top1_fraction"] = .2
    with pytest.raises((AssertionError, ValueError)):
        verify_score_audit(frame, .8, frame.score, result, ledger)


def test_fixed_calibration_cutoffs_ignore_future_labels_and_scores():
    frame = sample()
    cuts = tail_cutoffs(frame.score)
    before = deepcopy(cuts)
    changed = sample([.7] * 6, [1] * 6)
    result, ledger = audit_scores(changed, .8, cuts)
    verify_score_audit(changed, .8, frame.score, result, ledger)
    assert cuts == before
    assert result["tails"]["calibration_fixed"]["1%"]["source"]["cutoff"] == .9
    assert result["tails"]["contemporaneous_hindsight"]["1%"]["source"]["cutoff"] == .7


def test_invalid_cutoff_support_or_metadata():
    with pytest.raises(ValueError):
        tail_cutoffs([])
    with pytest.raises(ValueError):
        tail_cutoffs([[.1]])
    cuts = tail_cutoffs([.1, .2, .3])
    cuts["1%"]["source_rows"] = 4
    with pytest.raises(ValueError):
        validate_cutoffs(cuts)


def test_previous_metrics_and_proof_damage_rejected(tmp_path):
    path = tmp_path / "proof.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="지문"):
        checked_score_reference(path, "0" * 64)
    frame = sample()
    result, _ = audit_scores(frame, .8, tail_cutoffs(frame.score))
    metric = {"rows": 6, "positive": 3, "predicted_positive": 3, "true_positive": 2, "false_positive": 1, "false_negative": 1}
    verify_original(result, metric)
    metric["true_positive"] = 3
    with pytest.raises(ValueError):
        verify_original(result, metric)


def test_state_model_binding_and_preserved_context():
    frame = sample()
    current = frame[["end", "episode_id"]].copy()
    current["exit_score"], current["exit_request"], current["y_exit"] = .1, False, 0
    native = {group: {model + "_calibrated": {} for model in ["native", "capacity"]} for group in ["first", "full"]}
    shared, experts = {}, {}
    for action in ["reduce", "increase"]:
        current["y_" + action] = frame.label
        current["past_" + action + "_exists"] = [0., 0., 1., 1., 1., 1.]
        current["past_" + action + "_log_count_15m"] = [0., 0., 0., 0., np.log1p(1), np.log1p(1)]
        for model in ["baseline", "capacity", "tree"]:
            current[action + "_" + model + "_score"] = frame.score
            current[action + "_" + model + "_request"] = frame.request
        shared[action] = {s: {"tree": .8} for s in ["first", "restart"]}
        experts[action] = deepcopy(shared[action])
        for value in native.values():
            for limits in value.values():
                limits[action] = .8
    previous = current.copy()
    verified_context(current, current, previous)
    parts, thresholds = views(current, previous, native, shared, experts)
    assert len(parts) == len(thresholds) == 16
    for key, part in parts.items():
        assert len(part) == 2
        audit_scores(part, thresholds[key], tail_cutoffs(frame.score))
    previous.loc[0, "reduce_baseline_score"] += .01
    with pytest.raises(AssertionError):
        verified_context(current, current, previous)
