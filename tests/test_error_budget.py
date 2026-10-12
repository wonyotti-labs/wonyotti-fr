import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.count_region_management import FEATURES, REGIONS
from wonyotti_fr.error_budget import (
    MODELS,
    apply_budget_cutoff,
    choose_budget_cutoff,
    evaluate_budget,
    fit_budget_cutoffs,
    focal_retention,
    validate_budget_cutoff,
)
from wonyotti_fr.error_budget_reference import (
    reference_cutoff,
    verify_budget_cutoffs,
    verify_budget_evaluation,
    verify_focal_retention,
)
from wonyotti_fr.error_budget_research import checked_budget_reference
from wonyotti_fr.holding_stress import seal
from wonyotti_fr.management_capacity_learning import save_frame
from wonyotti_fr.management_rank_research import CONTEXT


def population(date="2021-02-01", n=96):
    rng = np.random.default_rng(138)
    i = np.arange(n)
    frame = pd.DataFrame(0.0, index=i * 10 + 100, columns=FEATURES)
    frame["end"] = pd.date_range(date, periods=n, freq="min", tz="UTC")
    frame["episode_id"] = i // 12 + 1
    frame["adds"] = i % 8
    frame["adds_capped"] = np.minimum(frame.adds, 5)
    frame["direction"] = np.where(i % 2, -1.0, 1.0)
    frame["y_exit"] = 0
    for shift, action in enumerate(["reduce", "increase"]):
        state = (i // 8 + shift) % 3
        frame["past_" + action + "_exists"] = (state > 0).astype(float)
        frame["past_" + action + "_log_count_15m"] = np.where(state == 2, np.log(2.0), 0.0)
        frame["y_" + action] = (rng.uniform(size=n) < 0.3).astype(int)
    predictions = {}
    for model in MODELS:
        data = frame[CONTEXT].copy()
        for action in ["reduce", "increase"]:
            p = rng.choice([0.0, 0.1, 0.5, 0.9, 1.0], size=n)
            data[action + "_tree_score"] = p
            data[action + "_tree_request"] = p >= (0.8 if model == "v127" else 0.4)
        predictions[model] = data
    return frame, predictions


@pytest.mark.parametrize(
    "negative,budget,expected",
    [
        ([0.8, 0.8, 0.2], 0, np.nextafter(0.8, np.inf)),
        ([0.8, 0.8, 0.2], 1, np.nextafter(0.8, np.inf)),
        ([0.8, 0.8, 0.2], 2, np.nextafter(0.2, np.inf)),
        ([0.8, 0.8, 0.2], 3, 0.0),
        ([0.0], 0, np.nextafter(0.0, np.inf)),
        ([1.0], 0, np.nextafter(1.0, np.inf)),
        ([0.5, np.nextafter(0.5, np.inf)], 1, np.nextafter(0.5, np.inf)),
        ([], 0, 0.0),
    ],
)
def test_order_statistic_ties_and_float_boundaries(negative, budget, expected):
    p = np.array([*negative, 0.95, 0.01])
    y = np.array([*[0] * len(negative), 1, 1])
    before = p.copy()
    actual = choose_budget_cutoff(y, p, budget)
    assert actual == reference_cutoff(y, p, budget)
    assert actual["threshold"] == expected
    np.testing.assert_array_equal(p, before)
    request = apply_budget_cutoff(p, actual)
    assert np.count_nonzero(request & (y == 0)) <= budget
    if expected:
        assert np.count_nonzero((p >= np.nextafter(expected, -np.inf)) & (y == 0)) > budget


def test_empty_calibration_is_not_all_positive_calibration():
    empty = choose_budget_cutoff(np.array([], int), np.array([], float), 0)
    positive = choose_budget_cutoff([1, 1], [0.2, 0.8], 0)
    assert empty["status"] == "no_calibration_rows" and empty["threshold"] is None
    assert positive["status"] == "supported" and positive["threshold"] == 0
    assert apply_budget_cutoff([0.1, 1.0], empty) is None
    np.testing.assert_array_equal(apply_budget_cutoff([0.1, 1.0], positive), [True, True])


def test_all_groups_independent_verification_and_nullable_parquet(tmp_path):
    cal, cp = population()
    cuts = fit_budget_cutoffs(cal, cp)
    verify_budget_cutoffs(cal, cp, cuts)
    assert len(cuts) == 18
    for date in ["2020-03-01", "2021-02-01", "2021-09-01"]:
        frame, pair = population(date)
        original = frame.copy(deep=True)
        evaluation, ledger = evaluate_budget(frame, pair, cuts)
        verify_budget_evaluation(frame, pair, cuts, evaluation, ledger)
        retained = focal_retention(frame, pair, cuts, ledger)
        verify_focal_retention(frame, pair, cuts, ledger, retained)
        save_frame(tmp_path / (date + ".parquet"), ledger)
        pd.testing.assert_frame_equal(frame, original, check_exact=True)
        empty, ep = frame.iloc[:0], {k: v.iloc[:0] for k, v in pair.items()}
        values, el = evaluate_budget(empty, ep, cuts)
        verify_budget_evaluation(empty, ep, cuts, values, el)
        assert all(value["support"]["rows"] == 0 for value in values.values())


def test_future_labels_cannot_change_thresholds_or_requests():
    cal, cp = population()
    cuts = fit_budget_cutoffs(cal, cp)
    future, pair = population("2021-08-01")
    _, expected = evaluate_budget(future, pair, cuts)
    for action in ["reduce", "increase"]:
        future["y_" + action] = 1 - future["y_" + action]
        for data in pair.values():
            data["y_" + action] = future["y_" + action]
    assert fit_budget_cutoffs(cal, cp) == cuts
    _, actual = evaluate_budget(future, pair, cuts)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    with pytest.raises(ValueError, match="상반기"):
        fit_budget_cutoffs(future, pair)


def test_positive_scores_do_not_change_negative_budget_threshold():
    frame, pair = population()
    cuts = fit_budget_cutoffs(frame, pair)
    for action in ["reduce", "increase"]:
        for data in pair.values():
            data.loc[frame["y_" + action].eq(1), action + "_tree_score"] = 1.0
    assert fit_budget_cutoffs(frame, pair) == cuts
    verify_budget_cutoffs(frame, pair, cuts)


def test_future_group_without_calibration_remains_unsupported():
    frame, pair = population()
    empty, ep = frame.iloc[:0], {k: v.iloc[:0] for k, v in pair.items()}
    cuts = fit_budget_cutoffs(empty, ep)
    verify_budget_cutoffs(empty, ep, cuts)
    future, fp = population("2021-09-01")
    values, ledger = evaluate_budget(future, fp, cuts)
    verify_budget_evaluation(future, fp, cuts, values, ledger)
    assert ledger.iloc[:, 2:].isna().all().all()
    for value in values.values():
        for m in value["models"].values():
            assert m["budget_matched"] is None and m["overlap"] is None
            assert m["unsupported_rows"] == value["support"]["rows"]
    retained = focal_retention(future, fp, cuts, ledger)
    verify_focal_retention(future, fp, cuts, ledger, retained)
    assert retained["budget_supported"] is False and retained["retained_new_true_positive"] is None


@pytest.mark.parametrize("label", [0, 1])
def test_no_positive_or_negative_labels_keep_defined_metrics(label):
    frame, pair = population()
    for action in ["reduce", "increase"]:
        frame["y_" + action] = label
        for values in pair.values():
            values["y_" + action] = label
    cuts = fit_budget_cutoffs(frame, pair)
    result, ledger = evaluate_budget(frame, pair, cuts)
    verify_budget_cutoffs(frame, pair, cuts)
    verify_budget_evaluation(frame, pair, cuts, result, ledger)
    for value in result.values():
        for m in value["models"].values():
            assert (
                m["original"]["false_positive_rate"] is None
                if label
                else m["original"]["average_precision"] is None
            )


@pytest.mark.parametrize("bad", [-1, 4, 0.5, True, np.int64(1)])
def test_invalid_budget_rejected(bad):
    with pytest.raises(ValueError):
        choose_budget_cutoff([0, 1, 0], [0.2, 0.3, 0.5], bad)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0, 1.1, True, "bad"])
def test_invalid_score_rejected(bad):
    scores = np.array([bad])
    with pytest.raises(ValueError):
        choose_budget_cutoff([0], scores, 0)


@pytest.mark.parametrize(
    "damage", ["threshold", "boundary", "ties", "fp", "slack", "source", "status", "extra"]
)
def test_corrupt_cutoff_rejected(damage):
    value = choose_budget_cutoff([0, 0, 1], [0.5, 0.5, 0.8], 1)
    changes = {
        "threshold": ("threshold", 0.5),
        "boundary": ("boundary_score", 0.6),
        "ties": ("boundary_negative_ties", 0),
        "fp": ("false_positive", 2),
        "slack": ("unused_budget", 0),
        "source": ("source_rows", 1),
        "status": ("status", "no_calibration_rows"),
        "extra": ("new", 0),
    }
    k, v = changes[damage]
    value[k] = v
    with pytest.raises(ValueError):
        validate_budget_cutoff(value)


@pytest.mark.parametrize(
    "damage",
    [
        "nan",
        "probability",
        "request",
        "context",
        "time",
        "naive",
        "duplicate",
        "count",
        "direction",
        "label",
        "model",
    ],
)
def test_input_damage_rejected(damage):
    frame, pair = population()
    if damage in {"nan", "probability"}:
        pair["v137"].iloc[0, pair["v137"].columns.get_loc("increase_tree_score")] = (
            np.nan if damage == "nan" else 2.0
        )
    elif damage == "request":
        pair["v137"]["increase_tree_request"] = pair["v137"].increase_tree_request.astype(int)
    elif damage == "context":
        pair["v137"].iloc[0, pair["v137"].columns.get_loc("episode_id")] += 1
    elif damage in {"time", "naive", "duplicate"}:
        if damage == "time":
            frame["end"] += pd.Timedelta(seconds=1)
        elif damage == "naive":
            frame["end"] = frame.end.dt.tz_localize(None)
        else:
            frame.iloc[0, frame.columns.get_loc("end")] = frame.end.iloc[1]
        for values in pair.values():
            values["end"] = frame.end
    elif damage == "count":
        frame.iloc[0, frame.columns.get_loc("adds_capped")] = 1
    elif damage == "direction":
        frame.iloc[0, frame.columns.get_loc("direction")] = 0
    elif damage == "label":
        frame["y_reduce"] = 2
        for values in pair.values():
            values["y_reduce"] = 2
    else:
        pair.pop("v137")
    with pytest.raises((ValueError, AssertionError)):
        fit_budget_cutoffs(frame, pair)


@pytest.mark.parametrize(
    "damage", ["ledger", "missing", "metrics", "overlap", "cutoff", "retention"]
)
def test_independent_checks_reject_damage(damage):
    frame, pair = population()
    cuts = fit_budget_cutoffs(frame, pair)
    result, ledger = evaluate_budget(frame, pair, cuts)
    key = "increase__quiet__all"
    column = "v137__" + key
    if damage == "cutoff":
        cuts[key]["models"]["v137"]["threshold"] = np.nextafter(
            cuts[key]["models"]["v137"]["threshold"], np.inf
        )
        with pytest.raises((ValueError, AssertionError)):
            verify_budget_cutoffs(frame, pair, cuts)
        return
    if damage == "retention":
        result = focal_retention(frame, pair, cuts, ledger)
        result["retained_new_true_positive"] += 1
        with pytest.raises(ValueError):
            verify_focal_retention(frame, pair, cuts, ledger, result)
        return
    if damage == "ledger":
        idx = ledger[column].first_valid_index()
        ledger.loc[idx, column] = not ledger.loc[idx, column]
    elif damage == "missing":
        idx = ledger.index[ledger[column].isna()][0]
        ledger.loc[idx, column] = False
    elif damage == "metrics":
        result[key]["models"]["v137"]["budget_matched"]["false_positive"] += 1
    else:
        result[key]["models"]["v137"]["overlap"]["added"]["true_positive"] += 1
    with pytest.raises((ValueError, AssertionError)):
        verify_budget_evaluation(frame, pair, cuts, result, ledger)


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "complete",
        "admitted",
        "accounts",
        "sources",
        "features",
        "regions",
        "iterations",
        "total",
        "max_adds",
        "file",
        "digest",
        "symlink",
    ],
)
def test_prior_proof_binding(tmp_path, monkeypatch, damage):
    run = tmp_path / "run"
    run.mkdir()
    summary = dict.fromkeys(
        ["complete", "independent_verification_complete", "all_inputs_and_code_unchanged"], True
    )
    summary.update(
        count_region_admitted=False, new_accounts_executed=False, profitability_accepted=False
    )
    if damage == "complete":
        summary["complete"] = False
    elif damage == "admitted":
        summary["count_region_admitted"] = True
    elif damage == "accounts":
        summary["new_accounts_executed"] = True
    settings = {
        "reference": "unused",
        "reference_sha256": "unused",
        "sources": {"teacher": "a"},
        "features": FEATURES.copy(),
        "regions": list(REGIONS),
        "iterations_per_region": 128,
        "total_new_trees": 512,
        "bot_max_adds_unchanged": 5,
    }
    for cause, key, wrong in [
        ("features", "features", []),
        ("regions", "regions", ["capped", "below"]),
        ("iterations", "iterations_per_region", 256),
        ("total", "total_new_trees", 1024),
        ("max_adds", "bot_max_adds_unchanged", 6),
    ]:
        if damage == cause:
            settings[key] = wrong
    save_json(run / "summary.json", summary)
    save_json(run / "manifest.json", {"settings": settings})
    seal(run)
    proof = tmp_path / "proof.json"
    save_json(
        proof,
        {
            "run": str(run),
            "complete": True,
            "files_sha256": sha256(run / "files.json"),
            "summary_sha256": sha256(run / "summary.json"),
            "new_accounts_executed": False,
            "profitability_accepted": False,
        },
    )
    monkeypatch.setattr(
        "wonyotti_fr.error_budget_research.checked_count_region_reference",
        lambda *args: {"teacher": "b" if damage == "sources" else "a"},
    )
    digest = "0" * 64 if damage == "digest" else sha256(proof)
    if damage == "file":
        save_json(run / "summary.json", {})
    elif damage == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(proof)
        proof = link
    if damage == "none":
        assert checked_budget_reference(proof, digest) == {
            "teacher": "a",
            "region_candidate": str(run),
        }
    else:
        with pytest.raises(ValueError):
            checked_budget_reference(proof, digest)
