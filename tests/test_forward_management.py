from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr import forward_weight_learning as learning
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.first_state_models import FEATURES
from wonyotti_fr.forward_management import (
    ARMS,
    CONTEXT,
    FOLDS,
    build_forward_weights,
    fit_forward_auxiliaries,
    fold_masks,
    forward_partition,
    forward_weight_groups,
    quarter_tail_weights,
    require_forward_support,
)
from wonyotti_fr.forward_management_reference import (
    reference_forward_partition,
    verify_forward_auxiliaries,
    verify_forward_weight_groups,
    verify_forward_weights,
)
from wonyotti_fr.holding_stress import seal
from wonyotti_fr.quiet_management import CLASSES
from wonyotti_fr.tail_weighted_management import SCOPE
from wonyotti_fr.tail_weighted_management import WEIGHT_RULE as PARENT_RULE


@pytest.fixture(scope="module")
def training():
    rng = np.random.default_rng(142)
    parts = []
    for quarter, date in enumerate(["2020-01-03", "2020-04-03", "2020-07-03", "2020-10-03"]):
        count = 1800
        frame = pd.DataFrame(rng.normal(size=(count, len(FEATURES))), columns=FEATURES)
        frame["end"] = pd.date_range(date, periods=count, freq="min", tz="UTC")
        frame["entry_time"] = frame.end - pd.Timedelta(hours=1)
        frame["label_end"] = frame.end + pd.Timedelta(minutes=1)
        frame["episode_id"] = np.arange(count) + quarter * count + 1
        frame["direction"] = 1
        frame["adds"] = np.arange(count) % 7
        frame["adds_capped"] = np.minimum(frame.adds, 5)
        for shift, action in enumerate(["reduce", "increase"]):
            state = (np.arange(count) + shift) % 3
            frame["past_" + action + "_exists"] = (state != 0).astype(float)
            frame["past_" + action + "_log_count_15m"] = np.where(state == 2, np.log1p(2), 0.0)
        for action in ["exit", "reduce", "increase"]:
            frame["y_" + action] = (
                rng.random(count) < 0.2 + 0.2 * frame[FEATURES[0]].gt(0.5)
            ).astype(int)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def fitted(training):
    return fit_forward_auxiliaries(training)


@pytest.fixture(scope="module")
def parents(training):
    result = {}
    for action in ["reduce", "increase"]:
        part = training.loc[training["past_" + action + "_log_count_15m"].eq(0)]
        x, y = part[FEATURES].to_numpy(float), part[["y_" + action]].to_numpy(int)
        model, _ = CLASSES[action].fit(x, y, x[:1])
        result[action] = {"tree": model.to_dict(), "linear": {}, "scope": SCOPE}
    return result


@pytest.fixture(scope="module")
def weights(training, fitted, parents):
    return build_forward_weights(training, parents, fitted[0], fitted[1])


def test_all_auxiliary_trees_and_target_predictions_independently_refitted(training, fitted):
    proof = verify_forward_auxiliaries(training, *fitted)
    assert proof["independent_auxiliary_trees"] == 1536
    assert proof["target_action_predictions"] == 7200


def test_both_weights_neutral_rows_and_all_groups(training, fitted, parents, weights):
    ledgers, details, overlap = weights
    assert verify_forward_weights(training, parents, fitted[1], *weights)
    for arm in ARMS:
        ledger = ledgers[arm]
        grouped = forward_weight_groups(training, ledger)
        verify_forward_weight_groups(training, ledger, grouped)
        for action in ["reduce", "increase"]:
            neutral = ledger[action + "_reason"].eq("warmup")
            assert ledger.loc[neutral, action + "_units"].eq(10).all()
            assert ledger.loc[neutral, action + "_score"].isna().all()
            assert grouped[action]["all_training"]["all"]["rows"] == 4800
            assert len(grouped[action]) == 37


def test_exact_calendar_boundaries_and_prior_entry(training):
    frame = training.copy()
    positions = list(range(1800, 1814))
    for i, stamp in zip(
        positions[:6],
        [
            "2020-04-01 00:00",
            "2020-04-01 23:59",
            "2020-04-02 00:00",
            "2020-06-29 23:58",
            "2020-06-29 23:59",
            "2020-06-30 00:00",
        ],
        strict=True,
    ):
        frame.loc[i, "end"] = pd.Timestamp(stamp, tz="UTC")
        frame.loc[i, "entry_time"] = frame.loc[i, "end"] - pd.Timedelta(minutes=1)
        frame.loc[i, "label_end"] = frame.loc[i, "end"] + pd.Timedelta(minutes=1)
        frame.loc[i, "past_reduce_log_count_15m"] = 0.0
    frame.loc[positions[6], "entry_time"] = pd.Timestamp("2020-03-30", tz="UTC")
    frame.loc[positions[6], "past_reduce_log_count_15m"] = 0.0
    frame = frame.sort_values("end").reset_index(drop=True)
    ledger, support = forward_partition(frame)
    reference_forward_partition(frame, ledger, support)
    keyed = ledger.set_index("episode_id")
    expected = [
        "left_boundary",
        "left_boundary",
        "eligible",
        "eligible",
        "right_boundary",
        "right_boundary",
        "prior_entry",
    ]
    for i, reason in zip(positions[:7], expected, strict=True):
        assert keyed.loc[i + 1, "reduce_reason"] == reason


def test_later_quarter_cannot_change_earlier_auxiliary_predictions_or_raw_units(
    training, fitted, parents, weights
):
    changed = training.copy()
    future = changed.end.ge(pd.Timestamp("2020-10-01", tz="UTC"))
    changed.loc[future, FEATURES[0]] += 100
    for action in ["reduce", "increase"]:
        changed.loc[future, "y_" + action] = 1 - changed.loc[future, "y_" + action]
    altered = fit_forward_auxiliaries(changed)
    altered_weights = build_forward_weights(changed, parents, altered[0], altered[1])
    for action in ["reduce", "increase"]:
        # 전체 연도 정규화는 이후 분기 정답에 따라 달라질 수 있다.
        for fold in ["Q2", "Q3"]:
            assert fitted[0][action][fold] == altered[0][action][fold]
            mask = fitted[1][action + "_fold"].eq(fold)
            pd.testing.assert_series_equal(
                fitted[1].loc[mask, action + "_forward_score"],
                altered[1].loc[mask, action + "_forward_score"],
            )
            for arm in ARMS:
                assert (
                    weights[1][arm][action]["folds"][fold]
                    == altered_weights[1][arm][action]["folds"][fold]
                )
                pd.testing.assert_series_equal(
                    weights[0][arm].loc[mask, action + "_units"],
                    altered_weights[0][arm].loc[mask, action + "_units"],
                )


@pytest.mark.parametrize("role", ["auxiliary", "target"])
def test_support_failure_prevents_any_fit(training, monkeypatch, role):
    from wonyotti_fr import forward_management as module

    frame = training.copy()
    a, t = fold_masks(frame, "reduce", "Q2")
    frame.loc[a if role == "auxiliary" else t, "y_reduce"] = 0

    def forbidden(*args, **kwargs):
        pytest.fail("지원 실패 뒤 학습 호출")

    monkeypatch.setattr(module.HistGradientBoostingClassifier, "fit", forbidden)
    _, support = forward_partition(frame)
    assert not support["reduce"]["Q2"][role]["supported"]
    with pytest.raises(ValueError, match="지원 부족"):
        fit_forward_auxiliaries(frame)


@pytest.mark.parametrize(
    "damage",
    ["future", "fractional_label", "timezone", "order", "episode", "nan", "count", "minute"],
)
def test_invalid_training_input_rejected(training, damage):
    frame = training.copy()
    if damage == "future":
        for column in CONTEXT[:3]:
            frame[column] += pd.DateOffset(years=1)
    elif damage == "fractional_label":
        frame["y_reduce"] = frame.y_reduce.astype(float)
        frame.loc[0, "y_reduce"] = 0.5
    elif damage == "timezone":
        frame["end"] = frame.end.dt.tz_convert("Asia/Seoul")
    elif damage == "order":
        frame = frame.iloc[::-1].reset_index(drop=True)
    elif damage == "episode":
        frame.loc[1, "episode_id"] = frame.loc[0, "episode_id"]
    elif damage == "nan":
        frame.loc[0, FEATURES[0]] = np.nan
    elif damage == "count":
        frame.loc[0, "past_reduce_log_count_15m"] = 0.1
    else:
        frame.loc[0, "end"] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError):
        forward_partition(frame)


def test_tail_ties_neutral_and_positive_scores():
    y = np.tile(np.r_[np.zeros(399, int), np.ones(21, int)], 7)
    ids = np.repeat(["", "", "", "", "Q2", "Q3", "Q4"], 420)
    scores = np.tile(np.r_[np.linspace(0, 0.8, 390), np.ones(9), np.zeros(21)], 7)
    scores[ids == ""] = np.nan
    w, hard, units, detail = quarter_tail_weights(y, scores, ids)
    assert all(
        v["rank"] == 4 and v["boundary_ties"] == v["hard_negative_rows"] == 9
        for v in detail["folds"].values()
    )
    assert (units[ids == ""] == 10).all() and (units[y == 1] == 10).all()
    assert hard.sum() == 27 and np.isclose(w.mean(), 1.0)
    scores[(y == 1) & (ids != "")] = 1
    changed = quarter_tail_weights(y, scores, ids)
    for a, b in zip((w, hard, units), changed[:3], strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "damage", ["context", "score", "support", "model", "fingerprint", "source", "missing_fold"]
)
def test_auxiliary_linkage_corruption_rejected(training, fitted, parents, damage):
    models, scores, _ = deepcopy(fitted)
    mask = scores.reduce_fold.eq("Q2")
    index = scores.index[mask][0]
    if damage == "context":
        scores.loc[index, "episode_id"] += 1
    elif damage == "score":
        scores.loc[index, "reduce_forward_score"] = 0.9
    elif damage == "support":
        scores.loc[0, "reduce_forward_score"] = 0.9
    elif damage == "model":
        models["reduce"]["Q2"]["tree"]["models"][0]["baseline"] += 1
    elif damage == "fingerprint":
        models["reduce"]["Q2"]["training_sha256"] = "a" * 64
    elif damage == "source":
        models["reduce"]["Q2"]["rows"] += 1
    else:
        models["reduce"].pop("Q3")
    with pytest.raises((ValueError, AssertionError)):
        build_forward_weights(training, parents, models, scores)


@pytest.mark.parametrize("damage", ["unit", "weight", "neutral", "boundary", "overlap", "group"])
def test_independent_weight_verifier_detects_corruption(training, fitted, parents, weights, damage):
    ledgers, details, overlap = deepcopy(weights)
    ledger = ledgers["v142forward"]
    index = ledger.index[ledger.reduce_fold.eq("Q2")][0]
    if damage == "unit":
        ledger.loc[index, "reduce_units"] = 7
    elif damage == "weight":
        ledger.loc[index, "reduce_weight"] *= 2
    elif damage == "neutral":
        ledger.loc[0, "reduce_units"] = 1
    elif damage == "boundary":
        details["v142forward"]["reduce"]["folds"]["Q2"]["boundary_score"] += 0.01
    elif damage == "overlap":
        overlap["reduce"]["Q2"]["intersection"] += 1
    else:
        saved = forward_weight_groups(training, ledger)
        saved["reduce"]["all_training"]["neutral_negative"]["rows"] += 1
        with pytest.raises(ValueError):
            verify_forward_weight_groups(training, ledger, saved)
        return
    with pytest.raises(AssertionError):
        verify_forward_weights(training, parents, fitted[1], ledgers, details, overlap)


def test_partition_and_support_damage_detected(training):
    ledger, support = forward_partition(training)
    broken = deepcopy(support)
    broken["reduce"]["Q2"]["target"]["positive"] += 1
    with pytest.raises(AssertionError):
        reference_forward_partition(training, ledger, broken)
    broken.pop("increase")
    with pytest.raises(ValueError):
        require_forward_support(broken)
    ledger.loc[1800, "reduce_fold"] = "Q4"
    with pytest.raises(AssertionError):
        reference_forward_partition(training, ledger, support)


def test_unsupported_fold_rejected(training):
    with pytest.raises(ValueError):
        fold_masks(training, "reduce", "Q1")
    assert list(FOLDS) == ["Q2", "Q3", "Q4"]


@pytest.mark.parametrize(
    "damage", [None, "hash", "incomplete", "accepted", "source", "exit", "linear", "models", "file"]
)
def test_fixed_parent_proof_required(tmp_path, monkeypatch, damage):
    root = tmp_path / "fixed"
    root.mkdir()
    summary = {
        k: True
        for k in [
            "complete",
            "independent_verification_complete",
            "all_inputs_and_code_unchanged",
            "new_models_fitted",
        ]
    }
    flags = {"new_accounts_executed": False, "profitability_accepted": False}
    summary.update(
        flags,
        fixed_bins_admitted=False,
        new_trees=512,
        new_linear_models=0,
        preserved_linear_models=2,
        equivalence_control_trees=512,
    )
    if damage == "incomplete":
        summary["independent_verification_complete"] = False
    if damage == "accepted":
        summary["fixed_bins_admitted"] = True
    if damage == "linear":
        summary["new_linear_models"] = 2
    settings = {
        "reference": "prior.json",
        "reference_sha256": "a" * 64,
        "sources": {"teacher": "bound"},
        "features": FEATURES,
        "weight_rule": PARENT_RULE,
        "models": ["v127", "v141"],
    }
    if damage == "models":
        settings["models"] = ["v127", "v140"]
    save_json(root / "summary.json", summary)
    save_json(root / "manifest.json", {"settings": settings})
    seal(root)
    proof = tmp_path / "proof.json"
    save_json(
        proof,
        {
            "complete": True,
            "run": str(root),
            "files_sha256": sha256(root / "files.json"),
            "summary_sha256": sha256(root / "summary.json"),
            "process_exit_code": False if damage == "exit" else 0,
            "new_models_fitted": True,
            **flags,
        },
    )
    monkeypatch.setattr(
        learning,
        "checked_fixed_reference",
        lambda *_: {"teacher": "wrong" if damage == "source" else "bound"},
    )
    if damage == "file":
        (root / "summary.json").write_text("{}")
    expected = "b" * 64 if damage == "hash" else sha256(proof)
    if damage is None:
        assert learning.checked_forward_reference(proof, expected) == {
            "teacher": "bound",
            "fixed": str(root),
        }
    else:
        with pytest.raises(ValueError):
            learning.checked_forward_reference(proof, expected)


@pytest.mark.parametrize("negatives,rank", [(399, 4), (400, 4), (401, 5)])
def test_quarter_integer_ceiling(negatives, rank):
    y = np.r_[np.zeros(3000, int), np.tile(np.r_[np.zeros(negatives, int), np.ones(20, int)], 3)]
    ids = np.r_[np.full(3000, ""), np.repeat(["Q2", "Q3", "Q4"], negatives + 20)]
    scores = np.r_[
        np.full(3000, np.nan), np.tile(np.r_[np.linspace(0, 0.99, negatives), np.ones(20)], 3)
    ]
    _, _, _, info = quarter_tail_weights(y, scores, ids)
    assert all(v["rank"] == v["hard_negative_rows"] == rank for v in info["folds"].values())


@pytest.mark.parametrize(
    "damage",
    [
        "neutral_score",
        "missing_score",
        "infinite",
        "negative",
        "future_fold",
        "fractional_label",
        "short",
        "ess",
    ],
)
def test_invalid_quarter_weights_rejected(damage):
    y = np.r_[np.zeros(3000, int), np.tile(np.r_[np.zeros(399, int), np.ones(21, int)], 3)].astype(
        float
    )
    ids = np.r_[np.full(3000, ""), np.repeat(["Q2", "Q3", "Q4"], 420)]
    scores = np.r_[np.full(3000, np.nan), np.tile(np.linspace(0, 1, 420), 3)]
    if damage == "neutral_score":
        scores[0] = 0
    elif damage == "missing_score":
        scores[3000] = np.nan
    elif damage == "infinite":
        scores[3000] = np.inf
    elif damage == "negative":
        scores[3000] = -0.1
    elif damage == "future_fold":
        ids[3000] = "Q1"
    elif damage == "fractional_label":
        y[3000] = 0.5
    elif damage == "short":
        y, scores, ids = y[:999], scores[:999], ids[:999]
    else:
        y, scores, ids = y[3000:], scores[3000:], ids[3000:]
    with pytest.raises(ValueError):
        quarter_tail_weights(y, scores, ids)


def test_auxiliary_label_boundary_is_strict(training):
    frame = training.copy()
    for i, end in enumerate(["2020-03-30 23:58", "2020-03-30 23:59", "2020-03-31 00:00"]):
        frame.loc[i, "end"] = pd.Timestamp(end, tz="UTC")
        frame.loc[i, "entry_time"] = frame.loc[i, "end"] - pd.Timedelta(hours=1)
        frame.loc[i, "label_end"] = frame.loc[i, "end"] + pd.Timedelta(minutes=1)
        frame.loc[i, "past_reduce_log_count_15m"] = 0.0
    frame = frame.sort_values("end").reset_index(drop=True)
    auxiliary, _ = fold_masks(frame, "reduce", "Q2")
    assert frame.loc[auxiliary, "episode_id"].isin([1, 2, 3]).sum() == 1
    assert 1 in set(frame.loc[auxiliary, "episode_id"])
    ledger, support = forward_partition(frame)
    reference_forward_partition(frame, ledger, support)
