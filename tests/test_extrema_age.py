import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.common import save_json
from wonyotti_fr.extrema_age import (
    EXTRA_FEATURES,
    SOURCE_COLUMNS,
    age_groups,
    attach_extrema_age,
    extrema_clock_table,
)
from wonyotti_fr.extrema_age_learning import (
    checked_age_reference,
    verify_age_reference_metrics,
    verify_age_reference_thresholds,
)
from wonyotti_fr.extrema_age_model_reference import (
    verify_age_admission,
    verify_age_predictions,
    verify_age_refits,
    verify_age_usage,
)
from wonyotti_fr.extrema_age_models import (
    FEATURES,
    NATIVE_FEATURES,
    ExtremaAgeBoost,
    ExtremaAgeLinear,
    age_admission,
    age_feature_usage,
    age_predictions,
    fit_age_models,
)
from wonyotti_fr.extrema_age_reference import (
    streaming_extrema,
    validate_clock_state,
    verify_age_groups,
    verify_extrema_inputs,
)
from wonyotti_fr.joint_history_models import evaluate_joint, joint_thresholds
from wonyotti_fr.joint_history_reference import verify_joint_evaluation, verify_joint_thresholds
from wonyotti_fr.management_capacity import LABELS, capacity_predictions, fit_outer_models
from wonyotti_fr.management_market_model_reference import verify_offsets
from wonyotti_fr.path_management import add_price_path


def basic_source():
    end = pd.date_range("2020-01-01 00:01", periods=12, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "end": end,
            "close": [
                100.0,
                102.0,
                102.0,
                101.0,
                99.0,
                100.0,
                98.0,
                98.0,
                99.0,
                101.0,
                101.0,
                101.0,
            ],
            "average_entry": [100.0] * 12,
            "entry_time": [end[0] - pd.Timedelta(minutes=1)] * 5
            + [end[5] - pd.Timedelta(minutes=1)] * 5
            + [pd.NaT] * 2,
            "direction": [1.0] * 5 + [-1.0] * 5 + [0.0] * 2,
            "episode_id": [1] * 5 + [2] * 5 + [0] * 2,
            "usable": True,
        }
    )
    return add_price_path(frame)[SOURCE_COLUMNS]


@pytest.fixture(scope="module")
def study():
    rng = np.random.default_rng(124)
    counts = [3040, 1240, 1240]
    end = pd.date_range("2020-12-29", periods=sum(counts), freq="min", tz="UTC")
    episodes = np.repeat([1, 2, 3], counts)
    direction = np.repeat([1.0, -1.0, 1.0], counts)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    entries = pd.DatetimeIndex(
        np.repeat([end[j] - pd.Timedelta(minutes=1) for j in starts], counts)
    )
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(end))))
    source = pd.DataFrame(
        {
            "end": end,
            "close": closes,
            "average_entry": np.where(np.arange(len(end)) % 1000 < 500, 100.0, 101.0),
            "entry_time": entries,
            "direction": direction,
            "episode_id": episodes,
            "usable": True,
        }
    )
    source = add_price_path(source)[SOURCE_COLUMNS]
    rows = {}
    for phase, start, count in zip(
        ["training", "calibration", "diagnosis"], starts, [3000, 1200, 1200], strict=True
    ):
        context = source.iloc[start + 20 : start + 20 + count].reset_index(drop=True)
        frame = pd.DataFrame(
            rng.normal(size=(count, len(NATIVE_FEATURES))), columns=NATIVE_FEATURES
        )
        for name in SOURCE_COLUMNS:
            frame[name] = context[name]
        frame["favorable_move"] = frame.direction * (frame.close / frame.average_entry - 1)
        frame["log_hold_minutes"] = np.log1p((frame.end - frame.entry_time).dt.total_seconds() / 60)
        frame["past_reduce_exists"] = np.arange(count) % 2
        frame["past_increase_exists"] = (np.arange(count) // 2) % 2
        frame["label_end"] = frame.end + pd.Timedelta(minutes=1)
        for label in LABELS:
            frame[label] = (rng.uniform(size=count) < 0.2 + 0.3 * frame.ret_5m.gt(0.5)).astype(int)
        rows[phase] = frame
    table = extrema_clock_table(source)
    expanded, links = attach_extrema_age(rows, table)
    refs, ref_offsets, _ = fit_outer_models(rows, "depth2_iter256")
    models, offsets, _ = fit_age_models(expanded)
    models, offsets = {**refs, **models}, {**ref_offsets, **offsets}
    constant = rows["calibration"][LABELS].mean().to_numpy()
    scores = {
        p: age_predictions(f, expanded[p], models, offsets, constant) for p, f in rows.items()
    }
    thresholds, details = joint_thresholds(rows["calibration"], scores["calibration"])
    return source, table, rows, expanded, links, models, offsets, scores, thresholds, details


def test_long_short_strict_extrema_ties_flat_and_average_change():
    source = basic_source()
    table = extrema_clock_table(source)
    np.testing.assert_array_equal(table.minutes_since_best, [0, 0, 1, 2, 3] * 2)
    np.testing.assert_array_equal(table.minutes_since_worst, [0, 1, 2, 3, 0] * 2)
    direct, state = streaming_extrema(source)
    pd.testing.assert_frame_equal(
        table[direct.columns], direct, check_exact=False, atol=1e-12, rtol=0
    )
    assert state["direction"] == 0 and state["end"] == source.end.iloc[-1].isoformat()
    changed = source.copy()
    changed.loc[2:4, "average_entry"] = 101.0
    pd.testing.assert_frame_equal(
        extrema_clock_table(changed).drop(columns="average_entry"),
        table.drop(columns="average_entry"),
        check_exact=True,
    )


@pytest.mark.parametrize("split", [1, 3, 5, 7, 10, 11])
def test_stream_restart_preserves_extrema_and_flat_cursor(split):
    source = basic_source()
    whole, final = streaming_extrema(source)
    before, state = streaming_extrema(source.iloc[:split])
    after, restored = streaming_extrema(source.iloc[split:], json.loads(json.dumps(state)))
    parts = [f for f in [before, after] if len(f)]
    pd.testing.assert_frame_equal(pd.concat(parts, ignore_index=True), whole, check_exact=True)
    assert restored == final


@pytest.mark.parametrize(
    "change",
    [
        "gap",
        "duplicate",
        "direction",
        "entry",
        "episode_return",
        "price",
        "average",
        "utc",
        "index",
    ],
)
def test_invalid_source_rejected(change):
    frame = basic_source()
    if change == "gap":
        frame = frame.drop(index=2)
    elif change == "duplicate":
        frame.loc[2, "end"] = frame.loc[1, "end"]
    elif change == "direction":
        frame.loc[2, "direction"] = -1
    elif change == "entry":
        frame.loc[2, "entry_time"] += pd.Timedelta(minutes=1)
    elif change == "episode_return":
        frame.loc[6:8, "episode_id"] = 1
    elif change == "price":
        frame.loc[2, "close"] = np.inf
    elif change == "average":
        frame.loc[2, "average_entry"] = 0
    elif change == "utc":
        frame["end"] = frame.end.dt.tz_localize(None)
    else:
        frame.index = [0] * len(frame)
    with pytest.raises(ValueError):
        extrema_clock_table(frame)


@pytest.mark.parametrize("change", ["future", "price", "missing", "naive", "first", "direction"])
def test_malformed_saved_state_rejected(change):
    _, state = streaming_extrema(basic_source().iloc[:4])
    if change == "future":
        state["high_end"] = "2020-01-01T01:00:00+00:00"
    elif change == "price":
        state["low"] = np.nan
    elif change == "missing":
        state.pop("entry")
    elif change == "naive":
        state["end"] = "2020-01-01 00:04:00"
    elif change == "first":
        state["first"] = state["entry"]
    else:
        state["direction"] = 3
    with pytest.raises(ValueError):
        validate_clock_state(state)


def test_restart_gap_is_rejected_after_flat():
    source = basic_source()
    _, state = streaming_extrema(source)
    later = source.iloc[-1:].copy()
    later["end"] += pd.Timedelta(minutes=2)
    with pytest.raises(ValueError):
        streaming_extrema(later, state)


def test_future_price_and_label_changes_preserve_past(study):
    source, table, rows, expanded = study[:4]
    changed = source.copy()
    changed.loc[changed.index[-10:], "close"] *= 2
    after = extrema_clock_table(changed)
    pd.testing.assert_frame_equal(after.iloc[:-10], table.iloc[:-10], check_exact=True)
    labels = {k: v.copy() for k, v in rows.items()}
    labels["diagnosis"][LABELS] = 1 - labels["diagnosis"][LABELS]
    extra, _ = attach_extrema_age(labels, table)
    for phase in rows:
        pd.testing.assert_frame_equal(
            extra[phase][EXTRA_FEATURES], expanded[phase][EXTRA_FEATURES], check_exact=True
        )


def test_all_original_inputs_refits_requests_and_groups(study):
    source, table, rows, expanded, links, models, offsets, scores, thresholds, details = study
    verified = verify_extrema_inputs(source, table, rows, expanded, links)
    assert sum(v["new_values"] for v in verified["periods"].values()) == 10800
    assert verify_age_refits(expanded, models)["independent_new_trees"] == 768
    verify_offsets(scores["calibration"], rows["calibration"][LABELS].to_numpy(), offsets)
    verify_joint_thresholds(rows["calibration"], scores["calibration"], thresholds, details)
    for phase, frame in rows.items():
        verify_age_predictions(expanded[phase], models, offsets, scores[phase])
        metrics, requests = evaluate_joint(frame, scores[phase], thresholds)
        verify_joint_evaluation(frame, scores[phase], thresholds, metrics, requests)
        for label in ["best", "worst"]:
            groups = age_groups(links[phase], label)
            verify_age_groups(links[phase], label, groups)
            for group in range(5):
                indices = np.flatnonzero(groups == group)
                part = frame.iloc[indices]
                ps = {k: v[indices] for k, v in scores[phase].items()}
                result, decisions = evaluate_joint(part, ps, thresholds)
                verify_joint_evaluation(part, ps, thresholds, result, decisions)
        if phase == "diagnosis":
            decision = age_admission(metrics)
            verify_age_admission(metrics, decision)
            assert len(decision["checks"]) == 31
    verify_age_usage(models["aged"], age_feature_usage(models["aged"]))


def test_truncated_prefix_is_not_silently_used(study):
    source, table, rows, expanded = study[:4]
    partial = extrema_clock_table(source.iloc[10:].reset_index(drop=True))
    assert partial.path_first_end.iloc[0] != table.path_first_end.iloc[0]
    with pytest.raises(ValueError, match="초기 경로 누락"):
        attach_extrema_age(rows, partial)
    for phase in rows:
        assert expanded[phase].path_log_minutes_since_best.iloc[0] >= 0


@pytest.mark.parametrize("change", ["missing", "old_path", "context", "new_feature", "order"])
def test_linkage_and_feature_mutations_detected(study, change):
    source, table, rows, expanded, links = study[:5]
    if change == "missing":
        bad = table.loc[~table.end.eq(rows["training"].end.iloc[0])]
        with pytest.raises(ValueError, match="누락"):
            attach_extrema_age(rows, bad)
    elif change in {"old_path", "context"}:
        bad = {k: v.copy() for k, v in rows.items()}
        name = "best_close_move" if change == "old_path" else "average_entry"
        bad["training"].loc[0, name] += 0.01
        with pytest.raises(AssertionError):
            attach_extrema_age(bad, table)
    else:
        bad = {k: v.copy() for k, v in expanded.items()}
        if change == "new_feature":
            bad["diagnosis"].loc[0, EXTRA_FEATURES[0]] += 0.01
        else:
            bad["diagnosis"] = bad["diagnosis"].iloc[::-1]
        with pytest.raises(AssertionError):
            verify_extrema_inputs(source, table, rows, bad, links)


def test_age_band_boundaries_and_zero_positive(study):
    link = pd.DataFrame({"minutes_since_best": [0, 1, 14, 15, 59, 60, 1439, 1440]})
    np.testing.assert_array_equal(age_groups(link, "best"), [0, 1, 1, 2, 2, 3, 3, 4])
    frame = study[2]["diagnosis"].iloc[:4].copy()
    frame[LABELS] = 0
    scores = {k: v[:4] for k, v in study[7]["diagnosis"].items()}
    result, requests = evaluate_joint(frame, scores, study[8])
    verify_joint_evaluation(frame, scores, study[8], result, requests)
    assert result["exit"]["all"]["aged_calibrated"]["average_precision"] is None


@pytest.mark.parametrize("both", [False, True])
def test_later_labels_preserve_models(study, both):
    source, table, rows, expanded, links, models, offsets, scores, thresholds, details = study
    changed = {k: v.copy() for k, v in expanded.items()}
    for phase in ["calibration", "diagnosis"] if both else ["diagnosis"]:
        changed[phase][LABELS] = 1 - changed[phase][LABELS]
    fitted, adjusted, _ = fit_age_models(changed)
    for name in fitted:
        assert fitted[name].to_dict() == models[name].to_dict()
        assert (adjusted[name].to_dict() != offsets[name].to_dict()) == both
    if not both:
        constant = rows["calibration"][LABELS].mean().to_numpy()
        cal = age_predictions(
            rows["calibration"],
            changed["calibration"],
            {**models, **fitted},
            {**offsets, **adjusted},
            constant,
        )
        assert joint_thresholds(rows["calibration"], cal) == (thresholds, details)


@pytest.mark.parametrize("name,cls", [("aged", ExtremaAgeBoost), ("aged_linear", ExtremaAgeLinear)])
def test_model_roundtrip_single_and_bad_features(study, name, cls):
    model = study[5][name]
    raw = model.to_dict()
    loaded = cls.from_dict(raw)
    values = study[3]["diagnosis"][FEATURES].iloc[:11].to_numpy()
    np.testing.assert_allclose(
        loaded.probabilities(values),
        np.vstack([model.probabilities(v[None, :]) for v in values]),
        atol=1e-12,
        rtol=0,
    )
    raw["features"] = raw["features"][:-1]
    with pytest.raises(ValueError):
        cls.from_dict(raw)
    assert len(loaded.features) == 52


def test_all_reference_thresholds_and_months_preserved(study, tmp_path):
    source, table, rows, expanded, links, models, offsets, scores, thresholds, details = study
    refs = {k: models[k] for k in ["native", "capacity", "linear"]}
    constant = rows["calibration"][LABELS].mean().to_numpy()
    old_scores = {p: capacity_predictions(f, refs, offsets, constant) for p, f in rows.items()}
    old_thresholds, old_details = joint_thresholds(rows["calibration"], old_scores["calibration"])
    save_json(tmp_path / "thresholds.json", old_thresholds)
    save_json(tmp_path / "threshold_selection.json", old_details)
    verify_age_reference_thresholds(tmp_path, thresholds, details)
    current, old, monthly, old_monthly = {}, {}, {}, {}
    for phase, frame in rows.items():
        for k, v in old_scores[phase].items():
            np.testing.assert_array_equal(v, scores[phase][k])
        current[phase], _ = evaluate_joint(frame, scores[phase], thresholds)
        old[phase], _ = evaluate_joint(frame, old_scores[phase], old_thresholds)
        monthly[phase], old_monthly[phase] = {}, {}
        for month, indices in frame.groupby(frame.end.dt.strftime("%Y-%m")).indices.items():
            part = frame.iloc[indices]
            monthly[phase][month], _ = evaluate_joint(
                part, {k: v[indices] for k, v in scores[phase].items()}, thresholds
            )
            old_monthly[phase][month], _ = evaluate_joint(
                part, {k: v[indices] for k, v in old_scores[phase].items()}, old_thresholds
            )
    save_json(tmp_path / "metrics.json", old)
    save_json(tmp_path / "monthly.json", old_monthly)
    verify_age_reference_metrics(tmp_path, current, monthly)
    changed = deepcopy(thresholds)
    changed["first"]["native_calibrated"]["reduce"] += 0.01
    with pytest.raises(AssertionError):
        verify_age_reference_thresholds(tmp_path, changed, details)


def test_changed_score_decision_and_proof_detected(study, tmp_path):
    source, table, rows, expanded, links, models, offsets, scores, thresholds, details = study
    broken = deepcopy(scores["diagnosis"])
    broken["aged_raw"][0, 0] += 0.01
    with pytest.raises(AssertionError):
        verify_age_predictions(expanded["diagnosis"], models, offsets, broken)
    metrics, _ = evaluate_joint(rows["diagnosis"], scores["diagnosis"], thresholds)
    decision = age_admission(metrics)
    decision["checks"]["exit_capacity_ap_preserved"] ^= True
    with pytest.raises(AssertionError):
        verify_age_admission(metrics, decision)
    path = tmp_path / "proof.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="지문"):
        checked_age_reference(path, "0" * 64)
