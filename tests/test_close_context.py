import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_close_utility import examples
from test_context_position import bars
from threadpoolctl import threadpool_limits

from wonyotti_fr.close_context import (
    ContextCloseModel,
    attach_close_context,
    close_context_source,
    context_close_admission,
)
from wonyotti_fr.close_utility import UtilityCloseModel
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.context_position import CONTEXT_FEATURES, context_features
from wonyotti_fr.event_features import MARKET_FEATURES, event_features
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS


def dataset(start=19000, size=100):
    market = bars(23000).assign(count=1.0)
    known = event_features(market).iloc[start : start + size].reset_index(drop=True)
    x, y, _ = examples()
    frame = pd.DataFrame(x[:size], columns=UtilityCloseModel.features)
    frame[MARKET_FEATURES] = known[MARKET_FEATURES]
    frame["decision_time"] = known.end
    frame["position_entry_time"] = known.end - pd.Timedelta(minutes=5)
    frame["close_advantage_bps"] = y[:size]
    return market, frame


def test_eight_context_inputs_preserve_old_features_and_cannot_use_future_market_changes():
    market, frame = dataset()
    result = attach_close_context({"training": frame}, market)["training"]
    pd.testing.assert_frame_equal(result.drop(columns=CONTEXT_FEATURES), frame, check_exact=True)
    expected = (
        context_features(market)
        .set_index("end")
        .loc[frame.decision_time, CONTEXT_FEATURES]
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(result[CONTEXT_FEATURES], expected, check_exact=True)
    future = market.copy()
    mask = future.end.gt(frame.decision_time.max())
    future.loc[mask, ["open", "high", "low", "close"]] *= 5
    future.loc[mask, "volume"] *= 10
    pd.testing.assert_frame_equal(
        attach_close_context({"training": frame}, future)["training"], result, check_exact=True
    )
    assert (
        ContextCloseModel.features[:56] == UtilityCloseModel.features
        and len(ContextCloseModel.features) == 64
    )


@pytest.mark.parametrize("damage", ["warmup", "unconfirmed", "changed_past", "gap"])
def test_missing_context_or_reconstructed_original_mismatch_fails_without_dropping_rows(damage):
    market, frame = dataset(start=1000 if damage == "warmup" else 19000)
    if damage == "unconfirmed":
        frame["decision_time"] += pd.Timedelta(minutes=1)
    elif damage == "changed_past":
        frame.loc[0, "ret_5m"] += 1.0
    elif damage == "gap":
        market = market.drop(index=10000).reset_index(drop=True)
        frame[MARKET_FEATURES] = (
            event_features(market)
            .set_index("end")
            .loc[frame.decision_time, MARKET_FEATURES]
            .reset_index(drop=True)
        )
    preserved = frame.copy(deep=True)
    with pytest.raises((ValueError, AssertionError)):
        attach_close_context({"training": frame}, market)
    pd.testing.assert_frame_equal(frame, preserved, check_exact=True)


def test_extended_cost_model_validates_pending_by_name_and_keeps_costs_identical_to_base():
    x, y, w = examples()
    extended = np.c_[x, np.random.default_rng(70).normal(size=(len(x), 8))]
    model, support, ledger = ContextCloseModel.fit(extended, y, w, extended[:200])
    base, original_support, original_ledger = UtilityCloseModel.fit(x, y, w, x[:200])
    assert support["normalizer"] == original_support["normalizer"]
    assert support["training_constant_score"] == original_support["training_constant_score"]
    pd.testing.assert_frame_equal(ledger, original_ledger, check_exact=True)
    fit = ledger.fit_used.to_numpy()
    with threadpool_limits(limits=1):
        independent = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(
            extended[fit], y[fit] > 0, sample_weight=ledger.fit_weight[fit]
        )
        np.testing.assert_allclose(
            model.probabilities(extended)[:, 0],
            independent.predict_proba(extended)[:, 1],
            rtol=0,
            atol=1e-12,
        )
    np.testing.assert_array_equal(
        ContextCloseModel.from_dict(model.to_dict()).probabilities(extended),
        model.probabilities(extended),
    )
    np.testing.assert_array_equal(
        UtilityCloseModel.from_dict(base.to_dict()).probabilities(x), base.probabilities(x)
    )
    with pytest.raises(ValueError):
        UtilityCloseModel.from_dict(model.to_dict())
    changed = extended.copy()
    changed[0, 52:56] = 0.0
    with pytest.raises(ValueError):
        model.probabilities(changed)
    future = extended[:200].copy()
    future[:, -8:] *= -100
    other, other_support, _ = ContextCloseModel.fit(extended, y, w, future)
    assert (
        other.to_dict() == model.to_dict() and other_support["normalizer"] == support["normalizer"]
    )


def test_market_source_uses_original_manifest_and_rejects_changed_data_and_reference_cycles(
    tmp_path,
):
    market, _ = dataset()
    folder, labels, reference = (tmp_path / name for name in ["market", "labels", "reference"])
    for path in [folder, labels, reference]:
        path.mkdir()
    market.to_parquet(folder / "bars.parquet", index=False)
    pd.DataFrame({"time": market.time.iloc[:1], "rate": [0.0]}).to_parquet(
        folder / "funding.parquet", index=False
    )
    save_json(
        folder / "manifest-5m.json",
        {
            "summary": {
                "BTCUSDT": {
                    "klines": {"file": "bars.parquet", "sha256": sha256(folder / "bars.parquet")},
                    "fundingRate": {
                        "file": "funding.parquet",
                        "sha256": sha256(folder / "funding.parquet"),
                    },
                }
            }
        },
    )
    save_json(
        labels / "manifest.json",
        {
            "settings": {
                "features": str(folder),
                "feature_manifest_sha256": sha256(folder / "manifest-5m.json"),
            }
        },
    )
    save_json(labels / "files.json", {"manifest.json": sha256(labels / "manifest.json")})
    save_json(
        reference / "manifest.json",
        {"settings": {"labels": str(labels), "labels_files_sha256": sha256(labels / "files.json")}},
    )
    loaded, proof = close_context_source(reference)
    pd.testing.assert_frame_equal(loaded, market, check_exact=True)
    assert (
        proof["market_manifest_sha256"] == sha256(folder / "manifest-5m.json")
        and proof["warmup_bars"] == 64 * 288
    )
    changed = market.copy()
    changed.loc[0, "close"] += 1
    changed.to_parquet(folder / "bars.parquet", index=False)
    with pytest.raises(ValueError, match="변경"):
        close_context_source(reference)
    save_json(reference / "files.json", {})
    save_json(
        reference / "manifest.json",
        {
            "settings": {
                "reference": str(reference),
                "reference_files_sha256": sha256(reference / "files.json"),
            }
        },
    )
    with pytest.raises(ValueError, match="순환"):
        close_context_source(reference)


def test_nineteen_gates_keep_original_fourteen_and_require_improvement_over_previous_utility():
    common = {
        "rows": 200,
        "positions": 40,
        "selected": 120,
        "selected_positions": 30,
        "selected_weighted_mean_bps": 2.0,
        "selected_mean_bps": 2.0,
    }
    metrics = {
        name: {**common, "weighted_regret_bps": value}
        for name, value in [
            ("context", 1.0),
            ("utility", 1.5),
            ("training_constant", 2.0),
            ("continuation", 2.0),
            ("weekly", 2.0),
        ]
    }
    probability = {
        name: {"rows": 200, "cost_log_loss": loss, "cost_brier": brier}
        for name, loss, brier in [
            ("context", 0.5, 0.2),
            ("utility", 0.55, 0.25),
            ("training_constant", 0.6, 0.3),
        ]
    }
    first = {
        name: {"positions": 40, "selected_positions": 30, "all_position_mean_common_bps": mean}
        for name, mean in [
            ("context", 2.0),
            ("utility", 1.5),
            ("continuation", 1.0),
            ("weekly", 1.0),
        ]
    }
    intervals = {"intervals": {"context": {"lower": 0.1}, "paired_difference": {"lower": 0.1}}}
    decision = context_close_admission(metrics, probability, first, intervals, intervals)
    assert len(decision["checks"]) == 19 and decision["context_admitted"]
    for field, value in [("cost_log_loss", 0.5), ("cost_brier", 0.19)]:
        changed = copy.deepcopy(probability)
        changed["utility"][field] = value
        assert not context_close_admission(metrics, changed, first, intervals, intervals)[
            "context_admitted"
        ]
    changed = copy.deepcopy(metrics)
    changed["utility"]["weighted_regret_bps"] = 1.0
    assert not context_close_admission(changed, probability, first, intervals, intervals)[
        "context_admitted"
    ]
    changed = copy.deepcopy(first)
    changed["utility"]["all_position_mean_common_bps"] = 2.0
    assert not context_close_admission(metrics, probability, changed, intervals, intervals)[
        "context_admitted"
    ]
    changed = copy.deepcopy(intervals)
    changed["intervals"]["paired_difference"]["lower"] = 0.0
    assert not context_close_admission(metrics, probability, first, intervals, changed)[
        "context_admitted"
    ]
    changed = copy.deepcopy(metrics)
    changed["context"]["selected_weighted_mean_bps"] = -1.0
    assert not context_close_admission(changed, probability, first, intervals, intervals)[
        "context_admitted"
    ]
    assert json.loads(json.dumps(decision)) == decision
