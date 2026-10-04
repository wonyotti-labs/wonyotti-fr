import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from test_first_close_margin import examples as first_examples
from threadpoolctl import threadpool_limits

from wonyotti_fr.close_utility import (
    UtilityCloseModel,
    cost_probability_metrics,
    cost_terms,
    cost_training,
    first_cost_positions,
    policy_effect_metrics,
    utility_admission,
)
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS


def examples():
    rng = np.random.default_rng(69)
    x = np.c_[rng.normal(size=(2000, 52)), np.tile(np.eye(4), (500, 1))]
    y = 7 * x[:, 0] + 3 * x[:, 1] + rng.normal(size=len(x))
    y[::10] = 0.0
    return x, y, np.linspace(0.25, 1.75, len(x))


def test_cost_weighted_classification_error_equals_original_regret_for_arbitrary_actions():
    _, y, w = examples()
    ledger, support = cost_training(y, w)
    assert len(ledger) == len(y) and support["zero_effect_rows"] == 200
    assert ledger.loc[~ledger.fit_used, "reason"].eq("zero_effect").all()
    assert ledger.loc[~ledger.fit_used, "cost_weight"].eq(0.0).all()
    np.testing.assert_array_equal(ledger.cost_weight, w * np.abs(y))
    np.testing.assert_array_equal(ledger.positive_effect, y > 0)
    np.testing.assert_allclose(
        ledger.fit_weight * support["normalizer"], ledger.cost_weight, rtol=1e-15
    )
    for action in [
        np.zeros(len(y), dtype=bool),
        np.ones(len(y), dtype=bool),
        y > 0,
        np.arange(len(y)) % 2 == 0,
    ]:
        np.testing.assert_allclose(
            ledger.cost_weight * (action != (y > 0)),
            w * (np.maximum(y, 0) - action * y),
            rtol=0,
            atol=0,
        )
    assert support["training_constant_score"] == pytest.approx(
        np.dot(w, np.maximum(y, 0)) / np.dot(w, np.abs(y)), abs=1e-15
    )


def test_cost_model_matches_independent_refit_without_zero_rows_and_preserves_future_independence():
    x, y, w = examples()
    model, support, ledger = UtilityCloseModel.fit(x, y, w, x[:200])
    mask = y != 0
    weight = w[mask] * abs(y[mask])
    weight /= weight.mean()
    with threadpool_limits(limits=1):
        independent = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(
            x[mask], y[mask] > 0, sample_weight=weight
        )
        np.testing.assert_allclose(
            model.probabilities(x)[:, 0], independent.predict_proba(x)[:, 1], rtol=0, atol=1e-12
        )
    assert support["fit_rows"] == 1800 and ledger.fit_used.sum() == 1800
    np.testing.assert_array_equal(
        UtilityCloseModel.from_dict(model.to_dict()).probabilities(x), model.probabilities(x)
    )
    changed = x[:200].copy()
    changed[:, :52] *= -100
    changed[:, -4:] = np.roll(changed[:, -4:], 1, axis=1)
    other, other_support, other_ledger = UtilityCloseModel.fit(x, y, w, changed)
    assert (
        other.to_dict() == model.to_dict() and other_support["normalizer"] == support["normalizer"]
    )
    pd.testing.assert_frame_equal(ledger, other_ledger, check_exact=True)
    damaged = model.to_dict()
    damaged["settings"]["class_weight"] = "balanced"
    with pytest.raises(ValueError):
        UtilityCloseModel.from_dict(damaged)
    changed[0, -4:] = 0
    with pytest.raises(ValueError):
        model.probabilities(changed)


@pytest.mark.parametrize(
    "case",
    ["negative_weight", "nan", "overflow", "underflow", "few_rows", "single_sign", "zero_effect"],
)
def test_invalid_cost_arithmetic_and_unsupported_training_are_rejected(case):
    _, y, w = examples()
    if case == "negative_weight":
        w[1] = -1.0
    elif case == "nan":
        y[1] = np.nan
    elif case == "overflow":
        y[1], w[1] = 1e308, 1e100
    elif case == "underflow":
        y[1], w[1] = 1e-300, 1e-300
    elif case == "few_rows":
        y, w = y[:900], w[:900]
    elif case == "single_sign":
        y = abs(y)
    else:
        y[:] = 0.0
    with pytest.raises(ValueError):
        cost_training(y, w)


def test_cost_scores_and_zero_cost_groups_are_distinct_from_account_forecast_errors():
    y, w, p = np.array([-5.0, 0.0, 2.0]), np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.9, 0.7])
    metrics = cost_probability_metrics(y, w, p)
    expected = -(5 * np.log(0.8) + 6 * np.log(0.7)) / 11
    assert metrics["cost_log_loss"] == pytest.approx(expected)
    assert metrics["cost_brier"] == pytest.approx((5 * 0.2**2 + 6 * 0.3**2) / 11)
    zero = cost_probability_metrics(np.zeros(3), w, p)
    assert zero["rows"] == 3 and zero["cost_mass"] == 0.0 and zero["cost_log_loss"] is None
    assert not any("mse" in name for name in metrics)
    for bad in [[np.nan, 0.0, 1.0], [1.1, 0.0, 1.0], [-0.1, 0.0, 1.0], [0.0, 1.0]]:
        with pytest.raises(ValueError):
            cost_probability_metrics(y, w, bad)
    with pytest.raises(ValueError):
        cost_terms([1.0, -1.0], [1.0])


def test_first_cost_choices_use_strict_fixed_boundary_keep_losses_and_store_score_unit():
    frame = first_examples()
    score = np.tile([0.5, 0.6, 0.9], 40)
    first = first_cost_positions(frame, score)
    assert first.first_cost_score.eq(0.6).all() and "first_prediction_bps" not in first
    assert first.selected_opportunities.eq(2).all()
    changed = frame.copy()
    changed.loc[
        changed.predicted_half.eq(2.0), ["close_advantage_pnl", "close_advantage_bps"]
    ] = -9.0
    changed_first = first_cost_positions(changed, score)
    pd.testing.assert_series_equal(first.first_selected_time, changed_first.first_selected_time)
    assert changed_first.first_effect_pnl.eq(-9.0).all()
    never = first_cost_positions(frame, np.zeros(len(frame)))
    assert (
        not never.chosen.any()
        and never.first_cost_score.isna().all()
        and never.first_effect_pnl.eq(0.0).all()
    )
    frame.loc[frame.predicted_half.eq(2.0), "original_intent"] = "exit"
    assert first_cost_positions(frame, score).first_cost_score.eq(0.9).all()


def test_policy_regret_preserves_all_rows_and_original_position_weights():
    frame = first_examples()
    action = frame.predicted_half.gt(0.5).to_numpy()
    metrics = policy_effect_metrics(frame, action)
    assert metrics["rows"] == 120 and metrics["positions"] == 40 and metrics["selected"] == 80
    assert metrics["weighted_effect_bps"] == pytest.approx(4 / 3)
    assert metrics["weighted_regret_bps"] == pytest.approx(0.0)
    assert metrics["selected_weighted_mean_bps"] == pytest.approx(2.0)
    none = policy_effect_metrics(frame, np.zeros(len(frame), dtype=bool))
    assert none["weighted_effect_bps"] == 0.0 and none["selected_mean_bps"] is None
    assert none["weighted_regret_bps"] == pytest.approx(4 / 3)
    frame.loc[action, "original_intent"] = "exit"
    with pytest.raises(ValueError):
        policy_effect_metrics(frame, action)


def test_all_fourteen_conditions_include_real_utility_and_first_choice_uncertainty():
    value = {
        "rows": 200,
        "positions": 40,
        "selected": 120,
        "selected_positions": 30,
        "selected_weighted_mean_bps": 2.0,
        "selected_mean_bps": 2.0,
        "weighted_regret_bps": 2.0,
    }
    metrics = {
        name: {**value, "weighted_regret_bps": regret}
        for name, regret in [
            ("utility", 1.0),
            ("training_constant", 2.0),
            ("continuation", 2.0),
            ("weekly", 2.0),
        ]
    }
    probability = {
        "utility": {"rows": 200, "cost_log_loss": 0.5, "cost_brier": 0.2},
        "training_constant": {"rows": 200, "cost_log_loss": 0.6, "cost_brier": 0.3},
    }
    first = {
        name: {"positions": 40, "selected_positions": 30, "all_position_mean_common_bps": mean}
        for name, mean in [("utility", 2.0), ("continuation", 1.0), ("weekly", 1.0)]
    }
    intervals = {"intervals": {"utility": {"lower": 0.1}, "paired_difference": {"lower": 0.1}}}
    decision = utility_admission(metrics, probability, first, intervals)
    assert decision["utility_admitted"] and len(decision["checks"]) == 14
    for field in ["cost_log_loss", "cost_brier"]:
        changed = copy.deepcopy(probability)
        changed["utility"][field] = 1.0
        assert not utility_admission(metrics, changed, first, intervals)["utility_admitted"]
    for name in ["training_constant", "continuation", "weekly"]:
        changed = copy.deepcopy(metrics)
        changed[name]["weighted_regret_bps"] = 1.0
        assert not utility_admission(changed, probability, first, intervals)["utility_admitted"]
    for field, bad in [
        ("selected", 99),
        ("selected_positions", 29),
        ("selected_weighted_mean_bps", 0.0),
        ("selected_mean_bps", 0.0),
    ]:
        changed, changed_first = copy.deepcopy(metrics), copy.deepcopy(first)
        changed["utility"][field] = bad
        if field == "selected_positions":
            changed_first["utility"][field] = bad
        assert not utility_admission(changed, probability, changed_first, intervals)[
            "utility_admitted"
        ]
    for name, mean in [("utility", 0.0), ("continuation", 2.0), ("weekly", 2.0)]:
        changed = copy.deepcopy(first)
        changed[name]["all_position_mean_common_bps"] = mean
        assert not utility_admission(metrics, probability, changed, intervals)["utility_admitted"]
    for name in ["utility", "paired_difference"]:
        changed = copy.deepcopy(intervals)
        changed["intervals"][name]["lower"] = 0.0
        assert not utility_admission(metrics, probability, first, changed)["utility_admitted"]
    changed = copy.deepcopy(probability)
    changed["utility"]["rows"] = 201
    with pytest.raises(ValueError):
        utility_admission(metrics, changed, first, intervals)
