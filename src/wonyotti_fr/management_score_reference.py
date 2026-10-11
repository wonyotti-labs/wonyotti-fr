from __future__ import annotations

import math

import numpy as np

from .management_score_audit import SCORE_QUANTILES
from .management_support import FEATURES
from .minute_management import ACTIONS
from .position_boost_verification import compare_statistics


def forest_by_leaf(model, x):
    output = np.full(len(x), model["baseline"], dtype=float)
    for tree in model["trees"]:

        def visit(node, indices, tree=tree):
            if tree["feature"][node] < 0:
                output[indices] += tree["value"][node]
            else:
                left = x[indices, tree["feature"][node]] <= tree["threshold"][node]
                visit(tree["left"][node], indices[left])
                visit(tree["right"][node], indices[~left])

        visit(0, np.arange(len(x)))
    return 1 / (1 + np.exp(-np.clip(output, -700, 700)))


def verify_scores(source, saved, policy):
    x = source[FEATURES].to_numpy(float)
    finite = np.isfinite(x).all(axis=1)
    assert np.array_equal(finite, saved.finite_features)
    raw = np.column_stack(
        [forest_by_leaf(model, x[finite]) for model in policy.manager.model.data["models"]]
    )
    clipped = np.clip(raw, np.finfo(float).eps, 1 - np.finfo(float).eps)
    adjusted = 1 / (
        1
        + np.exp(
            -np.clip(
                np.log(clipped) - np.log1p(-clipped) + policy.manager.offset.offsets, -700, 700
            )
        )
    )
    for i, action in enumerate(ACTIONS):
        np.testing.assert_allclose(
            saved.loc[finite, f"{action}_raw"], raw[:, i], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            saved.loc[finite, f"{action}_score"], adjusted[:, i], rtol=0, atol=1e-12
        )
        assert saved.loc[~finite, [f"{action}_raw", f"{action}_score"]].isna().all().all()
        for variant, multiplier in [("original", 1.5), ("direct", 1.0)]:
            if action == "exit":
                boundary = np.full(len(x), policy.thresholds[action])
            else:
                boundary = np.array(
                    [
                        policy.first_thresholds[action]
                        if value == 0
                        else policy.thresholds[action] * multiplier
                        for value in source[f"past_{action}_exists"]
                    ]
                )
            np.testing.assert_array_equal(saved[f"{variant}_{action}_threshold"], boundary)
            predicted = np.zeros(len(x), dtype=bool)
            predicted[finite] = adjusted[:, i] >= boundary[finite]
            np.testing.assert_array_equal(saved[f"{variant}_{action}_passed"], predicted)
    for variant in ["original", "direct"]:
        requested = []
        for masks in saved[[f"{variant}_{a}_passed" for a in ACTIONS]].itertuples(
            index=False, name=None
        ):
            requested.append(
                next((a for a, yes in zip(ACTIONS, masks, strict=True) if yes), "hold")
            )
        assert requested == saved[f"{variant}_requested"].tolist()
        choices, events = [], []
        for row, action in enumerate(requested):
            event, final = "action_" + action, action
            if not finite[row]:
                event = "action_unavailable"
            elif action == "reduce" and policy.size_model is not None:
                data = policy.size_model.data
                values = (x[row, :44] - np.asarray(data["mean"])) / np.asarray(data["scale"])
                fraction = float(np.clip(np.dot(values, data["coef"]) + data["intercept"], 0, 1))
                assert math.isclose(
                    fraction, saved.reduction_fraction.iloc[row], rel_tol=0, abs_tol=1e-12
                )
                assert (fraction == 0) == (saved.reduction_fraction.iloc[row] == 0)
                final, event = (
                    ("hold", "action_zero_reduction")
                    if fraction == 0
                    else ("reduce", "action_sized_reduce")
                )
            choices.append(final)
            events.append(event)
        assert choices == saved[f"{variant}_choice"].tolist()
        assert events == saved[f"{variant}_event"].tolist()


def manual_distribution(values):
    ordered = sorted(map(float, values))
    if not ordered:
        return None
    quantiles = {}
    for fraction in SCORE_QUANTILES:
        index = fraction * (len(ordered) - 1)
        left, right = math.floor(index), math.ceil(index)
        quantiles[str(fraction)] = ordered[left] + (ordered[right] - ordered[left]) * (index - left)
    return {
        "mean": math.fsum(ordered) / len(ordered),
        "maximum": ordered[-1],
        "quantiles": quantiles,
    }


def verify_summary(frame, summary):
    assert summary["rows"] == len(frame)
    for action in ACTIONS:
        for group, metrics in summary["actions"][action].items():
            mask = (
                np.ones(len(frame), dtype=bool)
                if group == "all"
                else (frame[f"past_{action}_exists"].to_numpy() == (0 if group == "first" else 1))
            )
            part = frame.loc[mask]
            good = part.loc[part.finite_features]
            assert [metrics["rows"], metrics["finite_rows"], metrics["nonfinite_rows"]] == [
                len(part),
                len(good),
                len(part) - len(good),
            ]
            compare_statistics(metrics["raw"], manual_distribution(good[f"{action}_raw"]))
            compare_statistics(metrics["calibrated"], manual_distribution(good[f"{action}_score"]))
            if f"y_{action}" in part:
                target = part[f"y_{action}"].to_numpy()
                assert np.isin(target, [0, 1]).all()
                assert metrics["positive"] == int(np.count_nonzero(target))
                compare_statistics(
                    metrics["positive_fraction"], float(target.mean()) if len(target) else None
                )
            for variant in ["original", "direct"]:
                passed = part[f"{variant}_{action}_passed"].to_numpy()
                requested = part[f"{variant}_requested"].eq(action).to_numpy()
                expected = metrics[variant]
                assert expected["passed"] == int(np.count_nonzero(passed))
                assert expected["requested"] == int(np.count_nonzero(requested))
                assert expected["priority_suppressed"] == int(np.count_nonzero(passed & ~requested))
                assert expected["final_choice"] == int(
                    np.count_nonzero(part[f"{variant}_choice"].eq(action))
                )
                if f"y_{action}" not in part or not len(part):
                    assert expected["raw_metrics"] is None
                    continue
                tp = int(np.count_nonzero((target == 1) & passed))
                fp = int(np.count_nonzero((target == 0) & passed))
                fn = int(np.count_nonzero((target == 1) & ~passed))
                b2 = {"exit": 4.0, "reduce": 1.0, "increase": 0.25}[action]
                denominator = (1 + b2) * tp + b2 * fn + fp
                reference = {
                    "rows": len(part),
                    "positive": tp + fn,
                    "predicted_positive": tp + fp,
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                    "recall": tp / (tp + fn) if tp + fn else 0.0,
                    "f_beta": (1 + b2) * tp / denominator if denominator else 0.0,
                }
                compare_statistics(expected["raw_metrics"], reference)
    choices = ["hold", "exit", "reduce", "increase"]
    counts = [[0] * 4 for _ in range(4)]
    changed = wait = events = 0
    zero = {"original": 0, "direct": 0}
    for row in frame.itertuples(index=False):
        counts[choices.index(row.original_choice)][choices.index(row.direct_choice)] += 1
        changed += row.original_choice != row.direct_choice
        wait += (row.original_requested == "hold") != (row.direct_requested == "hold")
        events += row.original_event != row.direct_event
        zero["original"] += row.original_event == "action_zero_reduction"
        zero["direct"] += row.direct_event == "action_zero_reduction"
    assert summary["choice_order"] == choices and summary["choice_transitions"] == counts
    assert [
        summary[k] for k in ["changed_choices", "changed_management_wait", "changed_event"]
    ] == [changed, wait, events]
    assert summary["zero_reduction"] == zero
    for name, flag in summary["unchanged_masks"].items():
        action = name.removeprefix("first_")
        part = frame if name == "exit" else frame.loc[frame[f"past_{action}_exists"] == 0]
        assert flag == bool(
            np.array_equal(part[f"original_{action}_passed"], part[f"direct_{action}_passed"])
        )
