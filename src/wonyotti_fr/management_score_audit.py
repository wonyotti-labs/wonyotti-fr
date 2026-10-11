from __future__ import annotations

import numpy as np
import pandas as pd

from .first_management import BETAS, decision_metrics
from .management_support import FEATURES, matrix
from .minute_management import ACTIONS

SCORE_QUANTILES = [0.5, 0.9, 0.99, 0.999]


def boundaries(values, thresholds, first_thresholds, multiplier):
    x = matrix(values)
    if (
        set(thresholds) != set(ACTIONS)
        or set(first_thresholds) != {"reduce", "increase"}
        or multiplier not in (1.0, 1.5)
        or not np.isfinite([*thresholds.values(), *first_thresholds.values()]).all()
        or any(not 0 <= v <= 1 for v in [*thresholds.values(), *first_thresholds.values()])
        or not np.isin(x[:, [44, 47]], [0, 1]).all()
    ):
        raise ValueError("관리 점수 진단의 고정 문턱·과거 상태 오류")
    result = np.tile([thresholds[a] for a in ACTIONS], (len(x), 1))
    for index, action, feature in [(1, "reduce", 47), (2, "increase", 44)]:
        result[:, index] = np.where(
            x[:, feature] == 0, first_thresholds[action], thresholds[action] * multiplier
        )
    return result


def resolve_requests(scores, limits, finite, fractions):
    p, limits = np.asarray(scores), np.asarray(limits)
    finite, fractions = np.asarray(finite), np.asarray(fractions)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or limits.shape != p.shape
        or finite.shape != (len(p),)
        or finite.dtype != bool
        or fractions.shape != (len(p),)
        or not np.isfinite(limits).all()
        or not np.isfinite(fractions).all()
        or not ((fractions >= 0) & (fractions <= 1)).all()
        or not np.isfinite(p[finite]).all()
        or ((p[finite] < 0) | (p[finite] > 1)).any()
        or not np.isnan(p[~finite]).all()
    ):
        raise ValueError("관리 점수 진단의 확률·수량·지원 오류")
    passed = finite[:, None] & (p >= limits)
    requested = np.where(
        passed[:, 0],
        "exit",
        np.where(passed[:, 1], "reduce", np.where(passed[:, 2], "increase", "hold")),
    )
    zero = (requested == "reduce") & (fractions == 0)
    choices = np.where(zero, "hold", requested)
    events = np.array(["action_" + a for a in requested], dtype=object)
    events[(requested == "reduce") & ~zero] = "action_sized_reduce"
    events[zero] = "action_zero_reduction"
    events[~finite] = "action_unavailable"
    return passed, requested, choices, events


def score_frame(frame, policy):
    x = frame[FEATURES].to_numpy(float)
    finite = np.isfinite(x).all(axis=1)
    raw = policy.manager.model.probabilities(x)
    calibrated = policy.manager.probabilities(x)
    result = frame[
        [
            k
            for k in [
                "end",
                "episode_id",
                "entry_time",
                "past_reduce_exists",
                "past_increase_exists",
                "y_exit",
                "y_reduce",
                "y_increase",
            ]
            if k in frame
        ]
    ].copy()
    result["finite_features"] = finite
    fractions = np.ones(len(x))
    limits = {
        key: boundaries(x, policy.thresholds, policy.first_thresholds, multiplier)
        for key, multiplier in [("original", 1.5), ("direct", 1.0)]
    }
    needs_size = np.zeros(len(x), dtype=bool)
    for values in limits.values():
        needs_size |= (
            finite & (calibrated[:, 0] < values[:, 0]) & (calibrated[:, 1] >= values[:, 1])
        )
    if policy.size_model is not None:
        for i in np.flatnonzero(needs_size):
            # 실제 정책과 같은 단일 행 연산으로 0 경계의 수량 판단을 보존한다.
            fractions[i] = policy.size_model.predict(x[i : i + 1, :44])[0]
    result["reduction_fraction"] = fractions
    for i, action in enumerate(ACTIONS):
        result[f"{action}_raw"] = raw[:, i]
        result[f"{action}_score"] = calibrated[:, i]
    for key, values in limits.items():
        passed, requested, choices, events = resolve_requests(calibrated, values, finite, fractions)
        if policy.size_model is None:
            events[requested == "reduce"] = "action_reduce"
        for i, action in enumerate(ACTIONS):
            result[f"{key}_{action}_passed"] = passed[:, i]
            result[f"{key}_{action}_threshold"] = values[:, i]
        result[f"{key}_requested"] = pd.Series(requested, index=result.index, dtype="str")
        result[f"{key}_choice"] = pd.Series(choices, index=result.index, dtype="str")
        result[f"{key}_event"] = pd.Series(events, index=result.index, dtype="str")
    if "intent" in frame and (
        not np.array_equal(result.original_choice, frame.intent)
        or not np.array_equal(result.original_event, frame.event)
    ):
        raise ValueError("관리 점수 진단과 실제 저장된 의도·사건 불일치")
    return result


def distribution(scores):
    values = np.asarray(scores, dtype=float)
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError("관리 점수 분포의 비유한 값 유입")
    return {
        "mean": float(values.mean()),
        "maximum": float(values.max()),
        "quantiles": dict(
            zip(
                map(str, SCORE_QUANTILES),
                np.quantile(values, SCORE_QUANTILES).tolist(),
                strict=True,
            )
        ),
    }


def summarize_scores(frame):
    actions = {}
    for action in ACTIONS:
        groups = {"all": frame}
        if action != "exit":
            groups.update(
                first=frame.loc[frame[f"past_{action}_exists"] == 0],
                repeat=frame.loc[frame[f"past_{action}_exists"] == 1],
            )
        actions[action] = {}
        for key, part in groups.items():
            good = part.loc[part.finite_features]
            row = {
                "rows": len(part),
                "finite_rows": len(good),
                "nonfinite_rows": len(part) - len(good),
                "raw": distribution(good[f"{action}_raw"]),
                "calibrated": distribution(good[f"{action}_score"]),
            }
            if f"y_{action}" in part:
                row["positive"] = int(part[f"y_{action}"].sum())
                row["positive_fraction"] = row["positive"] / len(part) if len(part) else None
            for variant in ["original", "direct"]:
                passed = part[f"{variant}_{action}_passed"].to_numpy(bool)
                requested = part[f"{variant}_requested"].eq(action)
                row[variant] = {
                    "passed": int(passed.sum()),
                    "priority_suppressed": int((passed & ~requested).sum()),
                    "requested": int(requested.sum()),
                    "final_choice": int(part[f"{variant}_choice"].eq(action).sum()),
                    "raw_metrics": decision_metrics(part[f"y_{action}"], passed, BETAS[action])
                    if f"y_{action}" in part and len(part)
                    else None,
                }
            actions[action][key] = row
    table = (
        pd.crosstab(frame.original_choice, frame.direct_choice)
        .reindex(index=["hold", *ACTIONS], columns=["hold", *ACTIONS], fill_value=0)
        .fillna(0)
        .astype(int)
    )
    original_active = frame.original_requested.ne("hold")
    direct_active = frame.direct_requested.ne("hold")
    return {
        "rows": len(frame),
        "actions": actions,
        "choice_order": ["hold", *ACTIONS],
        "choice_transitions": table.to_numpy().tolist(),
        "changed_choices": int(frame.original_choice.ne(frame.direct_choice).sum()),
        "changed_management_wait": int(original_active.ne(direct_active).sum()),
        "changed_event": int(frame.original_event.ne(frame.direct_event).sum()),
        "zero_reduction": {
            key: int(frame[f"{key}_event"].eq("action_zero_reduction").sum())
            for key in ["original", "direct"]
        },
        "unchanged_masks": {
            "exit": bool(frame.original_exit_passed.equals(frame.direct_exit_passed)),
            **{
                f"first_{a}": bool(
                    frame.loc[frame[f"past_{a}_exists"] == 0, f"original_{a}_passed"].equals(
                        frame.loc[frame[f"past_{a}_exists"] == 0, f"direct_{a}_passed"]
                    )
                )
                for a in ["reduce", "increase"]
            },
        },
    }


def repeat_admission(diagnosis, bot_summaries):
    checks = {}
    for action in ["reduce", "increase"]:
        groups = diagnosis["actions"][action]
        repeat, all_rows = groups["repeat"], groups["all"]
        old, direct = (repeat[k]["raw_metrics"] for k in ["original", "direct"])
        checks[action] = {
            "support": min(repeat["positive"], repeat["rows"] - repeat["positive"]) >= 20,
            "repeat_recall_improved": bool(
                old and direct and direct["recall"] > old["recall"] + 1e-12
            ),
            "repeat_f_improved": bool(old and direct and direct["f_beta"] > old["f_beta"] + 1e-12),
            "all_f_preserved": bool(
                all_rows["direct"]["raw_metrics"]
                and all_rows["original"]["raw_metrics"]
                and all_rows["direct"]["raw_metrics"]["f_beta"]
                >= all_rows["original"]["raw_metrics"]["f_beta"] - 1e-12
            ),
        }
    unchanged = all(diagnosis["unchanged_masks"].values())
    changed = any(
        row["changed_choices"] or row["changed_management_wait"] for row in bot_summaries.values()
    )
    return {
        "checks": checks,
        "unchanged_masks": unchanged,
        "bot_decisions_changed": bool(changed),
        "further_account_research_eligible": all(all(v.values()) for v in checks.values())
        and unchanged
        and changed,
        "new_models_fitted": False,
        "profitability_accepted": False,
    }
