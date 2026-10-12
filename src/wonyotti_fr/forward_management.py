from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .first_state_models import FEATURES
from .fixed_bin_management import export_histogram, model_digest, training_digest
from .management_capacity import validate_training
from .management_rank_audit import validate_scores
from .quiet_management import ACTIONS, CLASSES, groups, quiet_mask
from .tail_weighted_management import SCOPE, validate_tail_rows

ARMS = ("v142matched", "v142forward")
WEIGHT_RULE = "quarter_negative_top_one_percent_neutral1_high1_low01_mean1"
FOLDS = {
    "Q2": ("2020-03-31", "2020-04-01", "2020-06-30"),
    "Q3": ("2020-06-30", "2020-07-01", "2020-09-30"),
    "Q4": ("2020-09-30", "2020-10-01", "2020-12-31"),
}
CONTEXT = ["end", "entry_time", "label_end", "episode_id"]
REASONS = ("recent", "warmup", "left_boundary", "right_boundary", "prior_entry", "eligible")


def fold_masks(frame, action, fold):
    if fold not in FOLDS:
        raise ValueError("과거 예측의 미등록 분기")
    cut, start, stop = (pd.Timestamp(v, tz="UTC") for v in FOLDS[fold])
    quiet = quiet_mask(frame, action)
    auxiliary = quiet & frame.label_end.lt(cut).to_numpy()
    target = (
        quiet
        & frame.end.ge(start + pd.Timedelta(days=1)).to_numpy()
        & frame.label_end.lt(stop).to_numpy()
        & frame.entry_time.ge(start).to_numpy()
    )
    return auxiliary, target


def sample_support(frame, action, mask, minimum):
    part = frame.loc[mask]
    y = part["y_" + action].to_numpy(int)
    positive, negative = int(y.sum()), int(len(y) - y.sum())
    return {
        "rows": len(y),
        "positive": positive,
        "negative": negative,
        "episodes": int(part.episode_id.nunique()),
        "positive_episodes": int(part.loc[y == 1, "episode_id"].nunique()),
        "minimum_rows": minimum,
        "minimum_class_rows": 20,
        "supported": bool(len(y) >= minimum and min(positive, negative) >= 20),
    }


def forward_partition(training):
    validate_tail_rows(training, "training")
    validate_training(training)
    ledger = training[CONTEXT].copy()
    support = {}
    for action in ACTIONS:
        quiet = quiet_mask(training, action)
        reason = np.where(quiet, "warmup", "recent").astype(object)
        fold_ids = np.full(len(training), "", dtype=object)
        support[action] = {}
        seen = np.zeros(len(training), bool)
        for fold, dates in FOLDS.items():
            cut, start, stop = (pd.Timestamp(v, tz="UTC") for v in dates)
            aux, target = fold_masks(training, action, fold)
            quarter = (
                quiet
                & training.end.ge(start).to_numpy()
                & training.end.lt(start + pd.DateOffset(months=3)).to_numpy()
            )
            left = training.end.lt(start + pd.Timedelta(days=1)).to_numpy()
            right = training.label_end.ge(stop).to_numpy()
            prior = training.entry_time.lt(start).to_numpy()
            reason[quarter] = np.select(
                [left[quarter], right[quarter], prior[quarter]],
                ["left_boundary", "right_boundary", "prior_entry"],
                default="eligible",
            )
            fold_ids[target] = fold
            if (seen & target).any():
                raise ValueError("과거 예측의 대상 분기 중첩")
            seen |= target
            overlap = set(training.loc[aux, "episode_id"]) & set(training.loc[target, "episode_id"])
            time_ordered = bool(
                not aux.any()
                or not target.any()
                or training.loc[aux, "label_end"].max() < training.loc[target, "end"].min()
            )
            if overlap or not time_ordered:
                raise ValueError("과거 예측의 포지션·시간 경계 중첩")
            support[action][fold] = {
                "auxiliary": sample_support(training, action, aux, 1000),
                "target": sample_support(training, action, target, 100),
                "overlapping_episodes": len(overlap),
                "time_ordered": time_ordered,
                "auxiliary_label_end_max": (
                    training.loc[aux, "label_end"].max().isoformat() if aux.any() else None
                ),
                "target_end_min": (
                    training.loc[target, "end"].min().isoformat() if target.any() else None
                ),
                "cut": cut.isoformat(),
            }
        np.testing.assert_array_equal(reason == "eligible", seen)
        ledger[action + "_fold"] = fold_ids
        ledger[action + "_reason"] = reason
    return ledger, support


def require_forward_support(support):
    if set(support) != set(ACTIONS) or any(set(v) != set(FOLDS) for v in support.values()):
        raise ValueError("과거 예측의 행동·분기 지원 누락")
    for action in ACTIONS:
        for fold in FOLDS:
            for role, minimum in [("auxiliary", 1000), ("target", 100)]:
                item = support[action][fold][role]
                if (
                    item["rows"] < minimum
                    or min(item["positive"], item["negative"]) < 20
                    or item["positive"] + item["negative"] != item["rows"]
                    or item["supported"] is not True
                ):
                    raise ValueError(f"과거 예측의 분기 지원 부족: {action}/{fold}/{role}")


def fit_forward_auxiliaries(training):
    ledger, support = forward_partition(training)
    require_forward_support(support)
    models = {}
    with threadpool_limits(limits=1):
        for action in ACTIONS:
            models[action] = {}
            column = action + "_forward_score"
            ledger[column] = pd.array([pd.NA] * len(training), dtype="Float64")
            for fold in FOLDS:
                auxiliary, target = fold_masks(training, action, fold)
                part = training.loc[auxiliary]
                x, y = part[FEATURES].to_numpy(float), part["y_" + action].to_numpy(int)
                tx = training.loc[target, FEATURES].to_numpy(float)
                learner = HistGradientBoostingClassifier(**CLASSES[action].fixed_settings).fit(x, y)
                tree = export_histogram(learner, action)
                probabilities = CLASSES[action].from_dict(tree).probabilities(tx)[:, 0]
                np.testing.assert_allclose(
                    probabilities, learner.predict_proba(tx)[:, 1], atol=1e-12, rtol=0
                )
                models[action][fold] = {
                    "tree": tree,
                    "thresholds": [q.tolist() for q in learner._bin_mapper.bin_thresholds_],
                    "training_sha256": training_digest(part, action),
                    "tree_sha256": model_digest(tree),
                    "rows": len(x),
                }
                ledger.loc[target, column] = probabilities
    return models, ledger, support


def quarter_tail_weights(labels, scores, folds):
    y, p, fold_ids = np.asarray(labels), np.asarray(scores, float), np.asarray(folds)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or y.shape != fold_ids.shape
        or len(y) < 1000
        or not np.isin(y, [0, 1]).all()
        or not np.isin(fold_ids, ["", *FOLDS]).all()
        or not np.isnan(p[fold_ids == ""]).all()
    ):
        raise ValueError("분기 가중치의 정답·분기·비대상 점수 오류")
    eligible = fold_ids != ""
    validate_scores(p[eligible])
    hard, units, details = np.zeros(len(y), bool), np.full(len(y), 10, np.int64), {}
    for fold in FOLDS:
        selected = fold_ids == fold
        positive, negative = selected & (y == 1), selected & (y == 0)
        if selected.sum() < 100 or min(positive.sum(), negative.sum()) < 20:
            raise ValueError("분기 가중치의 대상 양성·음성 지원 부족")
        k = (int(negative.sum()) + 99) // 100
        q = float(np.sort(p[negative])[-k])
        hard |= negative & (p >= q)
        units[negative & (p < q)] = 1
        details[fold] = {
            "negative_rows": int(negative.sum()),
            "rank": k,
            "boundary_score": q,
            "boundary_ties": int((p[negative] == q).sum()),
            "hard_negative_rows": int((hard & selected).sum()),
        }
    total = int(units.sum())
    weights = units.astype(float) * len(y) / total
    effective_rows = float(len(y) ** 2 / np.square(weights).sum())
    if (
        not np.isfinite(weights).all()
        or (weights <= 0).any()
        or not math.isclose(math.fsum(weights.tolist()) / len(y), 1, abs_tol=1e-12, rel_tol=0)
        or effective_rows < 1000
    ):
        raise ValueError("분기 가중치의 평균·양수·유효 표본 오류")
    return weights, hard, units, {"folds": details, "normalization_units": total}


def build_forward_weights(training, parents, auxiliaries, scores):
    partition, support = forward_partition(training)
    require_forward_support(support)
    pd.testing.assert_frame_equal(partition, scores[partition.columns], check_exact=True)
    if set(parents) != set(ACTIONS) or set(auxiliaries) != set(ACTIONS):
        raise ValueError("분기 가중치의 부모·보조 행동 누락")
    ledgers = {arm: partition.copy() for arm in ARMS}
    details, overlap = {arm: {} for arm in ARMS}, {}
    for action in ACTIONS:
        parent = parents[action]
        if set(parent) != {"tree", "linear", "scope"} or parent["scope"] != SCOPE:
            raise ValueError("분기 가중치의 기존 점수 출처 오류")
        if set(auxiliaries[action]) != set(FOLDS):
            raise ValueError("분기 가중치의 보조 분기 누락")
        quiet = quiet_mask(training, action)
        targets = partition[action + "_fold"].ne("").to_numpy()
        actual = scores[action + "_forward_score"]
        if not np.array_equal(actual.notna(), targets):
            raise ValueError("분기 가중치의 과거 예측 지원 불일치")
        for fold in FOLDS:
            aux, target = fold_masks(training, action, fold)
            model = auxiliaries[action][fold]
            if (
                model["training_sha256"] != training_digest(training.loc[aux], action)
                or model["tree_sha256"] != model_digest(model["tree"])
                or model["rows"] != int(aux.sum())
            ):
                raise ValueError("분기 가중치의 보조 모델 연결 오류")
            expected = (
                CLASSES[action]
                .from_dict(model["tree"])
                .probabilities(training.loc[target, FEATURES].to_numpy(float))[:, 0]
            )
            np.testing.assert_array_equal(actual.loc[target].to_numpy(float), expected)
        matched = np.full(len(training), np.nan)
        matched[targets] = (
            CLASSES[action]
            .from_dict(parent["tree"])
            .probabilities(training.loc[targets, FEATURES].to_numpy(float))[:, 0]
        )
        for arm, probabilities in zip(
            ARMS, [matched, actual.to_numpy(float, na_value=np.nan)], strict=True
        ):
            values = probabilities[quiet]
            weights, hard, units, details[arm][action] = quarter_tail_weights(
                training.loc[quiet, "y_" + action].to_numpy(),
                values,
                partition.loc[quiet, action + "_fold"].to_numpy(),
            )
            for suffix, data, dtype in [
                ("score", values, "Float64"),
                ("weight", weights, "Float64"),
                ("hard_negative", hard, "boolean"),
                ("units", units, "Int64"),
            ]:
                column = action + "_" + suffix
                ledgers[arm][column] = pd.array([pd.NA] * len(training), dtype=dtype)
                ledgers[arm].loc[quiet, column] = data
        overlap[action] = {}
        a, b = [v[action + "_hard_negative"].fillna(False).to_numpy(bool) for v in ledgers.values()]
        for fold in FOLDS:
            target = partition[action + "_fold"].eq(fold).to_numpy()
            overlap[action][fold] = {
                "matched": int((target & a).sum()),
                "forward": int((target & b).sum()),
                "intersection": int((target & a & b).sum()),
                "union": int((target & (a | b)).sum()),
            }
    return ledgers, details, overlap


def forward_weight_groups(training, ledger):
    pd.testing.assert_frame_equal(training[CONTEXT], ledger[CONTEXT], check_exact=True)
    result = {}
    for action in ACTIONS:
        quiet = quiet_mask(training, action)
        y = training["y_" + action].to_numpy(int)
        hard = ledger[action + "_hard_negative"].fillna(False).to_numpy(bool)
        eligible = ledger[action + "_fold"].ne("").to_numpy()
        w = ledger[action + "_weight"].to_numpy(float, na_value=np.nan)
        if not np.array_equal(np.isfinite(w), quiet) or (w[quiet] <= 0).any():
            raise ValueError("분기 가중치 집계의 양수·지원 오류")
        partitions = {"all_training": quiet}
        for name, mask in groups(training, action).items():
            partitions[name] = quiet & mask
            for region, count_mask in [
                ("below", training.adds.lt(5)),
                ("capped", training.adds.ge(5)),
            ]:
                partitions[name + "__" + region] = quiet & mask & count_mask.to_numpy()
        for month in pd.date_range(
            "2020-01-01", "2021-01-01", freq="MS", inclusive="left"
        ).strftime("%Y-%m"):
            partitions[month] = quiet & training.end.dt.strftime("%Y-%m").eq(month).to_numpy()
        for reason in REASONS:
            partitions["reason_" + reason] = (
                quiet & ledger[action + "_reason"].eq(reason).to_numpy()
            )
        for fold in FOLDS:
            partitions[fold] = quiet & ledger[action + "_fold"].eq(fold).to_numpy()
        subsets = {
            "all": quiet,
            "positive": y == 1,
            "hard_negative": hard,
            "easy_eligible_negative": (y == 0) & eligible & ~hard,
            "neutral_negative": (y == 0) & ~eligible,
        }
        result[action] = {}
        for name, mask in partitions.items():
            total = math.fsum(w[mask].tolist())
            result[action][name] = {}
            for category, subset in subsets.items():
                values = w[mask & subset]
                mass = math.fsum(values.tolist())
                squares = math.fsum((values * values).tolist())
                result[action][name][category] = {
                    "rows": len(values),
                    "sum": mass,
                    "mean": mass / len(values) if len(values) else None,
                    "effective_rows": mass * mass / squares if len(values) else None,
                    "minimum": float(values.min()) if len(values) else None,
                    "maximum": float(values.max()) if len(values) else None,
                    "maximum_share": float(values.max() / total) if len(values) else None,
                }
    return result
