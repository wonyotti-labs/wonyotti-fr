from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .first_state_models import FEATURES
from .fixed_bin_reference import verify_nodes
from .management_capacity_reference import partition_forest
from .position_boost_verification import compare_statistics


def reference_forward_partition(training, ledger, support):
    contexts = ["end", "entry_time", "label_end", "episode_id"]
    pd.testing.assert_frame_equal(training[contexts], ledger[contexts], check_exact=True)
    dates = {
        "Q2": ("2020-03-31", "2020-04-01", "2020-06-30", "2020-07-01"),
        "Q3": ("2020-06-30", "2020-07-01", "2020-09-30", "2020-10-01"),
        "Q4": ("2020-09-30", "2020-10-01", "2020-12-31", "2021-01-01"),
    }
    bounds = {k: tuple(pd.Timestamp(v, tz="UTC").value for v in vs) for k, vs in dates.items()}
    ends, entries, labels = [
        training[k].astype("datetime64[ns, UTC]").array.asi8 for k in contexts[:3]
    ]
    day = 86_400_000_000_000
    masks, verified_support = {}, {}
    for action in ["reduce", "increase"]:
        quiet = training["past_" + action + "_log_count_15m"].eq(0).to_numpy()
        auxiliary, targets = ({fold: [] for fold in dates} for _ in range(2))
        reasons, fold_ids = [], []
        for i, (end, entry, label, is_quiet) in enumerate(
            zip(ends, entries, labels, quiet, strict=True)
        ):
            reason, selected = ("warmup" if is_quiet else "recent"), ""
            if is_quiet:
                for fold, (cut, start, stop, next_start) in bounds.items():
                    if label < cut:
                        auxiliary[fold].append(i)
                    if start <= end < next_start:
                        if end < start + day:
                            reason = "left_boundary"
                        elif label >= stop:
                            reason = "right_boundary"
                        elif entry < start:
                            reason = "prior_entry"
                        else:
                            reason, selected = "eligible", fold
                            targets[fold].append(i)
            reasons.append(reason)
            fold_ids.append(selected)
        np.testing.assert_array_equal(ledger[action + "_reason"], reasons)
        np.testing.assert_array_equal(ledger[action + "_fold"], fold_ids)
        masks[action], verified_support[action] = {}, {}
        for fold, (cut, _, _, _) in bounds.items():
            masks[action][fold] = {}
            parts = {}
            for role, ids, minimum in [
                ("auxiliary", auxiliary[fold], 1000),
                ("target", targets[fold], 100),
            ]:
                mask = np.zeros(len(training), bool)
                mask[ids] = True
                masks[action][fold][role] = mask
                part = training.iloc[ids]
                parts[role] = part
                counts = Counter(part["y_" + action].tolist())
                parts[role + "_support"] = {
                    "rows": len(ids),
                    "positive": counts[1],
                    "negative": counts[0],
                    "episodes": len(set(part.episode_id)),
                    "positive_episodes": len(
                        set(part.loc[part["y_" + action].eq(1), "episode_id"])
                    ),
                    "minimum_rows": minimum,
                    "minimum_class_rows": 20,
                    "supported": len(ids) >= minimum and min(counts[0], counts[1]) >= 20,
                }
            a, t = parts["auxiliary"], parts["target"]
            verified_support[action][fold] = {
                "auxiliary": parts["auxiliary_support"],
                "target": parts["target_support"],
                "overlapping_episodes": len(set(a.episode_id) & set(t.episode_id)),
                "time_ordered": bool(not len(a) or not len(t) or max(a.label_end) < min(t.end)),
                "auxiliary_label_end_max": max(a.label_end).isoformat() if len(a) else None,
                "target_end_min": min(t.end).isoformat() if len(t) else None,
                "cut": pd.Timestamp(cut, tz="UTC").isoformat(),
            }
    assert verified_support == support
    return masks


def verify_forward_auxiliaries(training, models, scores, support):
    masks = reference_forward_partition(training, scores, support)
    total_trees, total_predictions = 0, 0
    params = dict(
        loss="log_loss",
        max_iter=256,
        learning_rate=0.05,
        max_leaf_nodes=4,
        max_depth=2,
        min_samples_leaf=64,
        l2_regularization=1.0,
        max_bins=255,
        max_features=1.0,
        categorical_features=None,
        early_stopping=False,
        random_state=0,
        class_weight=None,
    )
    with threadpool_limits(limits=1):
        for action in ["reduce", "increase"]:
            targets = np.zeros(len(training), bool)
            for fold in ["Q2", "Q3", "Q4"]:
                aux, target = (masks[action][fold][k] for k in ["auxiliary", "target"])
                x = training.loc[aux, FEATURES].to_numpy(float)
                y = training.loc[aux, "y_" + action].to_numpy(int)
                tx = training.loc[target, FEATURES].to_numpy(float)
                learner = HistGradientBoostingClassifier(**params).fit(x, y)
                saved = models[action][fold]
                verify_nodes(learner, saved["tree"])
                assert saved["rows"] == len(x)
                assert saved["thresholds"] == [
                    q.tolist() for q in learner._bin_mapper.bin_thresholds_
                ]
                logits = partition_forest(saved["tree"]["models"][0], tx)
                numeric = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
                stored = scores.loc[target, action + "_forward_score"].to_numpy(float)
                np.testing.assert_allclose(stored, numeric, rtol=0, atol=1e-12)
                np.testing.assert_allclose(
                    stored, learner.predict_proba(tx)[:, 1], rtol=0, atol=1e-12
                )
                targets |= target
                total_trees += 256
                total_predictions += int(target.sum())
            np.testing.assert_array_equal(scores[action + "_forward_score"].notna(), targets)
    return {
        "independent_auxiliary_trees": total_trees,
        "target_action_predictions": total_predictions,
        "all_past_bins_nodes_and_target_probabilities_verified": True,
    }


def verify_forward_weights(training, parents, scores, ledgers, details, overlap):
    assert set(ledgers) == set(details) == {"v142matched", "v142forward"}
    expected_hard = {}
    for action in ["reduce", "increase"]:
        quiet = training["past_" + action + "_log_count_15m"].eq(0).to_numpy()
        targets = scores[action + "_fold"].ne("").to_numpy()
        y = training.loc[quiet, "y_" + action].to_numpy(int)
        fold_ids = scores.loc[quiet, action + "_fold"].to_numpy()
        x = training.loc[targets, FEATURES].to_numpy(float)
        matched = 1 / (
            1
            + np.exp(-np.clip(partition_forest(parents[action]["tree"]["models"][0], x), -700, 700))
        )
        expected_hard[action] = {}
        for arm in ["v142matched", "v142forward"]:
            ledger = ledgers[arm]
            for column in [
                "end",
                "entry_time",
                "label_end",
                "episode_id",
                action + "_fold",
                action + "_reason",
            ]:
                pd.testing.assert_series_equal(ledger[column], scores[column])
            for suffix in ["weight", "hard_negative", "units"]:
                np.testing.assert_array_equal(ledger[action + "_" + suffix].notna(), quiet)
            np.testing.assert_array_equal(ledger[action + "_score"].notna(), targets)
            expected_scores = (
                matched
                if arm == "v142matched"
                else scores.loc[targets, action + "_forward_score"].to_numpy(float)
            )
            np.testing.assert_allclose(
                ledger.loc[targets, action + "_score"].to_numpy(float),
                expected_scores,
                atol=1e-12,
                rtol=0,
            )
            p = ledger.loc[quiet, action + "_score"].to_numpy(float, na_value=np.nan)
            hard = np.zeros(len(y), bool)
            units = np.full(len(y), 10, dtype=int)
            expected_details = {}
            for fold in ["Q2", "Q3", "Q4"]:
                negatives = [
                    i
                    for i, (label, fid) in enumerate(zip(y, fold_ids, strict=True))
                    if label == 0 and fid == fold
                ]
                ranked = sorted((p[i] for i in negatives), reverse=True)
                k = math.ceil(len(negatives) / 100)
                q = ranked[k - 1]
                for i in negatives:
                    hard[i] = p[i] >= q
                    units[i] = 10 if hard[i] else 1
                expected_details[fold] = {
                    "negative_rows": len(negatives),
                    "rank": k,
                    "boundary_score": q,
                    "boundary_ties": ranked.count(q),
                    "hard_negative_rows": int(hard[fold_ids == fold].sum()),
                }
            total = sum(units.tolist())
            expected_weights = np.array([float(Fraction(int(u) * len(y), total)) for u in units])
            np.testing.assert_array_equal(
                ledger.loc[quiet, action + "_hard_negative"].to_numpy(bool), hard
            )
            np.testing.assert_array_equal(ledger.loc[quiet, action + "_units"].to_numpy(int), units)
            np.testing.assert_allclose(
                ledger.loc[quiet, action + "_weight"].to_numpy(float),
                expected_weights,
                rtol=0,
                atol=1e-12,
            )
            assert details[arm][action] == {"folds": expected_details, "normalization_units": total}
            full = np.zeros(len(training), bool)
            full[quiet] = hard
            expected_hard[action][arm] = full
        a, b = [expected_hard[action][arm] for arm in ["v142matched", "v142forward"]]
        for fold in ["Q2", "Q3", "Q4"]:
            ids = np.flatnonzero(scores[action + "_fold"].eq(fold))
            sa, sb = set(ids[a[ids]]), set(ids[b[ids]])
            assert overlap[action][fold] == {
                "matched": len(sa),
                "forward": len(sb),
                "intersection": len(sa & sb),
                "union": len(sa | sb),
            }
    return {"both_arms_all_raw_units_scores_and_normalized_weights_verified": True}


def verify_forward_weight_groups(training, ledger, saved):
    for action in ["reduce", "increase"]:
        quiet = training["past_" + action + "_log_count_15m"].eq(0).to_numpy()
        partitions = {"all_training": quiet}
        exists = training["past_" + action + "_exists"].eq(1).to_numpy()
        for name, mask in {
            "all": quiet,
            "quiet": quiet,
            "recent": np.zeros(len(training), bool),
            "first": quiet & ~exists,
            "restart": quiet & exists,
        }.items():
            partitions[name] = mask
            partitions[name + "__below"] = mask & training.adds.le(4).to_numpy()
            partitions[name + "__capped"] = mask & training.adds.ge(5).to_numpy()
        for month in range(1, 13):
            partitions[f"2020-{month:02d}"] = quiet & training.end.dt.month.eq(month).to_numpy()
        for reason in [
            "recent",
            "warmup",
            "left_boundary",
            "right_boundary",
            "prior_entry",
            "eligible",
        ]:
            partitions["reason_" + reason] = (
                quiet & ledger[action + "_reason"].eq(reason).to_numpy()
            )
        for fold in ["Q2", "Q3", "Q4"]:
            partitions[fold] = quiet & ledger[action + "_fold"].eq(fold).to_numpy()
        assert set(partitions) == set(saved[action])
        y = training["y_" + action].to_numpy(int)
        w = ledger[action + "_weight"].to_numpy(float, na_value=np.nan)
        hard = ledger[action + "_hard_negative"].fillna(False).to_numpy(bool)
        eligible = ledger[action + "_fold"].ne("").to_numpy()
        for name, mask in partitions.items():
            total = math.fsum(w[mask].tolist())
            expected = {}
            for category, subset in {
                "all": quiet,
                "positive": y == 1,
                "hard_negative": hard,
                "easy_eligible_negative": (y == 0) & eligible & ~hard,
                "neutral_negative": (y == 0) & ~eligible,
            }.items():
                values = w[mask & subset].tolist()
                mass = math.fsum(values)
                expected[category] = {
                    "rows": len(values),
                    "sum": mass,
                    "mean": mass / len(values) if values else None,
                    "effective_rows": mass**2 / math.fsum(v * v for v in values)
                    if values
                    else None,
                    "minimum": min(values) if values else None,
                    "maximum": max(values) if values else None,
                    "maximum_share": max(values) / total if values else None,
                }
            compare_statistics(saved[action][name], expected)
    return {"all_states_months_count_regions_reasons_and_weight_classes_verified": True}
