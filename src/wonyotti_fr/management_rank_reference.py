from __future__ import annotations

import math

import numpy as np

from .position_boost_verification import compare_statistics


def reference_cutoffs(scores):
    levels, counts = np.unique(scores, return_counts=True)
    result, cumulative = {}, counts[::-1].cumsum()
    for name, fraction in [("0.1%", .001), ("0.5%", .005), ("1%", .01), ("5%", .05)]:
        target = math.ceil(len(scores) * fraction)
        if not target:
            raise ValueError("순위 검산의 절단점 표본 부족")
        index = int(np.argmax(cumulative >= target))
        selected = int(cumulative[index])
        result[name] = {"cutoff": float(levels[::-1][index]), "source_rows": len(scores),
                        "nominal_rows": target, "selected_rows": selected, "tie_excess": selected - target}
    return result


def quantiles(values):
    result, ordered = {}, sorted(values)
    for name, fraction in [("min", 0), ("q10", .1), ("q25", .25), ("median", .5),
                           ("q75", .75), ("q90", .9), ("q99", .99), ("max", 1)]:
        if not ordered:
            result[name] = None
            continue
        position = fraction * (len(ordered) - 1)
        left, right = math.floor(position), math.ceil(position)
        result[name] = float(ordered[left] + (ordered[right] - ordered[left]) * (position - left))
    return result


def counts(labels, requests, original):
    codes = np.asarray(labels, int) * 4 + np.asarray(requests, int) * 2 + np.asarray(original, int)
    n = np.bincount(codes, minlength=8).tolist()
    tp, fp, fn = n[6] + n[7], n[2] + n[3], n[4] + n[5]
    return {"rows": sum(n), "positive": sum(n[4:]), "requests": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "added_tp": n[6], "added_fp": n[2], "removed_tp": n[5], "removed_fp": n[1]}


def verify_score_audit(frame, threshold, calibration_scores, result, ledger):
    assert set(result) == {"threshold", "original", "scores", "positive_max_below_threshold",
                           "positive_rank_start_fraction", "positive_rank_end_fraction", "concentration", "tails"}
    assert set(ledger) == {"positives", "episodes"}
    assert set(result["tails"]) == {"calibration_fixed", "contemporaneous_hindsight"}
    y, p, original = frame.label.to_numpy(int), frame.score.to_numpy(float), frame.request.to_numpy()
    assert np.array_equal(p >= threshold, original)
    compare_statistics(result["original"], counts(y, original, original))
    assert result["threshold"] == threshold
    assert result["positive_max_below_threshold"] == (bool(max(p[y == 1]) < threshold) if y.any() else None)
    for name, values in [("all", p), ("positive", p[y == 1]), ("negative", p[y == 0])]:
        compare_statistics(result["scores"][name], quantiles(values))
    levels, inverse, frequency = np.unique(p, return_inverse=True, return_counts=True)
    total_above = len(p) - frequency.cumsum()
    negative_frequency = np.bincount(inverse[y == 0], minlength=len(levels))
    negative_above = int((y == 0).sum()) - negative_frequency.cumsum()
    expected = []
    for i in np.flatnonzero(y):
        level = inverse[i]
        expected.append({"end": frame.end.iloc[i].isoformat(), "episode_id": int(frame.episode_id.iloc[i]),
                         "score": float(p[i]), "requested": bool(original[i]),
                         "rank_start": int(total_above[level] + 1),
                         "rank_end": int(total_above[level] + frequency[level]),
                         "negative_above": int(negative_above[level]),
                         "negative_equal": int(negative_frequency[level])})
    assert ledger["positives"] == expected
    for side in ["start", "end"]:
        compare_statistics(result["positive_rank_" + side + "_fraction"],
            quantiles([item["rank_" + side] / len(p) for item in expected]))
    episode_rows = {}
    for identity, label, request in zip(frame.episode_id, y, original, strict=True):
        identity = int(identity)
        item = episode_rows.setdefault(identity, {"episode_id": identity, "rows": 0, "positive": 0,
                                                   "requests": 0, "tp": 0, "fp": 0, "fn": 0})
        item["rows"] += 1
        item["positive"] += int(label)
        item["requests"] += int(request)
        item["tp"] += int(label == 1 and request)
        item["fp"] += int(label == 0 and request)
        item["fn"] += int(label == 1 and not request)
    episodes = [episode_rows[key] for key in sorted(episode_rows)]
    assert ledger["episodes"] == episodes
    concentration = {"episodes": len(episodes), "positive_episodes": sum(v["positive"] > 0 for v in episodes),
                     "requested_episodes": sum(v["requests"] > 0 for v in episodes),
                     "tp_episodes": sum(v["tp"] > 0 for v in episodes)}
    for key in ["positive", "fp"]:
        ordered = sorted([item[key] for item in episodes], reverse=True)
        for n in [1, 3]:
            concentration[key + "_top" + str(n) + "_fraction"] = sum(ordered[:n]) / sum(ordered) if sum(ordered) else None
    compare_statistics(result["concentration"], concentration)
    for name, values in [("calibration_fixed", calibration_scores), ("contemporaneous_hindsight", p)]:
        cuts = reference_cutoffs(values) if len(values) else None
        assert set(result["tails"][name]) == {"0.1%", "0.5%", "1%", "5%"}
        for key, item in result["tails"][name].items():
            spec = cuts[key] if cuts else None
            assert item["source"] == spec
            requests = np.greater_equal(p, spec["cutoff"]) if spec else np.zeros(len(p), bool)
            compare_statistics(item["metrics"], counts(y, requests, original))
    return {"verified": True, "rows": len(frame), "positive_ranks": len(expected), "episodes": len(episodes)}
