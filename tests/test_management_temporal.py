from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest
from test_error_budget import population

import wonyotti_fr.management_temporal_research as research
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.error_budget import evaluate_budget, fit_budget_cutoffs
from wonyotti_fr.holding_stress import seal
from wonyotti_fr.management_temporal import match_temporal, temporal_months, temporal_summary
from wonyotti_fr.management_temporal_reference import (
    graph_cardinality,
    verify_temporal,
    verify_temporal_summaries,
)
from wonyotti_fr.management_temporal_research import audit_temporal, checked_temporal_reference


def grid(n=40):
    return pd.DataFrame(
        {
            "end": pd.date_range("2021-01-31 23:50", periods=n, freq="min", tz="UTC"),
            "episode_id": np.ones(n, dtype=np.int64),
        },
        index=np.arange(n) * 10 + 7,
    )


def run(frame, request_indices, target_indices, horizon=15):
    requests, targets = np.zeros(len(frame), bool), np.zeros(len(frame), bool)
    requests[request_indices], targets[target_indices] = True, True
    rr, ee = match_temporal(frame, requests, targets, horizon)
    verify_temporal(frame, requests, targets, horizon, rr, ee)
    result = temporal_summary(rr, ee, horizon)
    months = temporal_months(rr, ee, horizon, ["2021-01", "2021-02"])
    verify_temporal_summaries(rr, ee, horizon, result, months)
    return rr, ee, result, months


@pytest.mark.parametrize("lead,expected", [(0, 1), (1, 1), (14, 1), (15, 0), (-1, 0)])
def test_forward_half_open_boundaries(lead, expected):
    rr, _, total, _ = run(grid(), [2], [2 + lead])
    assert total["matched_requests"] == expected
    assert total["original_true_positive"] == (lead == 0)
    if expected:
        assert rr.lead_minutes.iloc[0] == lead


def test_repeated_requests_cannot_multiply_one_target():
    _, _, total, _ = run(grid(), [0, 1, 2, 3], [4])
    assert total["requests_with_any_target"] == 4
    assert total["matched_requests"] == 1
    assert total["unmatched_requests"] == 3


def test_one_request_cannot_capture_several_targets():
    _, _, total, _ = run(grid(), [0], [1, 2, 3, 4])
    assert total["targets_with_any_request"] == 4
    assert total["matched_targets"] == 1


def test_earliest_request_preserves_maximum_count_and_exact_hit_is_separate():
    rr, _, total, _ = run(grid(), [0, 14], [14, 15])
    assert total["matched_requests"] == 2 and total["original_true_positive"] == 1
    assert rr.lead_minutes.tolist() == [14, 1]
    assert total["lead_minutes_counts"]["0"] == 0
    assert graph_cardinality([[0, 1], [0]]) == 2


def test_episode_boundary_blocks_matching():
    frame = grid()
    frame.loc[frame.index[2:], "episode_id"] = 2
    _, _, total, _ = run(frame, [1], [2])
    assert total["matched_requests"] == total["requests_with_any_target"] == 0


def test_month_boundary_keeps_two_different_cohorts():
    _, _, total, monthly = run(grid(), [8], [12])
    assert total["matched_requests"] == 1
    assert monthly["2021-01"]["matched_requests"] == 1
    assert monthly["2021-01"]["matched_targets"] == 0
    assert monthly["2021-02"]["matched_requests"] == 0
    assert monthly["2021-02"]["matched_targets"] == 1


@pytest.mark.parametrize("r,e", [([], []), ([0], []), ([], [0])])
def test_empty_sides_preserve_undefined_rates(r, e):
    _, _, total, _ = run(grid(), r, e)
    assert total["matched_requests"] == 0
    assert (total["request_match_rate"] is None) == (not r)
    assert (total["target_capture_rate"] is None) == (not e)


def test_empty_frame_and_missing_observation_grid():
    _, _, total, _ = run(grid(0), [], [])
    assert total["requests"] == total["targets"] == 0
    frame = grid().drop(grid().index[3])
    rr, _, total, _ = run(frame, [0, 38], [1])
    assert total["complete_observation_windows"] == 0
    assert total["windows_beyond_phase"] == 1
    assert rr.event_end.notna().tolist() == [True, False]


def exhaustive(adjacency, index=0, used=frozenset()):
    if index == len(adjacency):
        return 0
    return max(
        [exhaustive(adjacency, index + 1, used)]
        + [
            1 + exhaustive(adjacency, index + 1, used | {event})
            for event in adjacency[index]
            if event not in used
        ]
    )


def test_graph_algorithm_against_all_three_by_three_graphs():
    for bits in product([False, True], repeat=9):
        adjacency = [[e for e in range(3) if bits[r * 3 + e]] for r in range(3)]
        assert graph_cardinality(adjacency) == exhaustive(adjacency)


def test_temporal_matching_against_exhaustive_random_cases():
    rng = np.random.default_rng(139)
    frame = grid(6)
    frame["episode_id"] = [1, 1, 1, 2, 2, 2]
    for _ in range(80):
        requests, targets = rng.random((2, 6)) < 0.5
        for horizon in (1, 15):
            rr, ee = match_temporal(frame, requests, targets, horizon)
            ri, ei = np.flatnonzero(requests), np.flatnonzero(targets)
            adjacency = [
                [
                    j
                    for j, e in enumerate(ei)
                    if frame.episode_id.iloc[r] == frame.episode_id.iloc[e] and 0 <= e - r < horizon
                ]
                for r in ri
            ]
            assert int(rr.event_end.notna().sum()) == exhaustive(adjacency)
            verify_temporal(frame, requests, targets, horizon, rr, ee)


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "reverse",
        "timezone",
        "fractional_time",
        "nat",
        "fractional_id",
        "zero_id",
        "infinite_id",
        "bool_id",
    ],
)
def test_invalid_context_rejected(damage):
    frame = grid()
    if damage == "duplicate":
        frame.loc[frame.index[1], "end"] = frame.end.iloc[0]
    elif damage == "reverse":
        frame = frame.iloc[::-1]
    elif damage == "timezone":
        frame["end"] = frame.end.dt.tz_convert("Asia/Seoul")
    elif damage == "fractional_time":
        frame["end"] += pd.Timedelta(seconds=1)
    elif damage == "nat":
        frame.loc[frame.index[0], "end"] = pd.NaT
    else:
        frame["episode_id"] = {
            "fractional_id": 1.5,
            "zero_id": 0,
            "infinite_id": np.inf,
            "bool_id": True,
        }[damage]
    with pytest.raises(ValueError):
        match_temporal(frame, np.zeros(len(frame), bool), np.zeros(len(frame), bool), 15)


@pytest.mark.parametrize("horizon", [0, 2, 16, 15.0, True])
def test_invalid_horizon_rejected(horizon):
    with pytest.raises(ValueError):
        match_temporal(grid(), np.zeros(40, bool), np.zeros(40, bool), horizon)


@pytest.mark.parametrize("mask", [np.zeros(40, int), np.zeros((40, 1), bool), np.zeros(39, bool)])
def test_invalid_request_mask_rejected(mask):
    with pytest.raises(ValueError):
        match_temporal(grid(), mask, np.zeros(40, bool), 15)


@pytest.mark.parametrize("damage", ["pair", "lead", "availability", "observation", "reverse"])
def test_independent_verifier_rejects_corrupted_ledger(damage):
    frame = grid()
    r = np.zeros(40, bool)
    e = r.copy()
    r[[0, 1]], e[[2, 3]] = True, True
    rr, ee = match_temporal(frame, r, e, 15)
    if damage == "pair":
        rr.loc[1, "event_end"] = rr.event_end.iloc[0]
    elif damage == "lead":
        rr.loc[0, "lead_minutes"] = 14
    elif damage == "availability":
        rr.loc[0, "has_target"] = False
    elif damage == "observation":
        rr.loc[0, "window_grid_complete"] = False
    else:
        ee.loc[0, "request_end"] = pd.NaT
    with pytest.raises(AssertionError):
        verify_temporal(frame, r, e, 15, rr, ee)


def test_future_target_change_cannot_change_request_identity_or_inputs():
    frame = grid()
    before = frame.copy(deep=True)
    first, _, _, _ = run(frame, [0], [2])
    second, _, _, _ = run(frame, [0], [20])
    pd.testing.assert_frame_equal(
        first[["request_end", "episode_id"]], second[["request_end", "episode_id"]]
    )
    assert first.event_end.notna().all() and second.event_end.isna().all()
    pd.testing.assert_frame_equal(frame, before, check_exact=True)


def test_all_groups_exact_confusions_unsupported_and_parquet(tmp_path):
    frame, pair = population()
    frame["adds"] = frame.adds.clip(upper=4)
    frame["adds_capped"] = frame.adds
    cuts = fit_budget_cutoffs(frame, pair)
    metrics, ledger = evaluate_budget(frame, pair, cuts)
    results, rr, ee = audit_temporal(frame, pair, cuts, metrics, ledger, ["2021-02"])
    assert len(results) == 144
    unsupported = [r for r in results if r["status"] == "no_calibration_rows"]
    assert len(unsupported) == 24
    assert all(r["total"] is None and r["monthly"]["2021-02"] is None for r in unsupported)
    for name, table in [("requests", rr), ("targets", ee)]:
        table.to_parquet(tmp_path / (name + ".parquet"), index=False)
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / (name + ".parquet")), table, check_exact=True
        )
    damaged = ledger.copy()
    damaged.iloc[0, 2] = not bool(damaged.iloc[0, 2])
    with pytest.raises(AssertionError):
        audit_temporal(frame, pair, cuts, metrics, damaged, ["2021-02"])


@pytest.mark.parametrize(
    "damage", [None, "incomplete", "accepted", "wrong_source", "bool_exit", "bad_hash"]
)
def test_reference_requires_completed_unchanged_failed_candidate(tmp_path, monkeypatch, damage):
    root = tmp_path / "parent"
    root.mkdir()
    summary = {
        key: True
        for key in [
            "complete",
            "independent_verification_complete",
            "all_inputs_and_code_unchanged",
            "original_predictions_and_metrics_exact",
            "all_groups_months_cutoffs_and_requests_verified",
        ]
    }
    flags = {
        key: False
        for key in ["new_models_fitted", "new_accounts_executed", "profitability_accepted"]
    }
    summary.update(flags, original_candidate_admitted=False)
    if damage == "incomplete":
        summary["independent_verification_complete"] = False
    if damage == "accepted":
        summary["original_candidate_admitted"] = True
    save_json(root / "summary.json", summary)
    save_json(
        root / "manifest.json",
        {
            "settings": {
                "reference": "prior.json",
                "reference_sha256": "a" * 64,
                "sources": {"teacher": "bound"},
            }
        },
    )
    seal(root)
    proof = tmp_path / "proof.json"
    save_json(
        proof,
        {
            "complete": True,
            "run": str(root),
            "files_sha256": sha256(root / "files.json"),
            "summary_sha256": sha256(root / "summary.json"),
            "process_exit_code": False if damage == "bool_exit" else 0,
            **flags,
        },
    )
    monkeypatch.setattr(
        research,
        "checked_budget_reference",
        lambda *_: {"teacher": "wrong" if damage == "wrong_source" else "bound"},
    )
    digest = "b" * 64 if damage == "bad_hash" else sha256(proof)
    if damage is None:
        assert checked_temporal_reference(proof, digest) == (root, {"teacher": "bound"})
    else:
        with pytest.raises(ValueError):
            checked_temporal_reference(proof, digest)
