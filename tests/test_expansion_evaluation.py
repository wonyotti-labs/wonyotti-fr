import pytest

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.expansion_evaluation import ensure_expansion_period
from wonyotti_fr.expansion_research import expansion_gate
from wonyotti_fr.period_guard import guard_replay_period


def test_failed_gate_cannot_be_bypassed_by_flag_or_replay(tmp_path):
    metrics = {'total_return': -.01, 'closed_trades': 25, 'permanent_halt': False}
    frozen = {'protocol': 'expansion_v4', 'development_metrics': metrics,
              'new_evaluation_period': ['2026-09-01', '2026-10-01'],
              'seen_2026_period': ['2026-01-01', '2026-09-01']}
    save_json(tmp_path / 'frozen_selection.json', frozen)
    path = tmp_path / 'validation-2021/metrics.json'
    save_json(path, metrics)
    gate = expansion_gate(metrics, metrics)
    save_json(tmp_path / 'new_evaluation_gate.json', {**gate, 'may_open_new_period': True,
              'selection_sha256': sha256(tmp_path / 'frozen_selection.json'), 'validation_metrics_sha256': sha256(path)})
    with pytest.raises(ValueError, match='미충족'):
        ensure_expansion_period(tmp_path, frozen, 'new')
    with pytest.raises(ValueError, match='미충족'):
        guard_replay_period(tmp_path, frozen, '2026-08-01', '2026-09-02')
    guard_replay_period(tmp_path, frozen, '2026-01-01', '2026-09-01')
    with pytest.raises(ValueError, match='이후 기간'):
        guard_replay_period(tmp_path, frozen, '2026-09-01', '2026-11-01')
    save_json(path, {**metrics, 'total_return': .2})
    with pytest.raises(ValueError, match='지문'):
        ensure_expansion_period(tmp_path, frozen, 'new')
