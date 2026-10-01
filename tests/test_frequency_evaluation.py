import json

import pytest

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.frequency_evaluation import ensure_period_allowed
from wonyotti_fr.frequency_research import frequency_gate


def test_period_guard_recomputes_gate_and_rejects_modified_evidence(tmp_path):
    frozen = {'protocol': 'frequency_v3', 'development_metrics': {'total_return': -0.1},
              'new_evaluation_period': ['2026-09-01', '2026-10-01'],
              'seen_2026_period': ['2026-01-01', '2026-09-01']}
    save_json(tmp_path / 'frozen_selection.json', frozen)
    validation = {'total_return': 0.1, 'closed_trades': 20, 'permanent_halt': False}
    metrics_path = tmp_path / 'validation-2021/metrics.json'
    save_json(metrics_path, validation)
    gate = frequency_gate(frozen['development_metrics'], validation)
    gate.update(may_open_new_period=True, selection_sha256=sha256(tmp_path / 'frozen_selection.json'),
                validation_metrics_sha256=sha256(metrics_path))
    save_json(tmp_path / 'new_evaluation_gate.json', gate)
    with pytest.raises(ValueError, match='미충족'):
        ensure_period_allowed(tmp_path, frozen, 'new')
    assert ensure_period_allowed(tmp_path, frozen, 'seen_2026') == ('2026-01-01', '2026-09-01')
    metrics_path.write_text(json.dumps({**validation, 'total_return': 0.2}))
    with pytest.raises(ValueError, match='지문'):
        ensure_period_allowed(tmp_path, frozen, 'new')
