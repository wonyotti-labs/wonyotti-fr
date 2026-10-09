import json

import numpy as np
import pandas as pd
import pytest
from test_first_close_diagnostics import financial_examples
from test_minute_close_diagnostics import fixture, seal

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.minute_close_diagnostics import run_minute_close_diagnosis
from wonyotti_fr.minute_close_learning import MINUTE_POLICIES
from wonyotti_fr.visited_close_diagnostics import run_visited_close_diagnosis
from wonyotti_fr.visited_close_evaluation import VISITED_COMPARISONS
from wonyotti_fr.visited_close_reference import MINUTE_DIAGNOSIS_FILES


def signal_examples():
    frame = financial_examples()
    step = frame.groupby('position_entry_time').cumcount().to_numpy()
    frame['favorable_move'] = step/100
    frame['close_advantage_bps'] = np.where(step >= 5, 12., -10.)
    frame.loc[frame.index % 17 == 0, 'close_advantage_bps'] = 0.
    frame['close_advantage_pnl'] = frame.close_advantage_bps*frame.decision_equity/10000
    frame['close_cash'] = frame.continue_cash+frame.close_advantage_pnl
    return frame


def make_reference(tmp_path, monkeypatch):
    monkeypatch.setattr('test_minute_close_diagnostics.financial_examples', signal_examples)
    labels, legacy, minute, _ = fixture(tmp_path, monkeypatch)
    reference = run_minute_close_diagnosis(labels, legacy, tmp_path/'minute-runs')
    assert set(json.loads((reference/'files.json').read_text())) == MINUTE_DIAGNOSIS_FILES
    return reference, labels, legacy, minute


def test_complete_visited_pipeline_reproduces_old_results_and_keeps_every_original_row(tmp_path, monkeypatch):
    reference, _, _, _ = make_reference(tmp_path, monkeypatch)
    original = json.loads((reference/'files.json').read_text())
    out = run_visited_close_diagnosis(reference, tmp_path/'visited-runs')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['all_previous_outputs_reproduced']
    assert summary['same_full_position_population_and_first_equity'] and summary['original_training_and_full_diagnosis_rows_preserved']
    assert summary['teacher_models'] == 5 and summary['candidate_models'] == 1 and summary['candidate'] == 'visited_1m'
    assert len(summary['checks']) == 20 and not summary['profitability_accepted']
    assert summary['visited_training_rows'] < summary['training_rows']
    assert summary['training_positions'] == summary['visited_positions'] and summary['diagnosis_positions'] == 62
    for name, value in original.items():
        assert sha256(reference/name) == sha256(out/'baseline_source'/name) == value
    parity = json.loads((out/'reference_parity.json').read_text())
    assert parity['original_decision'] == json.loads((reference/'decision.json').read_text())
    for name in ['minute_ledger', 'feature_linkage', 'training_used', 'training_weights', 'diagnosis_used', 'diagnosis_weights', 'exclusion_ledger']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}.parquet'), pd.read_parquet(reference/f'{name}.parquet'), check_exact=True)
    old = pd.read_parquet(reference/'predictions.parquet')
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet')[old.columns], old, check_exact=True)
    for name in MINUTE_POLICIES:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'positions_{name}.parquet'), pd.read_parquet(reference/f'positions_{name}.parquet'), check_exact=True)
    for name in ['metrics', 'first_metrics', 'probability_metrics']:
        before, after = (json.loads((folder/f'{name}.json').read_text()) for folder in [reference, out])
        assert {key: after[key] for key in before} == before
    assert set(json.loads((out/'intervals.json').read_text())) == set(VISITED_COMPARISONS)
    visit = pd.read_parquet(out/'visitation_ledger.parquet')
    contribution = pd.read_parquet(out/'full_training_contribution.parquet')
    assert len(visit) == len(contribution) == summary['training_rows']
    assert contribution.loc[~visit.visited_prefix, 'reason'].eq('after_first_teacher_close').all()


def test_future_diagnosis_changes_leave_teachers_visited_training_and_final_choices_unchanged(tmp_path, monkeypatch):
    reference, labels, legacy, minute = make_reference(tmp_path, monkeypatch)
    first = run_visited_close_diagnosis(reference, tmp_path/'original')
    changed = minute.copy()
    selected = changed.decision_time.ge(pd.Timestamp('2021-10-02', tz='UTC')) & changed.decision_time.dt.minute.mod(5).ne(0)
    changed.loc[selected, ['close_advantage_bps', 'close_advantage_pnl']] *= -100
    changed.loc[selected, 'close_cash'] = changed.loc[selected, 'continue_cash']+changed.loc[selected, 'close_advantage_pnl']
    changed.to_parquet(labels/'opportunity_ledger.parquet', index=False)
    seal(labels)
    future_reference = run_minute_close_diagnosis(labels, legacy, tmp_path/'future-minute')
    future = run_visited_close_diagnosis(future_reference, tmp_path/'future')
    for name in ['teacher_models', 'teacher_support', 'model', 'training_support']:
        assert json.loads((first/f'{name}.json').read_text()) == json.loads((future/f'{name}.json').read_text())
    for name in ['teacher_training_membership', 'visitation_ledger', 'visited_training_used', 'visited_training_weights', 'visited_training_cost_ledger', 'full_training_contribution']:
        pd.testing.assert_frame_equal(pd.read_parquet(first/f'{name}.parquet'), pd.read_parquet(future/f'{name}.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(first/'predictions.parquet').filter(regex='score$|^selected_'),
        pd.read_parquet(future/'predictions.parquet').filter(regex='score$|^selected_'), check_exact=True)


@pytest.mark.parametrize('damage', ['score', 'model', 'protocol', 'snapshot', 'baseline_copy'])
def test_resigned_prior_damage_is_rejected_before_any_new_teacher_or_candidate(tmp_path, monkeypatch, damage):
    reference, _, _, _ = make_reference(tmp_path, monkeypatch)
    if damage == 'score':
        frame = pd.read_parquet(reference/'predictions.parquet')
        frame.loc[0, 'minute_score'] = 1-frame.loc[0, 'minute_score']
        frame.to_parquet(reference/'predictions.parquet', index=False)
    elif damage in ['model', 'protocol']:
        path = reference/('model.json' if damage == 'model' else 'manifest.json')
        value = json.loads(path.read_text())
        if damage == 'model':
            value['models'][0]['baseline'] += 1
        else:
            value['settings']['score_threshold'] = .6
        save_json(path, value)
    elif damage == 'snapshot':
        path = reference/'code_snapshot/minute_close_learning.py'
        path.write_text(path.read_text()+'\n# 합성 변조\n')
    else:
        path = reference/'baseline_source/model.json'
        value = json.loads(path.read_text())
        value['models'][0]['baseline'] += 1
        save_json(path, value)
        files = json.loads((reference/'baseline_files.json').read_text())
        files['model.json'] = sha256(path)
        save_json(reference/'baseline_files.json', files)
    seal(reference)
    calls = []

    def unexpected_fit(*_args):
        calls.append(True)
        raise RuntimeError('변조된 기준 뒤 새 학습 호출')

    monkeypatch.setattr('wonyotti_fr.visited_close_diagnostics.fit_visited_close', unexpected_fit)
    destination = tmp_path/'rejected'
    with pytest.raises((ValueError, AssertionError)):
        run_visited_close_diagnosis(reference, destination)
    assert calls == []
    run = next(destination.iterdir())
    assert (run/'failure.json').is_file()
    assert not (run/'teacher_models.json').exists() and not (run/'model.json').exists() and not (run/'summary.json').exists()
