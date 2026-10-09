import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_minute_close_diagnostics import seal
from test_visited_close_diagnostics import make_reference

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.minute_close_diagnostics import run_minute_close_diagnosis
from wonyotti_fr.retained_weight_close_diagnostics import (
    RETAINED_INPUT_FILES,
    run_retained_weight_close_diagnosis,
)
from wonyotti_fr.retained_weight_close_reference import VISITED_DIAGNOSIS_FILES
from wonyotti_fr.visited_close_diagnostics import run_visited_close_diagnosis
from wonyotti_fr.visited_close_evaluation import VISITED_POLICIES


def visited_reference(tmp_path, monkeypatch):
    minute, labels, legacy, frame = make_reference(tmp_path, monkeypatch)
    reference = run_visited_close_diagnosis(minute, tmp_path/'visited-runs')
    assert set(json.loads((reference/'files.json').read_text())) == VISITED_DIAGNOSIS_FILES
    return reference, labels, legacy, frame


def test_complete_retained_pipeline_preserves_teachers_prefix_old_results_and_full_population(tmp_path, monkeypatch):
    reference, _, _, _ = visited_reference(tmp_path, monkeypatch)
    files = json.loads((reference/'files.json').read_text())
    out = run_retained_weight_close_diagnosis(reference, tmp_path/'retained-runs')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['all_previous_outputs_reproduced'] and summary['same_teacher_scores_and_visitation_prefix']
    assert summary['original_full_training_and_diagnosis_preserved'] and summary['same_position_population_and_first_equity']
    assert summary['new_models_fitted'] == 1 and summary['candidate'] == 'retained_1m'
    assert len(summary['checks']) == 24 and summary['profitability_accepted'] is False
    assert summary['diagnosis_positions'] == 62 and summary['retained_training_rows'] < summary['original_training_rows']
    for name, value in files.items():
        assert sha256(reference/name) == value == sha256(out/'baseline_source'/name)
    for name in RETAINED_INPUT_FILES:
        assert sha256(reference/name) == sha256(out/name)
    parity = json.loads((out/'reference_parity.json').read_text())
    assert parity['original_decision'] == json.loads((reference/'decision.json').read_text())
    assert parity['all_previous_outputs_exact'] and len(parity['previous_reproductions']) == 2
    for item in parity['previous_reproductions'].values():
        assert sha256(Path(item['path'])/'files.json') == item['files_sha256']
    old = pd.read_parquet(reference/'predictions.parquet')
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet')[old.columns], old, check_exact=True)
    for name in VISITED_POLICIES:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'positions_{name}.parquet'), pd.read_parquet(reference/f'positions_{name}.parquet'), check_exact=True)
    for name in ['metrics', 'first_metrics', 'probability_metrics']:
        before, after = [json.loads((folder/f'{name}.json').read_text()) for folder in [reference, out]]
        assert {key: after[key] for key in before} == before
    old_used = pd.read_parquet(reference/'visited_training_used.parquet')
    pd.testing.assert_frame_equal(old_used, pd.read_parquet(out/'retained_training_used.parquet'), check_exact=True)
    visit = pd.read_parquet(out/'visitation_ledger.parquet')
    original = pd.read_parquet(out/'training_weights.parquet').sample_weight.to_numpy()[visit.visited_prefix]
    actual = pd.read_parquet(out/'retained_training_weights.parquet').sample_weight.to_numpy()
    np.testing.assert_array_equal(actual, original/original.mean())
    ledger = pd.read_parquet(out/'full_training_contribution.parquet')
    assert ledger.loc[~visit.visited_prefix, 'reason'].eq('after_first_teacher_close').all()
    assert ledger.loc[~visit.visited_prefix, ['retained_original_weight', 'cost_weight', 'fit_weight']].eq(0).all().all()


def test_future_diagnosis_changes_keep_fixed_prefix_and_retained_model_unchanged(tmp_path, monkeypatch):
    reference, labels, legacy, minute = visited_reference(tmp_path, monkeypatch)
    first = run_retained_weight_close_diagnosis(reference, tmp_path/'original')
    changed = minute.copy()
    selected = changed.decision_time.ge(pd.Timestamp('2021-10-02', tz='UTC')) & changed.decision_time.dt.minute.mod(5).ne(0)
    changed.loc[selected, ['close_advantage_bps', 'close_advantage_pnl']] *= -100
    changed.loc[selected, 'close_cash'] = changed.loc[selected, 'continue_cash']+changed.loc[selected, 'close_advantage_pnl']
    changed.to_parquet(labels/'opportunity_ledger.parquet', index=False)
    seal(labels)
    future_minute = run_minute_close_diagnosis(labels, legacy, tmp_path/'future-minute')
    future_visited = run_visited_close_diagnosis(future_minute, tmp_path/'future-visited')
    future = run_retained_weight_close_diagnosis(future_visited, tmp_path/'future')
    for name in ['teacher_models', 'teacher_support', 'model', 'training_support']:
        assert json.loads((first/f'{name}.json').read_text()) == json.loads((future/f'{name}.json').read_text())
    for name in ['teacher_training_membership', 'visitation_ledger', 'retained_training_used', 'retained_training_weights',
        'retained_training_cost_ledger', 'full_training_contribution']:
        pd.testing.assert_frame_equal(pd.read_parquet(first/f'{name}.parquet'), pd.read_parquet(future/f'{name}.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(first/'predictions.parquet').filter(regex='score$|^selected_'),
        pd.read_parquet(future/'predictions.parquet').filter(regex='score$|^selected_'), check_exact=True)


@pytest.mark.parametrize('damage', ['score', 'model', 'prefix', 'original_weight', 'snapshot', 'nested_reproduction'])
def test_resigned_prior_or_nested_damage_is_rejected_before_new_fit(tmp_path, monkeypatch, damage):
    reference, _, _, _ = visited_reference(tmp_path, monkeypatch)
    if damage == 'score':
        path = reference/'predictions.parquet'
        frame = pd.read_parquet(path)
        frame.loc[0, 'visited_score'] = 1-frame.loc[0, 'visited_score']
        frame.to_parquet(path, index=False)
    elif damage == 'model':
        path = reference/'model.json'
        value = json.loads(path.read_text())
        value['models'][0]['baseline'] += 1
        save_json(path, value)
    elif damage in ['prefix', 'original_weight']:
        path = reference/('visitation_ledger.parquet' if damage == 'prefix' else 'training_weights.parquet')
        value = pd.read_parquet(path)
        if damage == 'prefix':
            value.loc[0, 'visited_prefix'] = not value.visited_prefix.iloc[0]
        else:
            value.loc[0, 'sample_weight'] *= 2
        value.to_parquet(path, index=False)
    elif damage == 'snapshot':
        path = reference/'code_snapshot/visited_close.py'
        path.write_text(path.read_text()+'\n# 합성 변조\n')
    else:
        proof = json.loads((reference/'reference_parity.json').read_text())
        nested = Path(proof['reproduction'])
        copied = tmp_path/'modified-nested'
        shutil.copytree(nested, copied)
        value = json.loads((copied/'model.json').read_text())
        value['models'][0]['baseline'] += 1
        save_json(copied/'model.json', value)
        seal(copied)
        proof['reproduction'], proof['reproduction_files_sha256'] = str(copied), sha256(copied/'files.json')
        save_json(reference/'reference_parity.json', proof)
    seal(reference)
    calls = []

    def forbidden(*_args):
        calls.append(True)
        raise RuntimeError('변조된 이전 결과 뒤 새 학습 호출')

    monkeypatch.setattr('wonyotti_fr.retained_weight_close_diagnostics.fit_retained_weight_close', forbidden)
    destination = tmp_path/'rejected'
    with pytest.raises((ValueError, AssertionError)):
        run_retained_weight_close_diagnosis(reference, destination)
    assert calls == []
    out = next(destination.iterdir())
    assert (out/'failure.json').is_file() and not (out/'model.json').exists() and not (out/'summary.json').exists()
