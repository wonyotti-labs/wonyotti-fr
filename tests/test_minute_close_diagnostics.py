import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_close_context_diagnostics import synthetic_context
from test_close_flow_diagnostics import synthetic_flow
from test_first_close_diagnostics import financial_examples

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.close_economics import ECONOMIC_FEATURES
from wonyotti_fr.close_flow import FLOW_WINDOWS, FlowCloseModel
from wonyotti_fr.close_learning_inputs import CLOSE_SPLITS, close_learning_splits
from wonyotti_fr.close_utility import (
    cost_probability_metrics,
    first_cost_positions,
    policy_effect_metrics,
)
from wonyotti_fr.close_utility_diagnostics import fit_utility
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.continuation_inputs import PENDING_FEATURES, pending_values
from wonyotti_fr.first_close_diagnostics import first_close_metrics, paired_week_blocks
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.minute_close_diagnostics import run_minute_close_diagnosis
from wonyotti_fr.weekly_flow_diagnostics import FLOW_COMMON_FILES, validate_flow_reference


def seal(root):
    save_json(root/'files.json', {p.name: sha256(p) for p in root.iterdir() if p.is_file() and p.name != 'files.json'})


def synthetic_inputs(frame):
    result = frame.copy()
    result[ECONOMIC_FEATURES[0]] = .25
    result[ECONOMIC_FEATURES[1]] = result.favorable_move*10000
    result[PENDING_FEATURES] = pending_values(result.original_intent)
    return result


def extend(frame):
    return synthetic_flow(synthetic_context({'ledger': frame}, None), None)['ledger']


def fixture(tmp_path, monkeypatch):
    reference, parent, labels, legacy_labels = [tmp_path/name for name in ['reference', 'parent', 'labels', 'legacy-labels']]
    for path in [reference, parent, labels, legacy_labels]:
        path.mkdir()
    save_json(parent/'files.json', {})
    save_json(legacy_labels/'files.json', {'synthetic': 'closed-state-inputs-validated-separately'})
    save_json(labels/'manifest.json', {'settings': {'legacy_labels': str(legacy_labels), 'legacy_files_sha256': sha256(legacy_labels/'files.json')}})
    raw = financial_examples()
    raw['close_cash'] = raw.continue_cash+raw.close_advantage_pnl
    legacy = extend(synthetic_inputs(raw))
    minute = raw.loc[raw.index.repeat(5)].reset_index(drop=True)
    offset = np.tile(np.arange(-4, 1), len(raw))
    minute['decision_time'] += pd.to_timedelta(offset, unit='min')
    minute['decision_equity'] += offset*.01
    off_grid = offset != 0
    minute.loc[off_grid, 'close_advantage_bps'] = minute.loc[off_grid, 'close_advantage_pnl']/minute.loc[off_grid, 'decision_equity']*10000
    extras = []
    for i in range(2):
        part = minute.iloc[-3:].copy()
        part['position_entry_time'] = pd.Timestamp('2021-12-20', tz='UTC')+pd.Timedelta(days=i)
        part['decision_time'] = part.position_entry_time+pd.to_timedelta([1, 2, 3], unit='min')
        part['label_end'] = part.position_entry_time+pd.Timedelta(hours=2)
        part['continue_end'] = part.label_end
        extras.append(part)
    minute = pd.concat([minute, *extras], ignore_index=True)
    minute.to_parquet(labels/'opportunity_ledger.parquet', index=False)
    seal(labels)
    rows, assignments = close_learning_splits(legacy)
    weights = {name: frame[['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(frame)) for name, frame in rows.items()}
    model, support, costs = fit_utility(rows['training'], weights['training'], rows['diagnosis'], model_class=FlowCloseModel)
    frame = rows['diagnosis'].assign(sample_weight=weights['diagnosis'].sample_weight)
    score = model.probabilities(frame[model.features].to_numpy(dtype=float))[:, 0]
    for name in FLOW_COMMON_FILES:
        if name.endswith('.parquet'):
            pd.DataFrame({'synthetic_previous_artifact': [0]}).to_parquet(reference/name, index=False)
        elif name.endswith('.json'):
            save_json(reference/name, {})
        else:
            (reference/name).write_text('이전 합성 결과 보존\n')
    settings = {'reference': str(parent), 'reference_files_sha256': sha256(parent/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V71.md')),
        'periods': CLOSE_SPLITS, 'features': FlowCloseModel.features, 'settings': HISTOGRAM_SETTINGS,
        'new_model_count': 1, 'flow_windows': FLOW_WINDOWS, 'score_threshold': .5,
        'score_kind': 'cost_weighted_decision_score', 'cost_weight': 'original_position_weight_times_absolute_effect',
        'zero_effect_policy': 'zero_fit_contribution_preserved_in_evaluation',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True}
    save_json(reference/'manifest.json', {'settings': settings})
    save_json(reference/'summary.json', {**dict.fromkeys(['complete', 'all_previous_outputs_reproduced',
        'all_original_rows_and_evaluation_weights_preserved', 'original_cost_ledger_and_prior_preserved', 'zero_effect_rows_preserved'], True),
        'profitability_accepted': False})
    save_json(reference/'model.json', model.to_dict())
    save_json(reference/'training_support.json', support)
    save_json(reference/'decision.json', {'flow_admitted': False, 'checks': {f'prior_{i}': False for i in range(24)}, 'trading_returns_evaluated': False})
    for name in rows:
        rows[name].to_parquet(reference/f'{name}_used.parquet', index=False)
        weights[name].to_parquet(reference/f'{name}_weights.parquet', index=False)
    assignments.to_parquet(reference/'exclusion_ledger.parquet', index=False)
    costs.to_parquet(reference/'training_cost_ledger.parquet', index=False)
    keys = ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps', 'sample_weight']
    frame[keys].assign(flow_score=score).to_parquet(reference/'predictions.parquet', index=False)
    chosen = (score > .5) & frame.original_intent.ne('exit').to_numpy()
    positions = first_cost_positions(frame, score)
    positions.to_parquet(reference/'positions_flow.parquet', index=False)
    for name, value in [('metrics', policy_effect_metrics(frame, chosen)), ('first_metrics', first_close_metrics(positions)),
        ('probability_metrics', cost_probability_metrics(frame.close_advantage_bps, frame.sample_weight, score))]:
        save_json(reference/f'{name}.json', {'flow': value})
    for name, part, groups, metric in [
        ('breakdown', frame, [('direction', 'direction'), ('month', frame.decision_time.dt.strftime('%Y-%m'))],
            lambda group: policy_effect_metrics(group, chosen[group.index])),
        ('probability_breakdown', frame, [('direction', 'direction'), ('month', frame.decision_time.dt.strftime('%Y-%m'))],
            lambda group: cost_probability_metrics(group.close_advantage_bps, group.sample_weight, score[group.index])),
        ('first_breakdown', positions, [('direction', 'direction'), ('entry_month', positions.position_entry_time.dt.strftime('%Y-%m'))], first_close_metrics)]:
        entries = [{'kind': kind, 'group': str(key), 'model': 'flow', **metric(group)}
            for kind, grouper in groups for key, group in part.groupby(grouper)]
        save_json(reference/f'{name}.json', entries)
    for other in ['continuation', 'utility', 'context']:
        other_positions = first_cost_positions(frame, np.full(len(frame), .4))
        other_positions.to_parquet(reference/f'positions_{other}.parquet', index=False)
        result = paired_week_blocks({'flow': positions, other: other_positions}, model_names=['flow', other])
        prefix = '' if other == 'continuation' else other+'_'
        for name, data in zip(['blocks', 'block_draws', 'block_replicates'], result[:3], strict=True):
            data.to_parquet(reference/f'{prefix}{name}.parquet', index=False)
        save_json(reference/f'{prefix}block_intervals.json', result[3])
    seal(reference)
    market_proof = {'labels': str(legacy_labels), 'labels_files_sha256': sha256(legacy_labels/'files.json'), 'synthetic': True}
    source = {'market': market_proof, 'synthetic': True}
    # 원자·시세 연결은 별도 실제 엔진 검사로 검증하고 여기서는 전체 학습·대조 연결을 검사한다.
    monkeypatch.setattr('wonyotti_fr.minute_close_diagnostics.load_close_training',
        lambda path, **_k: (pd.read_parquet(path/'opportunity_ledger.parquet'), {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.minute_close_diagnostics.load_economic_inputs',
        lambda _p, data, **_k: (synthetic_inputs(data).drop(columns=PENDING_FEATURES), {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.minute_close_diagnostics.load_pending_inputs',
        lambda _p, data, **_k: (data.assign(**dict(zip(PENDING_FEATURES, pending_values(data.original_intent).T, strict=True))), {'synthetic': True}))
    monkeypatch.setattr('wonyotti_fr.minute_close_diagnostics.close_flow_source', lambda _p: (None, market_proof))
    monkeypatch.setattr('wonyotti_fr.minute_close_diagnostics.attach_minute_close_inputs',
        lambda data, _bars: (extend(data), data[['decision_time']].assign(confirmed_feature_end=data.decision_time.dt.floor('5min'))))
    for module in ['minute_close_diagnostics', 'minute_close_reference']:
        monkeypatch.setattr(f'wonyotti_fr.{module}.load_full_flow_ledger', lambda _p: (legacy.copy(), source))
    return labels, reference, minute, legacy


def test_complete_minute_pipeline_keeps_five_minute_reference_and_future_diagnosis_cannot_change_training(tmp_path, monkeypatch):
    labels, reference, minute, legacy = fixture(tmp_path, monkeypatch)
    validate_flow_reference(reference)
    original = json.loads((reference/'files.json').read_text())
    out = run_minute_close_diagnosis(labels, reference, tmp_path/'runs')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['same_position_population_and_first_equity']
    assert summary['new_models_fitted'] == 1 and summary['candidate'] == 'minute_1m' and len(summary['checks']) == 15
    assert summary['diagnosis_positions'] == 62 and not summary['profitability_accepted']
    pd.testing.assert_frame_equal(pd.read_parquet(out/'minute_ledger.parquet'), extend(synthetic_inputs(minute)), check_exact=True)
    for name, digest in original.items():
        assert sha256(reference/name) == sha256(out/'baseline_source'/name) == digest
    assert json.loads((out/'baseline_verification.json').read_text())['legacy_rows'] == len(legacy)
    altered = minute.copy()
    mask = altered.decision_time.ge(pd.Timestamp('2021-10-02', tz='UTC')) & altered.decision_time.dt.minute.mod(5).ne(0)
    altered.loc[mask, ['close_advantage_pnl', 'close_advantage_bps']] *= -100
    altered.loc[mask, 'close_cash'] = altered.loc[mask, 'continue_cash']+altered.loc[mask, 'close_advantage_pnl']
    altered.to_parquet(labels/'opportunity_ledger.parquet', index=False)
    seal(labels)
    other = run_minute_close_diagnosis(labels, reference, tmp_path/'future-changed')
    assert json.loads((out/'model.json').read_text()) == json.loads((other/'model.json').read_text())
    pd.testing.assert_frame_equal(pd.read_parquet(out/'training_cost_ledger.parquet'), pd.read_parquet(other/'training_cost_ledger.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet').filter(regex='score$|^selected_'),
        pd.read_parquet(other/'predictions.parquet').filter(regex='score$|^selected_'), check_exact=True)


@pytest.mark.parametrize('damage', ['score', 'model', 'protocol'])
def test_resigned_prior_reference_damage_is_rejected_before_new_candidate(tmp_path, monkeypatch, damage):
    labels, reference, _, _ = fixture(tmp_path, monkeypatch)
    if damage == 'score':
        frame = pd.read_parquet(reference/'predictions.parquet')
        frame.loc[0, 'flow_score'] = 1-frame.loc[0, 'flow_score']
        frame.to_parquet(reference/'predictions.parquet', index=False)
    else:
        path = reference/('model.json' if damage == 'model' else 'manifest.json')
        value = json.loads(path.read_text())
        if damage == 'model':
            value['models'][0]['baseline'] += 1
        else:
            value['settings']['score_threshold'] = .6
        save_json(path, value)
    seal(reference)
    output = tmp_path/'rejected'
    with pytest.raises((ValueError, AssertionError)):
        run_minute_close_diagnosis(labels, reference, output)
    run = next(output.iterdir())
    assert (run/'failure.json').is_file() and not (run/'model.json').exists() and not (run/'summary.json').exists()
