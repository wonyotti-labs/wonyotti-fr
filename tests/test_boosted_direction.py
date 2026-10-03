import copy
import json

import numpy as np
import pandas as pd
import pytest
from test_direction_diagnostics import source_fixture
from test_minute_inventory_research import ready_bars
from test_new_position import short_bars
from test_position_direction import selection as direct_selection

from wonyotti_fr.boosted_direction import BOOST_FILES, copy_direction_parent, validate_admission
from wonyotti_fr.boosted_direction_research import run_boosted_direction_selection
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.direction_diagnostics import direction_admission, run_direction_diagnostics
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.new_position import new_position_targets
from wonyotti_fr.position_direction import DIRECTION_PERIOD, position_direction_training
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation


def admission_fixture():
    metrics = {k: {'rows': 120, 'log_loss': v, 'actual_buy_fraction': .5}
               for k, v in [('logistic', .7), ('boosted', .6), ('constant', .69)]}
    decision = direction_admission(metrics)
    return {'metrics': metrics, 'decision': decision,
        'summary': {'complete': True, 'training_rows': 1200, 'diagnosis_rows': 120,
                    'episode_intersection': 0, **decision},
        'training_support': {k: {'rows': 1200, 'positive': 600, 'negative': 600} for k in ['logistic', 'boosted']},
        'settings': {'training_period': ['2018-01-01', '2021-01-01'],
            'diagnosis_period': ['2021-01-01', '2022-01-01'], 'model_count': 2,
            'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}}


def selection(tmp_path):
    parent = tmp_path / 'direct'
    old = direct_selection(parent, tmp_path / 'prior', tmp_path / 'position', tmp_path / 'recent',
                           tmp_path / 'minute', tmp_path / 'previous')
    root = tmp_path / 'boosted'
    root.mkdir()
    copy_direction_parent(parent, root)
    model = {'format': 'expansion_binary_v1', 'kind': 'boosted', 'features': MARKET_FEATURES,
        'learning_rate': .05, 'trees': [{'left': [-1], 'right': [-1], 'feature': [-2],
                                      'threshold': [-2.], 'value': [1.]} for _ in range(64)]}
    save_json(root / 'boosted_direction.json', model)
    save_json(root / 'boosted_direction_training.json', {'training_period': DIRECTION_PERIOD,
        'direction_threshold': .65, 'prior_offset_applied': False,
        'model': {'rows': 1200, 'positive': 600, 'negative': 600}})
    save_json(root / 'direction_admission.json', admission_fixture())
    frozen = {**old, 'protocol': 'boosted_direction_v28',
        'direction_selection_sha256': sha256(root / 'direction_selection.json'),
        'boost_files_sha256': {n: sha256(root / n) for n in BOOST_FILES}}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return root, parent, frozen


@pytest.mark.parametrize('damage', ['gate', 'support', 'overlap', 'period', 'returns'])
def test_failed_or_changed_temporal_admission_is_rejected(damage):
    evidence = copy.deepcopy(admission_fixture())
    if damage == 'gate':
        evidence['metrics']['boosted']['log_loss'] = .8
    elif damage == 'support':
        evidence['training_support']['boosted']['positive'] = 19
    elif damage == 'overlap':
        evidence['summary']['episode_intersection'] = 1
    elif damage == 'period':
        evidence['settings']['training_period'][1] = '2022-01-01'
    else:
        evidence['settings']['trading_returns_evaluated'] = True
    with pytest.raises(ValueError):
        validate_admission(evidence)


@pytest.mark.parametrize('damage', ['parent', 'model', 'risk', 'trees', 'period'])
def test_boosted_model_chain_rejects_tampering(tmp_path, damage):
    root, _, frozen = selection(tmp_path)
    if damage == 'parent':
        (root / 'direction_selection.json').write_text('{}')
    elif damage == 'model':
        (root / 'boosted_direction.json').write_text('{}')
    elif damage == 'risk':
        frozen['risk']['max_adds'] = 0
    else:
        name = 'boosted_direction.json' if damage == 'trees' else 'boosted_direction_training.json'
        data = json.loads((root / name).read_text())
        if damage == 'trees':
            data['trees'].pop()
        else:
            data['training_period'][0] = '2020-01-01'
        save_json(root / name, data)
        frozen['boost_files_sha256'][name] = sha256(root / name)
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_boosted_direction_actual_process_recovery(tmp_path, kind):
    root, _, _ = selection(tmp_path)
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73


def test_eleven_conditions_preserve_v26_and_v27(tmp_path, monkeypatch):
    root, parent, frozen = selection(tmp_path)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (short_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    strategies = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(strategies) == 11 and {'previous_v26', 'previous_v27'} <= strategies and 'previous_v25' not in strategies
    _, current = load_selection(out)
    for name, path in [('previous_v26', tmp_path / 'prior'), ('previous_v27', parent)]:
        _, previous = load_selection(path)
        assert previous.base.activity.data == current.base.activity.data
        assert previous.manager.to_dict() == current.manager.to_dict()
        backtest(short_bars(), previous, EngineConfig(**frozen['risk']), tmp_path / name)
        for filename in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'BTCUSDT' / name / filename),
                                          pd.read_parquet(tmp_path / name / filename), check_exact=True)
        assert json.loads((out / 'BTCUSDT' / name / 'final_state.json').read_text()) == json.loads((tmp_path / name / 'final_state.json').read_text())


def test_diagnosis_then_full_boosted_fit_preserves_training_and_management(tmp_path, monkeypatch):
    _, parent, frozen = selection(tmp_path)
    _, original = load_selection(parent)
    backtest(short_bars(), original, EngineConfig(**frozen['risk']), parent / 'candidate-00')
    data, actions, episodes = source_fixture()
    data.loc[5, 'usable'] = False
    data[MARKET_FEATURES[0]] = np.resize([-2., -.5, 2., .5], len(data))
    data[MARKET_FEATURES[1]] = 0.
    targets, _ = new_position_targets(data, actions)
    train, _ = position_direction_training(targets, episodes)
    train.to_parquet(parent / 'direction_training_used.parquet', index=False)
    for module in ['direction_diagnostics', 'boosted_direction_research']:
        monkeypatch.setattr(f'wonyotti_fr.{module}.source_inputs',
            lambda *_: ({'episodes': episodes, 'actions': actions}, {'synthetic': True}))
        monkeypatch.setattr(f'wonyotti_fr.{module}.make_expansion_data', lambda _: data)
    diagnosis = run_direction_diagnostics(tmp_path, tmp_path, tmp_path, tmp_path / 'diagnoses')
    assert json.loads((diagnosis / 'decision.json').read_text())['boosted_admitted']
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.boosted_direction_research.prepare_minute_period', lambda *_, **__: (short_bars(), {}))
    out = run_boosted_direction_selection(parent, diagnosis, tmp_path, tmp_path, tmp_path,
                                         tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    _, current = load_selection(out)
    assert current.base.direction.data['kind'] == 'boosted'
    assert current.base.activity.data == original.base.activity.data
    assert current.size_model.to_dict() == original.size_model.to_dict()
    pd.testing.assert_frame_equal(pd.read_parquet(out / 'direction_training_used.parquet'), train.reset_index(drop=True), check_exact=True)
    assert json.loads((out / 'baseline_parity.json').read_text())['full_outputs_and_state_exact']
