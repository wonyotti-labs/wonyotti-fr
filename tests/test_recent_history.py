import copy
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from test_calibration_diagnostics import source
from test_exit_state import fixture as exit_fixture
from test_first_state import fixed_budget  # noqa: F401
from test_history_calibration import refresh_files, setup
from test_history_state import selection as history_selection
from test_minute_inventory_research import ready_bars
from threadpoolctl import threadpool_limits

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.engine import EngineConfig
from wonyotti_fr.engine_stress import verify_stress
from wonyotti_fr.event_backtest import backtest, iter_events
from wonyotti_fr.event_research import load_selection
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.history_calibration_diagnostics import run_history_calibration_diagnostics
from wonyotti_fr.minute_management import ACTIONS, purged_window
from wonyotti_fr.order_history import HISTORY_FEATURES
from wonyotti_fr.period_guard import guard_replay_period
from wonyotti_fr.pullback_evaluation import run_pullback_evaluation
from wonyotti_fr.recent_history import (
    RECENT_HISTORY_FILES,
    RECENT_PERIODS,
    RecentHistoryPolicy,
    copy_exit_parent,
    fit_recent_history,
    load_recent_history_rows,
    validate_recent_history_rows,
)
from wonyotti_fr.recent_history_research import run_recent_history_selection


def rows():
    frame = source()
    frame[HISTORY_FEATURES] = 0.
    frame['past_reduce_exists'] = np.arange(len(frame)) % 2
    frame['past_increase_exists'] = np.arange(len(frame)) % 2
    return {n: purged_window(frame, *period).reset_index(drop=True) for n, period in RECENT_PERIODS.items()}


@lru_cache(maxsize=1)
def trained():
    return fit_recent_history(rows())


def fixture(tmp_path, monkeypatch):
    def with_evidence(path):
        root, prior, frozen = history_selection(path)
        evidence = json.loads((root / 'history_admission.json').read_text())
        evidence['files_sha256'] = 'a'*64
        evidence['settings']['history_files_sha256'] = 'b'*64
        save_json(root / 'history_admission.json', evidence)
        frozen['history_files_sha256']['history_admission.json'] = sha256(root / 'history_admission.json')
        save_json(root / 'frozen_selection.json', frozen)
        save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
        return root, prior, frozen
    monkeypatch.setattr('test_exit_state.history_selection', with_evidence)
    parent, ablation, _, _, old = exit_fixture(tmp_path, monkeypatch)
    root = tmp_path / 'recent-history'
    root.mkdir()
    copy_exit_parent(parent, root)
    model, offset, choices, support = copy.deepcopy(trained())
    sources = {'diagnosis_files_sha256': 'a'*64, 'history_files_sha256': 'b'*64,
               'training_sha256': 'c'*64, 'calibration_sha256': 'd'*64}
    support.update(sources=sources, reference_sha256=sha256(parent / 'frozen_selection.json'),
        periods=RECENT_PERIODS, protocol_sha256=sha256(Path('docs/EXPERIMENT_V50.md')),
        new_model_family=False, late_2021_is_calibration=True, profitability_accepted=False)
    for name, value in [('manager', model.to_dict()), ('offset', offset.to_dict()), ('thresholds', choices), ('training', support)]:
        save_json(root / f'recent_history_{name}.json', value)
    frozen = {**old, 'protocol': 'recent_history_v50', 'exit_selection_sha256': sha256(root / 'exit_selection.json'),
        'recent_history_files_sha256': {n: sha256(root / n) for n in RECENT_HISTORY_FILES}}
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    return root, parent, ablation, sources, frozen


def test_latest_fixed_fit_independent_predictions_offset_and_thresholds():
    data = rows()
    model, offset, choices, support = trained()
    train, cal = data['training'], data['calibration']
    x, cx = (p[model.features].to_numpy() for p in [train, cal])
    cp = offset.predict(model.probabilities(cx))
    for i, action in enumerate(ACTIONS):
        with threadpool_limits(limits=1):
            expected = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, train['y_'+action])
            raw = expected.predict_proba(cx)[:, 1]
        np.testing.assert_allclose(model.probabilities(cx)[:, i], raw, atol=1e-12, rtol=0)
        eps = np.finfo(float).eps
        q = np.clip(raw, eps, 1-eps)
        z = np.log(q)-np.log1p(-q)
        target = cal['y_'+action].mean()
        expected_offset = brentq(lambda v, z=z, target=target: expit(z+v).mean()-target, -64, 64, xtol=1e-14)
        assert abs(offset.offsets[i]-expected_offset) < 1e-12
        for phase in ['all', *(['first'] if action != 'exit' else [])]:
            mask = np.ones(len(cal), bool) if phase == 'all' else cal['past_'+action+'_exists'].eq(0).to_numpy()
            p, y = cp[mask, i], cal.loc[mask, 'y_'+action].to_numpy()
            beta = choices['betas'][action]
            options = []
            for t in np.unique(p):
                predicted = p >= t
                if predicted.sum() < 20:
                    continue
                tp, fp, fn = np.sum(predicted & (y == 1)), np.sum(predicted & (y == 0)), np.sum((~predicted) & (y == 1))
                f = (1+beta*beta)*tp/((1+beta*beta)*tp+beta*beta*fn+fp)
                options.append((f, t))
            maximum = max(f for f, _ in options)
            threshold = max(t for f, t in options if abs(f-maximum) < 1e-15)
            key = 'thresholds' if phase == 'all' else 'first_thresholds'
            assert choices[key][action] == threshold
    assert support['offset']['max_mean_residual'] < 1e-12
    assert support['splits']['calibration']['period'] == ['2021-07-01', '2022-01-01']


@pytest.mark.parametrize('damage', ['future', 'overlap', 'few_first', 'flags', 'nan', 'id', 'extra_split'])
def test_latest_fit_rejects_future_data_and_invalid_support(damage):
    data = rows()
    if damage == 'future':
        data['calibration'].loc[0, 'end'] = pd.Timestamp('2022-01-02', tz='UTC')
    elif damage == 'overlap':
        data['calibration'].loc[0, 'episode_id'] = data['training'].episode_id.iloc[0]
    elif damage == 'few_first':
        mask = data['calibration'].past_reduce_exists.eq(0)
        data['calibration'].loc[mask, 'y_reduce'] = 0
    elif damage == 'flags':
        data['calibration']['past_increase_exists'] = .5
    elif damage == 'nan':
        data['training'].loc[0, 'ret_5m'] = np.nan
    elif damage == 'id':
        data['training'].loc[0, 'episode_id'] = 0
    else:
        data['future'] = data['calibration'].copy()
    with pytest.raises((ValueError, AssertionError)):
        validate_recent_history_rows(data)


def test_existing_source_links_and_both_original_splits_are_exact(tmp_path, monkeypatch):
    history, reference, frame, _ = setup(tmp_path, monkeypatch)
    source_run = run_history_calibration_diagnostics(history, reference, tmp_path / 'sources')
    features = pd.read_parquet(history / 'history_features.parquet')
    full = frame.copy()
    full[HISTORY_FEATURES] = features[HISTORY_FEATURES].to_numpy()
    for name, period in RECENT_PERIODS.items():
        stored = 'training' if name == 'training' else 'diagnosis'
        purged_window(full, *period).reset_index(drop=True).to_parquet(history / f'{stored}_used.parquet', index=False)
    refresh_files(history)
    manifest = json.loads((source_run / 'manifest.json').read_text())
    manifest['settings']['history_files_sha256'] = sha256(history / 'files.json')
    save_json(source_run / 'manifest.json', manifest)
    refresh_files(source_run)
    parent = tmp_path / 'parent'
    parent.mkdir()
    save_json(parent / 'history_admission.json', {'files_sha256': sha256(source_run / 'files.json')})
    for name, stored in [('models', 'history_manager'), ('offsets', 'history_offset')]:
        save_json(parent / f'{stored}.json', json.loads((source_run / f'{name}.json').read_text())['histogram'])
    data, fingerprints = load_recent_history_rows(parent, source_run)
    for name, period in RECENT_PERIODS.items():
        pd.testing.assert_frame_equal(data[name], purged_window(full, *period).reset_index(drop=True), check_exact=True)
    assert fingerprints['history_files_sha256'] == sha256(history / 'files.json')
    (history / 'diagnosis_used.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError):
        load_recent_history_rows(parent, source_run)


@pytest.mark.parametrize('damage', ['parent', 'risk', 'old_model', 'model', 'period', 'plan', 'support', 'application', 'source'])
def test_latest_policy_rejects_unplanned_changes(tmp_path, monkeypatch, damage):
    root, _, _, _, frozen = fixture(tmp_path, monkeypatch)
    if damage == 'parent':
        (root / 'exit_selection.json').write_text('{}')
    elif damage == 'risk':
        frozen['risk']['entry_fraction'] *= 2
    elif damage == 'old_model':
        (root / 'history_manager.json').write_text('{}')
    elif damage == 'model':
        (root / 'recent_history_manager.json').write_text('{}')
    else:
        name = 'recent_history_thresholds.json' if damage in ['support', 'application'] else 'recent_history_training.json'
        value = json.loads((root / name).read_text())
        if damage == 'period':
            value['periods']['training'][1] = '2022-01-01'
        elif damage == 'plan':
            value['protocol_sha256'] = '0'*64
        elif damage == 'source':
            value['sources']['history_files_sha256'] = '0'*64
        elif damage == 'support':
            value['first_support']['increase']['actual_positive'] = 19
        else:
            value['application']['repeat'] = 1.
        save_json(root / name, value)
        frozen['recent_history_files_sha256'][name] = sha256(root / name)
    save_json(root / 'frozen_selection.json', frozen)
    save_json(root / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(root / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(root)


@pytest.mark.parametrize('kind', ['waiting', 'position'])
def test_latest_actual_process_exit_and_period_guard(tmp_path, monkeypatch, kind):
    root, _, _, _, frozen = fixture(tmp_path, monkeypatch)
    result = verify_stress(list(iter_events(ready_bars())), root, tmp_path / 'stress', {}, kind, True)
    assert result['all_passed'] and result['child_process_exit_code'] == 73
    with pytest.raises(ValueError):
        guard_replay_period(root, frozen, '2020-01-01', '2020-02-01')


def test_latest_eleven_conditions_preserve_original_controls(tmp_path, monkeypatch):
    root, parent, ablation, _, _ = fixture(tmp_path, monkeypatch)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    monkeypatch.setattr('wonyotti_fr.pullback_evaluation.block_interval', lambda _: {'synthetic': True})
    out = run_pullback_evaluation(root, tmp_path, tmp_path, tmp_path / 'runs', 'seen_2026', ['BTCUSDT'])
    names = {r['strategy'] for r in json.loads((out / 'results.json').read_text())}
    assert len(names) == 11 and {'previous_v48', 'previous_v46', 'unfiltered_v14'} <= names
    assert 'previous_v40' not in names and isinstance(load_selection(out)[1], RecentHistoryPolicy)
    for label, previous in [('previous_v48', parent), ('previous_v46', ablation)]:
        frozen, policy = load_selection(previous)
        expected = tmp_path / label
        backtest(ready_bars(), policy, EngineConfig(**frozen['risk']), expected)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(expected / name), pd.read_parquet(out / f'BTCUSDT/{label}' / name), check_exact=True)
        assert json.loads((expected / 'final_state.json').read_text()) == json.loads((out / f'BTCUSDT/{label}/final_state.json').read_text())
        assert json.loads((out / f'BTCUSDT/{label}/config.json').read_text()) == frozen['risk']


def test_full_latest_selection_refits_only_manager_and_keeps_parent_exact(tmp_path, monkeypatch):
    _, parent, _, sources, old = fixture(tmp_path, monkeypatch)
    _, original = load_selection(parent)
    backtest(ready_bars(), original, EngineConfig(**old['risk']), parent / 'candidate-00')
    diagnosis = tmp_path / 'family'
    diagnosis.mkdir()
    (diagnosis / 'files.json').write_text('{}')
    for n in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / n).write_text('{}')
    monkeypatch.setattr('wonyotti_fr.recent_history.load_recent_history_rows', lambda *_: (rows(), sources))
    monkeypatch.setattr('wonyotti_fr.recent_history_research.prepare_minute_period', lambda *_, **__: (ready_bars(), {}))
    out = run_recent_history_selection(parent, diagnosis, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs')
    frozen, bot = load_selection(out)
    assert frozen['risk'] == old['risk'] and bot.base.activity_threshold == original.base.activity_threshold
    assert bot.size_model.to_dict() == original.size_model.to_dict()
    assert bot.base.direction.to_dict() == original.base.direction.to_dict()
    assert bot.base.activity.to_dict() == original.base.activity.to_dict()
    for n in ['history_manager.json', 'history_offset.json', 'history_thresholds.json', 'first_policy_thresholds.json']:
        assert (out / n).read_bytes() == (parent / n).read_bytes()
    assert bot.manager.model.to_dict() != original.manager.model.to_dict()
    assert json.loads((out / 'baseline_parity.json').read_text())['full_outputs_and_state_exact']
    assert json.loads((out / 'manifest.json').read_text())['settings']['new_models_fitted'] is True
    for flags in [(0., 0.), (0., 1.), (1., 0.), (1., 1.)]:
        state = {'_fill_features': np.array([flags[0], 0., 0., flags[1], 0., 0.])}
        assert bot.action_threshold('exit', state) == bot.thresholds['exit']
        for i, action in [(0, 'increase'), (1, 'reduce')]:
            assert bot.action_threshold(action, state) == (bot.thresholds[action]*1.5 if flags[i] else bot.first_thresholds[action])
    def changed_future(*args, **kwargs):
        bars = ready_bars()
        if args[3] == '2022-01-01':
            bars[['open', 'high', 'low', 'close']] *= 2.
        return bars, {}
    monkeypatch.setattr('wonyotti_fr.recent_history_research.prepare_minute_period', changed_future)
    changed = run_recent_history_selection(parent, diagnosis, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'future')
    for name in RECENT_HISTORY_FILES:
        assert (out / name).read_bytes() == (changed / name).read_bytes()
