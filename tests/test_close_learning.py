import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from test_engine import config
from test_first_state import fixed_budget  # noqa: F401
from test_minute_inventory_research import ready_bars
from test_net_exit_state import bot
from threadpoolctl import threadpool_limits

from wonyotti_fr.addition_effect import position_weights
from wonyotti_fr.close_effect import CLOSE_FEATURES, run_close_effect_labels
from wonyotti_fr.close_learning import (
    CloseRegressionModel,
    CloseRidgeModel,
    close_admission,
    close_metrics,
    run_close_learning_diagnosis,
)
from wonyotti_fr.close_learning_inputs import close_learning_splits, load_close_training
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.entry_regression import REGRESSION_SETTINGS
from wonyotti_fr.exit_move_state import ExitMovePolicy
from wonyotti_fr.streaming_backtest import streaming_backtest


def examples():
    rng = np.random.default_rng(60)
    rows = []
    for start, positions, step in [('2021-01-01', 110, 2), ('2021-10-02', 40, 1)]:
        for i in range(positions):
            entry = pd.Timestamp(start, tz='UTC')+pd.Timedelta(days=i*step)
            for j in range(10+i%3):
                values = dict(zip(CLOSE_FEATURES, rng.normal(size=len(CLOSE_FEATURES)), strict=True))
                values['direction'] = 1 if i%2 else -1
                rows.append({**values, 'decision_time': entry+pd.Timedelta(minutes=5*(j+1)),
                    'position_entry_time': entry, 'label_end': entry+pd.Timedelta(hours=2),
                    'label_status': 'closed', 'original_intent': 'exit' if j == 10 else 'hold',
                    'close_advantage_bps': 0. if j == 10 else 10.*values[CLOSE_FEATURES[0]]+(-5 if i%2 else 5)})
    return pd.DataFrame(rows)


def test_equal_position_weights_and_disjoint_time_split_preserve_exclusions():
    frame = examples()
    extra = frame.iloc[:3].copy()
    extra['position_entry_time'] = pd.to_datetime(['2021-09-29T00:00Z', '2021-10-01T00:00Z', '2021-12-30T00:00Z'])
    extra['decision_time'] = pd.to_datetime(['2021-09-29T00:05Z', '2021-10-02T00:01Z', '2021-12-30T00:05Z'])
    extra['label_end'] = pd.to_datetime(['2021-09-30T00:00Z', '2021-10-02T01:00Z', '2021-12-31T00:00Z'])
    extra.loc[extra.index[-1], 'label_status'] = 'right_censored'
    frame = pd.concat([frame, extra]).sort_values('decision_time').reset_index(drop=True)
    rows, assigned = close_learning_splits(frame)
    assert len(assigned) == len(frame) and (assigned.split == 'excluded_boundary').sum() == 2
    assert (assigned.split == 'excluded_not_closed').sum() == 1
    assert not set(rows['training'].position_entry_time)&set(rows['diagnosis'].position_entry_time)
    for part in rows.values():
        weight = position_weights(part)
        by_position = part.assign(weight=weight).groupby('position_entry_time').weight.sum()
        np.testing.assert_allclose(by_position, len(part)/len(by_position), rtol=0, atol=1e-12)
        assert weight.mean() == pytest.approx(1.) and part.close_advantage_bps.lt(0).any()


@pytest.mark.parametrize('damage', ['duplicate', 'missing_end', 'reversed', 'few_positions', 'one_direction', 'nan'])
def test_split_rejects_bad_times_or_unsupported_inputs(damage):
    frame = examples()
    if damage == 'duplicate':
        frame.loc[1, 'decision_time'] = frame.decision_time.iloc[0]
    elif damage == 'missing_end':
        frame.loc[0, 'label_end'] = pd.NaT
    elif damage == 'reversed':
        frame.loc[0, 'label_end'] = frame.decision_time.iloc[0]
    elif damage == 'few_positions':
        frame.loc[frame.decision_time.dt.month.ge(10), 'position_entry_time'] = pd.Timestamp('2021-10-02', tz='UTC')
    elif damage == 'one_direction':
        frame['direction'] = 1
    else:
        frame.loc[0, CLOSE_FEATURES[0]] = np.nan
    with pytest.raises(ValueError):
        close_learning_splits(frame)


def test_weighted_models_match_normal_equations_and_independent_boosting():
    frame = examples().iloc[:500]
    x, y, w = frame[CLOSE_FEATURES].to_numpy(), frame.close_advantage_bps.to_numpy(), position_weights(frame)
    model, _ = CloseRidgeModel.fit(x, y, w)
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    scale[scale == 0] = 1
    z, target_mean = (x-mean)/scale, np.average(y, weights=w)
    expected = np.linalg.solve(z.T@(z*w[:, None])+100*np.eye(len(CLOSE_FEATURES)), z.T@((y-target_mean)*w))
    np.testing.assert_allclose(model.data['mean'], mean, atol=1e-12)
    np.testing.assert_allclose(model.data['scale'], scale, atol=1e-12)
    np.testing.assert_allclose(model.predict(x), z@expected+target_mean, rtol=0, atol=1e-10)
    boosted, _ = CloseRegressionModel.fit(x, y, w, x[:50])
    with threadpool_limits(limits=1):
        library = HistGradientBoostingRegressor(**REGRESSION_SETTINGS).fit(x, y, sample_weight=w)
        np.testing.assert_allclose(boosted.predict(x), library.predict(x), rtol=0, atol=1e-10)
    np.testing.assert_array_equal(CloseRidgeModel.from_dict(model.to_dict()).predict(x), model.predict(x))
    altered, _ = CloseRegressionModel.fit(x, y, w, x[:50]*-100)
    assert altered.to_dict() == boosted.to_dict()
    for cls, data in [(CloseRidgeModel, model.to_dict()), (CloseRegressionModel, boosted.to_dict())]:
        data['features'] = data['features'][:-1]
        with pytest.raises(ValueError):
            cls.from_dict(data)
    for bad in [np.zeros(len(w)), w*2, w[:-1]]:
        with pytest.raises(ValueError):
            CloseRidgeModel.fit(x, y, bad)


def test_original_exit_retained_in_error_but_not_selected_and_all_seven_gates_required():
    frame = examples().iloc[:110].assign(sample_weight=np.linspace(.5, 1.5, 110))
    frame.iloc[0, frame.columns.get_loc('original_intent')] = 'exit'
    prediction = np.ones(len(frame))
    result = close_metrics(frame, prediction)
    selected = frame.original_intent.ne('exit')
    assert result['rows'] == len(frame) and result['selected'] == selected.sum()
    assert result['weighted_mse'] == pytest.approx(np.average((frame.close_advantage_bps-1)**2, weights=frame.sample_weight))
    assert result['selected_positions'] == frame.loc[selected, 'position_entry_time'].nunique()
    assert result['selected_weighted_mean_bps'] == pytest.approx(np.average(frame.loc[selected, 'close_advantage_bps'], weights=frame.loc[selected, 'sample_weight']))
    assert close_metrics(frame, np.zeros(len(frame)))['selected_mean_bps'] is None
    candidate = {**result, 'weighted_mse': 50., 'mse': 50., 'selected': 100, 'selected_positions': 30,
        'selected_mean_bps': 1., 'selected_weighted_mean_bps': 1.}
    values = {'boosted': candidate, 'ridge': {**candidate, 'weighted_mse': 100., 'mse': 100.},
        'constant': {**candidate, 'weighted_mse': 100., 'mse': 100.}}
    assert close_admission(values)['boosted_admitted'] and len(close_admission(values)['checks']) == 7
    for model, field, value in [('boosted', 'weighted_mse', 99.), ('constant', 'weighted_mse', 50.),
        ('boosted', 'mse', 101.), ('boosted', 'selected', 99), ('boosted', 'selected_positions', 29),
        ('boosted', 'selected_mean_bps', 0.), ('boosted', 'selected_weighted_mean_bps', 0.)]:
        damaged = copy.deepcopy(values)
        damaged[model][field] = value
        assert not close_admission(damaged)['boosted_admitted']


def real_source(tmp_path, monkeypatch):
    frame = ready_bars().assign(count=1, volume=1.)
    frame[['time', 'end']] += pd.Timedelta(days=366)
    cfg = config(bar_seconds=60, max_hold_bars=20, fee_bps=5, slippage_bps=3)
    policy = ExitMovePolicy(bot((0., .05, .8)), .02)
    policy.manager.features = CLOSE_FEATURES
    parent = tmp_path/'parent'
    parent.mkdir()
    streaming_backtest(frame, policy, cfg, parent/'candidate-00', 8192)
    frozen = {'protocol': 'exit_move_v54', 'risk': asdict(cfg)}
    save_json(parent/'frozen_selection.json', frozen)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        save_json(tmp_path/name, {})
    for module in ['close_effect', 'close_learning_inputs']:
        monkeypatch.setattr(f'wonyotti_fr.{module}.load_selection', lambda _: (frozen, policy))
        monkeypatch.setattr(f'wonyotti_fr.{module}.prepare_minute_period', lambda *_, **__: (frame.copy(), {}))
    root = run_close_effect_labels(parent, tmp_path, tmp_path, tmp_path/'runs')
    code = root/'generation_source/wonyotti_fr'
    code.mkdir(parents=True)
    for path in Path('src/wonyotti_fr').glob('*.py'):
        (code/path.name).write_bytes(path.read_bytes())
    return root


def reseal(root):
    save_json(root/'files.json', {n: sha256(root/n) for n in json.loads((root/'files.json').read_text())})


def test_complete_source_validates_actual_inputs_cashflows_and_read_only_journal(tmp_path, monkeypatch):
    root = real_source(tmp_path, monkeypatch)
    before = sha256(root/'outcomes.sqlite')
    ledger, verification = load_close_training(root)
    assert len(ledger) and verification['all_actual_inputs_and_cashflows_verified']
    assert before == sha256(root/'outcomes.sqlite')


@pytest.mark.parametrize('damage', ['incomplete', 'parent', 'code', 'ledger', 'journal', 'open_journal', 'parity'])
def test_source_rejects_tampering_even_after_output_fingerprints_rewritten(tmp_path, monkeypatch, damage):
    root = real_source(tmp_path, monkeypatch)
    if damage == 'incomplete':
        value = json.loads((root/'summary.json').read_text())
        value['complete'] = False
        save_json(root/'summary.json', value)
    elif damage == 'parent':
        value = json.loads((root/'manifest.json').read_text())
        value['settings']['reference_sha256'] = 'changed'
        save_json(root/'manifest.json', value)
    elif damage == 'code':
        (root/'generation_source/wonyotti_fr/engine.py').write_text('changed')
    elif damage == 'ledger':
        frame = pd.read_parquet(root/'opportunity_ledger.parquet')
        frame.loc[0, 'close_advantage_bps'] += 1
        frame.to_parquet(root/'opportunity_ledger.parquet', index=False)
        frame[frame.label_status.eq('closed')].reset_index(drop=True).to_parquet(root/'training_labels.parquet', index=False)
    elif damage == 'journal':
        import sqlite3
        with sqlite3.connect(root/'outcomes.sqlite') as connection:
            connection.execute("UPDATE outcomes SET previous_hash='changed' WHERE sequence=0")
    elif damage == 'open_journal':
        (root/'outcomes.sqlite-wal').touch()
    else:
        value = json.loads((root/'summary.json').read_text())
        value['replay_parity_sha256'] = 'changed'
        save_json(root/'summary.json', value)
    reseal(root)
    with pytest.raises((ValueError, AssertionError)):
        load_close_training(root)


def test_complete_diagnosis_preserves_all_rows_and_future_changes_leave_fit_unchanged(tmp_path, monkeypatch):
    root = tmp_path/'source'
    root.mkdir()
    save_json(root/'files.json', {})
    frame = examples()
    monkeypatch.setattr('wonyotti_fr.close_learning.load_close_training', lambda _: (frame.copy(), {'labels_files_sha256': sha256(root/'files.json')}))
    out = run_close_learning_diagnosis(root, tmp_path/'runs')
    future = frame.decision_time.dt.month.ge(10)
    frame.loc[future, CLOSE_FEATURES[0]] *= -100
    frame.loc[future, 'close_advantage_bps'] *= -20
    frame.loc[future, 'label_end'] += pd.Timedelta(hours=1)
    altered = run_close_learning_diagnosis(root, tmp_path/'changed')
    for name in ['training_used.parquet', 'training_weights.parquet', 'models.json']:
        assert (out/name).read_bytes() == (altered/name).read_bytes()
    assert len(pd.read_parquet(out/'exclusion_ledger.parquet')) == len(frame)
    result = json.loads((out/'summary.json').read_text())
    assert result['complete'] and not result['profitability_accepted'] and not result['trading_returns_evaluated']
