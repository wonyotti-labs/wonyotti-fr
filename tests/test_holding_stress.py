import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_close_effect import CollectionPolicy, collection_fixture
from test_engine import config as engine_config

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.event_backtest import backtest
from wonyotti_fr.holding_stress import (
    STRESS_PERIODS,
    STRESS_VARIANTS,
    checked_stress_reference,
    run_holding_stress,
    seal,
    stress_configs,
    stress_details,
    stress_summary,
    verify_holding_stress,
)


def read(path):
    return json.loads(path.read_text())


def fixture(tmp_path, monkeypatch):
    bars, cfg, selection = collection_fixture(tmp_path)
    frozen = {'protocol': 'exit_state_v48', 'risk': asdict(cfg)}
    save_json(selection/'frozen_selection.json', frozen)
    review = tmp_path/'prior-review'
    records, baselines, retained, inputs, frames = [], {}, {}, {}, {}
    for period, bounds in STRESS_PERIODS.items():
        frame = bars.copy()
        shift = pd.Timestamp(bounds[0], tz='UTC')-pd.Timestamp(frame.time.iloc[0])
        frame[['time', 'end']] += shift
        frames[bounds[0]] = frame
        folder = tmp_path/'prior'/period
        backtest(frame, CollectionPolicy(), cfg, folder)
        baselines[period] = str(folder)
        inputs[period] = {'market': str(tmp_path/'market'), 'features': str(tmp_path/'market')}
        records.append({'run': str(folder), 'accounting': True, 'hashes': {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}})
    for period in ['2021_in_sample', '2023', '2024', '2025', 'previous_control']:
        folder = tmp_path/'prior'/period
        backtest(bars, CollectionPolicy(), cfg, folder)
        if period != 'previous_control':
            retained[period] = str(folder)
        records.append({'run': str(folder), 'accounting': True, 'hashes': {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}})
    # 기존 출처 계약용 합성 기록이며 실제 과거 검산을 뜻하지 않는다.
    save_json(review/'verification.json', {'complete': True, 'verified_runs': 8, 'profitability_accepted': False, 'synthetic_contract': True})
    save_json(review/'manifest.json', {'settings': {'selection': str(selection)}})
    save_json(review/'per_run_verification.json', records)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        save_json(tmp_path/'market'/name, {'synthetic_contract': True})
    sealed = {str(p): sha256(p) for root in [selection, review, tmp_path/'market'] for p in root.glob('*.json')}
    for row in records:
        sealed.update({str(Path(row['run'])/n): v for n, v in row['hashes'].items()})
    reference = tmp_path/'reference.json'
    save_json(reference, {'selection': str(selection), 'previous_review': str(review), 'inputs': inputs,
        'baselines': baselines, 'retained_runs': retained, 'sealed_files': sealed,
        'profitability_accepted': False, 'all_periods_already_observed': True})
    instances = []

    class SinglePreparePolicy(CollectionPolicy):
        prepared = False

        def prepare(self, frame):
            assert not self.prepared
            self.prepared = True

    def load(_):
        policy = SinglePreparePolicy()
        instances.append(policy)
        return copy.deepcopy(frozen), policy

    monkeypatch.setattr('wonyotti_fr.holding_stress.load_selection', load)
    monkeypatch.setattr('wonyotti_fr.holding_stress.prepare_minute_period', lambda *a, **_: (frames[a[3]].copy(), {'synthetic_contract': True}))
    return reference, cfg, instances


def test_twelve_executions_replay_accounting_baselines_delay_cost_and_fresh_state(tmp_path, monkeypatch):
    reference, cfg, instances = fixture(tmp_path, monkeypatch)
    out = run_holding_stress(reference, sha256(reference), tmp_path/'runs')
    rows = read(out/'results.json')
    assert len(rows) == 12 and read(out/'summary.json')['profitability_accepted'] is False
    review = verify_holding_stress(out, sha256(out/'files.json'), tmp_path/'reviews')
    assert read(review/'verification.json')['all_twelve_executions_independently_replayed']
    assert sum(policy.prepared for policy in instances) == 24
    spec = read(reference)
    assert read(out/'retained_results.json') == {k: read(Path(v)/'metrics.json') for k, v in spec['retained_runs'].items()}
    for period in STRESS_PERIODS:
        selected = {r['variant']: r for r in rows if r['period'] == period}
        assert selected['cost_x2']['fees'] > selected['base']['fees']
        assert selected['cost_x2']['total_return'] < selected['base']['total_return']
        base = pd.read_parquet(out/period/'base/fills.parquet')
        delayed = pd.read_parquet(out/period/'delay_1m/fills.parquet')
        assert pd.Timestamp(delayed.time.iloc[0])-pd.Timestamp(base.time.iloc[0]) == pd.Timedelta(minutes=1)
    assert checked_stress_reference(reference, sha256(reference))[1]['risk'] == asdict(cfg)


@pytest.mark.parametrize('damage', ['caller', 'market', 'proof', 'ledger', 'baseline', 'unsealed', 'period'])
def test_reference_damage_fails_before_any_new_simulation(tmp_path, monkeypatch, damage):
    reference, _, _ = fixture(tmp_path, monkeypatch)
    spec, digest = read(reference), sha256(reference)
    if damage == 'caller':
        digest = '0'*64
    elif damage == 'market':
        (Path(spec['inputs']['recent']['market'])/'manifest-1m.json').write_text('{}')
    elif damage == 'proof':
        (Path(spec['previous_review'])/'verification.json').write_text('{}')
    elif damage == 'ledger':
        path = Path(spec['previous_review'])/'per_run_verification.json'
        data = read(path)
        data[0]['hashes']['metrics.json'] = '0'*64
        save_json(path, data)
        spec['sealed_files'][str(path)] = sha256(path)
    elif damage == 'baseline':
        path = Path(spec['baselines']['confirmation'])/'trades.parquet'
        path.write_bytes(b'broken')
    elif damage == 'unsealed':
        del spec['sealed_files'][str(Path(spec['inputs']['recent']['market'])/'manifest-1m.json')]
    else:
        del spec['baselines']['recent']
    if damage in ['ledger', 'unsealed', 'period']:
        save_json(reference, spec)
        digest = sha256(reference)
    with pytest.raises(ValueError):
        run_holding_stress(reference, digest, tmp_path/'runs')
    assert not (tmp_path/'runs').exists()


@pytest.mark.parametrize('damage', ['cost', 'fill', 'summary', 'retained', 'missing', 'details'])
def test_semantic_tamper_rejected_even_after_output_resealing(tmp_path, monkeypatch, damage):
    reference, _, _ = fixture(tmp_path, monkeypatch)
    out = run_holding_stress(reference, sha256(reference), tmp_path/'runs')
    child = out/'confirmation/cost_x2'
    if damage == 'cost':
        data = read(child/'config.json')
        data['fee_bps'] = 5
        save_json(child/'config.json', data)
    elif damage == 'fill':
        frame = pd.read_parquet(child/'fills.parquet')
        frame.loc[0, 'fee'] += 1
        frame.to_parquet(child/'fills.parquet', index=False)
    elif damage == 'summary':
        data = read(out/'summary.json')
        data['profitability_accepted'] = True
        save_json(out/'summary.json', data)
    elif damage == 'retained':
        data = read(out/'retained_results.json')
        data['2021_in_sample']['total_return'] = 1.
        save_json(out/'retained_results.json', data)
    elif damage == 'missing':
        save_json(out/'results.json', read(out/'results.json')[:-1])
    else:
        data = read(child/'details.json')
        data['net_without_largest_trade'] = 1e9
        save_json(child/'details.json', data)
    seal(child)
    runs = read(out/'runs.json')
    runs['confirmation/cost_x2'] = sha256(child/'files.json')
    save_json(out/'runs.json', runs)
    seal(out)
    with pytest.raises((ValueError, AssertionError)):
        verify_holding_stress(out, sha256(out/'files.json'), tmp_path/'reviews')
    assert not list((tmp_path/'reviews').glob('*/verification.json'))


def test_configs_preserve_risk_and_summary_keeps_sample_failures():
    risk = asdict(engine_config(bar_seconds=60, fee_bps=5, slippage_bps=3))
    for config in stress_configs(risk).values():
        for key, value in risk.items():
            if key not in ['fee_bps', 'slippage_bps', 'signal_delay_bars']:
                assert asdict(config)[key] == value
    rows = [{'period': p, 'variant': v, 'total_return': .1, 'closed_trades': 8, 'permanent_halt': False} for p in STRESS_PERIODS for v in STRESS_VARIANTS]
    summary = stress_summary(rows)
    assert summary['cross_market_research_eligible'] and not summary['profitability_accepted']
    assert all(not check['at_least_30_trades'] for check in summary['checks'].values())
    with pytest.raises(ValueError):
        stress_summary(rows[:-1])
    with pytest.raises(ValueError):
        stress_summary(rows[:-1]+rows[:1])
    with pytest.raises(ValueError):
        stress_configs({**risk, 'fee_bps': 4.})


def test_block_diagnostic_matches_constant_daily_return_and_top_trade(tmp_path, monkeypatch):
    curve = pd.DataFrame({'time': pd.date_range('2023-01-01', periods=91, freq='D', tz='UTC')+pd.Timedelta(minutes=1), 'equity': 10000*1.001**np.arange(1, 92)})
    curve.to_parquet(tmp_path/'equity.parquet', index=False)
    pd.DataFrame({'net_pnl': [-10., 20., 5.]}).to_parquet(tmp_path/'trades.parquet', index=False)
    monkeypatch.setattr('wonyotti_fr.holding_stress.decompose_run', lambda *_: {'synthetic': True})
    details = stress_details(tmp_path, 10000)
    assert details['largest_trade_net_pnl'] == 20 and details['net_without_largest_trade'] == -5
    for key in ['annual_return_p025', 'annual_return_p50', 'annual_return_p975']:
        assert details['uncertainty'][key] == pytest.approx(1.001**365.25-1)
