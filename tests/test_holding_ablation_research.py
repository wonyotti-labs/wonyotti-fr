import copy
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
from test_activity_ablation import bots
from test_engine import config
from test_minute_inventory_research import ready_bars

from wonyotti_fr import holding_ablation_research as study
from wonyotti_fr.common import new_run, save_json, sha256
from wonyotti_fr.exit_state import ExitStatePolicy
from wonyotti_fr.holding_stress import (
    STRESS_PERIODS,
    STRESS_VARIANTS,
    read_json,
    seal,
    stress_configs,
    stress_summary,
)


def fixture(tmp_path, monkeypatch):
    cfg = config(bar_seconds=60, fee_bps=5, slippage_bps=3, max_hold_bars=20)
    frozen = {'protocol': 'exit_state_v48', 'risk': asdict(cfg)}
    selection = tmp_path/'selection'
    save_json(selection/'frozen_selection.json', frozen)
    paths = {'market': str(tmp_path/'market'), 'features': str(tmp_path/'market')}
    retained = {'2021_in_sample': {'total_return': -.1}, '2023': {}, '2024': {}, '2025': {}}
    for key, metric in retained.items():
        save_json(tmp_path/'legacy'/key/'metrics.json', metric)
    prior = {'selection': str(selection), 'inputs': {p: paths for p in STRESS_PERIODS},
        'retained_runs': {key: str(tmp_path/'legacy'/key) for key in retained}}
    ref = tmp_path/'prior-reference.json'
    save_json(ref, prior)
    monkeypatch.setattr(study, 'checked_stress_reference', lambda *_: (copy.deepcopy(prior), copy.deepcopy(frozen)))
    settings = {'reference': prior, 'reference_path': str(ref), 'reference_sha256': sha256(ref),
        'risk': frozen['risk'], 'periods': STRESS_PERIODS, 'variants': STRESS_VARIANTS,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V92.md'))}
    run = new_run(tmp_path, 'synthetic-prior', settings)
    rows, runs, verified = [], {}, {}
    # 이전 검산의 연결 계약만 표현한다. 실제 시세·계좌 검산 결과가 아니다.
    for period in STRESS_PERIODS:
        save_json(run/period/'input_verification.json', {'synthetic_contract': True})
        for variant in STRESS_VARIANTS:
            key = f'{period}/{variant}'
            folder = run/key
            metric = {'total_return': .1, 'closed_trades': 8, 'permanent_halt': False, 'bars': 90}
            for name, value in [('metrics.json', metric), ('config.json', asdict(stress_configs(frozen['risk'])[variant])), ('details.json', {'synthetic_contract': True}), ('final_state.json', {})]:
                save_json(folder/name, value)
            for name in ['equity', 'fills', 'trades']:
                (folder/(name+'.parquet')).write_bytes(b'synthetic reference contract')
            seal(folder)
            runs[key] = sha256(folder/'files.json')
            rows.append({'period': period, 'variant': variant, **metric, 'synthetic_contract': True})
            verified[key] = {'all_policy_decisions_fills_trades_equity_and_state_exact': True,
                'cash_fees_funding_and_net_verified': True, 'trades': 8, 'bars': 90}
        seal(run/period)
        runs[period] = sha256(run/period/'files.json')
    for name, value in [('results', rows), ('runs', runs), ('summary', stress_summary(rows)),
        ('retained_results', retained)]:
        save_json(run/(name+'.json'), value)
    seal(run)
    review = new_run(tmp_path, 'synthetic-review', {'source': str(run), 'source_files_sha256': sha256(run/'files.json')})
    save_json(review/'verification.json', {'complete': True, 'source_files_sha256': sha256(run/'files.json'),
        'all_twelve_executions_independently_replayed': True, 'cash_fees_funding_and_net_verified': True,
        'previous_failed_sample_counts_preserved': True, 'profitability_accepted': False, 'runs': verified})
    seal(review)
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        save_json(tmp_path/'market'/name, {'synthetic_contract': True})
    ref = tmp_path/'reference.json'
    save_json(ref, {'stress_run': str(run), 'stress_run_sha256': sha256(run/'files.json'),
        'stress_review': str(review), 'stress_review_sha256': sha256(review/'files.json'),
        'inputs': {s: {p: paths for p in ['observed', 'recent']} for s in ['ETHUSDT', 'SOLUSDT']},
        'sealed_files': {str(p): sha256(p) for p in (tmp_path/'market').iterdir()},
        'profitability_accepted': False, 'all_periods_already_observed': True})
    instances, markets = [], []

    class SinglePrepare(ExitStatePolicy):
        prepared = False

        def prepare(self, bars):
            assert not self.prepared
            self.prepared = True
            super().prepare(bars)

    def load(_):
        _, old = bots(manager=(0., 0., .9))
        policy = SinglePrepare(old)
        instances.append(policy)
        return copy.deepcopy(frozen), policy

    def bars(*args, **_):
        markets.append((args[2], args[3], args[4]))
        frame = ready_bars().copy()
        frame[['time', 'end']] += pd.Timestamp(args[3], tz='UTC')-frame.time.iloc[0]
        frame[['open', 'high', 'low', 'close']] = 100.
        frame.loc[frame.index >= 5, ['low', 'close']] = 99.8
        return frame, {'synthetic_contract': True, 'symbol': args[2], 'start': args[3]}

    monkeypatch.setattr(study, 'load_selection', load)
    monkeypatch.setattr(study, 'prepare_minute_period', bars)
    return ref, instances, markets


def test_all_twenty_eight_accounts_replayed_and_failed_samples_preserved(tmp_path, monkeypatch):
    ref, instances, markets = fixture(tmp_path, monkeypatch)
    run = study.run_holding_ablation(ref, sha256(ref), tmp_path/'runs')
    review = study.verify_holding_ablation(run, sha256(run/'files.json'), tmp_path/'reviews')
    assert read_json(review/'verification.json')['all_twenty_eight_executions_independently_replayed']
    assert len(instances) == 56 and all(p.prepared for p in instances)
    assert len(markets) == 26 and {s for s, _, _ in markets} == {'BTCUSDT', 'ETHUSDT', 'SOLUSDT'}
    rows = read_json(run/'results.json')
    assert len(rows) == 28 and not read_json(run/'summary.json')['profitability_accepted']
    assert read_json(run/'retained_results.json')['legacy']['2021_in_sample']['total_return'] < 0
    assert all(r['decomposition']['trade_additions'] == 0 for r in rows if r['variant'] in {'no_adds', 'hold_only'})


@pytest.mark.parametrize('damage', ['caller', 'market', 'proof', 'eligibility', 'unsealed', 'source', 'missing_run'])
def test_invalid_prerequisite_stops_before_new_simulation(tmp_path, monkeypatch, damage):
    ref, _, _ = fixture(tmp_path, monkeypatch)
    spec = read_json(ref)
    run, review = Path(spec['stress_run']), Path(spec['stress_review'])
    if damage == 'market':
        Path(next(iter(spec['sealed_files']))).write_text('{}')
    elif damage == 'proof':
        proof = read_json(review/'verification.json')
        proof['runs']['confirmation/base']['cash_fees_funding_and_net_verified'] = False
        save_json(review/'verification.json', proof)
        seal(review)
        spec['stress_review_sha256'] = sha256(review/'files.json')
    elif damage == 'eligibility':
        rows = read_json(run/'results.json')
        rows[0]['total_return'] = -.1
        metric = read_json(run/'confirmation/base/metrics.json')
        metric['total_return'] = -.1
        save_json(run/'confirmation/base/metrics.json', metric)
        seal(run/'confirmation/base')
        runs = read_json(run/'runs.json')
        runs['confirmation/base'] = sha256(run/'confirmation/base/files.json')
        save_json(run/'runs.json', runs)
        save_json(run/'results.json', rows)
        save_json(run/'summary.json', stress_summary(rows))
        seal(run)
        spec['stress_run_sha256'] = sha256(run/'files.json')
        for name in ['manifest.json', 'verification.json']:
            data = read_json(review/name)
            (data['settings'] if name == 'manifest.json' else data)['source_files_sha256'] = spec['stress_run_sha256']
            save_json(review/name, data)
        seal(review)
        spec['stress_review_sha256'] = sha256(review/'files.json')
    elif damage == 'unsealed':
        spec['sealed_files'].pop(next(iter(spec['sealed_files'])))
    elif damage == 'source':
        (run/'code_snapshot/engine.py').write_text('# 변경\n')
    elif damage == 'missing_run':
        (run/'confirmation/base/metrics.json').unlink()
    save_json(ref, spec)
    with pytest.raises((ValueError, FileNotFoundError)):
        study.run_holding_ablation(ref, '0'*64 if damage == 'caller' else sha256(ref), tmp_path/'new')
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('damage', ['risk', 'fill', 'summary', 'retained', 'omitted'])
def test_resealed_output_tampering_cannot_pass_independent_replay(tmp_path, monkeypatch, damage):
    ref, _, _ = fixture(tmp_path, monkeypatch)
    run = study.run_holding_ablation(ref, sha256(ref), tmp_path/'runs')
    key = 'ETHUSDT/observed/base'
    child = run/key
    if damage == 'risk':
        data = read_json(child/'config.json')
        data['fee_bps'] *= 2
        save_json(child/'config.json', data)
    elif damage == 'fill':
        frame = pd.read_parquet(child/'fills.parquet')
        frame.loc[0, 'fee'] += 1
        frame.to_parquet(child/'fills.parquet', index=False)
    elif damage == 'summary':
        data = read_json(run/'summary.json')
        data['profitability_accepted'] = True
        save_json(run/'summary.json', data)
    elif damage == 'retained':
        data = read_json(run/'retained_results.json')
        data['legacy']['2021_in_sample']['total_return'] = .1
        save_json(run/'retained_results.json', data)
    else:
        save_json(run/'results.json', read_json(run/'results.json')[:-1])
    seal(child)
    runs = read_json(run/'runs.json')
    runs[key] = sha256(child/'files.json')
    save_json(run/'runs.json', runs)
    seal(run)
    with pytest.raises((ValueError, AssertionError)):
        study.verify_holding_ablation(run, sha256(run/'files.json'), tmp_path/'reviews')
    assert not list((tmp_path/'reviews').glob('*/verification.json'))


def test_no_adds_changes_only_limit_and_twenty_eight_exact_conditions():
    risk = asdict(config(bar_seconds=60, fee_bps=5, slippage_bps=3))
    assert asdict(study.ablation_config(risk, 'no_adds')) == {**risk, 'max_adds': 0}
    assert asdict(study.ablation_config(risk, 'hold_only')) == risk
    groups = study.ablation_groups()
    rows = [{'symbol': g['symbol'], 'period': g['period'], 'variant': v, 'total_return': .1,
        'closed_trades': 8, 'permanent_halt': False} for g in groups for v in g['variants']]
    assert not study.ablation_summary(rows)['profitability_accepted']
    with pytest.raises(ValueError):
        study.ablation_summary(rows[:-1]+rows[:1])
    with pytest.raises(ValueError):
        study.ablation_config(risk, 'unknown')
