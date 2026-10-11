import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_exposure_backtest import data
from test_exposure_engine import bar, config

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.exposure_engine import ExposureEngine
from wonyotti_fr.exposure_reference import ReferenceAccount, require_near
from wonyotti_fr.holding_stress import read_json
from wonyotti_fr.recorded_change_backtest import backtest
from wonyotti_fr.recorded_change_policy import RecordedChangePolicy
from wonyotti_fr.recorded_change_reference import ReferenceRecordedPolicy, verify_path


def source():
    stamps = pd.date_range('2021-01-01T00:05Z', periods=300, freq='5min')
    sequence = np.tile([.4, 0., -.2, -.6, 0., .3], 50)
    return pd.DataFrame({'end': stamps, 'label_end': stamps+pd.Timedelta(minutes=5),
        'contract_supported': np.arange(300) % 17 != 0, 'contract_reason': ['synthetic']*300,
        'execution_change': sequence, 'xbtusd_execution_change': sequence*.7,
        'preserved_extra_column': np.arange(300)})


def options(timing='window_end', target='execution_change'):
    return {'target': target, 'timing': timing, 'historical_lookahead': timing == 'window_start'}


@pytest.mark.parametrize('timing', ['window_start', 'window_end'])
@pytest.mark.parametrize('target', ['execution_change', 'xbtusd_execution_change'])
@pytest.mark.parametrize('stop', [0., .04])
def test_all_eight_paths_full_independent_replay(tmp_path, timing, target, stop):
    frame, original = data(), source()
    settings = config(stop_fraction=stop, max_hold_bars=37, cooldown_bars=15)
    run = tmp_path/'run'
    metrics = backtest(frame, original, settings, run, batch_size=41, **options(timing, target))
    proof = verify_path(frame, original, settings, run, tmp_path/'review.json', **options(timing, target))
    assert proof['complete'] and proof['rows']['equity'] == 1500 and metrics['closed_trades'] > 0
    assert proof['all_checkpoint_states_compared'] == [500, 1000]
    assert proof['max_equity_difference'] < 1e-7
    assert proof['coverage']['source_rows'] == len(original)
    assert proof['coverage']['source_rows_not_invoked'] >= 1
    assert proof['coverage']['historical_lookahead_rows'] > 0 if timing == 'window_start' else proof['coverage']['historical_lookahead_rows'] == 0
    pd.testing.assert_frame_equal(pd.read_parquet(run/'source.parquet'), original)


@pytest.mark.parametrize('target,timing,flag', [
    ('execution_change', 'window_start', False), ('execution_change', 'window_end', True),
    ('execution_change', 'wrong', False), ('wrong', 'window_end', False),
    ('execution_change', 'window_start', 1)])
def test_lookahead_permission_and_metadata_are_explicit(target, timing, flag):
    with pytest.raises(ValueError):
        RecordedChangePolicy(source(), target=target, timing=timing, historical_lookahead=flag)


@pytest.mark.parametrize('change', ['naive', 'timezone', 'duplicate', 'unsorted', 'window', 'boundary',
    'support', 'reason', 'infinite', 'string_value', 'missing_column'])
def test_invalid_source_rejected(change):
    original = source()
    if change == 'naive':
        original['end'] = original.end.dt.tz_localize(None)
    elif change == 'timezone':
        original['end'] = original.end.dt.tz_convert('Asia/Seoul')
    elif change == 'duplicate':
        original = pd.concat([original, original.tail(1)])
    elif change == 'unsorted':
        original = original.iloc[::-1]
    elif change == 'window':
        original['label_end'] += pd.Timedelta(minutes=5)
    elif change == 'boundary':
        original['end'] += pd.Timedelta(seconds=1)
    elif change == 'support':
        original['contract_supported'] = original.contract_supported.astype(int)
    elif change == 'reason':
        original.loc[1, 'contract_reason'] = ''
    elif change == 'infinite':
        original.loc[1, 'execution_change'] = np.inf
    elif change == 'string_value':
        original['execution_change'] = original.execution_change.astype(str)
    else:
        original = original.drop(columns='xbtusd_execution_change')
    with pytest.raises(ValueError):
        RecordedChangePolicy(original, **options())


def test_exact_timing_missing_unsupported_zero_and_own_state():
    original = source()
    original.loc[0, ['execution_change', 'xbtusd_execution_change']] = np.nan
    policy = RecordedChangePolicy(original, **options())
    journal = []
    policy.audit_sink = journal.append
    own = {'equity': 10000., 'quantity': 20.}
    assert policy(bar(3), own) is None and journal == []
    assert policy(bar(4), own) is None and journal[-1]['reason'] == 'missing_source'
    assert policy(bar(9), own) is None and journal[-1]['reason'] == 'unsupported_source'
    assert policy(bar(14), own) is None and journal[-1]['reason'] == 'zero_change'
    assert policy(bar(19), own) == pytest.approx(0.)
    assert journal[-1]['current_exposure'] == .2 and journal[-1]['source_end'] == bar(14)['end']
    modified = dict(bar(19), original_current_exposure=1e15)
    assert policy(modified, own) == pytest.approx(0.)
    assert policy(modified, {'equity': 10000., 'quantity': -20.}) == pytest.approx(-.4)
    original.loc[2, 'execution_change'] = 100.
    assert policy(modified, own) == pytest.approx(0.)
    assert policy(bar(24), own) == pytest.approx(-.4)
    assert policy(bar(24), {'equity': 10000., 'quantity': -20.}) == -.5
    assert journal[-1]['unclipped_target'] == pytest.approx(-.8) and journal[-1]['clipped']


@pytest.mark.parametrize('timing', ['window_start', 'window_end'])
def test_future_sources_outside_declared_information_window_cannot_change_past(timing):
    original = source()
    modified = original.copy(deep=True)
    key = 'end' if timing == 'window_start' else 'label_end'
    last = pd.Timestamp(bar(29)['end'])
    modified.loc[modified[key] > last, 'execution_change'] = 1e9
    policies = [RecordedChangePolicy(s, **options(timing)) for s in [original, modified]]
    journals = [[], []]
    for p, rows in zip(policies, journals, strict=True):
        p.audit_sink = rows.append
        for i in range(30):
            p(bar(i), {'equity': 10000., 'quantity': -10.})
    assert journals[0] == journals[1]
    matched = [r for r in journals[0] if r['source_end'] is not None]
    assert {r['lookahead_seconds'] for r in matched} == ({300} if timing == 'window_start' else {0})


@pytest.mark.parametrize('stop', [0., .04])
def test_zero_change_keeps_quantity_and_only_price_stop_is_disabled(stop):
    original = source().iloc[:2].copy()
    original['contract_supported'] = True
    original['execution_change'] = [.4, 0.]
    policy = RecordedChangePolicy(original, **options('window_start'))
    engine = ExposureEngine(config(stop_fraction=stop, max_hold_bars=8))
    for i in range(6):
        result = engine.step(bar(i), policy)
    held = result['quantity']
    assert held > 0
    for i in range(6, 11):
        result = engine.step(bar(i, price=101., low=90.), policy)
        if stop:
            assert result['quantity'] == 0
        else:
            assert result['quantity'] == held and result['fills'] == []
    if not stop:
        for i in range(11, 14):
            result = engine.step(bar(i), policy)
        assert result['closed_trades'][0]['exit_reason'] == 'time_limit'


@pytest.mark.parametrize('limit', ['daily_loss_limit', 'max_drawdown'])
def test_disabling_price_stop_preserves_risk_halts_and_uninvoked_sources(tmp_path, limit):
    frame, original = data(), source()
    original['contract_supported'] = True
    original['execution_change'] = .5
    frame.loc[frame.index >= 7, ['open', 'high', 'low', 'close']] = 80.
    settings = config(stop_fraction=0., **{limit: .03})
    run = tmp_path/'run'
    backtest(frame, original, settings, run, **options('window_start'))
    proof = verify_path(frame, original, settings, run, tmp_path/'proof.json', **options('window_start'))
    assert proof['coverage']['not_invoked_risk_blocked'] > 0
    trades = pd.read_parquet(run/'account/trades.parquet')
    assert 'risk_halt' in set(trades.exit_reason)
    assert not any('stop' in v for v in trades.exit_reason)


@pytest.mark.parametrize('delay', [0, 1])
def test_no_trade_bars_restart_funding_and_reference_quantities(tmp_path, delay):
    frame, original = data(), source()
    frame.loc[frame.index % 11 == 5, ['count', 'volume']] = 0
    settings = config(signal_delay_bars=delay, stop_fraction=.04, max_hold_bars=11)
    run = tmp_path/'run'
    backtest(frame, original, settings, run, **options())
    proof = verify_path(frame, original, settings, run, tmp_path/'proof.json', **options())
    assert proof['metrics']['rejected']['market_no_trades'] > 0
    assert proof['metrics']['funding_cost'] != 0


def reseal(root):
    save_json(root/'files.json', {p.name: sha256(p) for p in root.iterdir() if p.is_file() and p.name != 'files.json'})


@pytest.mark.parametrize('tamper', ['decision', 'missing_decision', 'extra_decision', 'source', 'coverage',
    'checkpoint', 'policy', 'equity', 'fill', 'trade', 'manifest_omission'])
def test_independent_verifier_rejects_resealed_tampering(tmp_path, tamper):
    frame, original, settings = data(), source(), config(max_hold_bars=37)
    run = tmp_path/'run'
    backtest(frame, original, settings, run, **options())
    account = run/'account'
    if tamper in ['decision', 'missing_decision', 'extra_decision']:
        rows = pd.read_parquet(run/'decisions.parquet')
        if tamper == 'decision':
            rows.loc[rows.index[-1], 'lookahead_seconds'] = 300
        elif tamper == 'missing_decision':
            rows = rows.iloc[:-1]
        else:
            rows = pd.concat([rows, rows.tail(1)], ignore_index=True)
        rows.to_parquet(run/'decisions.parquet', index=False)
    elif tamper == 'source':
        rows = pd.read_parquet(run/'source.parquet')
        rows.loc[0, 'preserved_extra_column'] = -1
        rows.to_parquet(run/'source.parquet', index=False)
    elif tamper == 'coverage':
        values = read_json(run/'coverage.json')
        values['source_rows_not_invoked'] += 1
        save_json(run/'coverage.json', values)
    elif tamper in ['checkpoint', 'policy']:
        path = account/('checkpoint-500.json' if tamper == 'checkpoint' else 'policy.json')
        value = read_json(path)
        if tamper == 'checkpoint':
            value['state']['peak'] += 100.
        else:
            value['diagnostic_only'] = False
        save_json(path, value)
    elif tamper in ['equity', 'fill', 'trade']:
        name, field = {'equity': ('equity', 'quantity'), 'fill': ('fills', 'fee'), 'trade': ('trades', 'net_pnl')}[tamper]
        path = account/f'{name}.parquet'
        rows = pd.read_parquet(path)
        rows.loc[rows.index[-1], field] += 1.
        rows.to_parquet(path, index=False)
    reseal(account)
    storage = read_json(run/'storage.json')
    storage['account_files_sha256'] = sha256(account/'files.json')
    save_json(run/'storage.json', storage)
    reseal(run)
    if tamper == 'manifest_omission':
        sealed = read_json(run/'files.json')
        sealed.pop('source.parquet')
        save_json(run/'files.json', sealed)
    with pytest.raises(ValueError):
        verify_path(frame, original, settings, run, tmp_path/'proof.json', **options())
    assert not (tmp_path/'proof.json').exists()


def test_failed_execution_preserves_partial_journal_and_has_no_success_seal(tmp_path):
    frame = data()
    frame.loc[frame.index[-1], 'time'] += pd.Timedelta(minutes=1)
    run = tmp_path/'run'
    with pytest.raises(ValueError):
        backtest(frame, source(), config(), run, **options())
    assert (run/'failure.json').is_file() and (run/'account/failure.json').is_file()
    assert len(pd.read_parquet(run/'decisions.parquet')) > 0
    assert not (run/'files.json').exists()


def test_wrong_config_and_source_rejected_before_replay(tmp_path):
    frame, original, settings = data(), source(), config()
    run = tmp_path/'run'
    backtest(frame, original, settings, run, **options())
    with pytest.raises(ValueError, match='설정'):
        verify_path(frame, original, replace(settings, fee_bps=6.), run, tmp_path/'proof.json', **options())
    changed = original.copy()
    changed.loc[1, 'execution_change'] = .3
    with pytest.raises(ValueError, match='원본'):
        verify_path(frame, changed, settings, run, tmp_path/'proof.json', **options())


def test_independent_reference_decision_and_account_without_serialized_outputs():
    original = source()
    production = RecordedChangePolicy(original, **options('window_start'))
    reference = ReferenceRecordedPolicy(original, **options('window_start'))
    engine, account = ExposureEngine(config()), ReferenceAccount(config())
    for i in range(80):
        event = bar(i, price=100.+np.sin(i), funding=.0001)
        event['open'] = float(event['open'])
        event['close'] = float(event['close'])
        actual = engine.step(event, production, final=i == 79)
        expected = account.step(event, lambda b, cash, q: reference(b, cash, q)[0], final=i == 79)
        actual = copy.deepcopy(actual)
        actual.pop('accounting_residual')
        require_near(actual, expected)


@pytest.mark.parametrize('change,ordered', [(1e-12, False), (-1e-12, False), (1.0001e-12, True), (-1.0001e-12, True)])
def test_exact_zero_threshold(change, ordered):
    original = source().iloc[:1].copy()
    original['contract_supported'] = True
    original['execution_change'] = change
    policy = RecordedChangePolicy(original, **options('window_start'))
    result = policy(bar(4), {'equity': 10000., 'quantity': 0.})
    assert (result is not None) == ordered
    if ordered:
        assert result == change


def test_empty_source_preserves_market_rows_and_missing_decisions(tmp_path):
    original, frame = source().iloc[:0], data()
    run = tmp_path/'run'
    metrics = backtest(frame, original, config(), run, **options())
    proof = verify_path(frame, original, config(), run, tmp_path/'proof.json', **options())
    assert metrics['closed_trades'] == 0 and metrics['total_return'] == 0
    assert proof['rows']['equity'] == len(frame)
    assert proof['coverage']['source_rows'] == 0
    assert proof['coverage']['decision_reasons'] == {'missing_source': 299}
