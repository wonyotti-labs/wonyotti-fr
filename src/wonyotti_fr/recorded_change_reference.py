from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.exposure_backtest import SCHEMAS, events_with_feature_time, parquet_records
from wonyotti_fr.exposure_reference import ReferenceAccount, require_near
from wonyotti_fr.holding_stress import read_json
from wonyotti_fr.recorded_change_policy import metadata, validated_source


class ReferenceRecordedPolicy:
    def __init__(self, source, *, target, timing, historical_lookahead=False):
        self.settings = metadata(target, timing, historical_lookahead)
        self.source = validated_source(source)
        key = 'end' if timing == 'window_start' else 'label_end'
        self.times = [v.to_pydatetime() for v in self.source[key]]
        self.rows = list(self.source.itertuples(index=False))

    def __call__(self, bar, cash, quantity):
        end = datetime.fromisoformat(bar['end'])
        if end.second or end.microsecond or end.minute % 5:
            return None, None
        equity = cash+quantity*bar['close']
        current = quantity*bar['close']/equity
        index = bisect_left(self.times, end)
        source = self.rows[index] if index < len(self.times) and self.times[index] == end else None
        journal = {'time': bar['end'], 'source_end': None, 'source_label_end': None, 'available_at': None,
            'lookahead_seconds': 0, 'historical_lookahead': self.settings['historical_lookahead'],
            'source_supported': None, 'source_reason': None, 'supplied_change': None,
            'equity': equity, 'quantity': quantity, 'current_exposure': current,
            'unclipped_target': None, 'target': None, 'clipped': False, 'reason': 'missing_source'}
        if source is None:
            return None, journal
        journal.update(source_end=source.end.isoformat(), source_label_end=source.label_end.isoformat(),
            available_at=source.label_end.isoformat(), source_supported=source.contract_supported,
            source_reason=source.contract_reason,
            lookahead_seconds=int(max(0, (source.label_end.to_pydatetime()-end).total_seconds())))
        if not source.contract_supported:
            journal['reason'] = 'unsupported_source'
            return None, journal
        change = float(getattr(source, self.settings['target']))
        journal.update(supplied_change=change, reason='zero_change')
        if -1e-12 <= change <= 1e-12:
            return None, journal
        desired = current+change
        target = .5 if desired > .5 else -.5 if desired < -.5 else desired
        journal.update(unclipped_target=desired, target=target, clipped=abs(desired) > .5, reason='target_change')
        return target, journal


def checked_files(root):
    sealed = read_json(root/'files.json')
    if not isinstance(sealed, dict) or set(sealed) != {p.name for p in root.iterdir() if p.is_file() and p.name != 'files.json'}:
        raise ValueError('기록 재생 검산 파일 목록 불일치')
    for name, digest in sealed.items():
        if (name in ('.', '..') or '/' in name or '\\' in name or (root/name).is_symlink()
            or not isinstance(digest, str) or sha256(root/name) != digest):
            raise ValueError('기록 재생 검산 파일 해시 불일치')
    return sealed


def check_snapshot(snapshot, account, event, config, final=False):
    a = account
    active = None if a.active is None else {
        **{k: a.active[k] for k in ['entry_time', 'entry_price', 'direction', 'fees', 'funding_cost', 'max_quantity']},
        'gross_realized': a.active['cash_flow']+a.q*a.basis}
    state = {'cash': a.cash, 'quantity': a.q, 'entry_price': a.basis, 'index': a.i,
        'entry_index': a.entry_i, 'last_end': event['end'], 'last_close': event['close'],
        'peak': a.peak, 'day': a.day, 'day_equity': a.day_start, 'daily_halted': a.daily,
        'permanent_halted': a.halted, 'manual_halt': False, 'cooldown_until': a.resume_at,
        'pending': a.pending, 'deferred': a.delay, 'liquidity_exit_reason': a.liquidate,
        'active_trade': active, 'total_fees': a.fees, 'total_funding': a.funding,
        'closed_net': a.closed_net, 'closed_trades': a.trades, 'wins': a.wins,
        'sum_gains': a.gains, 'sum_losses': a.losses, 'max_drawdown': a.max_dd, 'completed': final}
    require_near(snapshot, {'format': 'exposure_engine_state_v1', 'config': asdict(config), 'state': state}, 'snapshot')


def verify_path(data, source, config, output, review, *, target, timing, historical_lookahead=False, progress=None):
    config.validate()
    if config.allocation != .5 or len(data) < 2:
        raise ValueError('기록 재생 검산 계좌 설정 오류')
    top_hash = sha256(output/'files.json')
    top = checked_files(output)
    account_root = output/'account'
    sealed = checked_files(account_root)
    account_hash = sha256(account_root/'files.json')
    settings = metadata(target, timing, historical_lookahead)
    reference = ReferenceRecordedPolicy(source, target=target, timing=timing, historical_lookahead=historical_lookahead)
    try:
        pd.testing.assert_frame_equal(pd.read_parquet(output/'source.parquet'), reference.source, check_exact=True)
    except AssertionError as error:
        raise ValueError('기록 재생 검산 원본 전체 행 불일치') from error
    if read_json(account_root/'config.json') != asdict(config) or read_json(account_root/'policy.json') != settings:
        raise ValueError('기록 재생 검산 설정·정책 불일치')
    storage = read_json(output/'storage.json')
    if storage['account_files_sha256'] != account_hash:
        raise ValueError('기록 재생 계좌 파일 목록 해시 불일치')
    account = ReferenceAccount(config)
    streams = {name: iter(parquet_records(account_root/f'{name}.parquet')) for name in SCHEMAS}
    journal_stream = iter(parquet_records(output/'decisions.parquet'))
    counts = dict.fromkeys(SCHEMAS, 0)
    daily, rejected, exposure = {}, {}, 0.
    max_equity_error, max_quantity_error, residual = 0., 0., 0.
    reasons, invoked = Counter(), set()
    linked = not_called = risk_blocked = final_only = liquidity_only = 0
    decision_count = clipped = lookahead = submitted = executed = 0
    checkpoints = sorted({len(data)//3, 2*len(data)//3})

    def decision(bar, cash, q):
        nonlocal decision_count, clipped, lookahead, submitted
        result, row = reference(bar, cash, q)
        if row is not None:
            require_near(next(journal_stream, None), row, 'decision.'+bar['end'])
            decision_count += 1
            reasons[row['reason']] += 1
            clipped += int(row['clipped'])
            lookahead += int(row['lookahead_seconds'] > 0)
            submitted += int(row['target'] is not None)
            if row['source_end'] is not None:
                invoked.add(datetime.fromisoformat(bar['end']))
        return result

    for i, event in enumerate(events_with_feature_time(data)):
        expected = account.step(event, decision, final=i == len(data)-1)
        stamp = datetime.fromisoformat(event['end'])
        at = bisect_left(reference.times, stamp)
        if at < len(reference.times) and reference.times[at] == stamp:
            linked += 1
            if stamp not in invoked:
                not_called += 1
                if expected['halted'] or expected['daily_halted']:
                    risk_blocked += 1
                elif expected['completed']:
                    final_only += 1
                else:
                    if account.liquidate is None:
                        raise ValueError('기록 신호 미호출 사유 누락')
                    liquidity_only += 1
        executed += int(expected['executed_target'] is not None)
        for name, key in [('trades', 'closed_trades'), ('fills', 'fills')]:
            for row in expected.pop(key):
                require_near(next(streams[name], None), row, f'{i}.{name}')
                counts[name] += 1
        for reason in expected.pop('rejected'):
            require_near(next(streams['rejections'], None), {'time': event['end'], 'reason': reason}, f'{i}.rejections')
            counts['rejections'] += 1
            rejected[reason] = rejected.get(reason, 0)+1
        saved = next(streams['equity'], None)
        if saved is None:
            raise ValueError('기록 재생 검산 계좌 행 누락')
        value = saved.pop('accounting_residual')
        if not math.isfinite(value) or abs(value) > max(1e-7, saved['equity']*1e-10):
            raise ValueError('기록 재생 회계 잔차 오류')
        residual = max(residual, abs(value))
        max_equity_error = max(max_equity_error, abs(saved['equity']-expected['equity']))
        max_quantity_error = max(max_quantity_error, abs(saved['quantity']-expected['quantity']))
        require_near(saved, expected, f'{i}.equity')
        counts['equity'] += 1
        daily[event['end'][:10]] = expected['equity']
        exposure += expected['exposure']
        if i+1 in checkpoints:
            check_snapshot(read_json(account_root/f'checkpoint-{i+1}.json'), account, event, config)
        if progress is not None and (i+1) % 100000 == 0:
            progress(i+1)
    for name, stream in {**streams, 'decisions': journal_stream}.items():
        if next(stream, None) is not None:
            raise ValueError(f'기록 재생 검산 추가 행: {name}')
    check_snapshot(read_json(account_root/'final_state.json'), account, event, config, final=True)
    returns = pd.Series(daily, dtype=float).pct_change().dropna()
    deviation = returns.std()
    metrics = {'total_return': account.cash/config.initial_equity-1,
        'annualized_return': (account.cash/config.initial_equity)**(365.25*1440/len(data))-1,
        'max_drawdown': account.max_dd, 'closed_trades': account.trades,
        'win_rate': account.wins/account.trades if account.trades else None,
        'profit_factor': account.gains/account.losses if account.losses else None,
        'daily_sharpe': float(np.sqrt(365.25)*returns.mean()/deviation) if deviation > 0 else None,
        'fees': account.fees, 'funding_cost': account.funding, 'average_exposure': exposure/len(data),
        'permanent_halt': account.halted, 'max_accounting_residual': residual, 'rejected': rejected,
        'final_equity': account.cash, 'bars': len(data)}
    require_near(read_json(account_root/'metrics.json'), metrics, 'metrics')
    if read_json(account_root/'storage.json') != {
        'batch_size': read_json(account_root/'storage.json')['batch_size'], 'rows': counts,
        'json_restored_checkpoints': checkpoints}:
        raise ValueError('기록 재생 계좌 저장 행·복원 시각 오류')
    if storage != {'decision_rows': decision_count, 'source_rows': len(reference.rows),
        'account_files_sha256': account_hash, 'account_rows': counts}:
        raise ValueError('기록 재생 판단·원본 저장 행 수 오류')
    support = sum(bool(row.contract_supported) for row in reference.rows)
    nonzero = sum(bool(row.contract_supported) and abs(float(getattr(row, target))) > 1e-12 for row in reference.rows)
    coverage = {'source_rows': len(reference.rows), 'source_supported_rows': support, 'source_nonzero_rows': nonzero,
        'source_rows_within_market_boundaries': linked, 'source_rows_outside_market_boundaries': len(reference.rows)-linked,
        'source_rows_invoked': len(invoked), 'source_rows_not_invoked': not_called, 'not_invoked_risk_blocked': risk_blocked,
        'not_invoked_final_only': final_only, 'not_invoked_liquidity_only': liquidity_only,
        'decision_rows': decision_count, 'decision_reasons': dict(reasons), 'clipped_targets': clipped,
        'historical_lookahead_rows': lookahead, 'submitted_targets': submitted, 'executed_targets': executed}
    require_near(read_json(output/'coverage.json'), coverage, 'coverage')
    if (sha256(output/'files.json') != top_hash or sha256(account_root/'files.json') != account_hash
        or checked_files(output) != top or checked_files(account_root) != sealed):
        raise ValueError('기록 재생 검산 중 파일 변경')
    proof = {'complete': True, 'source_files_sha256': top_hash, 'account_files_sha256': account_hash,
        'rows': counts, 'decision_rows': decision_count, 'coverage': coverage,
        'full_independent_policy_and_account_replay': True, 'every_fill_and_trade_compared': True,
        'all_checkpoint_states_compared': checkpoints, 'full_source_rows_preserved': True,
        'diagnostic_only': True, 'profitability_accepted': False,
        'max_equity_difference': max_equity_error, 'max_quantity_difference': max_quantity_error, 'metrics': metrics}
    save_json(review, proof)
    return proof
