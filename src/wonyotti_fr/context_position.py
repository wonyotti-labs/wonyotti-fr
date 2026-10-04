from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .activity_diagnostics import ACTIVITY_PERIODS
from .common import new_run, save_json, sha256
from .episode_balance import verified_files
from .event_features import MARKET_FEATURES, event_features
from .position_target import (
    PositionTargetModel,
    position_admission,
    position_metrics,
    position_reference,
)
from .reports import table

CONTEXT_FEATURES = ['ret_4d', 'ret_16d', 'ret_64d', 'trend_1d_4d', 'trend_4d_16d',
                    'trend_16d_64d', 'vol_16d', 'volume_ratio_16d']
CONTEXT_WINDOW = 64*288


class ContextPositionModel(PositionTargetModel):
    format = 'context_position_multinomial_v1'
    features = MARKET_FEATURES+CONTEXT_FEATURES


def context_features(bars):
    frame = bars.reset_index(drop=True)
    numeric = frame[['open', 'high', 'low', 'close', 'volume']].to_numpy(dtype=float)
    if (not len(frame) or frame[['time', 'end']].isna().any().any()
        or not frame.time.is_monotonic_increasing or frame.time.duplicated().any()
        or not frame.end.sub(frame.time).eq(pd.Timedelta(minutes=5)).all()
        or not np.isfinite(numeric).all() or (numeric[:, :4] <= 0).any() or (numeric[:, 4] < 0).any()
        or frame.high.lt(frame.low).any() or frame.close.gt(frame.high).any() or frame.close.lt(frame.low).any()):
        raise ValueError('다일 시장 맥락의 시세·시간·가격 오류')
    close = frame.close.astype(float)
    segments = frame.time.diff().ne(pd.Timedelta(minutes=5)).cumsum()
    result = frame[['end']].copy().astype({'end': 'datetime64[ns, UTC]'})
    for day in [4, 16, 64]:
        result[f'ret_{day}d'] = close.pct_change(day*288, fill_method=None)
    averages = {day: close.groupby(segments).transform(
        lambda part, span=day*288: part.ewm(span=span, adjust=False, min_periods=span).mean()
    ) for day in [1, 4, 16, 64]}
    for first, last in [(1, 4), (4, 16), (16, 64)]:
        result[f'trend_{first}d_{last}d'] = averages[first]/averages[last]-1
    returns = np.log(close).groupby(segments).diff()
    result['vol_16d'] = returns.groupby(segments).transform(
        lambda part: part.rolling(16*288, min_periods=16*288).std())*np.sqrt(16*288)
    volume_mean = frame.volume.groupby(segments).transform(
        lambda part: part.rolling(16*288, min_periods=16*288).mean())
    result['volume_ratio_16d'] = frame.volume/volume_mean.replace(0, np.nan)
    # 수익률의 시작 가격까지 같은 연속 구간이어야 한다.
    unsupported = frame.groupby(segments).cumcount() < CONTEXT_WINDOW
    result.loc[unsupported, CONTEXT_FEATURES] = np.nan
    result[CONTEXT_FEATURES] = result[CONTEXT_FEATURES].replace([np.inf, -np.inf], np.nan)
    return result


def attach_context(rows, bars):
    old = event_features(bars).set_index('end')
    new = context_features(bars).set_index('end')
    output = {}
    for name, frame in rows.items():
        if not frame.end.isin(old.index).all():
            raise ValueError('다일 시장 맥락의 기존 시각 누락')
        reconstructed = old.loc[frame.end, MARKET_FEATURES].reset_index(drop=True)
        pd.testing.assert_frame_equal(reconstructed, frame[MARKET_FEATURES].reset_index(drop=True), check_exact=True)
        extra = new.loc[frame.end, CONTEXT_FEATURES].reset_index(drop=True)
        if not np.isfinite(extra).all().all():
            raise ValueError('다일 시장 맥락의 원래 행 지원 부족')
        output[name] = pd.concat([frame.reset_index(drop=True), extra], axis=1)
    return output


def context_reference(reference):
    files = verified_files(reference)
    required = {'manifest.json', 'summary.json', 'training_support.json', 'model.json', 'metrics.json',
                'monthly.json', 'predictions.parquet', 'training_used.parquet', 'diagnosis_used.parquet',
                'training_ledger.parquet', 'diagnosis_ledger.parquet'}
    if not required <= set(files):
        raise ValueError('다일 시장 맥락의 원래 보유 진단 누락')
    settings = json.loads((reference / 'manifest.json').read_text())['settings']
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V44.md'))
        or settings['periods'] != ACTIVITY_PERIODS or settings['request_threshold'] != .65
        or settings['new_model_count'] != 1
        or json.loads((reference / 'summary.json').read_text()).get('complete') is not True):
        raise ValueError('다일 시장 맥락의 원래 계획·기간 오류')
    activity = Path(settings['activity'])
    if sha256(activity / 'files.json') != settings['activity_files_sha256']:
        raise ValueError('다일 시장 맥락의 원본 참조 변경')
    rows, ledgers, hashes = position_reference(activity)
    support = json.loads((reference / 'training_support.json').read_text())
    if support['source_sha256'] != hashes:
        raise ValueError('다일 시장 맥락의 원본 지문 불일치')
    for n in rows:
        pd.testing.assert_frame_equal(rows[n], pd.read_parquet(reference / f'{n}_used.parquet'), check_exact=True)
        pd.testing.assert_frame_equal(ledgers[n], pd.read_parquet(reference / f'{n}_ledger.parquet'), check_exact=True)
    model = PositionTargetModel.from_dict(json.loads((reference / 'model.json').read_text()))
    pred = pd.read_parquet(reference / 'predictions.parquet')
    val = rows['diagnosis']
    pd.testing.assert_frame_equal(pred[['end', 'episode_id', 'position_target_episode_id', 'position_target']],
        val[['end', 'episode_id', 'position_target_episode_id', 'position_target']], check_exact=True)
    scores = model.probabilities(val[MARKET_FEATURES].to_numpy())
    np.testing.assert_array_equal(scores, pred[['position_'+c for c in model.classes]].to_numpy())
    frequencies = np.bincount(rows['training'].position_target, minlength=3)/len(rows['training'])
    np.testing.assert_array_equal(frequencies, support['training_frequencies'])
    old_metrics = json.loads((reference / 'metrics.json').read_text())
    if old_metrics['position'] != position_metrics(val.position_target.to_numpy(), scores):
        raise ValueError('다일 시장 맥락의 기존 기준 지표 불일치')
    original = json.loads((activity / 'manifest.json').read_text())['settings']
    history = Path(original['history'])
    market = json.loads((history / 'manifest.json').read_text())
    path = (history / market['file']).resolve()
    if not path.is_relative_to(history.resolve()) or sha256(path) != hashes['history_sha256']:
        raise ValueError('다일 시장 맥락의 시장 경로·지문 오류')
    rows = attach_context(rows, pd.read_parquet(path))
    return rows, scores, frequencies, hashes


def context_admission(metrics, monthly):
    standard = position_admission({'position': metrics['context'], 'constant': metrics['constant']},
        [{**r, 'position_log_loss': r['context_log_loss']} for r in monthly])
    old, new = metrics['previous'], metrics['context']
    numbers = [old['log_loss'], new['log_loss'],
        *[m[c]['average_precision'] for m in [old, new] for c in ['short', 'long']]]
    if not np.isfinite(numbers).all() or min(numbers[:2]) < 0 or not all(0 <= p <= 1 for p in numbers[2:]):
        raise ValueError('다일 시장 맥락의 기존 모형 비교 오류')
    gain = 1-new['log_loss']/old['log_loss'] if old['log_loss'] > 0 else 0.
    checks = {**standard['checks'], 'previous_gain_passed': bool(gain > .01 and not np.isclose(gain, .01, rtol=0, atol=1e-12)),
        'previous_short_precision_preserved': new['short']['average_precision'] >= old['short']['average_precision'],
        'previous_long_precision_preserved': new['long']['average_precision'] >= old['long']['average_precision']}
    return {'context_position_admitted': all(checks.values()), 'checks': checks,
        'constant_relative_gain': standard['relative_log_loss_gain'], 'previous_relative_gain': gain,
        'improved_months': standard['improved_months'], 'profitability_accepted': False,
        'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}


def run_context_diagnostics(reference: Path, output: Path) -> Path:
    out = new_run(output, 'context-position-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V45.md')), 'periods': ACTIVITY_PERIODS,
        'new_features': CONTEXT_FEATURES, 'new_model_count': 1, 'request_threshold': .65,
        'trading_returns_evaluated': False})
    print(f'보유 방향의 다일 시장 맥락 진단: {out}', flush=True)
    try:
        rows, previous, frequencies, hashes = context_reference(reference)
        train, val = rows.values()
        model, support = ContextPositionModel.fit(train[ContextPositionModel.features].to_numpy(),
            train.position_target.to_numpy())
        scores = {'context': model.probabilities(val[ContextPositionModel.features].to_numpy()),
            'previous': previous, 'constant': np.tile(frequencies, (len(val), 1))}
        metrics = {n: position_metrics(val.position_target.to_numpy(), p) for n, p in scores.items()}
        pred = val[['end', 'episode_id', 'position_target_episode_id', 'position_target']].copy()
        for n, p in scores.items():
            for i, c in enumerate(model.classes):
                pred[n+'_'+c] = p[:, i]
            print(f'{n}: 로그 손실 {metrics[n]["log_loss"]:.6f}', flush=True)
        monthly = []
        for month, part in pred.groupby(pred.end.dt.strftime('%Y-%m')):
            row = {'month': month, 'rows': len(part), 'class_counts': np.bincount(part.position_target, minlength=3).tolist()}
            for n in scores:
                row[n+'_log_loss'] = float(log_loss(part.position_target,
                    part[[n+'_'+c for c in model.classes]], labels=[0, 1, 2]))
            monthly.append(row)
        decision = context_admission(metrics, monthly)
        support.update(training_frequencies=frequencies.tolist(), source_sha256=hashes)
        for n, value in [('model', model.to_dict()), ('training_support', support), ('metrics', metrics),
                         ('monthly', monthly), ('decision', decision)]:
            save_json(out / f'{n}.json', value)
        for n, frame in rows.items():
            frame.to_parquet(out / f'{n}_used.parquet', index=False)
        pred.to_parquet(out / 'predictions.parquet', index=False)
        save_json(out / 'summary.json', {'complete': True, 'rows': {n: len(f) for n, f in rows.items()},
            'original_rows_labels_features_and_scores_exact': True, **decision})
        (out / 'REPORT.md').write_text('# 보유 방향의 다일 시장 맥락 진단\n\n' + table(pd.DataFrame([
            {'model': n, 'log_loss': m['log_loss'], 'short_ap': m['short']['average_precision'],
             'long_ap': m['long']['average_precision']} for n, m in metrics.items()]))
            + '\n\n정답·지원 행과 기존 모형 점수를 보존하고 과거 시장 입력 8개만 추가했다. '
            '원본 상태·잔량·미래 자료는 입력에 넣지 않았다. 모든 기간은 이미 관찰했으며 예측 개선과 매매 수익성을 구분한다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'후속 다일 맥락 후보 허용: {decision["context_position_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
