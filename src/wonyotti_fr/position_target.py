from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss

from .activity_diagnostics import ACTIVITY_PERIODS
from .common import new_run, save_json, sha256
from .episode_balance import verified_files
from .event_features import MARKET_FEATURES
from .expansion_data import source_inputs
from .joint_entry import JointEntryModel, validate_joint_scores
from .management_diagnostics import management_metrics
from .reports import table


class PositionTargetModel(JointEntryModel):
    format = 'position_target_multinomial_v1'
    classes = ['flat', 'short', 'long']


def position_targets(frame, actions, episodes, name):
    if name not in ACTIVITY_PERIODS:
        raise ValueError('보유 방향 진단의 기간 오류')
    if (not len(actions) or not actions.time.is_monotonic_increasing or actions.time.isna().any()
        or not np.isfinite(actions[['before_qty', 'after_qty', 'episode_id']]).all().all()
        or not actions.before_qty.iloc[1:].reset_index(drop=True).equals(actions.after_qty.iloc[:-1].reset_index(drop=True))
        or episodes.episode_id.duplicated().any() or episodes.entry_time.isna().any()
        or not actions.loc[actions.after_qty.ne(0), 'episode_id'].isin(episodes.episode_id).all()):
        raise ValueError('보유 방향 정답의 원장 연속성·포지션 오류')
    if (not len(frame) or frame.end.isna().any() or frame.label_end.isna().any()
        or frame.end.duplicated().any() or not frame.end.is_monotonic_increasing
        or not frame.label_end.sub(frame.end).eq(pd.Timedelta(minutes=5)).all()
        or not frame.usable.eq(True).all()
        or not np.isfinite(frame[MARKET_FEATURES]).all().all()):
        raise ValueError('보유 방향 정답의 입력·시간 오류')
    first, last = (pd.Timestamp(v, tz='UTC') for v in ACTIVITY_PERIODS[name])
    if frame.end.min() < first+pd.Timedelta(days=1) or frame.label_end.max() >= last-pd.Timedelta(days=1):
        raise ValueError('보유 방향 정답의 원래 시간 배제 오류')
    times = actions.time.astype('datetime64[ns, UTC]').array.asi8
    current = np.searchsorted(times, frame.end.astype('datetime64[ns, UTC]').array.asi8, side='left')-1
    target = np.searchsorted(times, frame.label_end.astype('datetime64[ns, UTC]').array.asi8, side='left')-1
    if (current < 0).any() or frame.label_end.max() > actions.time.max():
        raise ValueError('보유 방향 정답의 원본 시간 지원 부족')
    qty = actions.after_qty.to_numpy()
    episode = np.where(qty == 0, 0, actions.episode_id.to_numpy())
    if (not np.array_equal(np.sign(qty[current]), frame.direction)
        or not np.array_equal(episode[current], frame.episode_id)):
        raise ValueError('보유 방향 정답의 현재 원본 상태 불일치')
    result = frame.copy()
    result['position_target'] = np.where(qty[target] == 0, 0, np.where(qty[target] < 0, 1, 2))
    result['position_target_episode_id'] = episode[target]
    result['position_target_time'] = actions.time.iloc[target].to_numpy()
    reasons = np.full(len(frame), 'included', dtype=object)
    for boundary, label in [(first, 'left_episode_boundary'), (last, 'right_episode_boundary')]:
        crossing = episodes.loc[episodes.entry_time.lt(boundary)
            & (episodes.exit_time.isna() | episodes.exit_time.ge(boundary)), 'episode_id']
        mask = result.episode_id.isin(crossing) | result.position_target_episode_id.isin(crossing)
        reasons[(reasons == 'included') & mask] = label
    ledger = result[['end', 'label_end', 'episode_id', 'position_target_episode_id', 'position_target_time', 'position_target']].assign(reason=reasons)
    selected = result[reasons == 'included'].reset_index(drop=True)
    counts = np.bincount(selected.position_target, minlength=3)
    if len(selected) < (1000 if name == 'training' else 100) or (counts < 20).any():
        raise ValueError('보유 방향 정답의 행·클래스 지원 부족')
    return selected, ledger


def position_metrics(labels, scores):
    y, p = np.asarray(labels), validate_joint_scores(scores)
    if (y.shape != (len(p),) or y.dtype.kind not in 'iu' or set(np.unique(y)) != {0, 1, 2}
        or (np.bincount(y, minlength=3) < 20).any()):
        raise ValueError('보유 방향 지표의 클래스 지원 부족')
    requested = np.where(p[:, 2] >= .65, 2, np.where(p[:, 1] >= .65, 1, 0))
    result = {'rows': len(y), 'class_counts': np.bincount(y, minlength=3).tolist(),
        'log_loss': float(log_loss(y, p, labels=[0, 1, 2])),
        'multiclass_brier': float(np.mean(np.sum((p-np.eye(3)[y])**2, axis=1))),
        'argmax_confusion': confusion_matrix(y, p.argmax(axis=1), labels=[0, 1, 2]).tolist(),
        'request_confusion': confusion_matrix(y, requested, labels=[0, 1, 2]).tolist(),
        'request_threshold': .65}
    for k, name in enumerate(PositionTargetModel.classes):
        result[name] = management_metrics(y == k, p[:, k])
        result[name]['requested'] = int((requested == k).sum())
        result[name]['correct_requests'] = int(((requested == k) & (y == k)).sum())
    return result


def position_admission(metrics, monthly):
    old, new = metrics['constant'], metrics['position']
    values = [old['log_loss'], new['log_loss']]
    precision = [m[k]['average_precision'] for m in [old, new] for k in ['short', 'long']]
    if not np.isfinite(values+precision).all() or min(values) < 0 or not all(0 <= v <= 1 for v in precision):
        raise ValueError('보유 방향 진단의 지표 범위 오류')
    months = [r['month'] for r in monthly]
    if months != [f'2021-{m:02}' for m in range(1, 13)] or any(
        not np.isfinite([r['position_log_loss'], r['constant_log_loss']]).all()
        or min(r['position_log_loss'], r['constant_log_loss']) < 0 for r in monthly
    ):
        raise ValueError('보유 방향 진단의 월별 비교 오류')
    gain = 1-new['log_loss']/old['log_loss'] if old['log_loss'] > 0 else 0.
    improved = sum(r['position_log_loss'] < r['constant_log_loss'] for r in monthly)
    checks = {'gain_passed': bool(gain > .01 and not np.isclose(gain, .01, atol=1e-12, rtol=0)),
        'short_precision_preserved': new['short']['average_precision'] >= old['short']['average_precision'],
        'long_precision_preserved': new['long']['average_precision'] >= old['long']['average_precision'],
        'monthly_consistency': improved >= 8}
    return {'position_target_admitted': all(checks.values()), 'checks': checks,
        'relative_log_loss_gain': gain, 'improved_months': improved, 'required_improved_months': 8,
        'profitability_accepted': False, 'trading_returns_evaluated': False,
        'all_source_periods_already_observed': True}


def position_reference(activity):
    files = verified_files(activity)
    if not {'manifest.json', 'summary.json', 'training_used.parquet', 'diagnosis_used.parquet'} <= set(files):
        raise ValueError('보유 방향 진단의 활동 참조 누락')
    settings = json.loads((activity / 'manifest.json').read_text())['settings']
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V42.md'))
        or settings['periods'] != ACTIVITY_PERIODS
        or json.loads((activity / 'summary.json').read_text()).get('complete') is not True):
        raise ValueError('보유 방향 진단의 원래 계획·기간 오류')
    source, hashes = source_inputs(*[Path(settings[k]) for k in ['audit', 'study', 'history']])
    if any(settings.get(k) != v for k, v in hashes.items()):
        raise ValueError('보유 방향 진단의 원본 입력 지문 오류')
    rows, ledgers = {}, {}
    for name in ACTIVITY_PERIODS:
        rows[name], ledgers[name] = position_targets(pd.read_parquet(activity / f'{name}_used.parquet'),
            source['actions'], source['episodes'], name)
    sets = [set(f.episode_id) | set(f.position_target_episode_id) for f in rows.values()]
    if (sets[0] & sets[1]) - {0} or rows['training'].label_end.max() >= rows['diagnosis'].end.min():
        raise ValueError('보유 방향 학습·진단의 포지션·시간 중첩')
    return rows, ledgers, hashes


def run_position_diagnostics(activity: Path, output: Path) -> Path:
    out = new_run(output, 'position-target-diagnosis', {'activity': str(activity),
        'activity_files_sha256': sha256(activity / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V44.md')), 'periods': ACTIVITY_PERIODS,
        'new_model_count': 1, 'request_threshold': .65, 'trading_returns_evaluated': False})
    print(f'다음 경계의 보유 방향 진단: {out}', flush=True)
    try:
        rows, ledgers, hashes = position_reference(activity)
        for n, frame in rows.items():
            frame.to_parquet(out / f'{n}_used.parquet', index=False)
            ledgers[n].to_parquet(out / f'{n}_ledger.parquet', index=False)
            frame.groupby('position_target_episode_id', sort=True).agg(rows=('end', 'size'),
                first_end=('end', 'min'), last_end=('end', 'max'), target=('position_target', 'first'),
                target_classes=('position_target', 'nunique')).reset_index().to_parquet(
                    out / f'{n}_position_support.parquet', index=False)
        train, val = rows.values()
        model, support = PositionTargetModel.fit(train[MARKET_FEATURES].to_numpy(), train.position_target.to_numpy())
        frequencies = np.bincount(train.position_target, minlength=3)/len(train)
        scores = {'position': model.probabilities(val[MARKET_FEATURES].to_numpy()),
            'constant': np.tile(frequencies, (len(val), 1))}
        pred = val[['end', 'episode_id', 'position_target_episode_id', 'position_target']].copy()
        metrics = {}
        for n, p in scores.items():
            for i, label in enumerate(PositionTargetModel.classes):
                pred[n+'_'+label] = p[:, i]
            metrics[n] = position_metrics(val.position_target.to_numpy(), p)
            print(f'{n}: 로그 손실 {metrics[n]["log_loss"]:.6f}', flush=True)
        monthly = []
        for month, part in pred.groupby(pred.end.dt.strftime('%Y-%m')):
            item = {'month': month, 'rows': len(part), 'class_counts': np.bincount(part.position_target, minlength=3).tolist(),
                'current_positions': int(part.episode_id[part.episode_id.ne(0)].nunique()),
                'target_positions': int(part.position_target_episode_id[part.position_target_episode_id.ne(0)].nunique())}
            for n in scores:
                p = part[[n+'_'+c for c in PositionTargetModel.classes]].to_numpy()
                item[n+'_log_loss'] = float(log_loss(part.position_target, p, labels=[0, 1, 2]))
            monthly.append(item)
        decision = position_admission(metrics, monthly)
        support.update(training_frequencies=frequencies.tolist(), source_sha256=hashes,
            training_positions=int(train.episode_id[train.episode_id.ne(0)].nunique()),
            diagnosis_positions=int(val.episode_id[val.episode_id.ne(0)].nunique()))
        for n, value in [('model', model.to_dict()), ('training_support', support), ('metrics', metrics),
                         ('monthly', monthly), ('decision', decision)]:
            save_json(out / f'{n}.json', value)
        pred.to_parquet(out / 'predictions.parquet', index=False)
        save_json(out / 'summary.json', {'complete': True, 'rows': {n: len(f) for n, f in rows.items()}, **decision})
        (out / 'REPORT.md').write_text('# 다음 경계의 보유 방향 진단\n\n' + table(pd.DataFrame([
            {'model': n, 'log_loss': m['log_loss'], 'short_ap': m['short']['average_precision'],
             'long_ap': m['long']['average_precision']} for n, m in metrics.items()]))
            + '\n\n당시 시장 입력만으로 다음 경계 직전 실제 보유 방향을 예측한다. 원본 포지션은 입력에 포함하지 않는다. '
            '신규 사건 없음과 수량 없음은 다른 정답이다. 지속된 보유 행은 독립 거래가 아니며 모든 기간은 이미 관찰했다. '
            '원본 보유 방향의 예측 개선은 진입 수익성의 증거가 아니다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'후속 보유 방향 후보 허용: {decision["position_target_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
