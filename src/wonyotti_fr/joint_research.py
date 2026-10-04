from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .activity_diagnostics import ACTIVITY_PERIODS
from .boosted_direction import read_admission
from .common import new_run, save_json, sha256
from .direction_diagnostics import direction_metrics
from .episode_balance import verified_files
from .event_features import MARKET_FEATURES
from .expansion_model import BinaryModel
from .joint_entry import JointEntryModel, factorized_scores, joint_targets, validate_joint_scores
from .management_diagnostics import management_metrics
from .reports import table


def joint_metrics(labels, scores, threshold):
    y, p = np.asarray(labels), validate_joint_scores(scores)
    if (y.shape != (len(p),) or y.dtype.kind not in 'iu' or set(np.unique(y)) != {0, 1, 2}
        or (np.bincount(y, minlength=3) < 20).any()
        or type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('결합 진입 진단의 정답·문턱·클래스 지원 부족')
    activity = 1-p[:, 0]
    total = p[:, 1]+p[:, 2]
    direction = np.divide(p[:, 2], total, out=np.full(len(p), .5), where=total > 0)
    requested = activity >= threshold
    classes = np.where(requested & (direction >= .65), 2, np.where(requested & (direction <= .35), 1, 0))
    result = {'rows': len(y), 'class_counts': np.bincount(y, minlength=3).tolist(),
        'log_loss': float(log_loss(y, p, labels=[0, 1, 2])),
        'multiclass_brier': float(np.mean(np.sum((p-np.eye(3)[y])**2, axis=1))),
        'activity': management_metrics(y != 0, activity), 'threshold': threshold,
        'conditional_direction': direction_metrics((y[y != 0] == 2).astype(int), direction[y != 0])}
    for k, name in [(1, 'short'), (2, 'long')]:
        result[name] = management_metrics(y == k, p[:, k])
        result[name]['requested'] = int((classes == k).sum())
        result[name]['correct_requests'] = int(((classes == k) & (y == k)).sum())
    return result


def joint_admission(metrics):
    old, new, constant = (metrics[n] for n in ['factorized', 'joint', 'constant'])
    values = [old['log_loss'], new['log_loss'], constant['log_loss'],
        old['activity']['log_loss'], new['activity']['log_loss']]
    precision = [m[a]['average_precision'] for m in [old, new] for a in ['short', 'long']]
    if not np.isfinite(values+precision).all() or min(values) < 0 or not all(0 <= v <= 1 for v in precision):
        raise ValueError('결합 진입 판정 지표의 범위 오류')
    gain = 1-new['log_loss']/old['log_loss'] if old['log_loss'] > 0 else 0.
    checks = {'gain_passed': bool(gain > .01 and not np.isclose(gain, .01, rtol=0, atol=1e-12)),
        'constant_beaten': new['log_loss'] < constant['log_loss'],
        'activity_loss_preserved': new['activity']['log_loss'] <= old['activity']['log_loss'],
        'short_precision_preserved': new['short']['average_precision'] >= old['short']['average_precision'],
        'long_precision_preserved': new['long']['average_precision'] >= old['long']['average_precision']}
    return {'joint_entry_admitted': all(checks.values()), 'checks': checks, 'relative_log_loss_gain': gain,
        'required_relative_gain': .01, 'profitability_accepted': False, 'trading_returns_evaluated': False,
        'all_source_periods_already_observed': True}


def joint_reference(activity, direction):
    files = verified_files(activity)
    required = {'manifest.json', 'summary.json', 'models.json', 'predictions.parquet', 'training_used.parquet', 'diagnosis_used.parquet'}
    if not required <= set(files):
        raise ValueError('결합 진입 진단의 활동 기준 파일 누락')
    settings = json.loads((activity / 'manifest.json').read_text())['settings']
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V42.md'))
        or settings['periods'] != ACTIVITY_PERIODS or settings['training_quantile'] != .975
        or json.loads((activity / 'summary.json').read_text()).get('complete') is not True):
        raise ValueError('결합 진입 진단의 활동 계획·분할 오류')
    hashes = {k: settings[k] for k in ['audit_sha256', 'history_sha256', 'events_sha256']}
    read_admission(direction, hashes)
    old_models = json.loads((direction / 'models.json').read_text())
    a_model = BinaryModel.from_dict(json.loads((activity / 'models.json').read_text())['logistic'])
    d_model = BinaryModel.from_dict(old_models['boosted'])
    rows = {n: pd.read_parquet(activity / f'{n}_used.parquet') for n in ['training', 'diagnosis']}
    for name, frame in rows.items():
        first, last = (pd.Timestamp(t, tz='UTC') for t in ACTIVITY_PERIODS[name])
        y = joint_targets(frame)
        if (len(frame) < (1000 if name == 'training' else 100) or (np.bincount(y, minlength=3) < 20).any()
            or frame.end.min() < first+pd.Timedelta(days=1) or frame.label_end.max() >= last-pd.Timedelta(days=1)
            or frame.end.duplicated().any() or not frame.end.is_monotonic_increasing
            or not np.isfinite(frame[MARKET_FEATURES]).all().all()):
            raise ValueError('결합 진입 진단의 행·클래스·시간 지원 부족')
    sets = [set(v.episode_id) | set(v.target_episode_id) for v in rows.values()]
    if ((sets[0] & sets[1]) - {0}) or rows['training'].label_end.max() >= rows['diagnosis'].end.min():
        raise ValueError('결합 진입 진단의 포지션·시간 중첩')
    active = rows['diagnosis'][rows['diagnosis'].active.eq(1)]
    previous = pd.read_parquet(direction / 'diagnosis_used.parquet')
    pd.testing.assert_frame_equal(active[['end', 'target_episode_id', 'buy']].reset_index(drop=True),
                                  previous[['end', 'target_episode_id', 'buy']].reset_index(drop=True), check_exact=True)
    prior_scores = pd.read_parquet(direction / 'predictions.parquet')
    np.testing.assert_array_equal(d_model.probabilities(active[MARKET_FEATURES].to_numpy()), prior_scores.boosted)
    validation = rows['diagnosis']
    old_activity = pd.read_parquet(activity / 'predictions.parquet')
    pd.testing.assert_frame_equal(validation[['end', 'active']], old_activity[['end', 'active']], check_exact=True)
    np.testing.assert_array_equal(a_model.probabilities(validation[MARKET_FEATURES].to_numpy()), old_activity.logistic)
    scores = {n: factorized_scores(a_model.probabilities(v[MARKET_FEATURES].to_numpy()),
                                  d_model.probabilities(v[MARKET_FEATURES].to_numpy())) for n, v in rows.items()}
    return rows, scores


def run_joint_diagnostics(activity: Path, direction: Path, output: Path) -> Path:
    out = new_run(output, 'joint-entry-diagnosis', {'activity': str(activity), 'direction': str(direction),
        'activity_files_sha256': sha256(activity / 'files.json'), 'direction_files_sha256': sha256(direction / 'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V43.md')), 'new_model_count': 1,
        'candidate_periods': ACTIVITY_PERIODS, 'direction_reference_training_period': ['2018-01-01', '2021-01-01'],
        'trading_returns_evaluated': False})
    print(f'신규 활동·방향의 결합 예측 진단: {out}', flush=True)
    try:
        rows, factorized = joint_reference(activity, direction)
        train, validation = (rows[n] for n in ['training', 'diagnosis'])
        y, vy = joint_targets(train), joint_targets(validation)
        model, support = JointEntryModel.fit(train[MARKET_FEATURES].to_numpy(), y)
        train_scores = model.probabilities(train[MARKET_FEATURES].to_numpy())
        frequencies = np.bincount(y, minlength=3)/len(y)
        scores = {'joint': model.probabilities(validation[MARKET_FEATURES].to_numpy()),
            'factorized': factorized['diagnosis'], 'constant': np.tile(frequencies, (len(validation), 1))}
        thresholds = {'joint': float(np.quantile(1-train_scores[:, 0], .975)),
            'factorized': float(np.quantile(1-factorized['training'][:, 0], .975)),
            'constant': float(1-frequencies[0])}
        predictions = validation[['end', 'episode_id', 'target_episode_id', 'active', 'buy']].copy()
        predictions['target'] = vy
        metrics = {}
        for name, p in scores.items():
            for i, label in enumerate(['hold', 'short', 'long']):
                predictions[name+'_'+label] = p[:, i]
            metrics[name] = joint_metrics(vy, p, thresholds[name])
            print(f'{name}: 로그 손실 {metrics[name]["log_loss"]:.6f}', flush=True)
        monthly = []
        for month, part in predictions.groupby(predictions.end.dt.strftime('%Y-%m')):
            for name in scores:
                p = part[[name+'_'+s for s in ['hold', 'short', 'long']]].to_numpy()
                monthly.append({'month': month, 'model': name, 'rows': len(part),
                    'class_counts': np.bincount(part.target, minlength=3).tolist(),
                    'log_loss': float(log_loss(part.target, p, labels=[0, 1, 2]))})
        decision = joint_admission(metrics)
        for name, value in [('model', model.to_dict()), ('training_support', support), ('metrics', metrics),
            ('thresholds', thresholds), ('decision', decision), ('monthly', monthly)]:
            save_json(out / f'{name}.json', value)
        predictions.to_parquet(out / 'predictions.parquet', index=False)
        save_json(out / 'summary.json', {'complete': True, 'reference_scores_exact': True, **decision})
        (out / 'REPORT.md').write_text('# 신규 활동·방향의 결합 예측 진단\n\n' + table(pd.DataFrame([
            {'model': n, 'log_loss': v['log_loss'], 'activity_log_loss': v['activity']['log_loss'],
             'short_ap': v['short']['average_precision'], 'long_ap': v['long']['average_precision']} for n, v in metrics.items()]))
            + '\n\n2020년 다항 학습을 과거 활동·방향 모형의 분해 결합과 대조했다. '
            '기준 방향 모형의 2018~2020년과 후보의 2020년 학습 범위 차이가 함께 있다. '
            '모든 기간은 이미 관찰했으며 실제 매매나 수익성 입증이 아니다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'후속 결합 진입 허용: {decision["joint_entry_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
