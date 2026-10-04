from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .common import new_run, save_json, sha256
from .event_features import MARKET_FEATURES
from .expansion_data import make_expansion_data, source_inputs
from .expansion_model import BinaryModel
from .histogram_management import HistogramManagementModels
from .management_diagnostics import management_metrics
from .new_position import new_position_targets
from .reports import table

ACTIVITY_PERIODS = {'training': ['2020-01-01', '2021-01-01'],
                    'diagnosis': ['2021-01-01', '2022-01-01']}


class ActivityHistogramModels(HistogramManagementModels):
    features = MARKET_FEATURES
    actions = ['active']
    format = 'new_position_activity_histogram_v1'


def activity_window(data, episodes, name):
    if name not in ACTIVITY_PERIODS:
        raise ValueError('신규 활동 진단의 사전 기간 오류')
    if (episodes.episode_id.duplicated().any()
        or not np.isin(data.active, [0, 1]).all()
        or not data.loc[data.active.eq(1), 'target_episode_id'].isin(episodes.episode_id).all()):
        raise ValueError('신규 활동 진단의 포지션·정답 원장 오류')
    first, last = (pd.Timestamp(v, tz='UTC') for v in ACTIVITY_PERIODS[name])
    cross = [episodes.loc[episodes.entry_time.lt(t) & (episodes.exit_time.isna() | episodes.exit_time.ge(t)),
                          'episode_id'] for t in [first, last]]
    reason = np.select([data.end.lt(first+pd.Timedelta(days=1)), data.label_end.ge(last-pd.Timedelta(days=1)),
        data.episode_id.isin(cross[0]) | data.target_episode_id.isin(cross[0]),
        data.episode_id.isin(cross[1]) | data.target_episode_id.isin(cross[1]), ~data.usable],
        ['before_or_left_embargo', 'after_or_right_embargo', 'left_episode_boundary',
         'right_episode_boundary', 'unusable_original_event'], default='included')
    ledger = data[['end', 'label_end', 'episode_id', 'target_episode_id', 'active']].assign(reason=reason)
    rows = data[ledger.reason.eq('included')].copy()
    positive = rows[rows.active.eq(1)]
    if (len(rows) < (1000 if name == 'training' else 100)
        or min(rows.active.sum(), (1-rows.active).sum()) < 20
        or rows.end.isna().any() or rows.label_end.isna().any()
        or not rows.end.is_monotonic_increasing or rows.end.duplicated().any()
        or not np.isfinite(rows[MARKET_FEATURES]).all().all()
        or not rows.label_end.sub(rows.end).eq(pd.Timedelta(minutes=5)).all()
        or positive.target_time.isna().any() or positive.target_time.lt(positive.end).any()
        or positive.target_time.ge(positive.label_end).any()
        or rows.loc[rows.active.eq(0), 'target_time'].notna().any()):
        raise ValueError('신규 활동 진단의 표본·시각·입력 지원 부족')
    return rows, ledger


def fit_activity_models(train, validation):
    x, vx = (v[MARKET_FEATURES].to_numpy(dtype=float) for v in [train, validation])
    y = train.active.to_numpy(dtype=int)
    logistic, logistic_support = BinaryModel.fit(x, y, 'logistic')
    histogram, histogram_support = ActivityHistogramModels.fit(x, y[:, None], vx)
    thresholds = {'logistic': float(np.quantile(logistic.probabilities(x), .975)),
                  'histogram': float(np.quantile(histogram.probabilities(x)[:, 0], .975))}
    return {'logistic': logistic, 'histogram': histogram}, thresholds, {'logistic': logistic_support, 'histogram': histogram_support}


def activity_metrics(labels, scores, threshold):
    if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('신규 활동 진단의 고정 문턱 오류')
    result = management_metrics(labels, scores)
    y, requested = np.asarray(labels).astype(bool), np.asarray(scores) >= threshold
    tp, fp = int((y & requested).sum()), int((~y & requested).sum())
    fn, tn = int((y & ~requested).sum()), int((~y & ~requested).sum())
    return {**result, 'threshold': threshold, 'true_positive': tp, 'false_positive': fp,
        'false_negative': fn, 'true_negative': tn, 'precision': tp/(tp+fp) if tp+fp else 0.,
        'recall': tp/(tp+fn), 'requested': tp+fp}


def activity_admission(metrics):
    old, new, constant = (metrics[n] for n in ['logistic', 'histogram', 'constant'])
    values = [old['log_loss'], new['log_loss'], constant['log_loss'], old['average_precision'], new['average_precision']]
    if (not np.isfinite(values).all() or min(values[:3]) < 0
        or not all(0 <= v <= 1 for v in values[3:])):
        raise ValueError('신규 활동 진단의 판정 지표 오류')
    gain = 1-new['log_loss']/old['log_loss'] if old['log_loss'] > 0 else 0.
    checks = {'relative_log_loss_gain': gain,
        'gain_passed': bool(gain > .01 and not np.isclose(gain, .01, atol=1e-12, rtol=0)),
        'constant_beaten': new['log_loss'] < constant['log_loss'],
        'average_precision_preserved': new['average_precision'] >= old['average_precision']}
    return {'activity_histogram_admitted': all(checks[k] for k in ['gain_passed', 'constant_beaten', 'average_precision_preserved']),
        'checks': checks, 'required_relative_gain': .01, 'profitability_accepted': False,
        'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}


def run_activity_diagnostics(audit: Path, study: Path, history: Path, output: Path) -> Path:
    source, hashes = source_inputs(audit, study, history)
    out = new_run(output, 'activity-model-diagnosis', {**hashes, 'audit': str(audit), 'study': str(study),
        'history': str(history), 'periods': ACTIVITY_PERIODS, 'training_quantile': .975,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V42.md')), 'model_count': 2,
        'all_source_periods_already_observed': True, 'trading_returns_evaluated': False})
    print(f'신규 진입 활동의 시간순 모형 진단: {out}', flush=True)
    try:
        data, events = new_position_targets(make_expansion_data(source), source['actions'])
        train, training_ledger = activity_window(data, source['episodes'], 'training')
        validation, diagnosis_ledger = activity_window(data, source['episodes'], 'diagnosis')
        sets = [set(v.episode_id) | set(v.target_episode_id) for v in [train, validation]]
        if ((sets[0] & sets[1]) - {0}) or train.label_end.max() >= validation.end.min():
            raise ValueError('신규 활동 학습·진단의 포지션·시각 중첩')
        for name, frame in [('training_used', train), ('diagnosis_used', validation),
            ('training_ledger', training_ledger), ('diagnosis_ledger', diagnosis_ledger), ('new_position_ledger', events)]:
            frame.to_parquet(out / f'{name}.parquet', index=False)
        models, thresholds, support = fit_activity_models(train, validation)
        vx = validation[MARKET_FEATURES].to_numpy()
        scores = {'logistic': models['logistic'].probabilities(vx),
            'histogram': models['histogram'].probabilities(vx)[:, 0],
            'constant': np.full(len(validation), train.active.mean())}
        thresholds['constant'] = float(train.active.mean())
        predictions = validation[['end', 'episode_id', 'target_episode_id', 'active']].copy()
        metrics = {}
        for name, values in scores.items():
            predictions[name] = values
            metrics[name] = activity_metrics(validation.active, values, thresholds[name])
            print(f'{name}: 로그 손실 {metrics[name]["log_loss"]:.6f}, 평균 정밀도 {metrics[name]["average_precision"]:.6f}', flush=True)
        decision = activity_admission(metrics)
        monthly = []
        for month, part in predictions.groupby(predictions.end.dt.strftime('%Y-%m')):
            for name in scores:
                monthly.append({'month': month, 'model': name, 'rows': len(part), 'positive': int(part.active.sum()),
                    'log_loss': float(log_loss(part.active, part[name], labels=[0, 1])),
                    'requested': int(part[name].ge(thresholds[name]).sum())})
        for name, value in [('models', {n: m.to_dict() for n, m in models.items()}), ('training_support', support),
            ('thresholds', {'training_quantile': .975, 'thresholds': thresholds}), ('metrics', metrics),
            ('decision', decision), ('monthly', monthly)]:
            save_json(out / f'{name}.json', value)
        predictions.to_parquet(out / 'predictions.parquet', index=False)
        save_json(out / 'summary.json', {'complete': True, 'training_rows': len(train),
            'diagnosis_rows': len(validation), 'episode_intersection': 0, **decision})
        (out / 'REPORT.md').write_text('# 신규 진입 활동의 시간순 모형 진단\n\n' + table(pd.DataFrame([
            {'model': n, **m} for n, m in metrics.items()]))
            + '\n\n2020년 학습·2021년 진단과 원본 수량 전이 정답·14개 시장 입력을 고정했다. '
            '각 문턱은 학습 점수의 같은 분위로 산출했다. 대부분 보유 중 반전인 원본과 무포지션 진입의 차이가 남으며 '
            '이 결과는 이미 관찰한 행동 예측 진단이다. 수익성이나 양방향 실행 균형의 입증이 아니다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'후속 활동 모형 허용: {decision["activity_histogram_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
