from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .common import new_run, save_json, sha256
from .event_features import MARKET_FEATURES
from .expansion_data import make_expansion_data, source_inputs
from .expansion_model import BinaryModel
from .new_position import new_position_targets
from .reports import table

TRAIN_PERIOD = ['2018-01-01', '2021-01-01']
DIAGNOSIS_PERIOD = ['2021-01-01', '2022-01-01']


def direction_window(data, episodes, period):
    if period not in [TRAIN_PERIOD, DIAGNOSIS_PERIOD]:
        raise ValueError('신규 방향 진단의 사전 고정 기간 오류')
    if episodes.episode_id.duplicated().any() or not data.loc[data.active.eq(1), 'target_episode_id'].isin(episodes.episode_id).all():
        raise ValueError('신규 방향 진단의 목표 에피소드 원장 오류')
    first, last = (pd.Timestamp(t, tz='UTC') for t in period)
    cross = [episodes.loc[episodes.entry_time.lt(t) & (episodes.exit_time.isna() | episodes.exit_time.ge(t)),
                         'episode_id'] for t in [first, last]]
    reason = np.select([data.end.lt(first + pd.Timedelta(days=1)), data.label_end.ge(last - pd.Timedelta(days=1)),
        data.episode_id.isin(cross[0]) | data.target_episode_id.isin(cross[0]),
        data.episode_id.isin(cross[1]) | data.target_episode_id.isin(cross[1]), ~data.usable, data.active.ne(1)],
        ['before_or_left_embargo', 'after_or_right_embargo', 'left_episode_boundary',
         'right_episode_boundary', 'unusable_original_event', 'no_new_position'], default='included')
    ledger = data[['end', 'label_end', 'episode_id', 'target_episode_id', 'active']].assign(reason=reason)
    rows = data[ledger.reason.eq('included')].copy()
    minimum = 1000 if period == TRAIN_PERIOD else 100
    if (len(rows) < minimum or not np.isin(rows.buy, [0, 1]).all()
        or min(rows.buy.eq(0).sum(), rows.buy.eq(1).sum()) < 20
        or rows.end.duplicated().any() or not rows.end.is_monotonic_increasing
        or rows.target_time.isna().any() or rows.target_time.lt(rows.end).any()
        or rows.target_time.ge(rows.label_end).any()):
        raise ValueError('신규 방향 진단의 표본·시각·정답 지원 부족')
    return rows, ledger


def direction_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or labels.shape != scores.shape or set(np.unique(labels)) != {0, 1}
        or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError('신규 방향 진단의 정답·점수 오류')
    long, short = scores >= .65, scores <= .35
    return {'rows': len(labels), 'log_loss': float(log_loss(labels, scores)),
        'brier': float(brier_score_loss(labels, scores)), 'roc_auc': float(roc_auc_score(labels, scores)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, scores >= .5)),
        'actual_buy_fraction': float(labels.mean()), 'predicted_buy_mean': float(scores.mean()),
        'long_decisions': int(long.sum()), 'short_decisions': int(short.sum()),
        'long_correct': int(labels[long].sum()), 'short_correct': int((1 - labels[short]).sum()),
        'abstentions': int((~(long | short)).sum())}


def direction_admission(metrics):
    values = [metrics[k]['log_loss'] for k in ['logistic', 'boosted', 'constant']]
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError('신규 방향 진단의 로그 손실 오류')
    gain = values[0] - values[1]
    # 수치 경계에서는 기존 모형을 유지한다.
    admitted = gain > .01 and not np.isclose(gain, .01, rtol=0, atol=1e-12) and values[1] < values[2]
    return {'boosted_admitted': bool(admitted), 'log_loss_improvement': gain, 'required_improvement': .01,
        'constant_beaten': values[1] < values[2], 'selection_uses_trading_returns': False,
        'profitability_accepted': False, 'all_source_periods_already_observed': True}


def run_direction_diagnostics(audit: Path, study: Path, history: Path, output: Path) -> Path:
    source, hashes = source_inputs(audit, study, history)
    out = new_run(output, 'direction-model-diagnosis', {**hashes, 'audit': str(audit), 'study': str(study),
        'history': str(history), 'training_period': TRAIN_PERIOD, 'diagnosis_period': DIAGNOSIS_PERIOD,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V28.md')), 'model_count': 2,
        'all_source_periods_already_observed': True, 'trading_returns_evaluated': False})
    print(f'신규 방향의 시간순 모형 진단: {out}', flush=True)
    try:
        data, events = new_position_targets(make_expansion_data(source), source['actions'])
        train, train_ledger = direction_window(data, source['episodes'], TRAIN_PERIOD)
        validation, validation_ledger = direction_window(data, source['episodes'], DIAGNOSIS_PERIOD)
        episode_sets = [set(rows.episode_id) | set(rows.target_episode_id) for rows in [train, validation]]
        if (episode_sets[0] & episode_sets[1]) - {0} or train.label_end.max() >= validation.end.min():
            raise ValueError('신규 방향의 학습·진단 에피소드 또는 시간 중첩')
        for name, rows in [('training_used', train), ('diagnosis_used', validation), ('training_ledger', train_ledger),
                           ('diagnosis_ledger', validation_ledger), ('new_position_ledger', events)]:
            rows.to_parquet(out / f'{name}.parquet', index=False)
        x, y = train[MARKET_FEATURES].to_numpy(dtype=float), train.buy.to_numpy(dtype=int)
        vx, vy = validation[MARKET_FEATURES].to_numpy(dtype=float), validation.buy.to_numpy(dtype=int)
        predictions = validation[['end', 'target_episode_id', 'buy']].copy()
        models, support, metrics = {}, {}, {}
        for kind in ['logistic', 'boosted']:
            models[kind], support[kind] = BinaryModel.fit(x, y, kind)
            scores = models[kind].probabilities(vx)
            predictions[kind] = scores
            metrics[kind] = direction_metrics(vy, scores)
            print(f'{kind}: 진단 {len(validation)}개, 로그 손실 {metrics[kind]["log_loss"]:.6f}', flush=True)
        predictions['constant'] = float(y.mean())
        metrics['constant'] = direction_metrics(vy, predictions.constant.to_numpy())
        save_json(out / 'models.json', {k: model.to_dict() for k, model in models.items()})
        save_json(out / 'training_support.json', support)
        save_json(out / 'metrics.json', metrics)
        predictions.to_parquet(out / 'predictions.parquet', index=False)
        monthly = []
        for month, part in predictions.groupby(predictions.end.dt.strftime('%Y-%m')):
            for name in ['logistic', 'boosted', 'constant']:
                monthly.append({'month': month, 'model': name, 'rows': len(part), 'buy': int(part.buy.sum()),
                    'log_loss': float(log_loss(part.buy, part[name], labels=[0, 1]))})
        save_json(out / 'monthly.json', monthly)
        decision = direction_admission(metrics)
        save_json(out / 'decision.json', decision)
        save_json(out / 'summary.json', {'complete': True, 'training_rows': len(train), 'diagnosis_rows': len(validation),
            'episode_intersection': 0, 'training_last_label_end': train.label_end.max(),
            'diagnosis_first_end': validation.end.min(), **decision})
        (out / 'REPORT.md').write_text('# 신규 방향 모형의 시간순 진단\n\n' + table(pd.DataFrame(
            [{'model': k, **v} for k, v in metrics.items()])) + '\n\n모든 기간은 이미 관찰했다. 내부 시간순 행동 예측 진단이며 수익성 검증이 아니다. '
            '2021년 행동 예측 개선 조건을 충족한 경우에만 같은 부스팅 설정의 다음 매매 후보를 허용한다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'후속 부스팅 후보 허용: {decision["boosted_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
