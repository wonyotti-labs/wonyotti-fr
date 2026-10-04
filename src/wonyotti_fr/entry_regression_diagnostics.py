from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .entry_regression import REGRESSION_SETTINGS, EntryRegressionModel
from .event_features import MARKET_FEATURES
from .event_research import load_selection
from .label_weighting import lifecycle_weights
from .net_edge_model import NetEdgeModel, net_values
from .policy_entry import load_policy_outcome_training

ENTRY_SPLITS = {'training': ['2021-01-01', '2021-09-30'], 'diagnosis': ['2021-10-02', '2021-12-31']}


def regression_splits(ledger):
    closed = ledger.label_status.eq('closed')
    if (ledger.decision_time.isna().any() or ledger.decision_time.duplicated().any()
        or not ledger.decision_time.is_monotonic_increasing
        or ledger.loc[closed, 'label_end'].isna().any()
        or ledger.loc[closed, 'decision_time'].ge(ledger.loc[closed, 'label_end']).any()):
        raise ValueError('진입 회귀의 원장 시각·중복·순서 오류')
    assignment = ledger.copy()
    assignment['split'] = np.where(closed, 'excluded_boundary', 'excluded_not_closed')
    rows = {}
    for name, (start, end) in ENTRY_SPLITS.items():
        mask = closed & ledger.decision_time.ge(pd.Timestamp(start, tz='UTC')) & ledger.label_end.lt(pd.Timestamp(end, tz='UTC'))
        frame = ledger.loc[mask].reset_index(drop=True)
        counts = frame.order_direction.value_counts()
        if (len(frame) < (200 if name == 'training' else 100)
            or any(counts.get(side, 0) < 20 for side in [-1, 1])
            or frame.decision_time.duplicated().any() or not frame.decision_time.is_monotonic_increasing
            or frame.decision_time.ge(frame.label_end).any()
            or not np.isfinite(frame.net_bps).all()
            or (name == 'training' and frame.decision_time.max()-frame.decision_time.min() < pd.Timedelta(days=180))):
            raise ValueError('진입 회귀 진단의 시간·방향·표본 지원 부족')
        assignment.loc[mask, 'split'] = name
        rows[name] = frame
    if rows['training'].label_end.max() >= rows['diagnosis'].decision_time.min():
        raise ValueError('진입 회귀의 학습·진단 정답 교차')
    return rows, assignment


def regression_metrics(target, prediction, weight):
    target, prediction, weight = (np.asarray(v, dtype=float) for v in [target, prediction, weight])
    if (target.ndim != 1 or not len(target) or prediction.shape != target.shape or weight.shape != target.shape
        or not all(np.isfinite(v).all() for v in [target, prediction, weight]) or (weight <= 0).any()):
        raise ValueError('진입 회귀 지표의 행·예측·가중치 오류')
    selected = prediction >= 8
    return {'rows': len(target), 'weighted_mse': float(np.average((target-prediction)**2, weights=weight)),
        'mse': float(np.mean((target-prediction)**2)), 'selected': int(selected.sum()),
        'selected_weighted_mean_bps': float(np.average(target[selected], weights=weight[selected])) if selected.any() else None,
        'selected_mean_bps': float(target[selected].mean()) if selected.any() else None,
        'mean_predicted_bps': float(prediction.mean()), 'mean_actual_bps': float(target.mean())}


def regression_admission(metrics):
    new, old, constant = (metrics[k] for k in ['boosted', 'ridge', 'constant'])
    if len({v['rows'] for v in metrics.values()}) != 1:
        raise ValueError('진입 회귀 진단의 비교 행 수 불일치')
    checks = {'weighted_mse_vs_ridge': new['weighted_mse'] < old['weighted_mse']*.99,
        'weighted_mse_vs_constant': new['weighted_mse'] < constant['weighted_mse']*.99,
        'unweighted_mse_not_worse': new['mse'] <= old['mse']+1e-9,
        'at_least_30_selected': new['selected'] >= 30,
        'positive_selected_weighted_mean': new['selected_weighted_mean_bps'] is not None and new['selected_weighted_mean_bps'] > 0,
        'positive_selected_mean': new['selected_mean_bps'] is not None and new['selected_mean_bps'] > 0}
    return {'checks': checks, 'boosted_admitted': all(checks.values()), 'trading_returns_evaluated': False}


def run_entry_regression_diagnosis(selection: Path, output: Path) -> Path:
    frozen, _ = load_selection(selection)
    if frozen['protocol'] != 'policy_entry_v56':
        raise ValueError('진입 회귀 진단에는 고정 v56 선택이 필요합니다.')
    settings = json.loads((selection/'manifest.json').read_text())['settings']
    labels, reference = Path(settings['labels']), Path(settings['reference'])
    source_settings = json.loads((labels/'manifest.json').read_text())['settings']
    market, features = Path(source_settings['market']), Path(source_settings['features'])
    out = new_run(output, 'entry-regression-diagnosis', {'selection': str(selection), 'labels': str(labels),
        'selection_sha256': sha256(selection/'frozen_selection.json'), 'labels_files_sha256': sha256(labels/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V57.md')), 'periods': ENTRY_SPLITS,
        'model_count': 2, 'settings': REGRESSION_SETTINGS, 'margin_bps': 8,
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'진입 손익 비선형 진단: {out}', flush=True)
    try:
        train_all, _, weights_all, weighting_all, _ = load_policy_outcome_training(reference, labels, market, features)
        pd.testing.assert_frame_equal(train_all, pd.read_parquet(selection/'entry_training_used.parquet'), check_exact=True)
        pd.testing.assert_frame_equal(weights_all, pd.read_parquet(selection/'entry_training_weights.parquet'), check_exact=True)
        if (json.loads((selection/'entry_weighting.json').read_text()) != weighting_all
            or settings['labels_files_sha256'] != sha256(labels/'files.json')
            or frozen['exit_move_selection_sha256'] != sha256(reference/'frozen_selection.json')):
            raise ValueError('진입 회귀 진단의 현재 정책 정답·가중치 연결 오류')
        rows, assignments = regression_splits(pd.read_parquet(labels/'opportunity_ledger.parquet'))
        assignments.to_parquet(out/'exclusion_ledger.parquet', index=False)
        weights, values, supports = {}, {}, {}
        for name, frame in rows.items():
            frame.to_parquet(out/f'{name}_used.parquet', index=False)
            weights[name], intervals, supports[name] = lifecycle_weights(frame)
            intervals.to_parquet(out/f'{name}_weights.parquet', index=False)
            values[name] = net_values(frame[MARKET_FEATURES], frame.order_direction, frame.favorable_bps, frame.wait_minutes)
        ridge, ridge_support = NetEdgeModel.fit(rows['training'], 100, sample_weight=weights['training'])
        boosted, boost_support = EntryRegressionModel.fit(values['training'], rows['training'].net_bps,
                                                          weights['training'], values['diagnosis'])
        constant = float(np.average(rows['training'].net_bps, weights=weights['training']))
        save_json(out/'models.json', {'ridge': ridge.to_dict(), 'boosted': boosted.to_dict(), 'constant': constant})
        save_json(out/'training_support.json', {'ridge': ridge_support, 'boosted': boost_support,
            'weighting': supports, 'excluded': assignments.split.value_counts().to_dict()})
        predicted = {'ridge': ridge.predict(values['diagnosis']), 'boosted': boosted.predict(values['diagnosis']),
                     'constant': np.full(len(rows['diagnosis']), constant)}
        frame = rows['diagnosis'][['decision_time', 'label_end', 'order_direction', 'net_bps']].copy()
        frame['sample_weight'] = weights['diagnosis']
        for name, prediction in predicted.items():
            frame[f'predicted_{name}'] = prediction
        frame.to_parquet(out/'predictions.parquet', index=False)
        metrics = {k: regression_metrics(frame.net_bps, v, frame.sample_weight) for k, v in predicted.items()}
        decision = regression_admission(metrics)
        details = []
        groups = [('direction', str(k), part) for k, part in frame.groupby('order_direction')]
        groups += [('month', k, part) for k, part in frame.groupby(frame.decision_time.dt.strftime('%Y-%m'))]
        for kind, group, part in groups:
            for name in predicted:
                details.append({'kind': kind, 'group': group, 'model': name,
                                **regression_metrics(part.net_bps, part[f'predicted_{name}'], part.sample_weight)})
        save_json(out/'metrics.json', metrics)
        save_json(out/'breakdown.json', details)
        save_json(out/'decision.json', decision)
        save_json(out/'summary.json', {'complete': True, 'training_rows': len(rows['training']),
            'diagnosis_rows': len(rows['diagnosis']), 'excluded_states_preserved': True,
            'profitability_accepted': False, **decision})
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        (out/'REPORT.md').write_text('# 신규 진입 손익의 비선형 예측 진단\n\n'
            f'시간순 진단 조건 통과: {decision["boosted_admitted"]}. '
            '같은 앞 구간에서 가중 회귀·단일 부스팅·상수를 비교했다. 후속 정답은 학습·가중치·반복 수에 쓰지 않았다. '
            '관리 정책과 전체 시스템은 이미 이 기간을 사용했다. 실제 매매·수익성 채택은 별도 검증이다.\n')
        print(json.dumps(decision, ensure_ascii=False), flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
