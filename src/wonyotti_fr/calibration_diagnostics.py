from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .event_research import load_selection
from .histogram_management import HistogramManagementModels
from .inventory_labels import verify_files
from .management_calibration import ManagementOffset
from .management_diagnostics import management_admission, management_metrics
from .minute_inventory import MinuteInventoryModels, load_minute_inventory_labels
from .minute_management import ACTIONS, purged_window
from .reports import table

PERIODS = {'training': ('2020-01-01', '2021-01-01'),
           'calibration': ('2021-01-01', '2021-07-01'), 'diagnosis': ('2021-07-01', '2022-01-01')}


def calibration_split(frame, reference):
    rows, support = {}, {}
    for name, period in PERIODS.items():
        part = purged_window(frame, *period).reset_index(drop=True)
        y = part[[f'y_{a}' for a in ACTIONS]].to_numpy()
        if (len(part) < (1000 if name == 'training' else 100)
            or not np.isfinite(part[MinuteInventoryModels.features]).all().all()
            or not np.isin(y, [0, 1]).all() or (y.sum(axis=0) < 20).any() or ((1 - y).sum(axis=0) < 20).any()
            or part.end.duplicated().any() or not part.end.is_monotonic_increasing):
            raise ValueError('시간순 보정 진단의 지원·입력·순서 오류')
        rows[name] = part
        support[name] = {'period': period, 'rows': len(part), 'episodes': int(part.episode_id.nunique()),
            'positive_minutes': dict(zip(ACTIONS, y.sum(axis=0).astype(int).tolist(), strict=True)),
            'independent_orders': {a: int(part[f'{a}_count'].sum()) for a in ACTIONS},
            'first_end': part.end.min(), 'last_label_end': part.label_end.max()}
    for first, second in [('training', 'calibration'), ('training', 'diagnosis'), ('calibration', 'diagnosis')]:
        if set(rows[first].episode_id) & set(rows[second].episode_id) or rows[first].label_end.max() >= rows[second].end.min():
            raise ValueError('시간순 보정 진단의 포지션·시각 중첩')
    pd.testing.assert_frame_equal(rows['diagnosis'], pd.read_parquet(reference / 'diagnosis_used.parquet'), check_exact=True)
    return rows, support


def calibration_admission(metrics):
    comparisons = {}
    for baseline in ['original_v21', 'logistic_calibrated']:
        mapped = {a: {'logistic': metrics[a][baseline], 'histogram': metrics[a]['histogram_calibrated'],
                      'constant': metrics[a]['constant']} for a in ACTIONS}
        comparisons[baseline] = management_admission(mapped)
    raw_preserved = {}
    for action in ACTIONS:
        raw, calibrated = (metrics[action][k]['log_loss'] for k in ['histogram_raw', 'histogram_calibrated'])
        if not np.isfinite([raw, calibrated]).all() or min(raw, calibrated) < 0:
            raise ValueError('보정 전후 로그 손실의 범위 오류')
        raw_preserved[action] = calibrated <= raw
    return {'calibrated_histogram_admitted': all(v['histogram_admitted'] for v in comparisons.values()) and all(raw_preserved.values()),
            'comparisons': comparisons, 'raw_histogram_loss_preserved': raw_preserved,
            'profitability_accepted': False, 'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}


def run_calibration_diagnostics(selection: Path, labels: Path, diagnosis: Path, output: Path, *, _history_context=None) -> Path:
    frozen, original = load_selection(selection)
    if frozen['protocol'] != 'minute_inventory_micro_v21':
        raise ValueError('시간순 보정 진단에는 고정 v21 모델이 필요합니다.')
    previous = json.loads((diagnosis / 'manifest.json').read_text())['settings']
    if (previous['reference_sha256'] != sha256(selection / 'frozen_selection.json')
        or previous['labels_files_sha256'] != sha256(labels / 'files.json')):
        raise ValueError('시간순 보정 진단의 기존 입력 지문 오류')
    verify_files(diagnosis, list(json.loads((diagnosis / 'files.json').read_text())))
    context = _history_context
    model_class = context['model_class'] if context else MinuteInventoryModels
    boost_class = context['boost_class'] if context else HistogramManagementModels
    protocol = 'docs/EXPERIMENT_V35.md' if context else 'docs/EXPERIMENT_V32.md'
    out = new_run(output, 'history-calibration-diagnosis' if context else 'management-calibration-diagnosis', {'selection': str(selection), 'labels': str(labels),
        'diagnosis': str(diagnosis), 'selection_sha256': sha256(selection / 'frozen_selection.json'),
        'labels_files_sha256': sha256(labels / 'files.json'), 'diagnosis_files_sha256': sha256(diagnosis / 'files.json'),
        'protocol_sha256': sha256(Path(protocol)), 'periods': PERIODS,
        'features': model_class.features, **(context['metadata'] if context else {}),
        'all_source_periods_already_observed': True, 'trading_returns_evaluated': False})
    print(f'관리 빈도의 시간순 보정 진단: {out}', flush=True)
    try:
        frame, _ = load_minute_inventory_labels(labels)
        rows, support = calibration_split(frame, diagnosis)
        del frame
        if context:
            from .history_calibration_diagnostics import attach_history_splits
            rows = attach_history_splits(rows, context)
        for name, part in rows.items():
            part.to_parquet(out / f'{name}_used.parquet', index=False)
        train, calibration, validation = (rows[n] for n in PERIODS)
        logistic, _, log_support = model_class.fit(train, calibration, 'logistic')
        x, cx, vx = (rows[n][model_class.features].to_numpy(dtype=float) for n in PERIODS)
        y, cy = (rows[n][[f'y_{a}' for a in ACTIONS]].to_numpy(dtype=int) for n in ['training', 'calibration'])
        histogram, hist_support = boost_class.fit(x, y, np.vstack([cx, vx]))
        del x
        models, offsets, offset_support = {}, {}, {}
        scores = {'original_v21': original.manager.probabilities(validation[MinuteInventoryModels.features].to_numpy()),
                  'constant': np.tile(cy.mean(axis=0), (len(vx), 1))}
        for name, model in [('logistic', logistic), ('histogram', histogram)]:
            offset, checks = ManagementOffset.fit(model.probabilities(cx), cy)
            models[name], offsets[name], offset_support[name] = model.to_dict(), offset.to_dict(), checks
            scores[name + '_raw'] = model.probabilities(vx)
            scores[name + '_calibrated'] = offset.predict(scores[name + '_raw'])
        old_scores = pd.read_parquet(diagnosis / 'predictions.parquet')
        columns = ['end', 'episode_id', *[f'y_{a}' for a in ACTIONS]]
        pd.testing.assert_frame_equal(old_scores[columns], validation[columns], check_exact=True)
        np.testing.assert_array_equal(scores['original_v21'], old_scores[[a + '_logistic' for a in ACTIONS]].to_numpy())
        if context:
            previous_scores = context['previous_predictions']
            pd.testing.assert_frame_equal(previous_scores[columns], validation[columns], check_exact=True)
            np.testing.assert_array_equal(scores['original_v21'], previous_scores[[a + '_original_v21' for a in ACTIONS]].to_numpy())
            scores['previous_calibrated'] = previous_scores[[a + '_histogram_calibrated' for a in ACTIONS]].to_numpy()
        predictions = validation[['end', 'episode_id', *[f'y_{a}' for a in ACTIONS]]].copy()
        metrics = {}
        for i, action in enumerate(ACTIONS):
            metrics[action] = {}
            for name, values in scores.items():
                predictions[f'{action}_{name}'] = values[:, i]
                metrics[action][name] = management_metrics(validation[f'y_{action}'], values[:, i])
            for name in ['logistic', 'histogram']:
                if abs(metrics[action][name + '_raw']['average_precision'] - metrics[action][name + '_calibrated']['average_precision']) > 1e-12:
                    raise ValueError('절편 보정의 예측 순위 보존 불일치')
            print(f'{action}: 기존 {metrics[action]["original_v21"]["log_loss"]:.6f}, 보정 부스팅 {metrics[action]["histogram_calibrated"]["log_loss"]:.6f}', flush=True)
        for name, value in [('models', models), ('offsets', offsets), ('split_support', support), ('metrics', metrics),
            ('training_support', {'logistic': log_support, 'histogram': hist_support, 'offsets': offset_support})]:
            save_json(out / f'{name}.json', value)
        predictions.to_parquet(out / 'predictions.parquet', index=False)
        decision = calibration_admission(metrics)
        if context:
            from .history_calibration_diagnostics import history_calibration_admission
            decision = history_calibration_admission(metrics)
        save_json(out / 'decision.json', decision)
        save_json(out / 'summary.json', {'complete': True, 'episode_intersection': 0,
            'original_predictions_exact': True, 'diagnosis_rows_exact': True, 'rank_preserved': True, **decision})
        values = [{'action': a, 'model': k, **v} for a, group in metrics.items() for k, v in group.items()]
        (out / 'REPORT.md').write_text('# 관리 행동 빈도의 시간순 보정 진단\n\n' + table(pd.DataFrame(values))
            + '\n\n2020년 학습·2021년 상반기 절편 보정·하반기 진단을 분리했다. 기존 18개월 학습을 12개월 학습과 6개월 보정으로 바꾼 차이도 포함한다. '
            '기울기와 예측 순위는 보존하며 보정 구간의 평균 일치는 이후 조건부 확률의 보장이 아니다. '
            '세 행동 모두의 기준을 요구하며 일부만 고르지 않는다. 이미 관찰한 자료이며 매매 수익성 진단이 아니다.\n')
        if context:
            with (out / 'REPORT.md').open('a') as stream:
                stream.write('\n기존 v32 분할의 모든 열·정답을 유지하고 검증된 과거 주문 입력 여섯 개를 더했다. '
                    '기존 v21·새 보정 로지스틱·v32 보정 부스팅 세 기준 모두를 넘는 조건이며 이전 실패 판단은 변경하지 않았다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'후속 연구 허용: {decision["calibrated_histogram_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
