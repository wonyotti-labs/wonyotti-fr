from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import new_run, save_json, sha256
from .entry_regression_diagnostics import (
    ENTRY_SPLITS,
    regression_metrics,
    run_entry_regression_diagnosis,
)
from .event_features import MARKET_FEATURES
from .histogram_management import HISTOGRAM_SETTINGS, HistogramManagementModels
from .net_edge_model import NET_FEATURES, net_values

REGRESSION_FILES = {'manifest.json', 'models.json', 'summary.json', 'decision.json', 'metrics.json',
    'breakdown.json', 'training_support.json', 'training_used.parquet', 'diagnosis_used.parquet',
    'training_weights.parquet', 'diagnosis_weights.parquet', 'exclusion_ledger.parquet', 'predictions.parquet'}
NATURAL_EXITS = {'intrabar_stop', 'gap_stop', 'time_limit', 'risk_halt', 'manual_halt',
    'signal_exit', 'signal_reduce', 'signal_reverse', 'liquidity_intrabar_stop', 'liquidity_gap_stop',
    'liquidity_time_limit', 'liquidity_risk_halt', 'liquidity_manual_halt'}


class EntryStopModel(HistogramManagementModels):
    features = NET_FEATURES
    actions = ['intrabar_stop']
    format = 'entry_stop_histogram_v1'


def reproduce_regression(source: Path, output: Path) -> Path:
    hashes = json.loads((source/'files.json').read_text())
    if (set(hashes) != REGRESSION_FILES or (source/'files.json').is_symlink()
        or any((source/n).is_symlink() or sha256(source/n) != h for n, h in hashes.items())):
        raise ValueError('손절 위험 진단의 이전 출력·지문 오류')
    settings = json.loads((source/'manifest.json').read_text())['settings']
    if (settings['periods'] != ENTRY_SPLITS or settings['model_count'] != 2 or settings['margin_bps'] != 8
        or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V57.md'))
        or settings['trading_returns_evaluated'] is not False or settings['whole_system_periods_already_observed'] is not True):
        raise ValueError('손절 위험 진단의 이전 기간·가정 오류')
    selection = Path(settings['selection'])
    if (settings['selection_sha256'] != sha256(selection/'frozen_selection.json')
        or settings['labels_files_sha256'] != sha256(Path(settings['labels'])/'files.json')):
        raise ValueError('손절 위험 진단의 이전 정책·정답 연결 오류')
    reproduced = run_entry_regression_diagnosis(selection, output)
    for name in REGRESSION_FILES-{'manifest.json'}:
        if name.endswith('.parquet'):
            pd.testing.assert_frame_equal(pd.read_parquet(source/name), pd.read_parquet(reproduced/name), check_exact=True)
        elif json.loads((source/name).read_text()) != json.loads((reproduced/name).read_text()):
            raise ValueError('손절 위험 진단의 원래 회귀 재현 불일치')
    return reproduced


def stop_inputs(frame, weights, *, training):
    if (frame.label_status.ne('closed').any() or not frame.exit_reason.isin(NATURAL_EXITS).all()
        or len(frame) < (1000 if training else 100)):
        raise ValueError('손절 위험의 종료 사유·상태·표본 오류')
    stop = frame.exit_reason.eq('intrabar_stop').to_numpy(dtype=int)
    if min(np.bincount(stop, minlength=2)) < (64 if training else 20):
        raise ValueError('손절 위험의 양쪽 목표 지원 부족')
    x = net_values(frame[MARKET_FEATURES], frame.order_direction, frame.favorable_bps, frame.wait_minutes)
    y, w = frame.net_bps.to_numpy(dtype=float), np.asarray(weights, dtype=float)
    if (w.shape != (len(frame),) or not all(np.isfinite(v).all() for v in [x, y, w])
        or (w <= 0).any() or not np.isclose(w.mean(), 1., rtol=0, atol=1e-12)):
        raise ValueError('손절 위험의 입력·손익·가중치 오류')
    return x, y, w, stop


def stop_expected_bps(probability, means):
    p = np.asarray(probability, dtype=float)
    if (p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
        or set(means) != {'intrabar_stop', 'non_stop'}
        or any(type(v) not in (int, float) or not np.isfinite(v) for v in means.values())):
        raise ValueError('손절 위험의 확률·조건부 평균 오류')
    return p*means['intrabar_stop']+(1-p)*means['non_stop']


def stop_probability_metrics(target, probability, weights):
    y, p, w = (np.asarray(v, dtype=float) for v in [target, probability, weights])
    if (y.ndim != 1 or not len(y) or p.shape != y.shape or w.shape != y.shape
        or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
        or not np.isfinite(w).all() or (w <= 0).any()):
        raise ValueError('손절 확률 지표의 정답·확률·가중치 오류')
    safe = np.clip(p, 1e-15, 1-1e-15)
    loss = -(y*np.log(safe)+(1-y)*np.log1p(-safe))
    return {'rows': len(y), 'stops': int(y.sum()), 'weighted_log_loss': float(np.average(loss, weights=w)),
        'weighted_brier': float(np.average((y-p)**2, weights=w)),
        'weighted_actual_stop_rate': float(np.average(y, weights=w)),
        'weighted_predicted_stop_rate': float(np.average(p, weights=w))}


def stop_admission(metrics, probability_metrics):
    candidate = metrics['stop_mixture']
    if len({v['rows'] for v in [*metrics.values(), *probability_metrics.values()]}) != 1:
        raise ValueError('손절 위험의 비교 행 수 불일치')
    checks = {f'weighted_mse_vs_{name}': candidate['weighted_mse'] < metrics[name]['weighted_mse']*.99
              for name in ['ridge', 'boosted', 'constant']}
    checks.update({f'unweighted_mse_vs_{name}': candidate['mse'] <= metrics[name]['mse']+1e-9
                   for name in ['ridge', 'boosted']})
    p, constant = probability_metrics['stop_mixture'], probability_metrics['constant']
    checks.update(at_least_30_selected=candidate['selected'] >= 30,
        positive_selected_weighted_mean=candidate['selected_weighted_mean_bps'] is not None and candidate['selected_weighted_mean_bps'] > 0,
        positive_selected_mean=candidate['selected_mean_bps'] is not None and candidate['selected_mean_bps'] > 0,
        weighted_log_loss_vs_constant=p['weighted_log_loss'] < constant['weighted_log_loss']*.99,
        weighted_brier_not_worse=p['weighted_brier'] <= constant['weighted_brier']+1e-12)
    return {'checks': checks, 'stop_risk_admitted': all(checks.values()), 'trading_returns_evaluated': False}


def run_entry_stop_diagnosis(diagnosis: Path, output: Path) -> Path:
    out = new_run(output, 'entry-stop-diagnosis', {'diagnosis': str(diagnosis),
        'diagnosis_files_sha256': sha256(diagnosis/'files.json'), 'periods': ENTRY_SPLITS,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V58.md')), 'settings': HISTOGRAM_SETTINGS,
        'new_models': 1, 'margin_bps': 8, 'trading_returns_evaluated': False,
        'whole_system_periods_already_observed': True})
    print(f'신규 진입의 손절 위험 분리 진단: {out}', flush=True)
    try:
        reproduced = reproduce_regression(diagnosis, out/'reference-reproduction')
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_models_predictions_rows_weights_exact': True,
            'reproduced_files_sha256': sha256(reproduced/'files.json')})
        parts = {}
        for name in ['training', 'diagnosis']:
            frame = pd.read_parquet(diagnosis/f'{name}_used.parquet')
            weight = pd.read_parquet(diagnosis/f'{name}_weights.parquet')
            parts[name] = stop_inputs(frame, weight.sample_weight, training=name == 'training')
            for suffix in ['used', 'weights']:
                (out/f'{name}_{suffix}.parquet').write_bytes((diagnosis/f'{name}_{suffix}.parquet').read_bytes())
        (out/'exclusion_ledger.parquet').write_bytes((diagnosis/'exclusion_ledger.parquet').read_bytes())
        x, y, w, stop = parts['training']
        vx, vy, vw, vstop = parts['diagnosis']
        means = {'intrabar_stop': float(np.average(y[stop == 1], weights=w[stop == 1])),
                 'non_stop': float(np.average(y[stop == 0], weights=w[stop == 0]))}
        prior = float(np.average(stop, weights=w))
        if not np.isclose(stop_expected_bps([prior], means)[0], np.average(y, weights=w), rtol=0, atol=1e-10):
            raise ValueError('손절 위험의 조건부 평균 분해 불일치')
        model, support = EntryStopModel.fit(x, stop[:, None], vx, sample_weight=w)
        save_json(out/'model.json', model.to_dict())
        save_json(out/'conditional_means.json', {'means_bps': means, 'weighted_stop_prior': prior,
            'weighted_target_mean_bps': float(np.average(y, weights=w))})
        save_json(out/'training_support.json', support)
        prediction = pd.read_parquet(diagnosis/'predictions.parquet')
        prediction['actual_intrabar_stop'] = vstop
        probability = model.probabilities(vx)[:, 0]
        prediction['probability_intrabar_stop'] = probability
        prediction['predicted_stop_mixture'] = stop_expected_bps(probability, means)
        prediction.to_parquet(out/'predictions.parquet', index=False)
        metrics = json.loads((diagnosis/'metrics.json').read_text())
        metrics['stop_mixture'] = regression_metrics(vy, prediction.predicted_stop_mixture, vw)
        probability_metrics = {'stop_mixture': stop_probability_metrics(vstop, probability, vw),
                               'constant': stop_probability_metrics(vstop, np.full(len(vstop), prior), vw)}
        decision = stop_admission(metrics, probability_metrics)
        breakdown = []
        groups = [('direction', str(k), part) for k, part in prediction.groupby('order_direction')]
        groups += [('month', k, part) for k, part in prediction.groupby(prediction.decision_time.dt.strftime('%Y-%m'))]
        for kind, group, part in groups:
            amounts = {name: regression_metrics(part.net_bps, part[f'predicted_{name}'], part.sample_weight) for name in metrics}
            probabilities = {'stop_mixture': stop_probability_metrics(part.actual_intrabar_stop, part.probability_intrabar_stop, part.sample_weight),
                'constant': stop_probability_metrics(part.actual_intrabar_stop, np.full(len(part), prior), part.sample_weight)}
            breakdown.append({'kind': kind, 'group': group, 'amount_metrics': amounts, 'probability_metrics': probabilities})
        for name, content in [('metrics', metrics), ('probability_metrics', probability_metrics), ('breakdown', breakdown), ('decision', decision)]:
            save_json(out/f'{name}.json', content)
        save_json(out/'summary.json', {'complete': True, 'training_rows': len(x), 'diagnosis_rows': len(vx),
            'all_previous_outputs_reproduced': True, 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 신규 진입의 손절 위험 분리 진단\n\n'
            f'사전 진단 조건 통과: {decision["stop_risk_admitted"]}. '
            '기존 전체 행·입력·구간별 가중치를 보존했다. 손절 확률과 앞 구간의 두 조건부 평균을 혼합했다. '
            '진단 종료 사유는 입력에 사용하지 않았다. 집단 안 손익 크기는 상수 근사이며 매매 수익성 검증은 별도다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(json.dumps(decision, ensure_ascii=False), flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
