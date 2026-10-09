from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_effect import CLOSE_FEATURES
from .close_learning import (
    CloseRegressionModel,
    CloseRidgeModel,
    close_metrics,
    run_close_learning_diagnosis,
)
from .close_learning_inputs import CLOSE_SPLITS
from .common import new_run, save_json, sha256
from .entry_regression import REGRESSION_SETTINGS
from .research_paths import reproduction_root

CALIBRATION_SPLITS = {'training': ['2021-01-01', '2021-06-30'],
    'calibration': ['2021-07-02', '2021-09-30'], 'diagnosis': ['2021-10-02', '2021-12-31']}
DIAGNOSIS_FILES = {'manifest.json', 'input_verification.json', 'exclusion_ledger.parquet',
    'training_used.parquet', 'training_weights.parquet', 'diagnosis_used.parquet', 'diagnosis_weights.parquet',
    'training_support.json', 'models.json', 'predictions.parquet', 'metrics.json', 'breakdown.json', 'decision.json', 'summary.json'}


class CloseAmountCalibration:
    def __init__(self, data):
        self.data = copy.deepcopy(data)

    @classmethod
    def fit(cls, prediction, target, weights):
        p, y, w = (np.asarray(v, dtype=float) for v in [prediction, target, weights])
        if (p.ndim != 1 or len(p) < 100 or y.shape != p.shape or w.shape != p.shape
            or not all(np.isfinite(v).all() for v in [p, y, w]) or (w <= 0).any()
            or not np.isclose(w.mean(), 1., rtol=0, atol=1e-12)):
            raise ValueError('청산 금액 보정의 표본·숫자·가중치 오류')
        mean_p, mean_y = np.average(p, weights=w), np.average(y, weights=w)
        variance = np.average((p-mean_p)**2, weights=w) if np.any(p != p[0]) else 0.
        covariance = np.average((p-mean_p)*(y-mean_y), weights=w)
        if not np.isfinite([mean_p, mean_y, variance, covariance]).all():
            raise ValueError('청산 금액 보정의 통계 범위 오류')
        # 음의 상관에서는 순위를 뒤집지 않고 보정 구간 평균으로 수축한다.
        slope = max(0., covariance/variance) if variance > 0 else 0.
        model = cls.from_dict({'format': 'close_amount_nonnegative_affine_v1',
            'slope': float(slope), 'intercept': float(mean_y-slope*mean_p)})
        return model, {'rows': len(p), 'weighted_prediction_mean': float(mean_p),
            'weighted_target_mean': float(mean_y), 'weighted_prediction_variance': float(variance),
            'weighted_covariance': float(covariance)}

    @classmethod
    def from_dict(cls, data):
        if (set(data) != {'format', 'slope', 'intercept'} or data['format'] != 'close_amount_nonnegative_affine_v1'
            or any(type(data[k]) not in (int, float) or not np.isfinite(data[k]) for k in ['slope', 'intercept'])
            or data['slope'] < 0):
            raise ValueError('청산 금액 보정의 형식·계수 오류')
        return cls(data)

    def predict(self, prediction):
        values = np.asarray(prediction, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError('청산 금액 보정의 예측 차원·숫자 오류')
        result = self.data['slope']*values+self.data['intercept']
        if not np.isfinite(result).all():
            raise ValueError('청산 금액 보정의 예측 범위 오류')
        return result

    def to_dict(self):
        return copy.deepcopy(self.data)


def calibration_splits(ledger):
    closed = ledger.label_status.eq('closed')
    if (ledger.decision_time.isna().any() or ledger.decision_time.duplicated().any()
        or not ledger.decision_time.is_monotonic_increasing
        or ledger.loc[closed, ['position_entry_time', 'label_end']].isna().any().any()
        or ledger.loc[closed, 'position_entry_time'].ge(ledger.loc[closed, 'decision_time']).any()
        or ledger.loc[closed, 'decision_time'].ge(ledger.loc[closed, 'label_end']).any()):
        raise ValueError('청산 보정 구간의 시각·정답 순서 오류')
    assignments = ledger.copy()
    assignments['split'] = np.where(closed, 'excluded_boundary', 'excluded_not_closed')
    rows, previous = {}, None
    for name, (start, end) in CALIBRATION_SPLITS.items():
        mask = (closed & ledger.position_entry_time.ge(pd.Timestamp(start, tz='UTC'))
            & ledger.decision_time.ge(pd.Timestamp(start, tz='UTC')) & ledger.label_end.lt(pd.Timestamp(end, tz='UTC')))
        frame = ledger.loc[mask].reset_index(drop=True)
        positions = frame.groupby('direction').position_entry_time.nunique()
        if (len(frame) < (1000 if name == 'training' else 100)
            or frame.position_entry_time.nunique() < (100 if name == 'training' else 30)
            or any(positions.get(side, 0) < 20 for side in [-1, 1])
            or frame.groupby('position_entry_time').direction.nunique().gt(1).any()
            or not np.isfinite(frame[CLOSE_FEATURES+['close_advantage_bps']]).all().all()
            or (name == 'training' and frame.decision_time.max()-frame.decision_time.min() < pd.Timedelta(days=150))):
            raise ValueError('청산 보정 구간의 기간·포지션·방향·숫자 지원 부족')
        if previous is not None and (previous.label_end.max() >= frame.decision_time.min()
            or set(previous.position_entry_time) & set(frame.position_entry_time)):
            raise ValueError('청산 보정 구간의 포지션·정답 교차')
        rows[name], previous = frame, frame
        assignments.loc[mask, 'split'] = name
    return rows, assignments


def reproduce_close_diagnosis(reference, output):
    files = json.loads((reference/'files.json').read_text())
    if (set(files) != DIAGNOSIS_FILES or (reference/'files.json').is_symlink()
        or any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in files.items())):
        raise ValueError('청산 보정의 기존 진단 파일·지문 오류')
    settings = json.loads((reference/'manifest.json').read_text())['settings']
    summary = json.loads((reference/'summary.json').read_text())
    labels = Path(settings['labels'])
    if (summary['complete'] is not True or summary['profitability_accepted'] is not False
        or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V60.md'))
        or settings['labels_files_sha256'] != sha256(labels/'files.json')
        or settings['periods'] != CLOSE_SPLITS or settings['settings'] != REGRESSION_SETTINGS
        or settings['features'] != CLOSE_FEATURES or settings['ridge_alpha'] != 100 or settings['margin_bps'] != 0
        or settings['model_count'] != 2 or settings['weighting'] != 'equal_original_position_within_split'
        or settings['trading_returns_evaluated'] is not False or settings['whole_system_periods_already_observed'] is not True):
        raise ValueError('청산 보정의 기존 진단 기간·계획·원장 연결 오류')
    reproduced = run_close_learning_diagnosis(labels, output)
    for name in sorted(DIAGNOSIS_FILES-{'manifest.json'}):
        if name.endswith('.parquet'):
            pd.testing.assert_frame_equal(pd.read_parquet(reference/name), pd.read_parquet(reproduced/name), check_exact=True)
        elif json.loads((reference/name).read_text()) != json.loads((reproduced/name).read_text()):
            raise ValueError('청산 보정의 기존 진단 재현 불일치: '+name)
    return reproduced


def fit_close_calibration(rows, output):
    weights, values, support = {}, {}, {}
    for name, frame in rows.items():
        frame.to_parquet(output/f'{name}_used.parquet', index=False)
        weights[name] = position_weights(frame)
        frame[['decision_time', 'position_entry_time']].assign(sample_weight=weights[name]).to_parquet(output/f'{name}_weights.parquet', index=False)
        values[name] = frame[CLOSE_FEATURES].to_numpy(dtype=float)
        support[name] = {'rows': len(frame), 'positions': int(frame.position_entry_time.nunique()),
            'direction_positions': {str(k): int(v) for k, v in frame.groupby('direction').position_entry_time.nunique().items()},
            'first_decision': frame.decision_time.min(), 'last_label_end': frame.label_end.max()}
    y, w = rows['training'].close_advantage_bps, weights['training']
    ridge, ridge_support = CloseRidgeModel.fit(values['training'], y, w)
    boosted, boost_support = CloseRegressionModel.fit(values['training'], y, w, values['calibration'])
    cal_prediction = boosted.predict(values['calibration'])
    calibration, cal_support = CloseAmountCalibration.fit(cal_prediction, rows['calibration'].close_advantage_bps, weights['calibration'])
    constants = {'training_constant': float(np.average(y, weights=w)),
        'calibration_constant': float(np.average(rows['calibration'].close_advantage_bps, weights=weights['calibration']))}
    save_json(output/'models.json', {'ridge': ridge.to_dict(), 'boosted': boosted.to_dict(), 'training_constant': constants['training_constant']})
    save_json(output/'calibration.json', {'model': calibration.to_dict(), 'calibration_constant': constants['calibration_constant']})
    save_json(output/'training_support.json', {'splits': support, 'ridge': ridge_support, 'boosted': boost_support, 'calibration': cal_support})
    rows['calibration'][['decision_time', 'position_entry_time', 'close_advantage_bps']].assign(
        sample_weight=weights['calibration'], predicted_boosted=cal_prediction,
        predicted_calibrated=calibration.predict(cal_prediction)).to_parquet(output/'calibration_predictions.parquet', index=False)
    frame = rows['diagnosis'][['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps']].copy()
    frame['sample_weight'] = weights['diagnosis']
    predictions = {'ridge': ridge.predict(values['diagnosis']), 'boosted': boosted.predict(values['diagnosis'])}
    predictions['calibrated'] = calibration.predict(predictions['boosted'])
    predictions.update({k: np.full(len(frame), v) for k, v in constants.items()})
    for name, prediction in predictions.items():
        frame[f'predicted_{name}'] = prediction
    frame.to_parquet(output/'predictions.parquet', index=False)
    return frame, {name: close_metrics(frame, p) for name, p in predictions.items()}


def calibration_admission(metrics):
    candidate = metrics['calibrated']
    references = ['boosted', 'ridge', 'training_constant', 'calibration_constant']
    if set(metrics) != {*references, 'calibrated'} or len({(v['rows'], v['positions']) for v in metrics.values()}) != 1:
        raise ValueError('청산 보정 진단의 대조·행·포지션 수 오류')
    checks = {f'weighted_mse_vs_{name}': candidate['weighted_mse'] < metrics[name]['weighted_mse']*.99 for name in references}
    checks.update({f'unweighted_mse_vs_{name}': candidate['mse'] <= metrics[name]['mse']+1e-9 for name in ['boosted', 'ridge']})
    checks.update(at_least_100_selected=candidate['selected'] >= 100, at_least_30_selected_positions=candidate['selected_positions'] >= 30,
        positive_selected_weighted_mean=candidate['selected_weighted_mean_bps'] is not None and candidate['selected_weighted_mean_bps'] > 0,
        positive_selected_mean=candidate['selected_mean_bps'] is not None and candidate['selected_mean_bps'] > 0)
    return {'checks': checks, 'close_calibration_admitted': all(checks.values()), 'trading_returns_evaluated': False}


def run_close_calibration_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'close-calibration-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V61.md')),
        'periods': CALIBRATION_SPLITS, 'features': CLOSE_FEATURES, 'settings': REGRESSION_SETTINGS,
        'ridge_alpha': 100, 'margin_bps': 0, 'calibration': 'nonnegative_affine',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'청산 금액의 시간순 보정 진단: {out}', flush=True)
    try:
        reproduced = reproduce_close_diagnosis(reference, reproduction_root(out))
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_models_predictions_rows_weights_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        rows, assignments = calibration_splits(pd.read_parquet(reproduced/'exclusion_ledger.parquet').drop(columns='split'))
        pd.testing.assert_frame_equal(rows['diagnosis'], pd.read_parquet(reference/'diagnosis_used.parquet'), check_exact=True)
        assignments.to_parquet(out/'exclusion_ledger.parquet', index=False)
        frame, metrics = fit_close_calibration(rows, out)
        decision, details = calibration_admission(metrics), []
        groups = [('direction', str(k), part) for k, part in frame.groupby('direction')]
        groups += [('month', k, part) for k, part in frame.groupby(frame.decision_time.dt.strftime('%Y-%m'))]
        for kind, group, part in groups:
            for name in metrics:
                details.append({'kind': kind, 'group': group, 'model': name, **close_metrics(part, part[f'predicted_{name}'])})
        for name, content in [('metrics', metrics), ('breakdown', details), ('decision', decision)]:
            save_json(out/f'{name}.json', content)
        save_json(out/'summary.json', {'complete': True, 'rows': {k: len(v) for k, v in rows.items()},
            'all_previous_outputs_reproduced': True, 'all_excluded_rows_preserved': True, 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 청산 순효과 금액의 시간순 보정\n\n'
            f'사전 진단 조건 통과: {decision["close_calibration_admitted"]}. '
            '앞 모델·중간 보정·뒤 진단의 원래 포지션과 시간을 분리했다. 보정 평균 상수와도 비교했다. '
            '이미 관찰한 기간의 진단이며 반복 선택 효과를 계좌 수익으로 합산하지 않았다. 매매 적용은 별도 검증이다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
