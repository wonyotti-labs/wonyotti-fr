from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .addition_effect import position_weights
from .close_effect import CLOSE_FEATURES
from .close_learning_inputs import CLOSE_SPLITS, close_learning_splits, load_close_training
from .common import new_run, save_json, sha256
from .entry_regression import REGRESSION_SETTINGS, EntryRegressionModel


class CloseRegressionModel(EntryRegressionModel):
    features = CLOSE_FEATURES
    format = 'close_advantage_histogram_v1'


class CloseRidgeModel:
    def __init__(self, data):
        self.data = copy.deepcopy(data)

    @classmethod
    def fit(cls, values, target, weights):
        x, y, w = (np.asarray(v, dtype=float) for v in [values, target, weights])
        if (x.ndim != 2 or x.shape[1] != len(CLOSE_FEATURES) or len(x) < 200
            or y.shape != (len(x),) or w.shape != (len(x),)
            or not all(np.isfinite(v).all() for v in [x, y, w]) or (w <= 0).any()
            or not np.isclose(w.mean(), 1., rtol=0, atol=1e-12)):
            raise ValueError('청산 회귀의 학습 차원·숫자·가중치 오류')
        with threadpool_limits(limits=1):
            scaler = StandardScaler().fit(x, sample_weight=w)
            learner = Ridge(alpha=100).fit(scaler.transform(x), y, sample_weight=w)
            expected = learner.predict(scaler.transform(x))
        model = cls.from_dict({'format': 'close_advantage_ridge_v1', 'features': CLOSE_FEATURES, 'alpha': 100,
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'coefficients': learner.coef_.tolist(), 'intercept': float(learner.intercept_)})
        error = float(np.max(np.abs(model.predict(x)-expected)))
        if error > 1e-10:
            raise ValueError('청산 회귀의 숫자 내보내기 불일치')
        return model, {'rows': len(x), 'export_max_error': error}

    @classmethod
    def from_dict(cls, data):
        if (data.get('format') != 'close_advantage_ridge_v1' or data.get('features') != CLOSE_FEATURES
            or type(data.get('alpha')) is not int or data['alpha'] != 100):
            raise ValueError('청산 회귀의 형식·설정 오류')
        values = [np.asarray(data[k], dtype=float) for k in ['mean', 'scale', 'coefficients']]
        if (any(v.shape != (len(CLOSE_FEATURES),) or not np.isfinite(v).all() for v in values)
            or (values[1] <= 0).any() or type(data.get('intercept')) not in (int, float)
            or not np.isfinite(data['intercept'])):
            raise ValueError('청산 회귀의 계수 오류')
        return cls(data)

    def predict(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(CLOSE_FEATURES):
            raise ValueError('청산 회귀의 예측 차원 오류')
        with np.errstate(over='ignore', invalid='ignore'):
            result = ((values-self.data['mean'])/self.data['scale']) @ np.asarray(self.data['coefficients'])+self.data['intercept']
        result[~np.isfinite(result)] = np.nan
        return result

    def to_dict(self):
        return copy.deepcopy(self.data)


def close_metrics(frame, prediction, *, margin_bps=0.):
    if type(margin_bps) not in (int, float) or not np.isfinite(margin_bps) or margin_bps < 0:
        raise ValueError('청산 진단 지표의 실행 문턱 오류')
    target, weight, prediction = (np.asarray(v, dtype=float) for v in [frame.close_advantage_bps, frame.sample_weight, prediction])
    if (not len(target) or prediction.shape != target.shape or weight.shape != target.shape
        or not all(np.isfinite(v).all() for v in [target, weight, prediction]) or (weight <= 0).any()
        or frame.position_entry_time.isna().any() or frame.original_intent.isna().any()):
        raise ValueError('청산 진단 지표의 행·예측·가중치 오류')
    # 원래 청산도 오차에 포함하되 추가 청산의 효과와 구분한다.
    selected = (prediction > margin_bps) & frame.original_intent.ne('exit').to_numpy()
    return {'rows': len(target), 'positions': int(frame.position_entry_time.nunique()),
        'weighted_mse': float(np.average((target-prediction)**2, weights=weight)),
        'mse': float(np.mean((target-prediction)**2)), 'selected': int(selected.sum()),
        'selected_positions': int(frame.loc[selected, 'position_entry_time'].nunique()),
        'selected_weighted_mean_bps': float(np.average(target[selected], weights=weight[selected])) if selected.any() else None,
        'selected_mean_bps': float(target[selected].mean()) if selected.any() else None,
        'mean_predicted_bps': float(prediction.mean()), 'mean_actual_bps': float(target.mean())}


def close_admission(metrics):
    new, old, constant = (metrics[k] for k in ['boosted', 'ridge', 'constant'])
    if len({(v['rows'], v['positions']) for v in metrics.values()}) != 1:
        raise ValueError('청산 진단의 비교 행·포지션 수 불일치')
    checks = {'weighted_mse_vs_ridge': new['weighted_mse'] < old['weighted_mse']*.99,
        'weighted_mse_vs_constant': new['weighted_mse'] < constant['weighted_mse']*.99,
        'unweighted_mse_not_worse': new['mse'] <= old['mse']+1e-9,
        'at_least_100_selected': new['selected'] >= 100,
        'at_least_30_selected_positions': new['selected_positions'] >= 30,
        'positive_selected_weighted_mean': new['selected_weighted_mean_bps'] is not None and new['selected_weighted_mean_bps'] > 0,
        'positive_selected_mean': new['selected_mean_bps'] is not None and new['selected_mean_bps'] > 0}
    return {'checks': checks, 'boosted_admitted': all(checks.values()), 'trading_returns_evaluated': False}


def run_close_learning_diagnosis(labels: Path, output: Path) -> Path:
    out = new_run(output, 'close-learning-diagnosis', {'labels': str(labels),
        'labels_files_sha256': sha256(labels/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V60.md')),
        'periods': CLOSE_SPLITS, 'features': CLOSE_FEATURES, 'settings': REGRESSION_SETTINGS,
        'ridge_alpha': 100, 'model_count': 2, 'margin_bps': 0, 'weighting': 'equal_original_position_within_split',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'보유 청산 순효과 진단: {out}', flush=True)
    try:
        ledger, verification = load_close_training(labels)
        save_json(out/'input_verification.json', verification)
        rows, assignment = close_learning_splits(ledger)
        assignment.to_parquet(out/'exclusion_ledger.parquet', index=False)
        weights, values, supports = {}, {}, {}
        for name, frame in rows.items():
            frame.to_parquet(out/f'{name}_used.parquet', index=False)
            weights[name] = position_weights(frame)
            frame[['decision_time', 'position_entry_time']].assign(sample_weight=weights[name]).to_parquet(out/f'{name}_weights.parquet', index=False)
            values[name] = frame[CLOSE_FEATURES].to_numpy(dtype=float)
            supports[name] = {'rows': len(frame), 'positions': int(frame.position_entry_time.nunique()),
                'direction_positions': {str(k): int(v) for k, v in frame.groupby('direction').position_entry_time.nunique().items()},
                'first_decision': frame.decision_time.min(), 'last_label_end': frame.label_end.max()}
        y, w = rows['training'].close_advantage_bps, weights['training']
        ridge, ridge_support = CloseRidgeModel.fit(values['training'], y, w)
        boosted, boost_support = CloseRegressionModel.fit(values['training'], y, w, values['diagnosis'])
        constant = float(np.average(y, weights=w))
        save_json(out/'models.json', {'ridge': ridge.to_dict(), 'boosted': boosted.to_dict(), 'constant': constant})
        save_json(out/'training_support.json', {'ridge': ridge_support, 'boosted': boost_support,
            'splits': supports, 'excluded': assignment.split.value_counts().to_dict()})
        frame = rows['diagnosis'][['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps']].copy()
        frame['sample_weight'] = weights['diagnosis']
        predictions = {'ridge': ridge.predict(values['diagnosis']), 'boosted': boosted.predict(values['diagnosis']),
                       'constant': np.full(len(frame), constant)}
        for name, prediction in predictions.items():
            frame[f'predicted_{name}'] = prediction
        frame.to_parquet(out/'predictions.parquet', index=False)
        metrics = {name: close_metrics(frame, prediction) for name, prediction in predictions.items()}
        decision = close_admission(metrics)
        details = []
        groups = [('direction', str(k), part) for k, part in frame.groupby('direction')]
        groups += [('month', k, part) for k, part in frame.groupby(frame.decision_time.dt.strftime('%Y-%m'))]
        for kind, group, part in groups:
            for name in predictions:
                details.append({'kind': kind, 'group': group, 'model': name, **close_metrics(part, part[f'predicted_{name}'])})
        save_json(out/'metrics.json', metrics)
        save_json(out/'breakdown.json', details)
        save_json(out/'decision.json', decision)
        if sha256(labels/'files.json') != verification['labels_files_sha256']:
            raise ValueError('청산 진단 중 원장 지문 변경')
        save_json(out/'summary.json', {'complete': True, 'training_rows': len(rows['training']),
            'diagnosis_rows': len(rows['diagnosis']), 'excluded_states_preserved': True, 'profitability_accepted': False, **decision})
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        (out/'REPORT.md').write_text('# 보유 청산 순효과의 시간순 예측 진단\n\n'
            f'시간순 진단 조건 통과: {decision["boosted_admitted"]}. '
            '원래 포지션별 가중 회귀·단일 부스팅·상수를 비교했다. 원래 청산도 오차와 학습에 포함했다. '
            '반복된 추가 청산 선택의 평균은 연속 계좌 수익이 아니다. 전체 시스템이 이미 사용한 기간이며 매매 적용은 별도 계획과 검증이 필요하다.\n')
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
