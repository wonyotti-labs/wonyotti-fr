from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from wonyotti_fr.common import new_run, save_json, sha256
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.histogram_management import HISTOGRAM_SETTINGS
from wonyotti_fr.holding_stress import seal
from wonyotti_fr.position_boost import PositionBoostModel
from wonyotti_fr.position_boost_reference import (
    checked_position_reference,
    checked_runtime_sources,
    position_baseline,
    read,
)
from wonyotti_fr.retained_weight_close_reference import checked_files


def compare_statistics(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('보유 방향 부스팅 검산의 지표 필드 오류')
        for key in expected:
            compare_statistics(actual[key], expected[key])
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('보유 방향 부스팅 검산의 목록 크기 오류')
        for left, right in zip(actual, expected, strict=True):
            compare_statistics(left, right)
    elif isinstance(expected, float):
        if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError('보유 방향 부스팅 검산의 숫자 불일치')
    elif actual != expected or type(actual) is not type(expected):
        raise ValueError('보유 방향 부스팅 검산의 값 불일치')


def manual_metrics(labels, scores):
    y, p = np.asarray(labels), np.asarray(scores)
    eps = np.finfo(float).eps
    clipped = np.clip(p, eps, 1-eps)
    requests = np.where(p[:, 2] >= .65, 2, np.where(p[:, 1] >= .65, 1, 0))
    result = {'rows': len(y), 'class_counts': np.bincount(y, minlength=3).tolist(),
        'log_loss': float(-np.log(clipped[np.arange(len(y)), y]).mean()),
        'multiclass_brier': float(np.mean(np.sum((p-np.eye(3)[y])**2, axis=1))), 'request_threshold': .65}
    for key, prediction in [('argmax_confusion', p.argmax(axis=1)), ('request_confusion', requests)]:
        matrix = np.zeros((3, 3), dtype=int)
        np.add.at(matrix, (y, prediction), 1)
        result[key] = matrix.tolist()
    for index, name in enumerate(['flat', 'short', 'long']):
        z, q = (y == index).astype(int), p[:, index]
        positive, negative = int(z.sum()), int((1-z).sum())
        order = np.argsort(-q, kind='stable')
        ends = np.r_[np.flatnonzero(np.diff(q[order]) != 0), len(q)-1]
        tp = np.cumsum(z[order])[ends]
        ap = np.sum(tp/(ends+1)*np.diff(np.r_[0, tp]))/positive
        auc = (rankdata(q)[z == 1].sum()-positive*(positive+1)/2)/(positive*negative)
        bounded = np.clip(q, eps, 1-eps)
        result[name] = {'rows': len(y), 'positive': positive, 'negative': negative,
            'log_loss': float(np.mean(-z*np.log(bounded)-(1-z)*np.log1p(-bounded))),
            'brier': float(np.mean((q-z)**2)), 'average_precision': float(ap), 'roc_auc': float(auc),
            'actual_positive_fraction': float(z.mean()), 'predicted_mean': float(q.mean()),
            'requested': int((requests == index).sum()), 'correct_requests': int(((requests == index) & (y == index)).sum())}
    return result


def manual_decision(metrics, monthly):
    comparisons = {}
    candidate = metrics['boosted']
    for name in ['position', 'constant']:
        baseline = metrics[name]
        gain = 1-candidate['log_loss']/baseline['log_loss'] if baseline['log_loss'] else 0.
        comparisons[name] = {'relative_log_loss_gain': gain,
            'gain_passed': gain > .01 and not math.isclose(gain, .01, abs_tol=1e-12, rel_tol=0),
            'short_precision_preserved': candidate['short']['average_precision'] >= baseline['short']['average_precision'],
            'long_precision_preserved': candidate['long']['average_precision'] >= baseline['long']['average_precision']}
    months = sum(item['boosted_log_loss'] < item['constant_log_loss'] for item in monthly)
    admitted = months >= 8 and all(item['gain_passed'] and item['short_precision_preserved']
        and item['long_precision_preserved'] for item in comparisons.values())
    return {'position_boost_admitted': admitted, 'comparisons': comparisons,
        'improved_months': months, 'required_improved_months': 8, 'monthly_consistency': months >= 8,
        'profitability_accepted': False, 'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}


def verify_position_boost(root, expected, output):
    required = {'manifest.json', 'baseline_files.json', 'reference_review.json', 'model.json',
        'training_support.json', 'metrics.json', 'monthly.json', 'decision.json', 'predictions.parquet', 'summary.json'}
    if set(checked_files(root)) != required or sha256(root/'files.json') != expected:
        raise ValueError('보유 방향 부스팅 검산의 실행 파일·봉인 오류')
    settings = read(root/'manifest.json')['settings']
    checked_runtime_sources(root)
    reference, reference_hash = Path(settings['reference_path']), settings['reference_sha256']
    spec, files = checked_position_reference(reference, reference_hash)
    if settings != {'reference': spec, 'reference_path': str(reference), 'reference_sha256': reference_hash,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V94.md')), 'new_model_count': 1,
        'request_threshold': .65, 'trading_returns_evaluated': False}:
        raise ValueError('보유 방향 부스팅 검산의 실행 계획·출처 오류')
    if (read(root/'baseline_files.json') != files
        or read(root/'reference_review.json') != read(Path(spec['review'])/'verification.json')
        or any(sha256(root/'baseline_source'/name) != digest for name, digest in files.items())):
        raise ValueError('보유 방향 부스팅 검산의 이전 사본·검산 근거 오류')
    out = new_run(output, 'position-boost-review', {'source': str(root), 'source_files_sha256': expected})
    print(out, flush=True)
    try:
        train, validation, scores = position_baseline(Path(spec['source']))
        x, y, vx = train[MARKET_FEATURES].to_numpy(), train.position_target.to_numpy(), validation[MARKET_FEATURES].to_numpy()
        model = PositionBoostModel.from_dict(read(root/'model.json'))
        with threadpool_limits(limits=1):
            learner = HistGradientBoostingClassifier(**HISTOGRAM_SETTINGS).fit(x, y)
            expected_training, expected_validation = learner.predict_proba(x), learner.predict_proba(vx)
        if learner.n_iter_ != 64 or learner.n_trees_per_iteration_ != 3 or learner.do_early_stopping_:
            raise ValueError('보유 방향 부스팅 독립 재학습의 구조 불일치')
        for index, item in enumerate(model.to_dict()['models']):
            if item['baseline'] != learner._baseline_prediction[0, index]:
                raise ValueError('보유 방향 부스팅 독립 초기 점수 불일치')
            for iteration, tree in enumerate(item['trees']):
                nodes = learner._predictors[iteration][index].nodes
                leaf = nodes['is_leaf'].astype(bool)
                expected_tree = {
                    'left': [-1 if end else int(value) for end, value in zip(leaf, nodes['left'], strict=True)],
                    'right': [-1 if end else int(value) for end, value in zip(leaf, nodes['right'], strict=True)],
                    'feature': [-2 if end else int(value) for end, value in zip(leaf, nodes['feature_idx'], strict=True)],
                    'threshold': [0. if end else float(value) for end, value in zip(leaf, nodes['num_threshold'], strict=True)],
                    'value': [float(value) if end else 0. for end, value in zip(leaf, nodes['value'], strict=True)]}
                if tree != expected_tree:
                    raise ValueError('보유 방향 부스팅 독립 트리 불일치')
        scores['boosted'] = model.probabilities(vx)
        np.testing.assert_allclose(scores['boosted'], expected_validation, atol=1e-12, rtol=0)
        train_error = float(np.max(np.abs(model.probabilities(x)-expected_training)))
        validation_error = float(np.max(np.abs(scores['boosted']-expected_validation)))
        compare_statistics(read(root/'training_support.json'), {'rows': len(x),
            'class_counts': np.bincount(y, minlength=3).tolist(), 'iterations': 64, 'trees_per_iteration': 3,
            'export_max_error': train_error, 'validation_export_max_error': validation_error, 'early_stopping': False})
        predictions = pd.read_parquet(root/'predictions.parquet')
        identity = ['end', 'episode_id', 'position_target_episode_id', 'position_target']
        expected_columns = identity + [name+'_'+label for name in scores for label in ['flat', 'short', 'long']]
        if list(predictions.columns) != expected_columns:
            raise ValueError('보유 방향 부스팅 검산의 예측 열 누락·추가')
        pd.testing.assert_frame_equal(predictions[identity], validation[identity], check_exact=True)
        metrics = {}
        for name, values in scores.items():
            np.testing.assert_array_equal(predictions[[name+'_'+label for label in ['flat', 'short', 'long']]], values)
            metrics[name] = manual_metrics(validation.position_target.to_numpy(), values)
        compare_statistics(read(root/'metrics.json'), metrics)
        monthly = []
        eps = np.finfo(float).eps
        for month in [f'2021-{number:02}' for number in range(1, 13)]:
            mask = validation.end.dt.strftime('%Y-%m').eq(month).to_numpy()
            part = validation[mask]
            targets = part.position_target.to_numpy()
            item = {'month': month, 'rows': len(part), 'class_counts': np.bincount(targets, minlength=3).tolist(),
                'current_positions': int(part.episode_id[part.episode_id.ne(0)].nunique()),
                'target_positions': int(part.position_target_episode_id[part.position_target_episode_id.ne(0)].nunique())}
            for name, values in scores.items():
                clipped = np.clip(values[mask], eps, 1-eps)
                item[name+'_log_loss'] = float(-np.log(clipped[np.arange(len(targets)), targets]).mean())
            monthly.append(item)
        compare_statistics(read(root/'monthly.json'), monthly)
        # 지표를 독립 계산한 뒤 저장된 같은 정밀도의 값으로 경계 판정을 확인한다.
        decision = manual_decision(read(root/'metrics.json'), read(root/'monthly.json'))
        compare_statistics(read(root/'decision.json'), decision)
        compare_statistics(read(root/'summary.json'), {'complete': True, 'new_models_fitted': 1,
            'training_rows': len(train), 'diagnosis_rows': len(validation), 'prior_states_and_ledgers_verification_reused': True,
            'original_baseline_scores_and_metrics_recomputed': True, **decision})
        checked_position_reference(reference, reference_hash)
        checked_runtime_sources(root)
        checked_files(root)
        if sha256(root/'files.json') != expected:
            raise ValueError('보유 방향 부스팅 검산 중 실행 변경')
        save_json(out/'verification.json', {'complete': True,
            'prior_original_state_and_ledger_audit_reused': True, 'independent_192_trees_and_baselines_exact': True,
            'all_predictions_metrics_months_and_decision_verified': True,
            'training_rows': len(train), 'diagnosis_rows': len(validation),
            'position_boost_admitted': decision['position_boost_admitted'], 'profitability_accepted': False})
        seal(out)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
