from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


def independent_target_check(source, joined, comparisons, part):
    if len(joined) != len(source):
        raise ValueError('독립 계좌 노출 전체 행 누락')
    pd.testing.assert_frame_equal(joined[source.columns], source, check_exact=True)
    lookup = {}
    for row in comparisons.loc[comparisons.part.eq(part)].to_dict('records'):
        key = (row['boundary'], row['source_end'])
        if key in lookup:
            raise ValueError('독립 계좌 노출 경계 중복')
        lookup[key] = row
    if len(lookup) != 2*len(source):
        raise ValueError('독립 계좌 노출 입력 행 누락')
    with localcontext() as ctx:
        ctx.prec = 70
        for row in joined.to_dict('records'):
            for boundary, key, prefix in [('current', 'end', 'current_'), ('target', 'label_end', '')]:
                value = lookup[boundary, row['end']]
                if (value['time'] != row[key] or value['source_episode_id'] != row['episode_id']
                    or value['source_target_episode_id'] != row['position_target_episode_id']):
                    raise ValueError('독립 계좌 노출 정답 시각·포지션 오류')
                reasons = []
                if value['all_prices_have_observed_trade'] is not True:
                    reasons.append('unobserved_price_trade')
                if value['external_uncertain_count'] != 0:
                    reasons.append('uncertain_external_transfer')
                if value['unsupported_reasons'] != '':
                    reasons.append('unsupported_valuation')
                elif Decimal(value['equity_min_satoshi']) <= 0 or Decimal(value['stress_equity_min_satoshi']) <= 0:
                    reasons.append('nonpositive_equity_lower_bound')
                reason = '|'.join(reasons) if reasons else 'included'
                if (row[prefix+'net_exposure_reason'] != reason
                    or bool(row[prefix+'net_exposure_supported']) != (not reasons)):
                    raise ValueError('독립 계좌 노출 지원 사유 불일치')
                if reasons:
                    if not math.isnan(row[prefix+'net_exposure']):
                        raise ValueError('독립 미지원 계좌 노출 대입 오류')
                else:
                    if (value['equity_min_satoshi'] != value['equity_max_satoshi']
                        or value['delta_min_satoshi'] != value['delta_max_satoshi']):
                        raise ValueError('독립 확정 계좌 노출 범위 오류')
                    beta = float(Decimal(value['delta_min_satoshi'])/Decimal(value['equity_min_satoshi']))
                    if not math.isclose(row[prefix+'net_exposure'], beta, rel_tol=0, abs_tol=1e-14):
                        raise ValueError('독립 Decimal 계좌 노출 불일치')
                for suffix, name in [('min', 'btc_partial_leverage_min'), ('max', 'btc_partial_leverage_max')]:
                    a, b = row[prefix+'net_exposure_'+suffix], value[name]
                    if not ((math.isnan(a) and math.isnan(b)) or a == b):
                        raise ValueError('독립 계좌 노출 민감도 구간 불일치')
    return len(joined)


def independent_model_check(model, ridge, x, y, all_x):
    settings = {'loss': 'squared_error', 'max_iter': 64, 'learning_rate': .05,
        'max_leaf_nodes': 4, 'max_depth': 2, 'min_samples_leaf': 64, 'l2_regularization': 1.,
        'max_bins': 255, 'max_features': 1., 'categorical_features': None,
        'early_stopping': False, 'random_state': 0}
    if model['settings'] != settings or len(model['trees']) != 64:
        raise ValueError('독립 계좌 노출 모형 설정 오류')
    with threadpool_limits(limits=1):
        direct = HistGradientBoostingRegressor(**settings).fit(x, y)
        reference = direct.predict(all_x)
        z = (x-x.mean(axis=0))/np.where(x.std(axis=0) == 0, 1., x.std(axis=0))
        coefficients = np.linalg.solve(z.T@z+np.eye(x.shape[1]), z.T@(y-y.mean()))
    if model['baseline'] != float(direct._baseline_prediction[0, 0]):
        raise ValueError('독립 계좌 노출 회귀 초기값 불일치')
    for nodes, saved in zip([row[0].nodes for row in direct._predictors], model['trees'], strict=True):
        expected = {key: [] for key in ['left', 'right', 'feature', 'threshold', 'value']}
        for node in nodes:
            leaf = bool(node['is_leaf'])
            expected['left'].append(-1 if leaf else int(node['left']))
            expected['right'].append(-1 if leaf else int(node['right']))
            expected['feature'].append(-2 if leaf else int(node['feature_idx']))
            expected['threshold'].append(0. if leaf else float(node['num_threshold']))
            expected['value'].append(float(node['value']) if leaf else 0.)
        if saved != expected:
            raise ValueError('독립 계좌 노출 전체 트리 재학습 불일치')
    scale = np.where(x.std(axis=0) == 0, 1., x.std(axis=0))
    if ridge['alpha'] != 1.:
        raise ValueError('독립 계좌 노출 릿지 벌점 오류')
    for saved, value in [(ridge['mean'], x.mean(axis=0)), (ridge['scale'], scale),
        (ridge['coef'], coefficients), (ridge['intercept'], y.mean())]:
        if not np.allclose(saved, value, rtol=1e-10, atol=1e-12):
            raise ValueError('독립 계좌 노출 릿지 해 불일치')
    return {'histogram': reference, 'ridge': ((all_x-x.mean(axis=0))/scale)@coefficients+y.mean(),
        'training_mean': np.full(len(all_x), math.fsum(map(float, y))/len(y)),
        'zero': np.zeros(len(all_x)), 'one': np.ones(len(all_x))}


def independent_metrics(y, p, lo, hi):
    rows = list(zip(map(float, y), map(float, p), map(float, lo), map(float, hi), strict=True))
    n = len(rows)
    def sign(value):
        return 1 if value > 0 else -1 if value < 0 else 0
    longs, shorts = [r for r in rows if r[0] > 0], [r for r in rows if r[0] < 0]
    ordered = sorted(r[1] for r in rows)
    quantiles = {}
    for name, fraction in [('min', 0), ('p01', .01), ('p10', .1), ('median', .5), ('p90', .9), ('p99', .99), ('max', 1)]:
        pos = (n-1)*fraction
        first, last = math.floor(pos), math.ceil(pos)
        quantiles[name] = ordered[first]+(ordered[last]-ordered[first])*(pos-first)
    return {'rows': n, 'mse': math.fsum((b-a)**2 for a, b, _, _ in rows)/n,
        'mae': math.fsum(abs(b-a) for a, b, _, _ in rows)/n,
        'interval_mse': math.fsum(max(a-b, b-c, 0)**2 for _, b, a, c in rows)/n,
        'interval_mae': math.fsum(max(a-b, b-c, 0) for _, b, a, c in rows)/n,
        'direction_accuracy': sum(sign(a) == sign(b) for a, b, _, _ in rows)/n,
        'long_rows': len(longs), 'short_rows': len(shorts), 'flat_rows': n-len(longs)-len(shorts),
        'long_recall': sum(r[1] > 0 for r in longs)/len(longs) if longs else None,
        'short_recall': sum(r[1] < 0 for r in shorts)/len(shorts) if shorts else None,
        'mean_prediction': math.fsum(ordered)/n, 'predicted_long_fraction': sum(v > 0 for v in ordered)/n,
        'prediction_quantiles': quantiles, 'independent_trade_count': None}


def require_exact_metrics(expected, actual):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError('독립 계좌 노출 지표 필드 불일치')
        for key in expected:
            require_exact_metrics(expected[key], actual[key])
    elif expected is None or type(expected) is int:
        if actual != expected:
            raise ValueError('독립 계좌 노출 지표 분모 불일치')
    elif not math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('독립 계좌 노출 지표 값 불일치')
