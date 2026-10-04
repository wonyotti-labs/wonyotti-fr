from __future__ import annotations

import numpy as np
import pandas as pd

from .continuation_inputs import ContinuationCloseModel, validate_pending_matrix
from .first_close_diagnostics import first_close_positions
from .histogram_management import HistogramManagementModels


def cost_terms(target, weights):
    y, w = (np.asarray(v, dtype=float) for v in [target, weights])
    if (y.ndim != 1 or not len(y) or w.shape != y.shape
        or not np.isfinite(y).all() or not np.isfinite(w).all() or (w <= 0).any()):
        raise ValueError('청산 비용의 정답·원래 가중치 오류')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        cost = w*np.abs(y)
    if not np.isfinite(cost).all() or ((y != 0) & (cost == 0)).any():
        raise ValueError('청산 비용 곱의 넘침·소실 오류')
    return y, w, cost


def cost_training(target, weights):
    y, w, cost = cost_terms(target, weights)
    fit = cost > 0
    if (not np.isclose(w.mean(), 1., rtol=0, atol=1e-12) or fit.sum() < 1000
        or min(np.bincount((y[fit] > 0).astype(int), minlength=2)) < 64):
        raise ValueError('청산 비용 학습의 원래 가중 평균·양쪽 지원 부족')
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        normalizer = float(cost[fit].mean())
        normalized = cost/normalizer
    if (not np.isfinite(normalizer) or normalizer <= 0 or not np.isfinite(normalized).all()
        or (normalized[fit] <= 0).any() or not np.isclose(normalized[fit].mean(), 1., rtol=0, atol=1e-12)):
        raise ValueError('청산 비용 정규화의 넘침·소실 오류')
    prior = float(np.average(y[fit] > 0, weights=normalized[fit]))
    if not 0 < prior < 1:
        raise ValueError('청산 비용의 학습 상수 지원 오류')
    ledger = pd.DataFrame({'original_weight': w, 'positive_effect': y > 0, 'absolute_effect_bps': np.abs(y),
        'cost_weight': cost, 'fit_weight': normalized, 'fit_used': fit,
        'reason': np.where(fit, 'positive_cost', 'zero_effect')})
    return ledger, {'normalizer': normalizer, 'training_constant_score': prior,
        'rows': len(y), 'fit_rows': int(fit.sum()), 'zero_effect_rows': int((~fit).sum())}


class UtilityCloseModel(HistogramManagementModels):
    features = ContinuationCloseModel.features
    actions = ['beneficial_close']
    format = 'cost_weighted_close_v1'

    @classmethod
    def fit(cls, values, target, weights, validation_values):
        x, vx = (np.asarray(v, dtype=float) for v in [values, validation_values])
        for matrix in [x, vx]:
            if matrix.ndim != 2 or matrix.shape[1] != len(cls.features) or not np.isfinite(matrix).all():
                raise ValueError('청산 비용 모델의 현재 입력 오류')
            validate_pending_matrix(matrix[:, -4:])
        ledger, support = cost_training(target, weights)
        if len(x) != len(ledger):
            raise ValueError('청산 비용 모델의 입력·정답 행 수 불일치')
        fit = ledger.fit_used.to_numpy()
        model, exported = super().fit(x[fit], ledger.positive_effect.to_numpy()[fit, None], vx,
            sample_weight=ledger.fit_weight.to_numpy()[fit])
        return model, {**support, 'export': exported}, ledger

    def probabilities(self, values):
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.features):
            raise ValueError('청산 비용 점수의 입력 차원 오류')
        validate_pending_matrix(matrix[:, -4:])
        return super().probabilities(matrix)


def cost_scores(values, rows):
    score = np.asarray(values, dtype=float)
    if score.shape != (rows,) or not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
        raise ValueError('청산 비용 점수의 행·범위 오류')
    return score


def cost_probability_metrics(target, weights, scores):
    y, _, cost = cost_terms(target, weights)
    p = cost_scores(scores, len(y))
    with np.errstate(over='ignore'):
        total = float(cost.sum())
    if not np.isfinite(total):
        raise ValueError('청산 평가 비용 합의 넘침 오류')
    fields = ['cost_log_loss', 'cost_brier', 'positive_cost_fraction', 'mean_cost_score']
    result = {'rows': len(y), 'nonzero_cost_rows': int((cost > 0).sum()), 'cost_mass': total}
    if total == 0:
        return {**result, **dict.fromkeys(fields)}
    weight, labels = cost/total, (y > 0).astype(float)
    safe = np.clip(p, 1e-15, 1-1e-15)
    values = [-(labels*np.log(safe)+(1-labels)*np.log1p(-safe)), (labels-p)**2, labels, p]
    return {**result, **{name: float(np.dot(weight, value)) for name, value in zip(fields, values, strict=True)}}


def policy_effect_metrics(frame, actions):
    y, w, _ = cost_terms(frame.close_advantage_bps, frame.sample_weight)
    a = np.asarray(actions)
    if (a.shape != y.shape or a.dtype != np.dtype(bool) or frame.position_entry_time.isna().any()
        or not frame.original_intent.isin(['hold', 'exit', 'reduce', 'increase']).all()
        or (a & frame.original_intent.eq('exit').to_numpy()).any()):
        raise ValueError('청산 행동 효과의 행·선택·원래 의도 오류')
    effect = a*y
    regret = np.maximum(y, 0)-effect
    with np.errstate(over='ignore', invalid='ignore'):
        result = {'rows': len(y), 'positions': int(frame.position_entry_time.nunique()),
            'selected': int(a.sum()), 'selected_positions': int(frame.loc[a, 'position_entry_time'].nunique()),
            'selected_weighted_mean_bps': float(np.average(y[a], weights=w[a])) if a.any() else None,
            'selected_mean_bps': float(y[a].mean()) if a.any() else None,
            'weighted_effect_bps': float(np.average(effect, weights=w)), 'effect_bps': float(effect.mean()),
            'weighted_regret_bps': float(np.average(regret, weights=w)), 'regret_bps': float(regret.mean())}
    if any(v is not None and not np.isfinite(v) for v in result.values()):
        raise ValueError('청산 행동 효과의 평균 넘침 오류')
    return result


def first_cost_positions(frame, scores):
    score = cost_scores(scores, len(frame))
    # 기존 최초 선택 회계를 재사용하고 점수에는 bp 단위를 붙이지 않는다.
    positions = first_close_positions(frame.assign(_cost_choice=(score > .5).astype(float)), '_cost_choice')
    positions = positions.drop(columns='first_prediction_bps')
    positions['first_cost_score'] = positions.first_selected_time.map(pd.Series(score, index=frame.decision_time))
    return positions


def utility_admission(metrics, probability, first, intervals):
    candidate, p, constant = metrics['utility'], probability['utility'], probability['training_constant']
    if (len({(v['rows'], v['positions']) for v in metrics.values()}) != 1
        or any(v['rows'] != candidate['rows'] for v in probability.values())
        or any(v['positions'] != candidate['positions'] for v in first.values())
        or first['utility']['selected_positions'] != candidate['selected_positions']):
        raise ValueError('청산 비용 진단의 행·최초 포지션 불일치')
    checks = {'cost_log_loss_vs_constant': p['cost_log_loss'] is not None and constant['cost_log_loss'] is not None and p['cost_log_loss'] < constant['cost_log_loss']*.99,
        'cost_brier_not_worse': p['cost_brier'] is not None and constant['cost_brier'] is not None and p['cost_brier'] <= constant['cost_brier']+1e-12}
    checks.update({f'weighted_regret_vs_{name}': candidate['weighted_regret_bps'] < metrics[name]['weighted_regret_bps']
        for name in ['training_constant', 'continuation', 'weekly']})
    mean = first['utility']['all_position_mean_common_bps']
    checks.update(at_least_100_selected=candidate['selected'] >= 100,
        at_least_30_selected_positions=candidate['selected_positions'] >= 30,
        positive_selected_weighted_mean=candidate['selected_weighted_mean_bps'] is not None and candidate['selected_weighted_mean_bps'] > 0,
        positive_selected_mean=candidate['selected_mean_bps'] is not None and candidate['selected_mean_bps'] > 0,
        positive_first_choice_mean=mean > 0,
        first_mean_vs_continuation=mean > first['continuation']['all_position_mean_common_bps'],
        first_mean_vs_weekly=mean > first['weekly']['all_position_mean_common_bps'])
    for key, name in [('positive_first_interval_lower', 'utility'), ('positive_paired_interval_lower', 'paired_difference')]:
        low = intervals['intervals'][name]['lower']
        checks[key] = bool(low is not None and np.isfinite(low) and low > 0)
    return {'checks': checks, 'utility_admitted': all(checks.values()), 'trading_returns_evaluated': False}
