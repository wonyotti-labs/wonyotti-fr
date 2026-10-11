import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from wonyotti_fr.action_model import select_threshold
from wonyotti_fr.first_management import decision_metrics
from wonyotti_fr.minute_management import ACTIONS
from wonyotti_fr.paired_management import direct_requests

BETAS = {'exit': 2., 'reduce': 1., 'increase': .5}


def choose_thresholds(labels, predictions):
    thresholds, details = {}, {}
    for name, scores in predictions.items():
        thresholds[name], details[name] = {}, {}
        for i, action in enumerate(ACTIONS):
            thresholds[name][action], details[name][action] = select_threshold(labels[:, i], scores[:, i], BETAS[action])
    return thresholds, details


def evaluate(labels, predictions, thresholds):
    y = np.asarray(labels)
    if y.ndim != 2 or y.shape[1] != 3 or not len(y) or not np.isin(y, [0, 1]).all() or set(predictions) != set(thresholds):
        raise ValueError('동일 관리 평가의 정답·모형 오류')
    result, decisions = {a: {} for a in ACTIONS}, {}
    for name, scores in predictions.items():
        masks, chosen = direct_requests(scores, thresholds[name])
        if scores.shape != y.shape:
            raise ValueError('동일 관리 평가의 예측 행 불일치')
        decisions[name] = {'requests': masks, 'chosen': chosen}
        for i, action in enumerate(ACTIONS):
            metrics = decision_metrics(y[:, i], masks[:, i], BETAS[action])
            metrics.update(log_loss=float(log_loss(y[:, i], scores[:, i], labels=[0, 1])),
                average_precision=float(average_precision_score(y[:, i], scores[:, i])) if y[:, i].sum() else None,
                brier=float(brier_score_loss(y[:, i], scores[:, i])))
            result[action][name] = metrics
    return result, decisions


def admission(metrics):
    checks = {}
    for action in ACTIONS:
        candidate = metrics[action]['all']
        for name in ('first', 'linear'):
            baseline = metrics[action][name]
            gain = 1-candidate['log_loss']/baseline['log_loss'] if baseline['log_loss'] else 0.
            checks[f'{action}_log_loss_{name}'] = bool(gain > .01+1e-12)
            checks[f'{action}_ap_{name}'] = bool(candidate['average_precision'] >= baseline['average_precision']-1e-12)
        checks[f'{action}_constant'] = bool(candidate['log_loss'] < metrics[action]['constant']['log_loss']-1e-12)
    for key in ('recall', 'f_beta'):
        checks['exit_'+key] = bool(metrics['exit']['all'][key] > metrics['exit']['first'][key]+1e-12)
    return {'management_admitted': all(checks.values()), 'checks': checks, 'profitability_accepted': False,
        'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}
