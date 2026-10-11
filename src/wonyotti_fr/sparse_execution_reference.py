import math
from datetime import datetime


class ReferenceSparsePolicy:
    def __init__(self, raw):
        self.classifiers = [self.pack(part) for part in raw['classifier']['models']]
        self.negative, self.positive = self.pack(raw['negative']), self.pack(raw['positive'])

    @staticmethod
    def pack(part):
        return (part['baseline'], [[tuple(node) for node in zip(tree['feature'], tree['threshold'], tree['left'], tree['right'], tree['value'], strict=True)]
            for tree in part['trees']])

    @staticmethod
    def value(head, x):
        result, forest = head
        for tree in forest:
            at = 0
            while tree[at][0] != -2:
                feature, threshold, left, right, _ = tree[at]
                at = left if x[feature] <= threshold else right
            result += tree[at][4]
        return result

    def __call__(self, bar, cash, quantity):
        end = datetime.fromisoformat(bar['end'])
        if end.minute % 5 or end.second or end.microsecond:
            return None, None
        assert bar['feature_end'] == bar['end']
        equity = cash+quantity*bar['close']
        current = quantity*bar['close']/equity
        journal = {'time': bar['end'], 'equity': equity, 'quantity': quantity, 'current_exposure': current,
            **dict.fromkeys(['probability_decrease', 'probability_no_change', 'probability_increase',
                'negative_size', 'positive_size', 'action', 'expected_change', 'decision_change', 'target']),
            'reason': 'missing_features'}
        if not all(v is not None and math.isfinite(v) for v in bar['features']):
            return None, journal
        x = [*bar['features'], current]
        logits = [self.value(head, x) for head in self.classifiers]
        exponent = [math.exp(value-max(logits)) for value in logits]
        probabilities = [v/sum(exponent) for v in exponent]
        negative, positive = max(0., self.value(self.negative, x)), max(0., self.value(self.positive, x))
        action = 0 if probabilities.count(max(probabilities)) > 1 else probabilities.index(max(probabilities))-1
        change = -negative if action == -1 else positive if action == 1 else 0.
        journal.update({'probability_decrease': probabilities[0], 'probability_no_change': probabilities[1],
            'probability_increase': probabilities[2], 'negative_size': negative, 'positive_size': positive,
            'action': action, 'expected_change': probabilities[2]*positive-probabilities[0]*negative,
            'decision_change': change, 'reason': 'no_change' if action == 0 else 'zero_size' if abs(change) <= 1e-12 else 'target'})
        if journal['reason'] == 'target':
            journal['target'] = min(.5, max(-.5, current+change))
        return journal['target'], journal
