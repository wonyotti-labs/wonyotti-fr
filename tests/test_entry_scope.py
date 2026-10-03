import numpy as np
import pytest
from test_action_model import FixedScores
from test_pullback_evaluation import bars
from test_rate_policy import inputs

from wonyotti_fr.entry_scope import direction_only_manager
from wonyotti_fr.expansion_model import ExpansionPolicy
from wonyotti_fr.minute_management import ACTIONS
from wonyotti_fr.net_edge_labels import potential_entries
from wonyotti_fr.pullback_policy import PullbackPolicy
from wonyotti_fr.rate_policy import RateActionPolicy


class ConstantBinary:
    def __init__(self, probability):
        self.probability = probability

    def probabilities(self, values):
        return np.full(len(values), self.probability)


def original(buy):
    base = ExpansionPolicy(ConstantBinary(0.), ConstantBinary(buy), .1, .65)
    return RateActionPolicy(PullbackPolicy(base, 16, 5), FixedScores([.1, .2, .3]),
                            dict.fromkeys(ACTIONS, .5), 1., dict.fromkeys(ACTIONS, 1.))


@pytest.mark.parametrize('buy,direction', [(.9, 1), (.1, -1), (.5, 0)])
def test_direction_scope_changes_activity_only_and_keeps_original(buy, direction):
    prior = original(buy)
    changed = direction_only_manager(prior)
    bar = {'end': '2021-01-01T00:05:00+00:00', 'features': np.zeros(14)}
    state = {'direction': 0, 'hold_bars': 0, 'halted': False}
    assert prior.base(bar, state) == 'hold' and prior.base.activity_threshold == .1
    assert changed.base(bar, state) == {1: 'enter_long', -1: 'enter_short', 0: 'hold'}[direction]
    assert changed.manager is prior.manager and changed.scales == prior.scales
    assert changed.base.direction is prior.base.direction and changed.base.direction_threshold == .65
    stored = {}
    for minute in range(1, 18):
        event, view = inputs(minute, stored)
        one, two = prior(event, view), changed(event, view)
        assert one == two
        stored = one.state
    with pytest.raises(ValueError, match='기반'):
        direction_only_manager(changed)


def test_broader_opportunities_keep_direction_wait_and_past_input_invariance():
    frame = bars().assign(volume=100., count=10)
    prior = original(.9)
    unchanged, _ = potential_entries(frame, prior)
    changed = direction_only_manager(prior)
    found, counts = potential_entries(frame, changed)
    assert unchanged.empty and len(found) > 0 and counts['triggered'] == len(found)
    assert found.order_direction.eq(1).all() and found.favorable_bps.ge(16).all() and found.wait_minutes.between(1, 5).all()
    modified = frame.copy()
    modified.loc[30:, ['open', 'high', 'low', 'close']] *= 2
    again, _ = potential_entries(modified, changed)
    import pandas as pd
    pd.testing.assert_frame_equal(found[found.decision_time < frame.time.iloc[30]].reset_index(drop=True),
        again[again.decision_time < frame.time.iloc[30]].reset_index(drop=True), check_exact=True)
