import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.event_model import EventModel
from wonyotti_fr.frequency_model import FrequencyModel, adjust_scores, score_quality


def model():
    n = len(MARKET_FEATURES)
    return EventModel(MARKET_FEATURES, ['enter_long', 'enter_short', 'hold'],
                      np.zeros(n), np.ones(n), np.zeros((3, n)), np.zeros(3))


def test_training_frequency_adjustment_has_known_endpoints():
    values = np.array([[1/3, 1/3, 1/3], [0, 0, 0]])
    prior = np.array([0.01, 0.04, 0.95])
    np.testing.assert_array_equal(adjust_scores(values, prior, 0), values)
    np.testing.assert_allclose(adjust_scores(values, prior, 1)[0], prior)
    np.testing.assert_array_equal(adjust_scores(values, prior, 1)[1], [0, 0, 0])
    for invalid in [[0, 0.5, 0.5], [0.1, 0.1, 0.1], [np.nan, 0.2, 0.8]]:
        with pytest.raises(ValueError):
            adjust_scores(values, invalid, 1)


def test_frequency_model_rejects_wrong_counts_and_keeps_missing_input_idle():
    base = model()
    counts = {'enter_long': 1, 'enter_short': 4, 'hold': 95}
    corrected = FrequencyModel.from_counts(base, counts, 1)
    frame = pd.DataFrame(np.zeros((2, len(MARKET_FEATURES))), columns=MARKET_FEATURES)
    frame.loc[1, MARKET_FEATURES[0]] = np.nan
    np.testing.assert_allclose(corrected.probabilities(frame.to_numpy())[0], [0.01, 0.04, 0.95])
    assert corrected.predict(frame).tolist() == ['hold', 'hold']
    with pytest.raises(ValueError):
        FrequencyModel.from_counts(base, {**counts, 'exit': 1}, 1)
    with pytest.raises(ValueError):
        FrequencyModel.from_counts(base, {**counts, 'hold': True}, 1)


def test_brier_and_log_loss_match_hand_calculation():
    frame = pd.DataFrame(np.zeros((3, len(MARKET_FEATURES))), columns=MARKET_FEATURES)
    frame['target'] = ['enter_long', 'enter_short', 'hold']
    metrics = score_quality(model(), frame)
    assert metrics['brier_multiclass'] == pytest.approx(2/3)
    assert metrics['log_loss'] == pytest.approx(np.log(3))
    assert sum(row['count'] for row in metrics['reliability']) == 9


def test_frequency_policy_keeps_alpha_zero_execution_and_restart_parity(tmp_path):
    from wonyotti_fr.engine import EngineConfig, TradingEngine
    from wonyotti_fr.event_backtest import EventPolicy
    from wonyotti_fr.event_features import STATE_FEATURES
    from wonyotti_fr.journal import EventJournal, canonical

    entry = model()
    entry.intercept = np.array([5, -5, 0])
    n = len(MARKET_FEATURES + STATE_FEATURES)
    management = EventModel(MARKET_FEATURES + STATE_FEATURES, ['enter_short', 'exit', 'hold'],
                            np.zeros(n), np.ones(n), np.zeros((3, n)), np.array([-5, 5, 0]))
    counts = [{'enter_long': 1, 'enter_short': 1, 'hold': 8}, {'enter_short': 1, 'exit': 1, 'hold': 8}]
    original = EventPolicy(entry, management, 0.5, 0.35)
    zero = EventPolicy(FrequencyModel.from_counts(entry, counts[0], 0), FrequencyModel.from_counts(management, counts[1], 0), 0.5, 0.35)
    adjusted = EventPolicy(FrequencyModel.from_counts(entry, counts[0], 0.5), FrequencyModel.from_counts(management, counts[1], 0.5), 0.5, 0.35)
    config = EngineConfig(stop_fraction=0, max_hold_bars=0)
    times = pd.date_range('2024-01-01', periods=8, freq='5min', tz='UTC')
    events = [{'time': t.isoformat(), 'open': 100, 'high': 100, 'low': 100, 'close': 100,
               'features': [0] * len(MARKET_FEATURES)} for t in times]
    left, right, memory = (TradingEngine(config) for _ in range(3))
    expected = []
    for i, event in enumerate(events):
        assert left.step(event, original, final=i == 7) == right.step(event, zero, final=i == 7)
        expected.append(memory.step(event, adjusted, final=i == 7))
    for start, stop in [(0, 3), (3, 8)]:
        with EventJournal(tmp_path / 'journal.sqlite', config, {'model': 'synthetic-frequency'}) as journal:
            for i in range(start, stop):
                journal.process(events[i], adjusted, final=i == 7)
            if stop == 8:
                assert canonical(journal.results()) == canonical(expected)
                assert canonical(journal.snapshot()) == canonical(memory.snapshot())
