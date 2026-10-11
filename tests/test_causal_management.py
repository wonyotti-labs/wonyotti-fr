from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.causal_management import (
    causal_admission,
    causal_predictions,
    fit_causal_calibration,
    map_management_history,
    mapping_support,
    reference_distributions,
)
from wonyotti_fr.causal_management_reference import (
    verify_causal_admission,
    verify_causal_calibration,
    verify_causal_predictions,
    verify_management_mapping,
    verify_mapping_support,
)
from wonyotti_fr.causal_management_research import checked_causal_reference
from wonyotti_fr.causal_score_distribution import (
    MINUTE_NS,
    PastScoreWindow,
    map_past_scores,
    verify_past_mapping,
)
from wonyotti_fr.quiet_management import evaluate_quiet
from wonyotti_fr.quiet_management_reference import verify_quiet_evaluation


def test_query_before_insert_and_thirty_day_boundary():
    times = pd.Series(pd.to_datetime(['2021-01-01', '2021-01-01T00:01', '2021-01-31',
                                     '2021-01-31T00:01', '2021-01-31T00:02'], format='mixed', utc=True))
    p, pool = np.array([.1, .1, .9, 0., 1.]), np.linspace(.02, .5, 1000)
    mapped = map_past_scores(times, p, pool)
    verify_past_mapping(times, p, pool, mapped)
    assert mapped[['count', 'less', 'equal']].values.tolist() == [[0, 0, 0], [1, 0, 1], [2, 2, 0], [2, 0, 0], [2, 2, 0]]
    for i, time in enumerate(times):
        earlier = p[(times < time) & (times >= time - pd.Timedelta(days=30))]
        assert mapped['count'].iloc[i] == len(earlier)
        assert mapped['less'].iloc[i] == sum(earlier < p[i])
        assert mapped['equal'].iloc[i] == sum(earlier == p[i])


@pytest.mark.parametrize('seed', range(4))
def test_future_scores_do_not_change_prior_ranks(seed):
    rng = np.random.default_rng(seed)
    times = pd.Series(pd.date_range('2021-01-01', periods=2000, freq='h', tz='UTC'))
    p, pool = rng.integers(0, 30, 2000) / 30, np.sort(rng.random(1000))
    mapped = map_past_scores(times, p, pool)
    verify_past_mapping(times, p, pool, mapped)
    changed = p.copy()
    changed[1250:] = rng.random(750)
    newer = map_past_scores(times, changed, pool)
    verify_past_mapping(times, changed, pool, newer)
    pd.testing.assert_frame_equal(mapped.iloc[:1250], newer.iloc[:1250], check_exact=True)


def test_microsecond_time_empty_and_constant_pool():
    times = pd.Series(pd.date_range('2021-01-01', periods=1002, freq='min', tz='UTC'))
    for values in [np.full(1002, .4), np.r_[np.zeros(1001), 1.]]:
        mapped = map_past_scores(times, values, np.full(1000, .15))
        verify_past_mapping(times, values, np.full(1000, .15), mapped)
        other = map_past_scores(times.astype('datetime64[us, UTC]'), values, np.full(1000, .15))
        pd.testing.assert_frame_equal(mapped, other, check_exact=True)
        assert mapped.mapped.eq(.15).all()
        assert mapped['count'].iloc[1000] == 1000
    empty = map_past_scores(times.iloc[:0], np.array([], dtype=float), np.full(1000, .15))
    verify_past_mapping(times.iloc[:0], np.array([], dtype=float), np.full(1000, .15), empty)


@pytest.mark.parametrize('kind', ['nan', 'range', 'duplicate', 'naive', 'nonminute', 'unsorted_pool', 'small_pool'])
def test_invalid_mapping_inputs_rejected(kind):
    times = pd.Series(pd.date_range('2021-01-01', periods=4, freq='min', tz='UTC'))
    values, pool = np.array([.1, .2, .3, .4]), np.linspace(0, 1, 1000)
    if kind == 'nan':
        values[0] = np.nan
    elif kind == 'range':
        values[0] = 1.1
    elif kind == 'duplicate':
        times.iloc[0] = times.iloc[1]
    elif kind == 'naive':
        times = times.dt.tz_localize(None)
    elif kind == 'nonminute':
        times.iloc[0] += pd.Timedelta(seconds=1)
    elif kind == 'unsorted_pool':
        pool = pool[::-1]
    else:
        pool = pool[:999]
    with pytest.raises(ValueError):
        map_past_scores(times, values, pool)


def test_stream_rejects_duplicate_and_boolean_score():
    window = PastScoreWindow()
    window.observe(0, .3)
    with pytest.raises(ValueError):
        window.observe(0, .3)
    with pytest.raises(ValueError):
        window.observe(MINUTE_NS, True)


def test_large_reference_tail_gap_keeps_linear_quantile_accuracy():
    times = pd.Series(pd.date_range('2021-01-01', periods=1000, freq='min', tz='UTC'))
    values = np.r_[np.zeros(999), 1.]
    pool = np.zeros(361041)
    position = ((999 + .5) / 1000) * (len(pool) - 1)
    pool[int(np.ceil(position)):] = 1.
    result = map_past_scores(times, values, pool)
    verify_past_mapping(times, values, pool, result)
    assert abs(result.mapped.iloc[-1] - (position - np.floor(position))) < 1e-12


@pytest.fixture(scope='module')
def prepared():
    originals = {}
    rng = np.random.default_rng(41)
    for phase, start in [('training', '2020-12-25'), ('calibration', '2021-01-03'), ('diagnosis', '2021-07-03')]:
        n, i = 9000, np.arange(9000)
        frame = pd.DataFrame({'end': pd.date_range(start, periods=n, freq='min', tz='UTC'),
            'episode_id': i // 60 + len(originals) * 1000, 'exit_score': .01, 'exit_request': False, 'y_exit': 0})
        for action in ['reduce', 'increase']:
            y = i % (11 if action == 'reduce' else 13) == 0
            frame['y_' + action] = y.astype(int)
            frame['past_' + action + '_exists'] = (i % 3 != 0).astype(float)
            frame['past_' + action + '_log_count_15m'] = np.where(i % 3 == 2, np.log1p(1.), 0.)
            for name in ['baseline', 'capacity', 'tree_raw', 'linear_raw', 'tree', 'linear', 'constant']:
                scores = .01 + .04 * y + rng.random(n) * .08
                frame[action + '_' + name + '_score'] = scores
                frame[action + '_' + name + '_request'] = scores >= .07
        originals[phase] = frame
    references = reference_distributions(originals['training'])
    ledgers = map_management_history(originals, references)
    fitted = fit_causal_calibration(originals['calibration'], ledgers['calibration'])
    return originals, references, ledgers, fitted


def test_full_mapping_calibration_preservation_and_admission(prepared):
    original, reference, ledgers, fitted = prepared
    proof = verify_management_mapping(original, reference, ledgers)
    assert len(proof) == 8
    offsets, thresholds, details, constants = fitted
    verify_causal_calibration(original['calibration'], ledgers['calibration'], *fitted)
    for phase, frame in original.items():
        support = mapping_support(frame, ledgers[phase])
        verify_mapping_support(frame, ledgers[phase], support)
        saved = causal_predictions(frame, ledgers[phase], offsets, thresholds, constants)
        unchanged = verify_causal_predictions(frame, ledgers[phase], saved, offsets, thresholds, constants)
        metrics = evaluate_quiet(saved)
        verify_quiet_evaluation(saved, metrics)
        if phase == 'diagnosis':
            old = evaluate_quiet(frame)
            decision = causal_admission(metrics, old, unchanged)
            verify_causal_admission(metrics, old, unchanged, decision)
            assert len(decision['checks']) == 35
            assert support['increase']['first']['fallback']['rows'] == 1000
            assert support['increase']['first']['supported']['rows'] == 2000
            broken = deepcopy(decision)
            broken['checks']['increase_restart_recall_improved'] = not broken['checks']['increase_restart_recall_improved']
            with pytest.raises(AssertionError):
                verify_causal_admission(metrics, old, unchanged, broken)


def test_future_labels_leave_mapping_and_fit_unchanged(prepared):
    original, reference, ledgers, fitted = prepared
    changed = {p: f.copy() for p, f in original.items()}
    for action in ['reduce', 'increase']:
        changed['diagnosis']['y_' + action] = 1 - changed['diagnosis']['y_' + action]
    newer = map_management_history(changed, reference)
    for phase in ledgers:
        pd.testing.assert_frame_equal(ledgers[phase], newer[phase], check_exact=True)
    assert fit_causal_calibration(changed['calibration'], newer['calibration']) == fitted


@pytest.mark.parametrize('field', ['count', 'less', 'equal', 'mapped'])
def test_history_damage_rejected(prepared, field):
    original, reference, ledgers, _ = prepared
    bad = {p: f.copy() for p, f in ledgers.items()}
    bad['diagnosis'].loc[0, 'reduce_tree_' + field] += 1
    with pytest.raises(AssertionError):
        verify_management_mapping(original, reference, bad)


@pytest.mark.parametrize('column,index', [('exit_score', 0), ('reduce_baseline_score', 5000),
    ('reduce_tree_score', 0), ('reduce_tree_score', 5001), ('increase_constant_score', 5002),
    ('increase_linear_request', 5001)])
def test_fallback_recent_and_active_damage_rejected(prepared, column, index):
    original, _, ledgers, (offsets, thresholds, _, constants) = prepared
    frame, ledger = original['diagnosis'], ledgers['diagnosis']
    saved = causal_predictions(frame, ledger, offsets, thresholds, constants)
    saved.loc[index, column] = not saved.loc[index, column] if column.endswith('request') else saved.loc[index, column] + .01
    with pytest.raises(AssertionError):
        verify_causal_predictions(frame, ledger, saved, offsets, thresholds, constants)


def test_insufficient_calibration_and_broken_source(prepared, tmp_path):
    original, _, ledgers, _ = prepared
    changed = original['calibration'].copy()
    changed['y_increase'] = 0
    with pytest.raises(ValueError, match='지원'):
        fit_causal_calibration(changed, ledgers['calibration'])
    path = tmp_path / 'proof.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='지문'):
        checked_causal_reference(path, '0' * 64)
