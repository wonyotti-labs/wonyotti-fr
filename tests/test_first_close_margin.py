import copy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.close_learning import close_metrics
from wonyotti_fr.first_close_diagnostics import first_close_positions
from wonyotti_fr.first_close_margin import FIRST_MARGINS, margin_admission, select_first_margin


def examples():
    rows = []
    for i, entry in enumerate(pd.date_range('2021-07-02', periods=40, freq='D', tz='UTC')):
        for j, (prediction, effect) in enumerate([(.5, -3.), (2., 3.), (5., 1.)]):
            rows.append({'position_entry_time': entry, 'decision_time': entry+pd.Timedelta(minutes=5*(j+1)),
                'label_end': entry+pd.Timedelta(hours=1), 'continue_end': entry+pd.Timedelta(hours=1),
                'continue_cash': 10000., 'decision_equity': 10000., 'direction': 1 if i%2 else -1,
                'label_status': 'closed', 'original_intent': 'hold', 'sample_weight': 1.,
                'close_advantage_bps': effect, 'close_advantage_pnl': effect, 'predicted_half': prediction})
    return pd.DataFrame(rows)


def test_margin_selects_strict_first_crossing_and_keeps_original_prediction_and_account_error():
    frame = examples()
    pd.testing.assert_frame_equal(first_close_positions(frame, 'predicted_half'),
        first_close_positions(frame, 'predicted_half', margin_bps=0.), check_exact=True)
    original = close_metrics(frame, frame.predicted_half)
    shifted = close_metrics(frame, frame.predicted_half, margin_bps=.5)
    positions = first_close_positions(frame, 'predicted_half', margin_bps=.5)
    assert positions.first_prediction_bps.eq(2.).all()
    np.testing.assert_allclose(positions.first_effect_common_bps, 3., rtol=0, atol=1e-12)
    assert original['selected'] == 120 and shifted['selected'] == 80
    for key in ['weighted_mse', 'mse', 'mean_predicted_bps', 'mean_actual_bps']:
        assert original[key] == shifted[key]
    frame.loc[frame.predicted_half.eq(2.), 'original_intent'] = 'exit'
    assert first_close_positions(frame, 'predicted_half', margin_bps=.5).first_prediction_bps.eq(5.).all()
    for bad in [-1., np.nan, np.inf, True, '1']:
        with pytest.raises(ValueError):
            first_close_positions(frame, 'predicted_half', margin_bps=bad)
        with pytest.raises(ValueError):
            close_metrics(frame, frame.predicted_half, margin_bps=bad)


def test_selection_keeps_all_six_candidates_and_uses_first_effect_with_higher_tie_margin():
    frame = examples()
    selection, metrics, positions = select_first_margin(frame)
    assert selection['selection_passed'] and selection['chosen_margin_bps'] == 1.
    assert selection['chosen_candidate'] == 'candidate-02'
    assert len(metrics) == len(positions) == 6
    assert [m['margin_bps'] for m in metrics] == FIRST_MARGINS
    assert metrics[0]['selected_negative'] == 40
    assert metrics[0]['all_position_mean_common_bps'] == pytest.approx(-3., abs=1e-12)
    assert metrics[-1]['selected_positions'] == 0 and metrics[-1]['all_position_mean_common_bps'] == 0.
    frame['future_diagnosis_prediction'], frame['future_diagnosis_profit'] = 1e8, -1e8
    assert select_first_margin(frame)[0] == selection


@pytest.mark.parametrize('case', ['negative', 'few_positions', 'no_improvement'])
def test_unsupported_nonpositive_or_unimproved_selection_blocks_final_diagnosis(case):
    frame = examples()
    if case == 'negative':
        frame['close_advantage_bps'], frame['close_advantage_pnl'] = -1., -1.
    elif case == 'few_positions':
        frame = frame.iloc[:29*3]
    else:
        frame['close_advantage_bps'], frame['close_advantage_pnl'] = 2., 2.
    selection, metrics, positions = select_first_margin(frame)
    assert not selection['selection_passed'] and selection['chosen_margin_bps'] is None
    assert len(metrics) == len(positions) == 6
    decision = margin_admission(selection)
    assert not decision['margin_admitted'] and not decision['final_diagnosis_evaluated']
    assert decision['checks'] == {'selection_supported': False}


def test_ten_final_conditions_include_paired_uncertainty_and_unchanged_forecast_errors():
    selection = {'selection_passed': True, 'chosen_margin_bps': 1.}
    value = {'rows': 200, 'positions': 40, 'weighted_mse': 50., 'mse': 50.,
        'mean_predicted_bps': 1., 'mean_actual_bps': 2., 'selected': 120, 'selected_positions': 30,
        'selected_weighted_mean_bps': 2., 'selected_mean_bps': 2.}
    metrics = {'margin': dict(value), 'half_zero': dict(value)}
    first = {name: {'positions': 40, 'selected_positions': 30, 'all_position_mean_common_bps': mean}
        for name, mean in [('margin', 2.), ('half_zero', 1.), ('continuation', 1.)]}
    intervals = {'intervals': {'margin': {'lower': .1}, 'paired_difference': {'lower': .1}}}
    result = margin_admission(selection, metrics, first, intervals)
    assert result['margin_admitted'] and len(result['checks']) == 10
    for field, bad in [('selected', 99), ('selected_positions', 29), ('selected_weighted_mean_bps', 0.), ('selected_mean_bps', 0.)]:
        altered, altered_first = copy.deepcopy(metrics), copy.deepcopy(first)
        altered['margin'][field] = bad
        if field == 'selected_positions':
            altered_first['margin'][field] = bad
        assert not margin_admission(selection, altered, altered_first, intervals)['margin_admitted']
    for name, bad in [('margin', 0.), ('half_zero', 2.), ('continuation', 2.)]:
        altered = copy.deepcopy(first)
        altered[name]['all_position_mean_common_bps'] = bad
        assert not margin_admission(selection, metrics, altered, intervals)['margin_admitted']
    for name in ['margin', 'paired_difference']:
        altered = copy.deepcopy(intervals)
        altered['intervals'][name]['lower'] = 0.
        assert not margin_admission(selection, metrics, first, altered)['margin_admitted']
    altered = copy.deepcopy(metrics)
    altered['margin']['mse'] += 1
    with pytest.raises(ValueError):
        margin_admission(selection, altered, first, intervals)
