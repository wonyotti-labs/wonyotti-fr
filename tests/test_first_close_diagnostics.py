import json

import numpy as np
import pandas as pd
import pytest
from test_close_calibration import calibration_examples, source_fixture
from test_first_state import fixed_budget  # noqa: F401

from wonyotti_fr.close_economics import ECONOMIC_FEATURES, run_close_economic_diagnosis
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.first_close_diagnostics import (
    first_close_metrics,
    first_close_positions,
    paired_week_blocks,
    reproduce_economic_diagnosis,
    run_first_close_diagnosis,
)


def cases():
    frame = pd.DataFrame({'position_entry_time': pd.to_datetime(['2021-10-02T00:00Z']*4+['2021-10-03T00:00Z']*4),
        'direction': [1]*4+[-1]*4, 'original_intent': ['hold', 'exit', 'reduce', 'increase']*2,
        'label_status': 'closed', 'decision_equity': [10000., 9990., 9980., 9995.]*2,
        'close_advantage_pnl': [10., 0., -2., 20., -1., 0., 2., -3.],
        'predicted_economic': [-1., 100., .2, 100., -1., 10., 0., -1.],
        'predicted_boosted': [-1., 1., -1., .1, -1., 1., -1., .2], 'continue_cash': 10000.})
    frame['decision_time'] = frame.position_entry_time+pd.to_timedelta([5, 10, 15, 20]*2, unit='min')
    frame['label_end'] = frame.position_entry_time+pd.Timedelta(hours=1)
    frame['continue_end'] = frame.label_end
    frame['close_advantage_bps'] = frame.close_advantage_pnl/frame.decision_equity*10000
    return frame


def test_first_choice_keeps_loss_skips_original_exit_and_preserves_no_choice_position():
    frame = cases()
    positions = first_close_positions(frame, 'predicted_economic')
    assert positions.chosen.tolist() == [True, False]
    assert positions.first_selected_time.iloc[0] == frame.decision_time.iloc[2]
    assert positions.first_effect_pnl.tolist() == [-2., 0.]
    assert positions.first_effect_common_bps.tolist() == [-2., 0.]
    assert positions.selected_opportunities.tolist() == [2, 0]
    assert positions.first_effect_decision_bps.iloc[0] == pytest.approx(-2/9980*10000)
    assert positions.first_effect_decision_bps.iloc[0] != positions.first_effect_common_bps.iloc[0]
    metrics = first_close_metrics(positions)
    assert metrics['positions'] == 2 and metrics['selected_positions'] == 1 and metrics['selected_negative'] == 1
    assert metrics['all_position_mean_common_bps'] == -1. and metrics['selected_mean_common_bps'] == -2.
    assert metrics['later_selected_opportunities'] == 1


def test_later_predictions_and_targets_cannot_replace_first_choice_and_targets_do_not_select():
    frame = cases()
    original = first_close_positions(frame, 'predicted_economic')
    frame.loc[3, ['predicted_economic', 'close_advantage_pnl']] = [1e6, 1e6]
    frame['close_advantage_bps'] = frame.close_advantage_pnl/frame.decision_equity*10000
    altered = first_close_positions(frame, 'predicted_economic')
    pd.testing.assert_frame_equal(original, altered, check_exact=True)
    frame['close_advantage_pnl'] *= -1
    frame['close_advantage_bps'] *= -1
    changed = first_close_positions(frame, 'predicted_economic')
    pd.testing.assert_series_equal(original.first_selected_time, changed.first_selected_time, check_exact=True)
    assert changed.first_effect_pnl.iloc[0] == 2.


@pytest.mark.parametrize('damage', ['duplicate', 'order', 'mixed_direction', 'nonpositive_equity', 'nonfinite', 'bad_amount', 'mismatched_end'])
def test_invalid_episode_or_financial_inputs_are_rejected(damage):
    frame = cases()
    if damage == 'duplicate':
        frame.loc[1, 'decision_time'] = frame.decision_time.iloc[0]
    elif damage == 'order':
        frame = frame.iloc[::-1].reset_index(drop=True)
    elif damage == 'mixed_direction':
        frame.loc[0, 'direction'] = -1
    elif damage == 'nonpositive_equity':
        frame.loc[0, 'decision_equity'] = 0
    elif damage == 'nonfinite':
        frame.loc[0, 'predicted_economic'] = np.nan
    elif damage == 'bad_amount':
        frame.loc[0, 'close_advantage_pnl'] += 1
    else:
        frame.loc[0, 'continue_end'] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        first_close_positions(frame, 'predicted_economic')


def test_calendar_bootstrap_includes_zero_choices_empty_blocks_and_identical_paired_draws():
    frame = cases()
    positions = {name: first_close_positions(frame, 'predicted_'+name) for name in ['economic', 'boosted']}
    blocks, draws, replicates, intervals = paired_week_blocks(positions)
    assert len(blocks) == 13 and intervals['empty_blocks'] == 12
    assert blocks.end.iloc[-1] == pd.Timestamp('2021-12-31', tz='UTC')
    np.testing.assert_array_equal(draws.to_numpy(), np.random.default_rng(63).integers(0, 13, size=(1000, 13)))
    assert replicates.positions.eq(0).any() and replicates.economic.isna().sum() == replicates.positions.eq(0).sum()
    for i, selected in enumerate(draws.to_numpy()):
        samples = [0, 1]*int((selected == 0).sum())
        for name in positions:
            expected = positions[name].first_effect_common_bps.iloc[samples].mean() if samples else np.nan
            assert replicates[name].iloc[i] == pytest.approx(expected, nan_ok=True)
    valid = replicates.economic.dropna()
    assert intervals['intervals']['economic']['lower'] == pytest.approx(np.percentile(valid, 2.5))
    np.testing.assert_allclose(replicates.paired_difference, replicates.economic-replicates.boosted, equal_nan=True)
    repeated = paired_week_blocks(positions)
    for left, right in zip([blocks, draws, replicates], repeated[:3], strict=True):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert intervals == repeated[3]


def test_no_selection_keeps_every_position_and_zero_effect():
    frame = cases().assign(predicted_economic=-1., predicted_boosted=-1.)
    positions = {name: first_close_positions(frame, 'predicted_'+name) for name in ['economic', 'boosted']}
    metrics = first_close_metrics(positions['economic'])
    assert metrics['positions'] == 2 and metrics['selected_positions'] == 0
    assert metrics['selected_mean_common_bps'] is None and metrics['all_position_mean_common_bps'] == 0
    _, _, replicates, intervals = paired_week_blocks(positions)
    assert replicates.economic.dropna().eq(0).all()
    assert intervals['intervals']['economic']['lower'] == intervals['intervals']['economic']['upper'] == 0.


def financial_examples():
    frame = calibration_examples()
    frame['decision_equity'] = 10000.+np.arange(len(frame))*.01
    frame['close_advantage_pnl'] = frame.close_advantage_bps*frame.decision_equity/10000
    frame['continue_end'] = frame.label_end
    frame['continue_cash'] = 10000.
    return frame


def test_full_source_reproduction_and_first_effects_preserve_all_positions_without_admission(tmp_path, monkeypatch):
    monkeypatch.setattr('test_close_calibration.calibration_examples', financial_examples)
    previous = source_fixture(tmp_path, monkeypatch)
    def extra(_labels, ledger):
        augmented = ledger.copy()
        augmented[ECONOMIC_FEATURES[0]] = .25
        augmented[ECONOMIC_FEATURES[1]] = augmented.favorable_move*10000
        return augmented, {'rows': len(ledger), 'future_fill_or_outcome_used': False}
    monkeypatch.setattr('wonyotti_fr.close_economics.load_economic_inputs', extra)
    reference = run_close_economic_diagnosis(previous, tmp_path/'economic')
    out = run_first_close_diagnosis(reference, tmp_path/'first')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['positions'] == 60 and summary['opportunities'] == 600
    assert not any(summary[k] for k in ['new_models_fitted', 'profitability_accepted', 'trading_returns_evaluated', 'admission_decision_made'])
    for name in ['economic', 'boosted']:
        frame = pd.read_parquet(out/f'positions_{name}.parquet')
        assert len(frame) == 60 and frame.position_entry_time.nunique() == 60
    assert (out/'exclusion_ledger.parquet').read_bytes() == (reference/'exclusion_ledger.parquet').read_bytes()
    predictions = pd.read_parquet(reference/'predictions.parquet')
    predictions.loc[0, 'predicted_economic'] += 1
    predictions.to_parquet(reference/'predictions.parquet', index=False)
    hashes = json.loads((reference/'files.json').read_text())
    hashes['predictions.parquet'] = sha256(reference/'predictions.parquet')
    save_json(reference/'files.json', hashes)
    with pytest.raises(AssertionError):
        reproduce_economic_diagnosis(reference, tmp_path/'rejected')
