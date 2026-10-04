import copy
import json

import numpy as np
import pandas as pd
import pytest
from test_close_learning import examples
from test_first_state import fixed_budget  # noqa: F401

from wonyotti_fr.close_calibration import (
    CloseAmountCalibration,
    calibration_admission,
    calibration_splits,
    fit_close_calibration,
    reproduce_close_diagnosis,
    run_close_calibration_diagnosis,
)
from wonyotti_fr.close_effect import CLOSE_FEATURES
from wonyotti_fr.close_learning import run_close_learning_diagnosis
from wonyotti_fr.common import save_json, sha256


def calibration_examples():
    source = examples().iloc[:1000].copy()
    frames = []
    for start, count in [('2021-01-01', 160), ('2021-07-02', 60), ('2021-10-02', 60)]:
        frame = pd.concat([source]*2, ignore_index=True).iloc[:count*10].copy()
        positions = np.repeat(pd.date_range(start, periods=count, freq='D', tz='UTC'), 10)
        frame['position_entry_time'] = positions
        frame['decision_time'] = frame.position_entry_time+pd.to_timedelta(np.tile(np.arange(1, 11)*5, count), unit='min')
        frame['label_end'] = frame.position_entry_time+pd.Timedelta(hours=2)
        frame['direction'] = np.repeat(np.where(np.arange(count)%2, 1., -1.), 10)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_affine_fit_matches_independent_normal_equations_and_roundtrip():
    p = np.linspace(-4., 8., 200)
    y = 2*p+3+np.sin(p)
    w = np.linspace(.5, 1.5, 200)
    model, support = CloseAmountCalibration.fit(p, y, w)
    x = np.c_[p, np.ones(len(p))]
    expected = np.linalg.solve(x.T@(w[:, None]*x), x.T@(w*y))
    np.testing.assert_allclose([model.data['slope'], model.data['intercept']], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.predict(p), x@expected, rtol=0, atol=1e-12)
    assert support['rows'] == len(p)
    restored = CloseAmountCalibration.from_dict(model.to_dict())
    model.data['slope'] += 1
    np.testing.assert_allclose(restored.predict(p), x@expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize('kind', ['constant', 'negative'])
def test_constant_prediction_or_negative_correlation_shrinks_without_reversing_rank(kind):
    p = np.full(200, 1.3) if kind == 'constant' else np.linspace(-1., 1., 200)
    y, w = -np.arange(200, dtype=float), np.linspace(.5, 1.5, 200)
    model, _ = CloseAmountCalibration.fit(p, y, w)
    assert model.data['slope'] == 0
    np.testing.assert_allclose(model.predict(p), np.average(y, weights=w), atol=1e-12)


@pytest.mark.parametrize('damage', ['short', 'zero_weight', 'nan', 'negative_slope', 'nonfinite_intercept'])
def test_calibration_rejects_invalid_inputs_or_models(damage):
    p, y, w = np.arange(100, dtype=float), np.arange(100, dtype=float), np.ones(100)
    if damage in {'negative_slope', 'nonfinite_intercept'}:
        data = {'format': 'close_amount_nonnegative_affine_v1', 'slope': -1. if damage == 'negative_slope' else 1.,
            'intercept': 0. if damage == 'negative_slope' else float('inf')}
        with pytest.raises(ValueError):
            CloseAmountCalibration.from_dict(data)
        return
    if damage == 'short':
        p, y, w = p[:99], y[:99], w[:99]
    elif damage == 'zero_weight':
        w[0] = 0
    else:
        p[0] = np.nan
    with pytest.raises(ValueError):
        CloseAmountCalibration.fit(p, y, w)


def test_three_disjoint_splits_keep_boundary_and_censored_rows():
    frame = calibration_examples()
    extra = frame.iloc[:3].copy()
    extra['position_entry_time'] = pd.to_datetime(['2021-06-29T00:00Z', '2021-07-01T00:00Z', '2021-12-30T00:00Z'])
    extra['decision_time'] = pd.to_datetime(['2021-06-29T00:05Z', '2021-07-02T00:01Z', '2021-12-30T00:05Z'])
    extra['label_end'] = pd.to_datetime(['2021-06-30T00:00Z', '2021-07-02T01:00Z', '2021-12-31T00:00Z'])
    extra.loc[2, 'label_status'] = 'right_censored'
    frame = pd.concat([frame, extra]).sort_values('decision_time').reset_index(drop=True)
    rows, assignment = calibration_splits(frame)
    assert {k: len(v) for k, v in rows.items()} == {'training': 1600, 'calibration': 600, 'diagnosis': 600}
    assert len(assignment) == len(frame) and (assignment.split == 'excluded_boundary').sum() == 2
    assert (assignment.split == 'excluded_not_closed').sum() == 1
    for left, right in [('training', 'calibration'), ('training', 'diagnosis'), ('calibration', 'diagnosis')]:
        assert not set(rows[left].position_entry_time)&set(rows[right].position_entry_time)
        assert rows[left].label_end.max() < rows[right].decision_time.min()
    for field, value in [('label_end', pd.NaT), (CLOSE_FEATURES[0], float('nan')), ('direction', 0)]:
        damaged = frame.copy()
        damaged.loc[:2000, field] = value
        with pytest.raises(ValueError):
            calibration_splits(damaged)


def test_all_ten_admission_checks_are_necessary():
    candidate = {'rows': 200, 'positions': 40, 'weighted_mse': 50., 'mse': 50., 'selected': 100,
        'selected_positions': 30, 'selected_weighted_mean_bps': 1., 'selected_mean_bps': 1.}
    references = ['ridge', 'boosted', 'training_constant', 'calibration_constant']
    metrics = {k: {**candidate, 'weighted_mse': 100., 'mse': 100.} for k in references}
    metrics['calibrated'] = candidate
    assert calibration_admission(metrics)['close_calibration_admitted']
    assert len(calibration_admission(metrics)['checks']) == 10
    cases = [(k, 'weighted_mse', 50.) for k in references]+[(k, 'mse', 49.) for k in ['ridge', 'boosted']]
    cases += [('calibrated', 'selected', 99), ('calibrated', 'selected_positions', 29),
        ('calibrated', 'selected_weighted_mean_bps', 0.), ('calibrated', 'selected_mean_bps', 0.)]
    for name, field, value in cases:
        altered = copy.deepcopy(metrics)
        altered[name][field] = value
        assert not calibration_admission(altered)['close_calibration_admitted']


def test_later_inputs_targets_and_ends_do_not_change_earlier_models_weights_or_calibration(tmp_path):
    rows, _ = calibration_splits(calibration_examples())
    outputs = {}
    for name in ['original', 'future', 'calibration']:
        altered = copy.deepcopy(rows)
        if name != 'original':
            part = altered['diagnosis' if name == 'future' else 'calibration']
            part[CLOSE_FEATURES[0]] *= -100
            part['close_advantage_bps'] += 100
            part['label_end'] += pd.Timedelta(hours=1)
        out = tmp_path/name
        out.mkdir()
        fit_close_calibration(altered, out)
        outputs[name] = out
    for name in ['future', 'calibration']:
        for file in ['training_used.parquet', 'training_weights.parquet', 'models.json']:
            assert (outputs['original']/file).read_bytes() == (outputs[name]/file).read_bytes()
    for file in ['calibration_used.parquet', 'calibration_weights.parquet', 'calibration.json', 'calibration_predictions.parquet']:
        assert (outputs['original']/file).read_bytes() == (outputs['future']/file).read_bytes()
    assert (outputs['original']/'calibration.json').read_bytes() != (outputs['calibration']/'calibration.json').read_bytes()


def source_fixture(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    save_json(source/'files.json', {})
    frame = calibration_examples()
    monkeypatch.setattr('wonyotti_fr.close_learning.load_close_training', lambda _: (frame.copy(), {'labels_files_sha256': sha256(source/'files.json')}))
    return run_close_learning_diagnosis(source, tmp_path/'previous')


def test_full_diagnosis_reproduces_all_original_outputs_and_rejects_refingerprinted_predictions(tmp_path, monkeypatch):
    reference = source_fixture(tmp_path, monkeypatch)
    out = run_close_calibration_diagnosis(reference, tmp_path/'new')
    saved = json.loads((out/'summary.json').read_text())
    assert saved['complete'] and saved['all_previous_outputs_reproduced'] and not saved['profitability_accepted']
    pd.testing.assert_frame_equal(pd.read_parquet(reference/'diagnosis_used.parquet'), pd.read_parquet(out/'diagnosis_used.parquet'), check_exact=True)
    assert len(pd.read_parquet(out/'exclusion_ledger.parquet')) == 2800
    prediction = pd.read_parquet(reference/'predictions.parquet')
    prediction.loc[0, 'predicted_boosted'] += 1
    prediction.to_parquet(reference/'predictions.parquet', index=False)
    fingerprints = json.loads((reference/'files.json').read_text())
    fingerprints['predictions.parquet'] = sha256(reference/'predictions.parquet')
    save_json(reference/'files.json', fingerprints)
    with pytest.raises(AssertionError):
        reproduce_close_diagnosis(reference, tmp_path/'rejected')
