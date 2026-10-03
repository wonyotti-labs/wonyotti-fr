import json

import numpy as np
import pandas as pd
import pytest
from test_inventory_research import inventory_selection
from test_pullback_evaluation import bars
from test_rate_policy import rate_selection

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.inventory_labels import sizing_training
from wonyotti_fr.inventory_management import (
    CALIBRATION_PERIODS,
    TRAINING_PERIODS,
    InventoryActionModels,
    ReductionModel,
)
from wonyotti_fr.inventory_research import load_inventory_selection, run_inventory_selection
from wonyotti_fr.minute_management import ACTIONS, purged_window
from wonyotti_fr.rate_policy import calibrate_rates


def source_frame():
    times = pd.date_range('2020-01-03', periods=1100, freq='10h', tz='UTC').append(
        pd.date_range('2021-07-03', periods=300, freq='8h', tz='UTC'))
    rng = np.random.default_rng(52)
    frame = pd.DataFrame(rng.normal(size=(len(times), 36)), columns=InventoryActionModels.features)
    frame['remaining_fraction'] = rng.uniform(.01, 1, len(times))
    for i, a in enumerate(ACTIONS):
        frame[f'y_{a}'] = frame.iloc[:, i].gt(.5).astype(int)
    frame = frame.assign(end=times, entry_time=times-pd.Timedelta(minutes=5),
        episode_id=np.arange(len(times))+1, label_end=times+pd.Timedelta(minutes=1), usable=True)
    ends = times[::5]
    ledger = pd.DataFrame({'window_end': ends, 'size_label_end': ends+pd.Timedelta(minutes=1),
        'order_key': [str(i) for i in range(len(ends))], 'size_reason': 'supported', 'reduction_target': .3})
    return frame, ledger


def test_recent_sizing_and_calibration_keep_new_window_boundaries():
    frame, ledger = source_frame()
    recent = sizing_training(frame, ledger, TRAINING_PERIODS[1])
    old = sizing_training(frame, ledger)
    assert len(recent) > len(old) and recent.end.max() < pd.Timestamp('2021-07-01', tz='UTC')
    size, report = ReductionModel.fit(recent, TRAINING_PERIODS[1])
    assert report['rows'] == len(recent)
    with pytest.raises(ValueError, match='시간'):
        ReductionModel.fit(recent)
    calibration = purged_window(frame, *CALIBRATION_PERIODS[1])
    training = purged_window(frame, *TRAINING_PERIODS[1])
    manager, _, _ = InventoryActionModels.fit(training, calibration, 'logistic')
    rates = calibrate_rates(manager, calibration, CALIBRATION_PERIODS[1])
    assert rates['first_end'] >= pd.Timestamp('2021-07-02', tz='UTC')
    with pytest.raises(ValueError, match='시간'):
        calibrate_rates(manager, calibration)
    with pytest.raises(ValueError, match='사전 고정'):
        ReductionModel.fit(recent, ('2010-01-01', '2030-01-01'))
    np.testing.assert_allclose(size.predict(recent[size.features]), .3, atol=1e-12)


@pytest.mark.parametrize('change', ['training', 'calibration', 'in_sample', 'risk'])
def test_recent_loader_rejects_wrong_period_or_omitted_training_overlap(tmp_path, change):
    root = tmp_path / 'selection'
    frozen = inventory_selection(root)
    frozen.update(protocol='minute_inventory_recent_v20', training_period=list(TRAINING_PERIODS[1]),
                  calibration_period=list(CALIBRATION_PERIODS[1]), development_in_sample=True)
    load_inventory_selection(root, frozen)
    if change == 'training':
        frozen['training_period'] = list(TRAINING_PERIODS[0])
    elif change == 'calibration':
        frozen['calibration_period'] = list(CALIBRATION_PERIODS[0])
    elif change == 'in_sample':
        del frozen['development_in_sample']
    else:
        frozen['risk']['max_adds'] += 1
    with pytest.raises(ValueError):
        load_inventory_selection(root, frozen)


def test_recent_selection_fits_once_and_marks_in_sample_without_profit_selection(tmp_path, monkeypatch):
    reference, labels = tmp_path / 'reference', tmp_path / 'labels'
    parent = rate_selection(reference)
    labels.mkdir()
    (labels / 'files.json').write_text('{}')
    for name in ['manifest-1m.json', 'manifest-5m.json']:
        (tmp_path / name).write_text('{}')
    frame, ledger = source_frame()
    monkeypatch.setattr('wonyotti_fr.inventory_research.verify_files', lambda *_: None)
    monkeypatch.setattr('wonyotti_fr.inventory_research.load_inventory_labels', lambda *_: (frame, ledger))
    monkeypatch.setattr('wonyotti_fr.inventory_research.prepare_minute_period', lambda *_: (bars(), {}))
    out = run_inventory_selection(reference, labels, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / 'runs', latest_source=True)
    frozen = json.loads((out / 'frozen_selection.json').read_text())
    assert frozen['development_in_sample'] and frozen['risk'] == parent['risk']
    assert frozen['training_period'] == list(TRAINING_PERIODS[1])
    assert frozen['calibration_period'] == list(CALIBRATION_PERIODS[1])
    assert frozen['protocol'] == 'minute_inventory_recent_v20'
    assert frozen['candidate'] == 0 and not json.loads((out / 'summary.json').read_text())['selected_by_profit']
    assert {r['period'] for r in json.loads((out / 'development.json').read_text())} == {'2021_in_sample'}
    assert sha256(out / 'action_model.json') == sha256(reference / 'action_model.json')
    from wonyotti_fr.event_research import load_selection
    load_selection(out)
    # 새로운 기간 메타데이터가 기존 v19 로딩의 기간 검사를 우회하지 못하게 한다.
    frozen['protocol'] = 'minute_inventory_v19'
    save_json(out / 'frozen_selection.json', frozen)
    save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
    with pytest.raises(ValueError):
        load_selection(out)
