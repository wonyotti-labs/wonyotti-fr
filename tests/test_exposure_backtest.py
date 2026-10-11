from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_exposure_engine import bar, config
from test_exposure_execution import model

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.exposure_backtest import backtest, verify_path
from wonyotti_fr.exposure_policy import ExposurePolicy
from wonyotti_fr.holding_stress import read_json


def data():
    random = np.random.default_rng(98)
    rows = []
    for i in range(1500):
        row = bar(i, price=100.+2*np.sin(i/80), funding=.0001 if i % 480 == 0 else 0.)
        row['time'], row['end'] = pd.Timestamp(row['time']), pd.Timestamp(row['end'])
        row['feature_end'] = row['end'].floor('5min')
        row.update(zip(MARKET_FEATURES, random.normal(size=14), strict=True))
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize('kind', ['change', 'absolute', 'constant'])
@pytest.mark.parametrize('delay', [0, 1])
def test_streamed_all_rows_reference_and_checkpoints(tmp_path, kind, delay):
    frame = data()
    learner = None if kind == 'constant' else model(kind)
    policy = ExposurePolicy(kind, learner, constant_exposure=.25 if kind == 'constant' else None)
    settings = config(signal_delay_bars=delay, stop_fraction=.04, cooldown_bars=15, max_hold_bars=333)
    output = tmp_path/'run'
    metrics = backtest(frame, policy, settings, output, batch_size=51)
    proof = verify_path(frame, kind, None if learner is None else learner.to_dict(), settings, output, tmp_path/'review.json', constant_exposure=.25 if kind == 'constant' else None)
    assert proof['complete'] and proof['rows']['equity'] == len(frame)
    assert proof['metrics']['final_equity'] == pytest.approx(metrics['final_equity'])
    assert read_json(output/'storage.json')['json_restored_checkpoints'] == [500, 1000]
    curve = pd.read_parquet(output/'equity.parquet')
    assert curve.decision.iloc[:4].isna().all() and pd.notna(curve.decision.iloc[4])
    assert len(pd.read_parquet(output/'trades.parquet')) == metrics['closed_trades']


@pytest.mark.parametrize('tamper', ['equity', 'fills', 'trades', 'policy', 'extra_equity'])
def test_full_verifier_rejects_tampering_even_when_output_hash_is_resealed(tmp_path, tamper):
    frame, settings, output = data(), config(max_hold_bars=111), tmp_path/'run'
    backtest(frame, ExposurePolicy('constant', constant_exposure=.25), settings, output, batch_size=51)
    if tamper == 'policy':
        save_json(output/'policy.json', {'kind': 'change', 'model': None})
    else:
        name = 'equity' if tamper == 'extra_equity' else tamper
        target = output/f'{name}.parquet'
        table = pd.read_parquet(target)
        if tamper == 'extra_equity':
            table = pd.concat([table, table.tail(1)], ignore_index=True)
        else:
            field = {'equity': 'equity', 'fills': 'fee', 'trades': 'net_pnl'}[tamper]
            table.loc[table.index[-1], field] += 1.
        table.to_parquet(target, index=False)
    save_json(output/'files.json', {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != 'files.json'})
    with pytest.raises(ValueError):
        verify_path(frame, 'constant', None, settings, output, tmp_path/'review.json', constant_exposure=.25)
    assert not (tmp_path/'review.json').exists()


def test_no_finite_features_preserves_empty_execution_files(tmp_path):
    frame = data()
    frame[MARKET_FEATURES] = np.nan
    learner = model('change')
    settings = config()
    output = tmp_path/'run'
    metrics = backtest(frame, ExposurePolicy('change', learner), settings, output)
    assert metrics['closed_trades'] == 0 and metrics['total_return'] == 0
    assert pd.read_parquet(output/'fills.parquet').empty
    verify_path(frame, 'change', learner.to_dict(), settings, output, tmp_path/'review.json')


def test_changed_configuration_rejected_before_replay(tmp_path):
    frame, settings, output = data(), config(), tmp_path/'run'
    backtest(frame, ExposurePolicy('constant', constant_exposure=.25), settings, output)
    with pytest.raises(ValueError, match='설정'):
        verify_path(frame, 'constant', None, replace(settings, fee_bps=6), output, tmp_path/'review.json', constant_exposure=.25)


def test_constant_mismatch_or_omission_rejected_before_replay(tmp_path):
    frame, settings, output = data(), config(), tmp_path/'run'
    backtest(frame, ExposurePolicy('constant', constant_exposure=.25), settings, output)
    for value in [None, .2, float('nan')]:
        with pytest.raises(ValueError, match='설정'):
            verify_path(frame, 'constant', None, settings, output, tmp_path/'review.json', constant_exposure=value)
    assert not (tmp_path/'review.json').exists()
    assert read_json(output/'policy.json')['constant_exposure'] == .25
    assert verify_path(frame, 'constant', None, settings, output, tmp_path/'review.json', constant_exposure=.25)['complete']
