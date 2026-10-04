import copy
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from test_close_calibration import source_fixture
from test_close_learning import examples, real_source, reseal
from test_first_state import fixed_budget  # noqa: F401
from test_market_liquidity import event, hold
from test_net_exit_state import bot
from threadpoolctl import threadpool_limits

from wonyotti_fr.close_economics import (
    ECONOMIC_FEATURES,
    EconomicCloseModel,
    current_economic_values,
    economic_admission,
    load_economic_inputs,
    run_close_economic_diagnosis,
)
from wonyotti_fr.close_learning_inputs import snapshot_close_values
from wonyotti_fr.common import sha256
from wonyotti_fr.engine import EngineConfig, TradingEngine
from wonyotti_fr.entry_regression import REGRESSION_SETTINGS
from wonyotti_fr.exit_move_state import ExitMovePolicy


@pytest.mark.parametrize('direction', [-1, 1])
def test_current_values_match_manual_partial_realization_fees_funding_and_engine(direction):
    risk = EngineConfig(bar_seconds=60, stop_fraction=0., daily_loss_limit=1., max_drawdown=1., cooldown_bars=0,
        fee_bps=5, slippage_bps=3, entry_fraction=.5, addition_fraction=.5, allow_adverse_add=True)
    engine, gross, fees, funding, entry = TradingEngine(risk), 0., 0., 0., 0.
    for minute, price, intent, rate in [(0, 100., 'enter_long' if direction == 1 else 'enter_short', 0.),
        (1, 102., 'reduce', .001), (2, 101., 'hold', -.001)]:
        oldq = engine.state['quantity']
        funding += oldq*price*rate
        engine.state['pending'] = intent
        result = engine.step(event(minute, price, funding=rate), hold)
        for fill in result['fills']:
            fees += fill['fee']
            if oldq == 0:
                entry = fill['price']
            else:
                gross += direction*abs(fill['delta_quantity'])*(fill['price']-entry)
    state = engine.snapshot()
    q, price = state['quantity'], state['last_close']
    equity, execution = state['cash']+q*price, price*(1-direction*.0003)
    manual_net = gross-fees-funding+q*(execution-entry)-abs(q)*execution*.0005
    values = current_economic_values(state, risk)
    np.testing.assert_allclose(values, [abs(q)*price/equity, manual_net/equity*10000], rtol=0, atol=1e-10)
    assert values[1] == pytest.approx(engine.view(price)['estimated_exit_net']/equity*10000)
    altered = copy.deepcopy(state)
    altered['future_exit_time'], altered['future_fill_price'], altered['future_outcome'] = '2099-01-01', 1e9, -1e9
    np.testing.assert_array_equal(values, current_economic_values(altered, risk))


def test_original_features_alias_cost_state_that_changes_actual_exit_threshold(tmp_path, monkeypatch):
    root = real_source(tmp_path, monkeypatch)
    with sqlite3.connect((root/'outcomes.sqlite').as_uri()+'?mode=ro&immutable=1', uri=True) as connection:
        op = json.loads(connection.execute('SELECT payload FROM outcomes ORDER BY sequence LIMIT 1').fetchone()[0])['opportunity']
    state = op['state']
    state['last_close'] = state['entry_price']*(1+.03*np.sign(state['quantity']))
    state['active_trade'].update(gross_realized=0., funding_cost=0., fees=0.)
    expensive = copy.deepcopy(op)
    expensive['state']['active_trade']['fees'] = abs(state['quantity'])*state['last_close']
    risk = EngineConfig(fee_bps=5, slippage_bps=3)
    bar = {'features': np.ones(14), 'minute_features': np.ones(4)}
    np.testing.assert_array_equal(snapshot_close_values(op, bar), snapshot_close_values(expensive, bar))
    original, changed = current_economic_values(state, risk), current_economic_values(expensive['state'], risk)
    assert original[1] > 0 > changed[1]
    policy = ExitMovePolicy(bot((0., .05, .8)), .02)
    assert policy.action_threshold('exit', {'estimated_exit_net': float(original[1]), 'favorable_move': .03}) < policy.action_threshold('exit', {'estimated_exit_net': float(changed[1]), 'favorable_move': .03})
    larger_equity = copy.deepcopy(op)
    larger_equity['state']['cash'] += 10000
    np.testing.assert_array_equal(snapshot_close_values(op, bar), snapshot_close_values(larger_equity, bar))
    assert current_economic_values(larger_equity['state'], risk)[0] < original[0]


@pytest.mark.parametrize('damage', ['zero_quantity', 'nonpositive_equity', 'nan_price'])
def test_invalid_financial_state_is_rejected_instead_of_silently_excluded(damage):
    state = {'quantity': 1., 'last_close': 100., 'entry_price': 99., 'cash': 1000.,
        'active_trade': {'gross_realized': 0., 'fees': 1., 'funding_cost': 2.}}
    state[{'zero_quantity': 'quantity', 'nonpositive_equity': 'cash', 'nan_price': 'last_close'}[damage]] = {'zero_quantity': 0., 'nonpositive_equity': -100., 'nan_price': np.nan}[damage]
    with pytest.raises(ValueError):
        current_economic_values(state, EngineConfig())


def test_all_snapshot_inputs_preserve_original_rows_and_readonly_journal(tmp_path, monkeypatch):
    root = real_source(tmp_path, monkeypatch)
    ledger = pd.read_parquet(root/'opportunity_ledger.parquet')
    before = sha256(root/'outcomes.sqlite')
    augmented, verification = load_economic_inputs(root, ledger)
    pd.testing.assert_frame_equal(augmented.drop(columns=ECONOMIC_FEATURES), ledger, check_exact=True)
    assert verification['rows'] == len(ledger) and verification['journal_read_only']
    assert before == sha256(root/'outcomes.sqlite') and np.isfinite(augmented[ECONOMIC_FEATURES]).all().all()
    with sqlite3.connect(root/'outcomes.sqlite') as connection:
        connection.execute("UPDATE outcomes SET previous_hash='changed' WHERE sequence=0")
    connection.close()
    reseal(root)
    with pytest.raises(ValueError, match='해시'):
        load_economic_inputs(root, ledger)


def test_extended_numeric_model_matches_independent_refit_and_ignores_future_validation_values():
    frame = examples().iloc[:500]
    rng = np.random.default_rng(62)
    x = np.c_[frame[EconomicCloseModel.features[:-2]], rng.uniform(.05, .5, len(frame)), rng.normal(size=len(frame))]
    y, w = frame.close_advantage_bps.to_numpy(), np.ones(len(frame))
    model, _ = EconomicCloseModel.fit(x, y, w, x[:50])
    with threadpool_limits(limits=1):
        library = HistGradientBoostingRegressor(**REGRESSION_SETTINGS).fit(x, y, sample_weight=w)
        np.testing.assert_allclose(model.predict(x), library.predict(x), rtol=0, atol=1e-10)
    changed, _ = EconomicCloseModel.fit(x, y, w, x[:50]*-100)
    assert model.to_dict() == changed.to_dict() and len(model.features) == 52
    altered = model.to_dict()
    altered['features'] = altered['features'][:-2]
    with pytest.raises(ValueError):
        EconomicCloseModel.from_dict(altered)


def test_all_nine_gates_required():
    candidate = {'rows': 200, 'positions': 40, 'weighted_mse': 50., 'mse': 50., 'selected': 100,
        'selected_positions': 30, 'selected_weighted_mean_bps': 1., 'selected_mean_bps': 1.}
    metrics = {k: {**candidate, 'weighted_mse': 100., 'mse': 100.} for k in ['boosted', 'ridge', 'constant']}
    metrics['economic'] = candidate
    assert economic_admission(metrics)['economic_inputs_admitted'] and len(economic_admission(metrics)['checks']) == 9
    cases = [(k, 'weighted_mse', 50.) for k in ['boosted', 'ridge', 'constant']]
    cases += [(k, 'mse', 49.) for k in ['boosted', 'ridge']]
    cases += [('economic', 'selected', 99), ('economic', 'selected_positions', 29),
        ('economic', 'selected_weighted_mean_bps', 0.), ('economic', 'selected_mean_bps', 0.)]
    for name, field, value in cases:
        damaged = copy.deepcopy(metrics)
        damaged[name][field] = value
        assert not economic_admission(damaged)['economic_inputs_admitted']


def test_full_pipeline_preserves_reference_predictions_and_extra_inputs_do_not_change_old_rows(tmp_path, monkeypatch):
    reference = source_fixture(tmp_path, monkeypatch)
    def extra(_labels, ledger):
        augmented = ledger.copy()
        augmented[ECONOMIC_FEATURES[0]] = .25
        augmented[ECONOMIC_FEATURES[1]] = augmented.favorable_move*10000
        return augmented, {'rows': len(ledger), 'future_fill_or_outcome_used': False}
    monkeypatch.setattr('wonyotti_fr.close_economics.load_economic_inputs', extra)
    out = run_close_economic_diagnosis(reference, tmp_path/'new')
    result = json.loads((out/'summary.json').read_text())
    assert result['complete'] and not result['profitability_accepted'] and result['all_original_rows_and_weights_preserved']
    pd.testing.assert_frame_equal(pd.read_parquet(out/'predictions.parquet').drop(columns='predicted_economic'), pd.read_parquet(reference/'predictions.parquet'), check_exact=True)
    assert (out/'previous_models.json').read_bytes() == (reference/'models.json').read_bytes()
    for name in ['training', 'diagnosis']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}_weights.parquet'), pd.read_parquet(reference/f'{name}_weights.parquet'), check_exact=True)
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{name}_used.parquet').drop(columns=ECONOMIC_FEATURES), pd.read_parquet(reference/f'{name}_used.parquet'), check_exact=True)
    assert Path(out/'input_verification.json').exists()
