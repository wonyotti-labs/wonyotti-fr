import json

import pandas as pd
import pytest
from test_close_effect import CollectionPolicy
from test_first_exit_research import fixture

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.first_exit_research import run_first_exit_control
from wonyotti_fr.first_exit_verification import verify_first_exit_run


def make_run(tmp_path, monkeypatch):
    source, proof, parent, _, _ = fixture(tmp_path, monkeypatch)
    from wonyotti_fr import first_exit_research as research
    monkeypatch.setattr('wonyotti_fr.first_exit_verification.prepare_minute_period', research.prepare_minute_period)
    monkeypatch.setattr('wonyotti_fr.first_exit_verification.load_selection', lambda _: (json.loads((parent/'frozen_selection.json').read_text()), CollectionPolicy()))
    out = run_first_exit_control(source, proof, sha256(proof), tmp_path/'runs')
    return out


def reseal(folder):
    save_json(folder/'files.json', {p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p.name != 'files.json'})


def test_independent_all_four_replays_and_manual_cash_flow(tmp_path, monkeypatch):
    out = make_run(tmp_path, monkeypatch)
    review = verify_first_exit_run(out, sha256(out/'files.json'), tmp_path/'reviews')
    proof = json.loads((review/'verification.json').read_text())
    assert proof['complete'] and proof['all_four_executions_independently_replayed']
    assert proof['cash_fees_funding_and_net_verified'] and not proof['profitability_accepted']
    assert len(proof['runs']) == 4
    assert proof['runs']['2021/always_first_exit']['control_decisions'] > 0
    with pytest.raises(ValueError, match='지정 지문'):
        verify_first_exit_run(out, '0'*64, tmp_path/'wrong')


@pytest.mark.parametrize('damage', ['trace', 'fill', 'state', 'risk', 'breakdown', 'summary'])
def test_resigned_semantic_tampering_is_rejected(tmp_path, monkeypatch, damage):
    out = make_run(tmp_path, monkeypatch)
    child = out/'2021/always_first_exit'
    if damage == 'trace':
        path = out/'2021/control_decisions.parquet'
        frame = pd.read_parquet(path)
        frame.loc[frame.changed, 'original_intent'] = 'exit'
        frame.to_parquet(path, index=False)
    elif damage == 'fill':
        path = child/'fills.parquet'
        frame = pd.read_parquet(path)
        frame.loc[0, 'fee'] += 1
        frame.to_parquet(path, index=False)
    elif damage == 'state':
        path = child/'final_state.json'
        data = json.loads(path.read_text())
        data['cash'] += 1
        save_json(path, data)
    elif damage == 'risk':
        path = child/'config.json'
        data = json.loads(path.read_text())
        data['fee_bps'] += 1
        save_json(path, data)
    elif damage == 'breakdown':
        path = child/'breakdown.json'
        data = json.loads(path.read_text())
        data[0]['net_pnl'] += 1
        save_json(path, data)
    else:
        path = out/'summary.json'
        data = json.loads(path.read_text())
        data['profitability_accepted'] = True
        save_json(path, data)
    reseal(child)
    reseal(out/'2021')
    runs = json.loads((out/'runs.json').read_text())
    runs['2021'] = sha256(out/'2021/files.json')
    runs['2021/always_first_exit'] = sha256(child/'files.json')
    save_json(out/'runs.json', runs)
    reseal(out)
    with pytest.raises((ValueError, AssertionError)):
        verify_first_exit_run(out, sha256(out/'files.json'), tmp_path/'reviews')
    review = next((tmp_path/'reviews').iterdir())
    assert (review/'failure.json').exists() and not (review/'verification.json').exists()
