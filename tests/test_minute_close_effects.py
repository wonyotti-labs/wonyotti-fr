import copy
import json

import pandas as pd
import pytest
from test_close_effect import CollectionPolicy, collection_fixture
from test_close_learning import real_source, reseal

from wonyotti_fr.close_effect import collect_close_effects
from wonyotti_fr.close_learning_inputs import load_close_training
from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.minute_close_effects import (
    preserve_generation_source,
    run_minute_close_effect_labels,
    verify_legacy_grid,
)
from wonyotti_fr.outcome_journal import OutcomeJournal


def test_all_minute_opportunities_preserve_every_legacy_record_and_resume_before_first_grid(tmp_path):
    frame, cfg, parent = collection_fixture(tmp_path)
    results, records = {}, {}
    for seconds in [300, 60]:
        with OutcomeJournal(tmp_path/f'{seconds}.sqlite', {'decision_seconds': seconds}) as journal:
            if seconds == 60:
                partial, _, complete = collect_close_effects(frame, CollectionPolicy(), cfg, parent/'candidate-00',
                    tmp_path/'partial', journal, frame.end.iloc[30], 1, decision_seconds=60)
                assert not complete and len(partial) == 1
                assert pd.Timestamp(partial[0]['decision_time']).minute % 5
            rows, counts, complete = collect_close_effects(frame, CollectionPolicy(), cfg, parent/'candidate-00',
                tmp_path/f'full-{seconds}', journal, frame.end.iloc[30], decision_seconds=seconds)
            assert complete and counts['boundaries'] == len(frame)//(seconds//60)
            assert counts['boundaries'] == counts['opportunities']+sum(v for k, v in counts.items() if k.startswith('excluded_'))
            records[seconds] = {json.loads(row[0])['opportunity']['decision_time']: json.loads(row[0])
                for row in journal.connection.execute('SELECT payload FROM outcomes ORDER BY sequence')}
        results[seconds] = rows
    assert len(results[60]) > len(results[300])
    assert {time: records[60][time] for time in records[300]} == records[300]
    expected = pd.read_parquet(parent/'candidate-00/equity.parquet')
    eligible = expected.loc[expected.quantity.ne(0) & expected.policy_event.eq('test_management'), 'time']
    assert [row['decision_time'] for row in results[60]] == eligible.tolist()
    assert any(r['label_status'] == 'right_censored' for r in results[60])
    assert any(r['label_status'] == 'closed' and r['original_intent'] == 'exit' for r in results[60])


@pytest.mark.parametrize('seconds', [True, 60., 0, 120])
def test_unsupported_grid_is_rejected_before_output(tmp_path, seconds):
    frame, cfg, parent = collection_fixture(tmp_path)
    with OutcomeJournal(tmp_path/'invalid.sqlite', {'case': 'invalid'}) as journal:
        with pytest.raises(ValueError, match='간격|특징'):
            collect_close_effects(frame, CollectionPolicy(), cfg, parent/'candidate-00', tmp_path/'invalid',
                journal, frame.end.iloc[-1], decision_seconds=seconds)
    assert not (tmp_path/'invalid').exists()


def test_complete_minute_pipeline_and_partial_resume_validate_all_cashflows_and_legacy_grid(tmp_path, monkeypatch):
    legacy = real_source(tmp_path, monkeypatch)
    original_hashes = {p: sha256(legacy/p) for p in json.loads((legacy/'files.json').read_text())}
    out = run_minute_close_effect_labels(legacy, tmp_path/'minute-runs', max_opportunities=1)
    partial = json.loads((out/'legacy_parity.json').read_text())
    assert not partial['complete_source'] and partial['matched_rows'] == 0
    run_minute_close_effect_labels(legacy, tmp_path/'unused', resume=out)
    full, proof = load_close_training(out, decision_seconds=60)
    original, _ = load_close_training(legacy)
    assert proof['all_actual_inputs_and_cashflows_verified'] and proof['legacy_grid']['matched_rows'] == len(original)
    assert proof['legacy_grid']['additional_rows'] == len(full)-len(original) > 0
    assert proof['legacy_grid']['complete_source']
    assert all(sha256(legacy/p) == digest for p, digest in original_hashes.items())
    for name, checksum in json.loads((out/'files.json').read_text()).items():
        assert sha256(out/name) == checksum
    with pytest.raises(ValueError, match='파일'):
        load_close_training(out)
    with pytest.raises(ValueError, match='완료'):
        run_minute_close_effect_labels(legacy, tmp_path/'unused', resume=out)


@pytest.mark.parametrize('damage', ['grid_proof', 'minute_ledger', 'legacy_connection', 'generation_code'])
def test_minute_loader_rejects_resigned_tampering(tmp_path, monkeypatch, damage):
    legacy = real_source(tmp_path, monkeypatch)
    out = run_minute_close_effect_labels(legacy, tmp_path/'minute-runs')
    if damage == 'grid_proof':
        proof = json.loads((out/'legacy_parity.json').read_text())
        proof['matched_rows'] += 1
        save_json(out/'legacy_parity.json', proof)
    elif damage == 'minute_ledger':
        rows = pd.read_parquet(out/'opportunity_ledger.parquet')
        off_grid = rows.decision_time.dt.minute.mod(5).ne(0) & rows.label_status.eq('closed')
        rows.loc[rows.index[off_grid][0], 'close_advantage_bps'] += 1
        rows.to_parquet(out/'opportunity_ledger.parquet', index=False)
        rows[rows.label_status.eq('closed')].reset_index(drop=True).to_parquet(out/'training_labels.parquet', index=False)
    elif damage == 'legacy_connection':
        settings = json.loads((out/'manifest.json').read_text())
        settings['settings']['legacy_files_sha256'] = 'changed'
        save_json(out/'manifest.json', settings)
    else:
        (out/'generation_source/wonyotti_fr/engine.py').write_text('changed')
    reseal(out)
    with pytest.raises((ValueError, AssertionError)):
        load_close_training(out, decision_seconds=60)


def test_prefix_grid_comparison_and_source_preservation_reject_mutations(tmp_path):
    times = pd.date_range('2021-01-01T00:01Z', periods=15, freq='min')
    frame = pd.DataFrame({'decision_time': times, 'effect': range(15)})
    legacy = frame[frame.decision_time.dt.minute.mod(5).eq(0)].reset_index(drop=True)
    assert verify_legacy_grid(frame.iloc[:3], legacy, False)['matched_rows'] == 0
    assert verify_legacy_grid(frame.iloc[:7], legacy, False)['matched_rows'] == 1
    assert verify_legacy_grid(frame, legacy, True)['matched_rows'] == 3
    changed = frame.copy()
    changed.loc[4, 'effect'] += 1
    with pytest.raises(AssertionError):
        verify_legacy_grid(changed, legacy, True)
    with pytest.raises(AssertionError):
        verify_legacy_grid(frame.iloc[:7], legacy, True)
    source = tmp_path/'code_snapshot'
    source.mkdir()
    (source/'engine.py').write_text('original')
    hashes = {'engine.py': sha256(source/'engine.py')}
    preserve_generation_source(tmp_path, hashes)
    preserve_generation_source(tmp_path, hashes)
    target = tmp_path/'generation_source/wonyotti_fr/engine.py'
    target.write_text('changed')
    assert (source/'engine.py').read_text() == 'original'
    with pytest.raises(ValueError, match='지문'):
        preserve_generation_source(tmp_path, hashes)


def test_resume_rejects_changed_grid_and_source_identity(tmp_path, monkeypatch):
    legacy = real_source(tmp_path, monkeypatch)
    out = run_minute_close_effect_labels(legacy, tmp_path/'minute-runs', max_opportunities=1)
    manifest = json.loads((out/'manifest.json').read_text())
    damaged = copy.deepcopy(manifest)
    damaged['settings']['decision_seconds'] = 300
    save_json(out/'manifest.json', damaged)
    with pytest.raises(ValueError, match='변경'):
        run_minute_close_effect_labels(legacy, tmp_path/'unused', resume=out)
    save_json(out/'manifest.json', manifest)
    (legacy/'support.json').write_text('[]')
    reseal(legacy)
    with pytest.raises(ValueError, match='변경'):
        run_minute_close_effect_labels(legacy, tmp_path/'unused', resume=out)
