from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.cross_market import (
    CROSS_FEATURES,
    RELATIVE_FEATURES,
    attach_cross_market,
    cross_market_support,
    cross_market_table,
)
from wonyotti_fr.cross_market_reference import verify_cross_market, verify_cross_support
from wonyotti_fr.cross_market_research import checked_cross_market_source, checked_cross_reference


def source():
    i = np.arange(700)
    time = pd.date_range('2020-01-01', periods=len(i), freq='5min', tz='UTC')
    btc = pd.DataFrame({'time': time, 'end': time + pd.Timedelta(minutes=5),
        'close': 10000 * np.exp(np.cumsum(.0003 * np.cos(i / 8)))})
    eth = btc.copy()
    eth['close'] = 100 * np.exp(np.cumsum(.001 * np.sin(i / 10) + .0002))
    pivot = btc.end.iloc[300]
    ends = pd.to_datetime([pivot - pd.Timedelta(nanoseconds=1), pivot,
        pivot + pd.Timedelta(nanoseconds=1), pivot + pd.Timedelta(minutes=4), btc.end.iloc[301]], utc=True)
    frame = pd.DataFrame({'end': ends, 'episode_id': [1, 1, 2, 2, 2], 'direction': [1, -1, 1, -1, 1],
        'y_reduce': [0, 1, 1, 0, 1], 'y_increase': [1, 0, 0, 1, 0]}, index=[10, 20, 30, 40, 50])
    for action in ['reduce', 'increase']:
        frame['past_' + action + '_exists'] = [0, 1, 1, 0, 1]
        frame['past_' + action + '_log_count_15m'] = [0., 0., np.log(2), 0., np.log(3)]
    return frame, btc, eth


def test_latest_completed_candle_all_windows_and_original_columns():
    frame, btc, eth = source()
    expanded, links = attach_cross_market(frame, cross_market_table(btc, eth))
    verify_cross_market(frame, btc, eth, expanded, links)
    assert links.source_row.tolist() == [299, 300, 300, 300, 301]
    assert links.lag_row_1d.tolist() == [11, 12, 12, 12, 13]
    assert links.market_age_seconds.iloc[1] == 0
    assert links.market_age_seconds.iloc[2] == 1e-9
    pd.testing.assert_frame_equal(expanded[frame.columns], frame, check_exact=True)
    verify_cross_support(expanded, cross_market_support(expanded))
    for data in [btc, eth]:
        for col in ['time', 'end']:
            data[col] = data[col].astype('datetime64[ns, UTC]')
    updated, _ = attach_cross_market(frame, cross_market_table(btc, eth))
    pd.testing.assert_frame_equal(expanded, updated, check_exact=True)


def test_market_currency_unit_invariance_and_position_direction():
    frame, btc, eth = source()
    original, _ = attach_cross_market(frame, cross_market_table(btc, eth))
    btc['close'] *= 10000
    eth['close'] *= .001
    scaled, links = attach_cross_market(frame, cross_market_table(btc, eth))
    verify_cross_market(frame, btc, eth, scaled, links)
    np.testing.assert_allclose(scaled[CROSS_FEATURES], original[CROSS_FEATURES], atol=1e-12, rtol=0)
    frame['direction'] *= -1
    flipped, _ = attach_cross_market(frame, cross_market_table(btc, eth))
    np.testing.assert_array_equal(flipped[RELATIVE_FEATURES], scaled[RELATIVE_FEATURES])
    np.testing.assert_array_equal(flipped[CROSS_FEATURES[5:]], -scaled[CROSS_FEATURES[5:]])
    eth['close'] = btc.close * .5
    common, _ = attach_cross_market(frame, cross_market_table(btc, eth))
    np.testing.assert_allclose(common[CROSS_FEATURES], 0, atol=1e-12, rtol=0)


def test_future_close_change_cannot_change_past_inputs():
    frame, btc, eth = source()
    expected, links = attach_cross_market(frame, cross_market_table(btc, eth))
    btc.loc[302:, 'close'] *= 4
    eth.loc[302:, 'close'] *= .01
    actual, new_links = attach_cross_market(frame, cross_market_table(btc, eth))
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(new_links, links, check_exact=True)


@pytest.mark.parametrize('damage', ['nan', 'infinite', 'zero', 'negative', 'naive', 'gap', 'both_gap', 'duplicate', 'reverse', 'end', 'shift', 'short', 'missing'])
def test_invalid_market_prices_or_calendar_rejected(damage):
    _, btc, eth = source()
    if damage in {'nan', 'infinite', 'zero', 'negative'}:
        eth.loc[290, 'close'] = {'nan': np.nan, 'infinite': np.inf, 'zero': 0., 'negative': -1.}[damage]
    elif damage == 'naive':
        btc['time'] = btc.time.dt.tz_localize(None)
    elif damage in {'gap', 'both_gap'}:
        eth = eth.drop(index=250)
        if damage == 'both_gap':
            btc = btc.drop(index=250)
    elif damage == 'duplicate':
        eth = pd.concat([eth.iloc[:250], eth.iloc[[249]], eth.iloc[250:]])
    elif damage == 'reverse':
        btc = btc.iloc[::-1]
    elif damage == 'end':
        eth.loc[290, 'end'] += pd.Timedelta(seconds=1)
    elif damage == 'shift':
        eth['time'] += pd.Timedelta(minutes=5)
        eth['end'] += pd.Timedelta(minutes=5)
    elif damage == 'short':
        btc, eth = btc.iloc[:288], eth.iloc[:288]
    else:
        eth = eth.drop(columns='close')
    with pytest.raises(ValueError):
        cross_market_table(btc, eth)


@pytest.mark.parametrize('damage', ['direction', 'episode', 'naive', 'duplicate', 'order', 'new_column', 'missing', 'source_row', 'nan', 'warmup', 'stale'])
def test_invalid_join_or_missing_history_is_rejected(damage):
    frame, btc, eth = source()
    table = cross_market_table(btc, eth)
    if damage == 'direction':
        frame.loc[10, 'direction'] = 0
    elif damage == 'episode':
        frame['episode_id'] = frame.episode_id.astype(float)
        frame.loc[10, 'episode_id'] = np.nan
    elif damage == 'naive':
        frame['end'] = frame.end.dt.tz_localize(None)
    elif damage == 'duplicate':
        frame.loc[20, 'end'] = frame.loc[10, 'end']
    elif damage == 'order':
        frame = frame.iloc[::-1]
    elif damage == 'new_column':
        frame[CROSS_FEATURES[0]] = 0.
    elif damage == 'missing':
        table = table.drop(columns=RELATIVE_FEATURES[0])
    elif damage == 'source_row':
        table.loc[300, 'source_row'] += 1
    elif damage == 'nan':
        table.loc[300, RELATIVE_FEATURES[0]] = np.nan
    elif damage == 'warmup':
        frame = frame.iloc[[0]].copy()
        frame['end'] = btc.end.iloc[287]
    else:
        frame = frame.iloc[[0]].copy()
        frame['end'] = btc.end.iloc[-1] + pd.Timedelta(minutes=5)
    with pytest.raises(ValueError):
        attach_cross_market(frame, table)


@pytest.mark.parametrize('damage', ['feature', 'directional', 'source', 'time', 'age', 'lag', 'original', 'support'])
def test_independent_verification_rejects_output_damage(damage):
    frame, btc, eth = source()
    expanded, links = attach_cross_market(frame, cross_market_table(btc, eth))
    if damage in {'feature', 'directional'}:
        expanded.loc[10, CROSS_FEATURES[0 if damage == 'feature' else 5]] += .01
    elif damage == 'time':
        links.loc[10, 'market_end'] += pd.Timedelta(minutes=5)
    elif damage in {'source', 'age', 'lag'}:
        links.loc[10, {'source': 'source_row', 'age': 'market_age_seconds', 'lag': 'lag_row_1d'}[damage]] += 1
    elif damage == 'original':
        expanded.loc[10, 'episode_id'] += 1
    else:
        support = deepcopy(cross_market_support(expanded))
        support['reduce']['quiet']['features'][CROSS_FEATURES[0]]['median'] += .01
        with pytest.raises(ValueError):
            verify_cross_support(expanded, support)
        return
    with pytest.raises(AssertionError):
        verify_cross_market(frame, btc, eth, expanded, links)


def test_empty_state_distributions_preserved():
    frame, btc, eth = source()
    for action in ['reduce', 'increase']:
        frame['past_' + action + '_exists'] = 0
        frame['past_' + action + '_log_count_15m'] = 0.
    expanded, _ = attach_cross_market(frame, cross_market_table(btc, eth))
    support = cross_market_support(expanded)
    verify_cross_support(expanded, support)
    assert support['increase']['restart']['rows'] == 0
    assert all(v is None for v in support['increase']['restart']['features'].values())


def test_invalid_proof_digest_or_symlink_rejected(tmp_path):
    proof = tmp_path / 'proof.json'
    proof.write_text('{}')
    with pytest.raises(ValueError, match='지문'):
        checked_cross_reference(proof, '0' * 64)
    linked = tmp_path / 'linked.json'
    linked.symlink_to(proof)
    with pytest.raises(ValueError, match='지문'):
        checked_cross_reference(linked, sha256(proof))


@pytest.mark.parametrize('damage', ['none', 'manifest', 'file', 'rows', 'path', 'evidence', 'source', 'symbol'])
def test_market_source_binding_and_metadata(tmp_path, damage):
    _, btc, _ = source()
    p = tmp_path / 'BTCUSDT-5m.parquet'
    btc.to_parquet(p, index=False)
    evidence = tmp_path / 'paired.json'
    save_json(evidence, {'source_sha256': {'1m': 'a'}, 'requested_symbols': ['BTCUSDT'],
        'symbols': {'BTCUSDT': {'after': {'buckets': 700, 'matched': 700}}}})
    metadata = {'interval': '5m', 'summary': {'BTCUSDT': {'klines': {'file': p.name, 'sha256': sha256(p),
        'rows': len(btc), 'first': btc.time.min().isoformat(), 'last': btc.time.max().isoformat(), 'gaps': 0}}},
        'paired_reconciliation': {'evidence_file': evidence.name, 'evidence_sha256': sha256(evidence), 'source_sha256': {'1m': 'a'}}}
    if damage == 'rows':
        metadata['summary']['BTCUSDT']['klines']['rows'] -= 1
    elif damage == 'path':
        metadata['summary']['BTCUSDT']['klines']['file'] = '../BTCUSDT-5m.parquet'
    elif damage == 'source':
        metadata['paired_reconciliation']['source_sha256']['1m'] = 'b'
    manifest = tmp_path / 'manifest-5m.json'
    save_json(manifest, metadata)
    digest = sha256(manifest)
    if damage == 'manifest':
        digest = '0' * 64
    elif damage == 'file':
        btc.loc[0, 'close'] += 1
        btc.to_parquet(p, index=False)
    elif damage == 'evidence':
        save_json(evidence, {})
    if damage == 'none':
        got, provenance = checked_cross_market_source(tmp_path, 'BTCUSDT', digest)
        pd.testing.assert_frame_equal(got, btc, check_exact=True)
        assert provenance['reconciliation_covers_entire_training_period_claimed'] is False
    else:
        with pytest.raises(ValueError):
            checked_cross_market_source(tmp_path, 'SOLUSDT' if damage == 'symbol' else 'BTCUSDT', digest)


@pytest.mark.parametrize('damage', ['none', 'complete', 'accepted', 'accounts', 'profits', 'features', 'sources', 'summary'])
def test_previous_learning_evidence_binding(tmp_path, monkeypatch, damage):
    from wonyotti_fr.holding_stress import seal
    from wonyotti_fr.uncapped_management import FEATURES as PREVIOUS_FEATURES

    run = tmp_path / 'run'
    run.mkdir()
    summary = {'complete': True, 'independent_verification_complete': True, 'all_inputs_and_code_unchanged': True,
        'uncapped_count_admitted': False, 'new_accounts_executed': False, 'profitability_accepted': False}
    if damage == 'complete':
        summary['complete'] = False
    elif damage == 'accepted':
        summary['uncapped_count_admitted'] = True
    elif damage == 'accounts':
        summary['new_accounts_executed'] = True
    elif damage == 'profits':
        summary['profitability_accepted'] = True
    settings = {'reference': 'unused', 'reference_sha256': 'unused', 'sources': {'teacher': 'a'},
        'features': PREVIOUS_FEATURES.copy()}
    if damage == 'features':
        settings['features'].reverse()
    save_json(run / 'summary.json', summary)
    save_json(run / 'manifest.json', {'settings': settings})
    seal(run)
    proof = tmp_path / 'proof.json'
    save_json(proof, {'run': str(run), 'complete': True, 'files_sha256': sha256(run / 'files.json'),
        'summary_sha256': sha256(run / 'summary.json'), 'new_accounts_executed': False, 'profitability_accepted': False})
    monkeypatch.setattr('wonyotti_fr.cross_market_research.checked_uncapped_reference',
        lambda *args: {'teacher': 'b' if damage == 'sources' else 'a'})
    if damage == 'summary':
        save_json(run / 'summary.json', {})
    if damage == 'none':
        assert checked_cross_reference(proof, sha256(proof)) == {'teacher': 'a', 'uncapped': str(run)}
    else:
        with pytest.raises(ValueError):
            checked_cross_reference(proof, sha256(proof))
