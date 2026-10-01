import pandas as pd
import pytest
from test_minute_repair import source

from wonyotti_fr.common import save_json, sha256
from wonyotti_fr.minute_repair import BAR_COLUMNS
from wonyotti_fr.paired_repair import canonical_window, context_ends, repair_paired_market
from wonyotti_fr.research import load_market


def inputs():
    date, trades, prior, truth, five = source()
    aggregates = pd.DataFrame({'aggregate_id': range(len(trades)), 'price': trades.price, 'qty': trades.qty,
                              'first_id': trades.id, 'last_id': trades.id, 'time': trades.time,
                              'is_buyer_maker': trades.is_buyer_maker})
    for column in set(BAR_COLUMNS) - set(five.columns):
        five[column] = truth[column].sum()
    return date, trades, prior, truth, five, aggregates


def test_both_resolutions_can_be_wrong_and_neither_is_forced_as_truth():
    _, trades, prior, truth, five, aggregates = inputs()
    wrong_five = five.assign(volume=9., count=9)
    updates, checks = canonical_window(prior, wrong_five, trades, aggregates, pd.DatetimeIndex(five.end))
    assert checks['changed_minutes'] == checks['changed_five_minutes'] == 1
    assert checks['corroboration']['target_trades'] == 10
    assert updates['1m'].close.iloc[0] == 103
    assert updates['5m'].volume.iloc[0] == 10
    assert prior.close.iloc[1] == 102 and wrong_five.volume.iloc[0] == 9
    updates, _ = canonical_window(truth, wrong_five, trades, aggregates, pd.DatetimeIndex(five.end))
    assert updates['1m'].empty and len(updates['5m']) == 1


def test_all_target_trades_require_independent_evidence_even_without_id_gaps():
    _, trades, prior, _, five, aggregates = inputs()
    with pytest.raises(ValueError, match='한 번씩'):
        canonical_window(prior, five, trades, aggregates.iloc[1:], pd.DatetimeIndex(five.end))
    with pytest.raises(ValueError, match='구간 부족'):
        canonical_window(prior, five, trades.iloc[2:], aggregates, pd.DatetimeIndex(five.end))
    with pytest.raises(ValueError, match='불일치'):
        canonical_window(prior, five, trades, aggregates.assign(qty=1.1), pd.DatetimeIndex(five.end))
    # 개별 체결 하나가 통째로 없으면 나머지 체결의 일대일 연결만으로 통과시키지 않는다.
    with pytest.raises(ValueError, match='전체가 개별 체결에서 누락'):
        canonical_window(prior, five, trades.drop(index=3), aggregates, pd.DatetimeIndex(five.end))


def test_cross_boundary_group_requires_context_and_missing_context_is_rejected():
    date, trades, prior, _, five, aggregates = inputs()
    context = trades.iloc[[0]].copy()
    context['id'], context['time'] = 0, date.value // 10**6 - 50
    aggregates.loc[0, ['qty', 'first_id', 'time']] = [2., 0, date.value // 10**6 - 50]
    with pytest.raises(ValueError, match='한 번씩'):
        canonical_window(prior, five, trades, aggregates, pd.DatetimeIndex(five.end))
    updates, checks = canonical_window(prior, five, pd.concat([context, trades], ignore_index=True),
                                       aggregates, pd.DatetimeIndex(five.end))
    assert len(updates['1m']) == 1 and checks['corroboration']['target_trades'] == 10
    with pytest.raises(ValueError, match='날짜 경계'):
        context_ends(pd.DatetimeIndex(five.end), date)


def test_output_hashes_funding_preservation_and_failed_pair_unavailable(tmp_path, monkeypatch):
    date, trades, prior, truth, five, aggregates = inputs()
    delta = pd.Timedelta(hours=12)
    for frame in [prior, truth, five]:
        frame['time'], frame['end'] = frame.time + delta, frame.end + delta
    for frame in [trades, aggregates]:
        frame['time'] += delta.value // 10**6
    one_source, five_source = tmp_path / 'one', tmp_path / 'five'
    for interval, path, bars in [('1m', one_source, prior), ('5m', five_source, five.assign(count=8, volume=8.))]:
        path.mkdir()
        bars.to_parquet(path / 'bars.parquet', index=False)
        pd.DataFrame({'time': [date], 'rate': [.0001]}).to_parquet(path / 'funding.parquet', index=False)
        save_json(path / f'manifest-{interval}.json', {'interval': interval, 'summary': {'BTCUSDT': {
            'klines': {'file': 'bars.parquet', 'sha256': sha256(path / 'bars.parquet')},
            'fundingRate': {'file': 'funding.parquet', 'sha256': sha256(path / 'funding.parquet')}}}})
    monkeypatch.setattr('wonyotti_fr.paired_repair.verified_archive', lambda url, cache: (url, {'checksum_verified': True}))
    monkeypatch.setattr('wonyotti_fr.paired_repair.archive_csv', lambda url:
                        (aggregates if '/aggTrades/' in url else trades).to_csv(index=False, header=False).encode())
    before = [sha256(p / 'bars.parquet') for p in [one_source, five_source]]
    output, feature_output = tmp_path / 'out-one', tmp_path / 'out-five'
    checks = repair_paired_market(one_source, five_source, output, feature_output, tmp_path / 'cache')
    assert checks['symbols']['BTCUSDT']['after']['mismatched'] == 0
    for interval, path, source_path in [('1m', output, one_source), ('5m', feature_output, five_source)]:
        repaired, _ = load_market(path, 'BTCUSDT', interval)
        assert repaired['count'].sum() == 10
        assert sha256(path / 'funding.parquet') == sha256(source_path / 'funding.parquet')
    assert before == [sha256(p / 'bars.parquet') for p in [one_source, five_source]]
    aggregates.loc[0, 'qty'] = 2.
    broken_one, broken_five = tmp_path / 'bad-one', tmp_path / 'bad-five'
    with pytest.raises(ValueError, match='불일치'):
        repair_paired_market(one_source, five_source, broken_one, broken_five, tmp_path / 'cache')
    assert not (broken_one / 'manifest-1m.json').exists()
    assert not (broken_five / 'manifest-5m.json').exists()
    assert (broken_one / 'failure.json').exists()
