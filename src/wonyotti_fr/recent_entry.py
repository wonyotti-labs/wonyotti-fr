from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .action_research import action_diagnostics
from .addition_research import copy_minute_parent
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_features import MARKET_FEATURES
from .event_research import load_selection
from .expansion_data import make_expansion_data, source_inputs
from .expansion_model import BinaryModel, ExpansionPolicy
from .minute_data import prepare_minute_period
from .minute_inventory import MinuteInventoryPolicy
from .minute_inventory_research import load_minute_inventory_selection
from .reports import table

ENTRY_FILES = ['recent_entry_models.json', 'recent_entry_training.json']
ENTRY_PERIOD = ['2020-01-01', '2022-01-01']


def recent_entry_training(data, episodes):
    first, last = (pd.Timestamp(t, tz='UTC') for t in ENTRY_PERIOD)
    crossing = []
    for boundary in [first, last]:
        crossing.append(episodes.loc[episodes.entry_time.lt(boundary)
            & (episodes.exit_time.isna() | episodes.exit_time.ge(boundary)), 'episode_id'])
    reason = np.select([
        data.end.lt(first + pd.Timedelta(days=1)), data.label_end.ge(last - pd.Timedelta(days=1)),
        data.episode_id.isin(crossing[0]) | data.target_episode_id.isin(crossing[0]),
        data.episode_id.isin(crossing[1]) | data.target_episode_id.isin(crossing[1]), ~data.usable,
    ], ['before_training_or_left_embargo', 'after_training_or_right_embargo',
        'left_episode_boundary', 'right_episode_boundary', 'unusable_original_event'], default='included')
    ledger = data[['end', 'label_end', 'episode_id', 'target_episode_id', 'active']].assign(reason=reason)
    train = data[ledger.reason.eq('included')].copy()
    active = train[train.active.eq(1)]
    if (train.empty or not train.end.is_monotonic_increasing or train.end.duplicated().any()
        or not np.isin(train.active, [0, 1]).all() or not np.isin(active.buy, [0, 1]).all()
        or active.target_time.isna().any() or active.target_time.lt(active.end).any() or active.target_time.ge(active.label_end).any()):
        raise ValueError('최근 진입 정답의 순서·활동·방향·시각 오류')
    return train, ledger


class RecentEntryPolicy(MinuteInventoryPolicy):
    def __init__(self, parent, activity, direction, threshold):
        super().__init__(parent, parent.manager, parent.thresholds, parent.multiplier, parent.scales, parent.size_model)
        self.base = ExpansionPolicy(activity, direction, threshold, .65, 12)


def load_recent_entry_selection(selection, frozen):
    path = selection / 'minute_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['minute_selection_sha256']:
        raise ValueError('최근 진입 정책의 원래 관리 선택 지문 오류')
    parent = json.loads(path.read_text())
    if (frozen.get('protocol') != 'recent_entry_v23' or parent.get('protocol') != 'minute_inventory_micro_v21'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or frozen.get('entry_training_period') != ENTRY_PERIOD or frozen.get('entry_activity_quantile') != .975
        or frozen.get('entry_direction_threshold') != .65 or frozen.get('entry_kind') != 'logistic'
        or set(frozen['entry_files_sha256']) != set(ENTRY_FILES)
        or any((selection / n).stat().st_size > 1024**2
               or sha256(selection / n) != frozen['entry_files_sha256'][n] for n in ENTRY_FILES)):
        raise ValueError('최근 진입 정책의 학습·고정 설정·모델 지문 오류')
    _, original = load_minute_inventory_selection(selection, parent)
    models = json.loads((selection / 'recent_entry_models.json').read_text())
    if set(models) != {'activity', 'direction'}:
        raise ValueError('최근 진입 정책의 모델 종류 오류')
    activity, direction = (BinaryModel.from_dict(models[n]) for n in ['activity', 'direction'])
    metadata = json.loads((selection / 'recent_entry_training.json').read_text())
    if (any(m.data['kind'] != 'logistic' for m in [activity, direction])
        or metadata['training_period'] != ENTRY_PERIOD or metadata['activity_quantile'] != .975
        or metadata['activity_threshold'] != frozen['entry_activity_threshold']):
        raise ValueError('최근 진입 모델의 고정 모형·활동 문턱 오류')
    return frozen, RecentEntryPolicy(original, activity, direction, frozen['entry_activity_threshold'])


def run_recent_entry_selection(reference: Path, audit: Path, study: Path, history: Path, market: Path,
                                features: Path, confirmation_market: Path, confirmation_features: Path,
                                output: Path) -> Path:
    parent, original = load_selection(reference)
    base = json.loads((reference / 'base_selection.json').read_text())
    if (parent['protocol'] != 'minute_inventory_micro_v21' or base['kind'] != 'logistic'
        or base['direction_threshold'] != .65 or base['development_metrics']['activity_quantile'] != .975):
        raise ValueError('최근 진입 비교에는 기존 v21과 고정 진입 설정이 필요합니다.')
    source, hashes = source_inputs(audit, study, history)
    data = make_expansion_data(source)
    train, ledger = recent_entry_training(data, source['episodes'])
    out = new_run(output, 'recent-entry-selection', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'), **hashes,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V23.md')), 'candidate_count': 1,
        'entry_training_period': ENTRY_PERIOD, 'activity_quantile': .975, 'development_in_sample': True,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'최근 원본의 신규 진입 학습: {out}', flush=True)
    try:
        copy_minute_parent(reference, out)
        train.to_parquet(out / 'entry_training_used.parquet', index=False)
        ledger.to_parquet(out / 'entry_training_ledger.parquet', index=False)
        models, details = {}, {}
        for name, rows, label in [('activity', train, 'active'), ('direction', train[train.active.eq(1)], 'buy')]:
            models[name], details[name] = BinaryModel.fit(rows[MARKET_FEATURES].to_numpy(dtype=float),
                                                         rows[label].to_numpy(dtype=int), 'logistic')
        scores = models['activity'].probabilities(train[MARKET_FEATURES].to_numpy(dtype=float))
        threshold = float(np.quantile(scores, .975))
        save_json(out / 'recent_entry_models.json', {n: m.to_dict() for n, m in models.items()})
        metadata = {'training_period': ENTRY_PERIOD, 'activity_quantile': .975, 'activity_threshold': threshold,
            'models': details, 'first_end': train.end.min(), 'last_label_end': train.label_end.max(),
            'exclusion_counts': ledger.reason.value_counts().to_dict(), 'original_management_unchanged': True}
        save_json(out / 'recent_entry_training.json', metadata)
        frozen = {**parent, 'protocol': 'recent_entry_v23', 'minute_selection_sha256': sha256(out / 'minute_selection.json'),
            'entry_training_period': ENTRY_PERIOD, 'entry_activity_quantile': .975, 'entry_activity_threshold': threshold,
            'entry_direction_threshold': .65, 'entry_kind': 'logistic',
            'entry_files_sha256': {n: sha256(out / n) for n in ENTRY_FILES}}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        print(f'활동 {len(train)}개·방향 {int(train.active.sum())}개, 학습 분위 {threshold:.8f}', flush=True)
        del data, train, ledger, source
        config = EngineConfig(**frozen['risk'])
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        comparisons = []
        for name, bot, folder in [('recent_entry', policy, 'candidate-00'), ('previous_v21', original, 'control-previous')]:
            metrics = backtest(bars, bot, config, out / folder)
            action_diagnostics(out / folder, bars, config)
            comparisons.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', comparisons)
            print(f'2021 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'control-previous' / name),
                                          pd.read_parquet(reference / 'candidate-00' / name), check_exact=True)
        if json.loads((out / 'control-previous/final_state.json').read_text()) != json.loads((reference / 'candidate-00/final_state.json').read_text()):
            raise ValueError('이전 진입 대조와 원래 v21의 전체 상태 불일치')
        save_json(out / 'baseline_parity.json', {'full_outputs_and_state_exact': True})
        frozen['development_metrics'] = {'candidate': 0, **comparisons[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        comparisons.append({'policy': 'recent_entry', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', comparisons)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 최근 원본의 신규 진입 학습 비교\n\n' + table(pd.DataFrame(comparisons)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n진입 활동·방향의 학습 기간과 학습 점수 분위 값만 갱신했다. 관리·축소·위험과 원래 진입 대조는 그대로다. '
            '2021년은 적합 진단이며 이후 2022년 결과로 모델을 재선택하지 않았다. 모든 기간은 이미 관찰했다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
