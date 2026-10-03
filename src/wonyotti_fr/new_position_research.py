from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .action_research import action_diagnostics
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_features import MARKET_FEATURES
from .event_research import load_selection
from .expansion_data import make_expansion_data, source_inputs
from .expansion_model import BinaryModel
from .minute_data import prepare_minute_period
from .new_position import POSITION_FILES, copy_recent_parent, new_position_targets
from .recent_entry import ENTRY_PERIOD, recent_entry_training
from .reports import table


def run_new_position_selection(reference: Path, audit: Path, study: Path, history: Path, market: Path,
                                features: Path, confirmation_market: Path, confirmation_features: Path,
                                output: Path) -> Path:
    parent, original = load_selection(reference)
    if parent['protocol'] != 'recent_entry_v23':
        raise ValueError('신규 포지션 활동 대조에는 고정 v23 모델이 필요합니다.')
    source, hashes = source_inputs(audit, study, history)
    data, events = new_position_targets(make_expansion_data(source), source['actions'])
    train, ledger = recent_entry_training(data, source['episodes'])
    out = new_run(output, 'new-position-selection', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'), **hashes,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V25.md')), 'candidate_count': 1,
        'entry_training_period': ENTRY_PERIOD, 'activity_quantile': .975, 'development_in_sample': True,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'신규 포지션 시작 활동 학습: {out}', flush=True)
    try:
        copy_recent_parent(reference, out)
        train.to_parquet(out / 'entry_training_used.parquet', index=False)
        ledger.to_parquet(out / 'entry_training_ledger.parquet', index=False)
        events.to_parquet(out / 'new_position_ledger.parquet', index=False)
        models, details = {}, {}
        for name, rows, label in [('activity', train, 'active')]:
            models[name], details[name] = BinaryModel.fit(rows[MARKET_FEATURES].to_numpy(dtype=float),
                                                         rows[label].to_numpy(dtype=int), 'logistic')
        scores = models['activity'].probabilities(train[MARKET_FEATURES].to_numpy(dtype=float))
        threshold = float(np.quantile(scores, .975))
        save_json(out / 'new_position_activity.json', models['activity'].to_dict())
        metadata = {'training_period': ENTRY_PERIOD, 'activity_quantile': .975, 'activity_threshold': threshold,
            'models': details, 'first_end': train.end.min(), 'last_label_end': train.label_end.max(),
            'exclusion_counts': ledger.reason.value_counts().to_dict(), 'original_management_unchanged': True, 'direction_unchanged': True,
            'new_position_ledger_reasons': events.reason.value_counts().to_dict()}
        save_json(out / 'new_position_training.json', metadata)
        frozen = {**parent, 'protocol': 'new_position_v25', 'recent_selection_sha256': sha256(out / 'recent_selection.json'),
            'position_activity_threshold': threshold, 'position_files_sha256': {n: sha256(out / n) for n in POSITION_FILES}}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        print(f'활동 {len(train)}개·신규 시작 양성 {int(train.active.sum())}개, 학습 분위 {threshold:.8f}', flush=True)
        del data, train, ledger, source, events
        config = EngineConfig(**frozen['risk'])
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        comparisons = []
        for name, bot, folder in [('new_position', policy, 'candidate-00'), ('previous_v23', original, 'control-previous')]:
            metrics = backtest(bars, bot, config, out / folder)
            action_diagnostics(out / folder, bars, config)
            comparisons.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', comparisons)
            print(f'2021 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'control-previous' / name),
                                          pd.read_parquet(reference / 'candidate-00' / name), check_exact=True)
        if json.loads((out / 'control-previous/final_state.json').read_text()) != json.loads((reference / 'candidate-00/final_state.json').read_text()):
            raise ValueError('기존 확대 활동 대조와 원래 v23의 전체 상태 불일치')
        save_json(out / 'baseline_parity.json', {'full_outputs_and_state_exact': True})
        frozen['development_metrics'] = {'candidate': 0, **comparisons[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        comparisons.append({'policy': 'new_position', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', comparisons)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 신규 포지션 시작 활동 학습 비교\n\n' + table(pd.DataFrame(comparisons)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n활동 정답만 실제 신규 포지션 시작으로 바꿨다. 방향·관리·축소·위험과 원래 v23 대조는 그대로다. '
            '2021년은 적합 진단이며 이후 2022년 결과로 모델을 재선택하지 않았다. 모든 기간은 이미 관찰했다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
