from __future__ import annotations

import json
from pathlib import Path

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
from .new_position import new_position_targets
from .position_direction import (
    DIRECTION_FILES,
    DIRECTION_PERIOD,
    copy_prior_parent,
    position_direction_training,
)
from .reports import table


def run_position_direction_selection(reference: Path, audit: Path, study: Path, history: Path, market: Path,
                                features: Path, confirmation_market: Path, confirmation_features: Path,
                                output: Path) -> Path:
    parent, original = load_selection(reference)
    if parent['protocol'] != 'position_prior_v26':
        raise ValueError('신규 방향 대조에는 고정 v26 모델이 필요합니다.')
    source, hashes = source_inputs(audit, study, history)
    data, events = new_position_targets(make_expansion_data(source), source['actions'])
    train, ledger = position_direction_training(data, source['episodes'])
    out = new_run(output, 'position-direction-selection', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'), **hashes,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V27.md')), 'candidate_count': 1,
        'direction_training_period': DIRECTION_PERIOD, 'direction_threshold': .65, 'development_in_sample': True,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'실제 신규 포지션 방향 학습: {out}', flush=True)
    try:
        copy_prior_parent(reference, out)
        train.to_parquet(out / 'direction_training_used.parquet', index=False)
        ledger.to_parquet(out / 'direction_training_ledger.parquet', index=False)
        events.to_parquet(out / 'new_position_ledger.parquet', index=False)
        direction, support = BinaryModel.fit(train[MARKET_FEATURES].to_numpy(dtype=float), train.buy.to_numpy(dtype=int), 'logistic')
        save_json(out / 'new_position_direction.json', direction.to_dict())
        save_json(out / 'new_direction_training.json', {'training_period': DIRECTION_PERIOD,
            'direction_threshold': .65, 'prior_offset_applied': False, 'model': support,
            'first_end': train.end.min(), 'last_label_end': train.label_end.max(),
            'exclusion_counts': ledger.reason.value_counts().to_dict(),
            'yearly_rows': train.end.dt.year.value_counts().sort_index().to_dict(), 'activity_and_management_unchanged': True})
        frozen = {**parent, 'protocol': 'position_direction_v27', 'prior_selection_sha256': sha256(out / 'prior_selection.json'),
            'direction_files_sha256': {n: sha256(out / n) for n in DIRECTION_FILES}}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        print(f'실제 신규 방향 {len(train)}개, 롱 {support["positive"]}개·숏 {support["negative"]}개', flush=True)
        del data, train, ledger, source, events
        config = EngineConfig(**frozen['risk'])
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        comparisons = []
        for name, bot, folder in [('position_direction', policy, 'candidate-00'), ('previous_v26', original, 'control-previous')]:
            metrics = backtest(bars, bot, config, out / folder)
            action_diagnostics(out / folder, bars, config)
            comparisons.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', comparisons)
            print(f'2021 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'control-previous' / name),
                                          pd.read_parquet(reference / 'candidate-00' / name), check_exact=True)
        if json.loads((out / 'control-previous/final_state.json').read_text()) != json.loads((reference / 'candidate-00/final_state.json').read_text()):
            raise ValueError('기존 방향 빈도 보정 대조와 원래 v26의 전체 상태 불일치')
        save_json(out / 'baseline_parity.json', {'full_outputs_and_state_exact': True})
        frozen['development_metrics'] = {'candidate': 0, **comparisons[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        comparisons.append({'policy': 'position_direction', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', comparisons)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 실제 신규 포지션 방향 학습 비교\n\n' + table(pd.DataFrame(comparisons)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n방향을 전체 원본의 실제 신규 포지션 목표로 다시 학습했다. 활동·관리·축소·위험과 원래 v26 대조는 그대로다. 새 점수에 기존 절편 보정을 더하지 않았다. '
            '2021년은 적합 진단이며 이후 2022년 결과로 모델을 재선택하지 않았다. 모든 기간은 이미 관찰했다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
