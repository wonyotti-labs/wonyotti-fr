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
from .expansion_model import BinaryModel
from .minute_data import prepare_minute_period
from .position_prior import PRIOR_FILES, adjust_direction, copy_position_parent, prior_offset
from .recent_entry import ENTRY_PERIOD
from .reports import table


def run_position_prior_selection(reference: Path, market: Path, features: Path,
                                  confirmation_market: Path, confirmation_features: Path, output: Path) -> Path:
    parent, original = load_selection(reference)
    if parent['protocol'] != 'new_position_v25':
        raise ValueError('방향 빈도 보정에는 고정 v25 모델이 필요합니다.')
    old_root = Path(json.loads((reference / 'manifest.json').read_text())['settings']['reference'])
    old_frozen, old_policy = load_selection(old_root)
    if old_frozen['protocol'] != 'recent_entry_v23' or sha256(old_root / 'frozen_selection.json') != parent['recent_selection_sha256']:
        raise ValueError('방향 빈도 보정의 원래 학습 선택 연결 오류')
    old, new = (pd.read_parquet(p / 'entry_training_used.parquet') for p in [old_root, reference])
    columns = [*MARKET_FEATURES, 'end', 'label_end', 'episode_id', 'usable']
    time_types = {'end': 'datetime64[ns, UTC]', 'label_end': 'datetime64[ns, UTC]'}
    pd.testing.assert_frame_equal(old[columns].astype(time_types), new[columns].astype(time_types), check_exact=True)
    counts = []
    for frame in [old, new]:
        active = frame[frame.active.eq(1)]
        if not np.isin(active.buy, [0, 1]).all():
            raise ValueError('방향 빈도 집계의 이진 정답 오류')
        counts.append({'buy': int(active.buy.sum()), 'sell': int(len(active) - active.buy.sum())})
    old_details = json.loads((reference / 'recent_entry_training.json').read_text())['models']['direction']
    if counts[0] != {'buy': old_details['positive'], 'sell': old_details['negative']}:
        raise ValueError('방향 빈도와 원래 학습 메타데이터 불일치')
    active = old[old.active.eq(1)]
    recreated, _ = BinaryModel.fit(active[MARKET_FEATURES].to_numpy(), active.buy.to_numpy(dtype=int), 'logistic')
    if recreated.to_dict() != original.base.direction.to_dict() or recreated.to_dict() != old_policy.base.direction.to_dict():
        raise ValueError('방향 빈도 집계의 원래 학습 모형 재구성 불일치')
    direction = adjust_direction(original.base.direction, *counts)
    offset = prior_offset(*counts)
    out = new_run(output, 'position-prior-selection', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'), 'original_selection': str(old_root),
        'training_files_sha256': {str(p / 'entry_training_used.parquet'): sha256(p / 'entry_training_used.parquet') for p in [old_root, reference]},
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V26.md')), 'candidate_count': 1,
        'entry_training_period': ENTRY_PERIOD, 'activity_quantile': .975, 'development_in_sample': True,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'신규 포지션 방향 빈도 보정: {out}', flush=True)
    try:
        copy_position_parent(reference, out)
        save_json(out / 'position_direction_model.json', direction.to_dict())
        save_json(out / 'direction_prior.json', {'training_period': ENTRY_PERIOD, 'old_counts': counts[0],
            'new_counts': counts[1], 'offset': offset, 'original_direction_reproduced': True,
            'feature_coefficients_unchanged': True, 'exact_probability_calibration_claimed': False})
        frozen = {**parent, 'protocol': 'position_prior_v26', 'position_selection_sha256': sha256(out / 'position_selection.json'),
            'direction_prior_offset': offset, 'prior_files_sha256': {n: sha256(out / n) for n in PRIOR_FILES}}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        print(f'방향 절편 이동 {offset:.8f}, 기존 {counts[0]}, 신규 {counts[1]}', flush=True)
        del old, new, active
        config = EngineConfig(**frozen['risk'])
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        comparisons = []
        for name, bot, folder in [('position_prior', policy, 'candidate-00'), ('previous_v25', original, 'control-previous')]:
            metrics = backtest(bars, bot, config, out / folder)
            action_diagnostics(out / folder, bars, config)
            comparisons.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', comparisons)
            print(f'2021 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'control-previous' / name),
                                          pd.read_parquet(reference / 'candidate-00' / name), check_exact=True)
        if json.loads((out / 'control-previous/final_state.json').read_text()) != json.loads((reference / 'candidate-00/final_state.json').read_text()):
            raise ValueError('기존 방향 대조와 원래 v25의 전체 상태 불일치')
        save_json(out / 'baseline_parity.json', {'full_outputs_and_state_exact': True})
        frozen['development_metrics'] = {'candidate': 0, **comparisons[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        comparisons.append({'policy': 'position_prior', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', comparisons)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 신규 포지션 방향 빈도 보정 비교\n\n' + table(pd.DataFrame(comparisons)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n원본 목표의 롱·숏 빈도 차이만큼 방향 절편만 이동했다. 계수·활동·관리·축소·위험과 원래 v25 대조는 그대로다. '
            '2021년은 적합 진단이며 이후 2022년 결과로 모델을 재선택하지 않았다. 모든 기간은 이미 관찰했다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
