from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .action_research import action_diagnostics
from .boosted_direction import BOOST_FILES, copy_direction_parent, read_admission
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_features import MARKET_FEATURES
from .event_research import load_selection
from .expansion_data import make_expansion_data, source_inputs
from .expansion_model import BinaryModel
from .minute_data import prepare_minute_period
from .new_position import new_position_targets
from .position_direction import DIRECTION_PERIOD, position_direction_training
from .reports import table


def run_boosted_direction_selection(reference: Path, diagnosis: Path, audit: Path, study: Path, history: Path, market: Path,
                                features: Path, confirmation_market: Path, confirmation_features: Path,
                                output: Path) -> Path:
    parent, original = load_selection(reference)
    if parent['protocol'] != 'position_direction_v27':
        raise ValueError('신규 방향 대조에는 고정 v27 모델이 필요합니다.')
    source, hashes = source_inputs(audit, study, history)
    out = new_run(output, 'boosted-direction-selection', {'reference': str(reference), 'diagnosis': str(diagnosis), 'diagnosis_files_sha256': sha256(diagnosis / 'files.json'),
        'reference_sha256': sha256(reference / 'frozen_selection.json'), **hashes,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V28.md')), 'candidate_count': 1,
        'direction_training_period': DIRECTION_PERIOD, 'direction_threshold': .65, 'development_in_sample': True,
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'진단 통과 방향 부스팅 학습: {out}', flush=True)
    try:
        admission = read_admission(diagnosis, hashes)
        data, events = new_position_targets(make_expansion_data(source), source['actions'])
        train, ledger = position_direction_training(data, source['episodes'])
        pd.testing.assert_frame_equal(train.reset_index(drop=True), pd.read_parquet(reference / 'direction_training_used.parquet'), check_exact=True)
        copy_direction_parent(reference, out)
        save_json(out / 'direction_admission.json', admission)
        train.to_parquet(out / 'direction_training_used.parquet', index=False)
        ledger.to_parquet(out / 'direction_training_ledger.parquet', index=False)
        events.to_parquet(out / 'new_position_ledger.parquet', index=False)
        direction, support = BinaryModel.fit(train[MARKET_FEATURES].to_numpy(dtype=float), train.buy.to_numpy(dtype=int), 'boosted')
        save_json(out / 'boosted_direction.json', direction.to_dict())
        save_json(out / 'boosted_direction_training.json', {'training_period': DIRECTION_PERIOD,
            'direction_threshold': .65, 'prior_offset_applied': False, 'model': support,
            'first_end': train.end.min(), 'last_label_end': train.label_end.max(),
            'exclusion_counts': ledger.reason.value_counts().to_dict(),
            'yearly_rows': train.end.dt.year.value_counts().sort_index().to_dict(), 'activity_and_management_unchanged': True})
        frozen = {**parent, 'protocol': 'boosted_direction_v28', 'direction_selection_sha256': sha256(out / 'direction_selection.json'),
            'boost_files_sha256': {n: sha256(out / n) for n in BOOST_FILES}}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        print(f'실제 신규 방향 {len(train)}개, 롱 {support["positive"]}개·숏 {support["negative"]}개', flush=True)
        del data, train, ledger, source, events
        config = EngineConfig(**frozen['risk'])
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01', minute_inputs=True)
        save_json(out / 'development_input.json', checks)
        comparisons = []
        for name, bot, folder in [('boosted_direction', policy, 'candidate-00'), ('previous_v27', original, 'control-previous')]:
            metrics = backtest(bars, bot, config, out / folder)
            action_diagnostics(out / folder, bars, config)
            comparisons.append({'policy': name, 'period': '2021_in_sample', **metrics})
            save_json(out / 'development.json', comparisons)
            print(f'2021 적합 {name}: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래', flush=True)
        for name in ['equity.parquet', 'fills.parquet', 'trades.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(out / 'control-previous' / name),
                                          pd.read_parquet(reference / 'candidate-00' / name), check_exact=True)
        if json.loads((out / 'control-previous/final_state.json').read_text()) != json.loads((reference / 'candidate-00/final_state.json').read_text()):
            raise ValueError('기존 직접 방향 대조와 원래 v27의 전체 상태 불일치')
        save_json(out / 'baseline_parity.json', {'full_outputs_and_state_exact': True})
        frozen['development_metrics'] = {'candidate': 0, **comparisons[0]}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01', minute_inputs=True)
        save_json(out / 'confirmation_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        comparisons.append({'policy': 'boosted_direction', 'period': '2022', **metrics})
        save_json(out / 'comparison.json', comparisons)
        save_json(out / 'confirmation_checks.json', {'positive': metrics['total_return'] > 0,
            'at_least_30_trades': metrics['closed_trades'] >= 30, 'no_halt': not metrics['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'selected_by_profit': False})
        (out / 'REPORT.md').write_text('# 시간순 진단 통과 방향 부스팅 비교\n\n' + table(pd.DataFrame(comparisons)[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n시간순 방향 진단을 통과한 고정 부스팅 모형을 전체 원본의 같은 신규 방향 정답으로 학습했다. 활동·관리·축소·위험과 원래 v27 대조는 그대로다. 새 점수에 기존 절편 보정을 더하지 않았다. '
            '2021년은 적합 진단이며 이후 2022년 결과로 모델을 재선택하지 않았다. 모든 기간은 이미 관찰했다.\n')
        print(f'2022 확인: {metrics["total_return"]:.2%} / {metrics["closed_trades"]}거래 / 중지 {metrics["permanent_halt"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
