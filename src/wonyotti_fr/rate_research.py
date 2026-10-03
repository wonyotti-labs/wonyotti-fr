from __future__ import annotations

import json
from pathlib import Path

from .action_research import action_diagnostics
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_backtest import backtest
from .event_research import load_selection
from .minute_data import prepare_minute_period
from .minute_management import purged_window
from .rate_policy import RateActionPolicy, calibrate_rates
from .reports import table


def run_rate_selection(reference: Path, market: Path, features: Path, confirmation_market: Path,
                           confirmation_features: Path, output: Path) -> Path:
    import pandas as pd

    parent, existing = load_selection(reference)
    if parent.get('protocol') != 'minute_path_v12':
        raise ValueError('누적 빈도 대조는 고정 v12 후보가 필요합니다.')
    manifest = json.loads((reference / 'manifest.json').read_text())['settings']
    labels = Path(manifest['labels'])
    hashes = json.loads((labels / 'files.json').read_text())
    if sha256(labels / 'files.json') != manifest['files_sha256'] or sha256(labels / 'events.parquet') != hashes['events.parquet']:
        raise ValueError('빈도 보정 기반 정답의 지문 오류')
    data = pd.read_parquet(reference / 'calibration_used.parquet')
    expected = purged_window(pd.read_parquet(labels / 'events.parquet'), '2020-07-01', '2021-01-01')
    pd.testing.assert_frame_equal(data.reset_index(drop=True), expected.reset_index(drop=True), check_exact=True)
    calibration = calibrate_rates(existing.manager, data)
    calibration['calibration_sha256'] = sha256(reference / 'calibration_used.parquet')
    del data, expected
    out = new_run(output, 'action-rate-selection', {'reference': str(reference),
        'reference_sha256': sha256(reference / 'frozen_selection.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V14.md')), 'candidate_count': 1, 'refit': False, 'calibration_sha256': calibration['calibration_sha256'],
        'input_manifests': {str(p / n): sha256(p / n) for p, n in [(market, 'manifest-1m.json'),
            (features, 'manifest-5m.json'), (confirmation_market, 'manifest-1m.json'), (confirmation_features, 'manifest-5m.json')]}})
    print(f'고정 모델 사건 빈도 누적 대조: {out}', flush=True)
    for name in ['pullback_selection.json', 'base_selection.json', 'expansion_models.json', 'action_model.json']:
        (out / name).write_bytes((reference / name).read_bytes())
    (out / 'path_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())
    save_json(out / 'candidate_plan.json', [{'candidate': 0, 'decision_rule': 'cumulative_expected_events', 'refit': False}])
    save_json(out / 'rate_calibration.json', calibration)
    policy = RateActionPolicy(existing, existing.manager, existing.thresholds, existing.multiplier, calibration['scales'])
    config = EngineConfig(**parent['risk'])
    try:
        bars, checks = prepare_minute_period(market, features, 'BTCUSDT', '2021-01-01', '2022-01-01')
        save_json(out / 'development_input.json', checks)
        metrics = backtest(bars, policy, config, out / 'candidate-00')
        action_diagnostics(out / 'candidate-00', bars, config)
        rows = [{'candidate': 0, **metrics}]
        save_json(out / 'development.json', rows)
        frozen = {**parent, 'protocol': 'minute_rate_v14', 'candidate': 0, 'development_metrics': rows[0],
                  'path_selection_sha256': sha256(out / 'path_selection.json'), 'rate_scales': calibration['scales'],
                  'rate_calibration_sha256': sha256(out / 'rate_calibration.json')}
        save_json(out / 'frozen_selection.json', frozen)
        save_json(out / 'frozen_integrity.json', {'frozen_selection_sha256': sha256(out / 'frozen_selection.json')})
        _, policy = load_selection(out)
        print(f'2021년 개발: {metrics["total_return"]:.2%}, {metrics["closed_trades"]}거래', flush=True)
        del bars
        bars, checks = prepare_minute_period(confirmation_market, confirmation_features, 'BTCUSDT', '2022-01-01', '2023-01-01')
        save_json(out / 'confirmation_input.json', checks)
        confirmation = backtest(bars, policy, config, out / 'confirmation-2022')
        action_diagnostics(out / 'confirmation-2022', bars, config)
        save_json(out / 'confirmation_checks.json', {'positive': confirmation['total_return'] > 0,
            'at_least_30_trades': confirmation['closed_trades'] >= 30, 'no_halt': not confirmation['permanent_halt'],
            'profitability_accepted': False, 'all_periods_already_observed': True})
        comparison_rows = [{'policy': 'v12_flat', 'period': '2021', **parent['development_metrics']},
            {'policy': 'v14_rate', 'period': '2021', **metrics},
            {'policy': 'v12_flat', 'period': '2022', **json.loads((reference / 'confirmation-2022' / 'metrics.json').read_text())},
            {'policy': 'v14_rate', 'period': '2022', **confirmation}]
        # 서로 다른 부가 열을 표로 합칠 때 생기는 결측값을 원래 JSON에 주입하지 않는다.
        save_json(out / 'comparison.json', comparison_rows)
        comparison = pd.DataFrame(comparison_rows)
        (out / 'REPORT.md').write_text('# 모델 사건 빈도의 누적 명령 대조\n\n' + table(comparison[
            ['policy', 'period', 'total_return', 'max_drawdown', 'closed_trades', 'permanent_halt']])
            + '\n\n관리 모델·신규 진입·위험은 기존 v12와 같다. 즉시 명령 문턱 대신 누적량 1의 단일 규칙을 사용하며 '
            '2020년 하반기의 사건 분 수와 점수 합으로 빈도 배율을 고정했다. 위험 청산은 평탄 상태로 유지한다. '
            '이미 관찰한 기간의 반복 연구이며 수익성 채택과 전체 goal 완료는 별도다.\n')
        save_json(out / 'summary.json', {'complete': True, 'profitability_accepted': False, 'refit': False})
        print(f'2022년 확인: {confirmation["total_return"]:.2%}, {confirmation["closed_trades"]}거래', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
