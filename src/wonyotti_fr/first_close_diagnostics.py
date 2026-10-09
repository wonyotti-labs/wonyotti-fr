from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .close_calibration import DIAGNOSIS_FILES
from .close_economics import EconomicCloseModel, run_close_economic_diagnosis
from .close_learning import close_metrics
from .close_learning_inputs import CLOSE_SPLITS
from .common import new_run, save_json, sha256
from .entry_regression import REGRESSION_SETTINGS
from .research_paths import reproduction_root

ECONOMIC_FILES = {'manifest.json', 'input_verification.json', 'economic_ledger.parquet', 'exclusion_ledger.parquet',
    'training_used.parquet', 'training_weights.parquet', 'diagnosis_used.parquet', 'diagnosis_weights.parquet',
    'model.json', 'previous_models.json', 'training_support.json', 'predictions.parquet', 'metrics.json',
    'breakdown.json', 'decision.json', 'summary.json', 'reference_parity.json', 'REPORT.md'}
FIRST_MODELS = ['economic', 'boosted']
BLOCK_START, BLOCK_END = pd.Timestamp('2021-10-02', tz='UTC'), pd.Timestamp('2021-12-31', tz='UTC')


def first_close_positions(frame, prediction, *, margin_bps=0.):
    if type(margin_bps) not in (int, float) or not np.isfinite(margin_bps) or margin_bps < 0:
        raise ValueError('최초 청산 진단의 실행 문턱 오류')
    columns = ['close_advantage_pnl', 'close_advantage_bps', 'decision_equity', prediction]
    if (frame.empty or frame.decision_time.isna().any() or frame.decision_time.duplicated().any()
        or not frame.decision_time.is_monotonic_increasing
        or frame[['position_entry_time', 'label_end']].isna().any().any()
        or frame.position_entry_time.ge(frame.decision_time).any() or frame.decision_time.ge(frame.label_end).any()
        or not frame.label_status.eq('closed').all() or not frame.direction.isin([-1, 1]).all()
        or not frame.original_intent.isin(['hold', 'exit', 'reduce', 'increase']).all()
        or not np.isfinite(frame[columns]).all().all() or frame.decision_equity.le(0).any()
        or not np.allclose(frame.close_advantage_bps, frame.close_advantage_pnl/frame.decision_equity*10000, rtol=0, atol=1e-9)):
        raise ValueError('최초 청산 진단의 시각·정답·금액·순자산 오류')
    result = []
    for entry, part in frame.groupby('position_entry_time', sort=True):
        if part.direction.nunique() != 1 or part.continue_end.nunique() != 1 or part.continue_cash.nunique() != 1:
            raise ValueError('최초 청산 진단의 원래 포지션 연결 오류')
        selected = part.loc[part[prediction].gt(margin_bps) & part.original_intent.ne('exit')]
        first, anchor = (selected.iloc[0] if len(selected) else None), part.iloc[0]
        amount = float(first.close_advantage_pnl) if first is not None else 0.
        result.append({'position_entry_time': entry, 'direction': int(anchor.direction),
            'first_available_time': anchor.decision_time, 'reference_equity': float(anchor.decision_equity),
            'opportunities': len(part), 'selected_opportunities': len(selected),
            'later_selected_opportunities': max(len(selected)-1, 0), 'chosen': first is not None,
            'first_selected_time': first.decision_time if first is not None else pd.NaT,
            'first_prediction_bps': float(first[prediction]) if first is not None else None,
            'first_label_end': first.label_end if first is not None else pd.NaT,
            'first_effect_pnl': amount, 'first_effect_decision_bps': float(first.close_advantage_bps) if first is not None else 0.,
            'first_effect_common_bps': amount/anchor.decision_equity*10000})
    return pd.DataFrame(result)


def first_close_metrics(positions):
    if positions.empty or positions.position_entry_time.duplicated().any() or not np.isfinite(positions[['first_effect_pnl', 'first_effect_common_bps']]).all().all():
        raise ValueError('최초 청산 효과의 포지션·금액 오류')
    chosen = positions[positions.chosen]
    return {'positions': len(positions), 'selected_positions': len(chosen),
        'opportunities': int(positions.opportunities.sum()), 'selected_opportunities': int(positions.selected_opportunities.sum()),
        'later_selected_opportunities': int(positions.later_selected_opportunities.sum()),
        'all_position_mean_common_bps': float(positions.first_effect_common_bps.mean()),
        'all_position_mean_pnl': float(positions.first_effect_pnl.mean()),
        'selected_mean_common_bps': float(chosen.first_effect_common_bps.mean()) if len(chosen) else None,
        'selected_mean_decision_bps': float(chosen.first_effect_decision_bps.mean()) if len(chosen) else None,
        'selected_mean_pnl': float(chosen.first_effect_pnl.mean()) if len(chosen) else None,
        'selected_positive': int(chosen.first_effect_pnl.gt(0).sum()),
        'selected_negative': int(chosen.first_effect_pnl.lt(0).sum()),
        'selected_zero': int(chosen.first_effect_pnl.eq(0).sum())}


def paired_week_blocks(positions, *, model_names=None):
    names = FIRST_MODELS if model_names is None else model_names
    if (not isinstance(names, list) or len(names) != 2 or any(not isinstance(n, str) or not n for n in names)
        or len(set(names)) != 2 or set(positions) != set(names)):
        raise ValueError('최초 청산 블록 진단의 비교 모형 오류')
    first, second = (positions[k] for k in names)
    keys = ['position_entry_time', 'direction', 'first_available_time', 'reference_equity']
    pd.testing.assert_frame_equal(first[keys], second[keys], check_exact=True)
    if (first.empty or first.position_entry_time.duplicated().any() or not first.position_entry_time.is_monotonic_increasing
        or first.position_entry_time.lt(BLOCK_START).any() or first.position_entry_time.ge(BLOCK_END).any()
        or any(not np.isfinite(p.first_effect_common_bps).all() for p in positions.values())):
        raise ValueError('최초 청산 블록 진단의 기간·포지션·숫자 오류')
    block = ((first.position_entry_time-BLOCK_START)/pd.Timedelta(days=7)).astype(int).to_numpy()
    blocks = pd.DataFrame({'block': np.arange(13), 'start': pd.date_range(BLOCK_START, periods=13, freq='7D')})
    blocks['end'] = blocks.start.add(pd.Timedelta(days=7)).clip(upper=BLOCK_END)
    blocks['positions'] = np.bincount(block, minlength=13)
    for name in names:
        # 블록 안 합은 재표본 평균의 분자이며 연속 계좌 수익이 아니다.
        blocks[f'effect_sum_{name}'] = np.bincount(block, weights=positions[name].first_effect_common_bps, minlength=13)
    draws = np.random.default_rng(63).integers(0, 13, size=(1000, 13))
    denominator = blocks.positions.to_numpy()[draws].sum(axis=1)
    replicates = pd.DataFrame({'replicate': np.arange(1000), 'positions': denominator})
    for name in names:
        numerator = blocks[f'effect_sum_{name}'].to_numpy()[draws].sum(axis=1)
        replicates[name] = np.divide(numerator, denominator, out=np.full(1000, np.nan), where=denominator > 0)
    replicates['paired_difference'] = replicates[names[0]]-replicates[names[1]]
    intervals = {}
    for name in [*names, 'paired_difference']:
        valid = replicates[name].dropna()
        intervals[name] = {'valid_replicates': len(valid), 'lower': float(valid.quantile(.025)) if len(valid) else None,
            'upper': float(valid.quantile(.975)) if len(valid) else None}
    return blocks, pd.DataFrame(draws, columns=[f'draw_{i}' for i in range(13)]), replicates, {
        'seed': 63, 'replicates': 1000, 'calendar_blocks': 13, 'empty_blocks': int(blocks.positions.eq(0).sum()),
        'block_days': 7, 'intervals': intervals, 'profitability_accepted': False}


def reproduce_economic_diagnosis(reference, output):
    files = json.loads((reference/'files.json').read_text())
    if (set(files) != ECONOMIC_FILES or (reference/'files.json').is_symlink()
        or any((reference/n).is_symlink() or sha256(reference/n) != h for n, h in files.items())):
        raise ValueError('최초 청산 진단의 기존 경제적 입력 파일·지문 오류')
    settings = json.loads((reference/'manifest.json').read_text())['settings']
    summary = json.loads((reference/'summary.json').read_text())
    previous = Path(settings['reference'])
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V62.md'))
        or settings['reference_files_sha256'] != sha256(previous/'files.json')
        or settings['periods'] != CLOSE_SPLITS or settings['features'] != EconomicCloseModel.features
        or settings['settings'] != REGRESSION_SETTINGS or settings['margin_bps'] != 0 or settings['new_model_count'] != 1
        or settings['trading_returns_evaluated'] is not False or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['profitability_accepted'] is not False):
        raise ValueError('최초 청산 진단의 기존 기간·설정·완료 연결 오류')
    reproduced = run_close_economic_diagnosis(previous, output)
    for name in sorted(ECONOMIC_FILES-{'manifest.json', 'reference_parity.json'}):
        if name.endswith('.parquet'):
            pd.testing.assert_frame_equal(pd.read_parquet(reference/name), pd.read_parquet(reproduced/name), check_exact=True)
        elif name.endswith('.json'):
            if json.loads((reference/name).read_text()) != json.loads((reproduced/name).read_text()):
                raise ValueError('최초 청산 진단의 기존 재현 불일치: '+name)
        elif (reference/name).read_bytes() != (reproduced/name).read_bytes():
            raise ValueError('최초 청산 진단의 기존 보고서 재현 불일치')
    for folder in [reference, reproduced]:
        parity = json.loads((folder/'reference_parity.json').read_text())
        child = Path(parity['reproduction'])
        if (parity['complete'] is not True or parity['all_previous_models_predictions_rows_weights_exact'] is not True
            or parity['reproduced_files_sha256'] != sha256(child/'files.json')):
            raise ValueError('최초 청산 진단의 기존 재현 경로 연결 오류')
        for name in DIAGNOSIS_FILES-{'manifest.json'}:
            if name.endswith('.parquet'):
                pd.testing.assert_frame_equal(pd.read_parquet(previous/name), pd.read_parquet(child/name), check_exact=True)
            elif json.loads((previous/name).read_text()) != json.loads((child/name).read_text()):
                raise ValueError('최초 청산 진단의 기존 대조 출력 오류')
    return reproduced


def run_first_close_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'first-close-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V63.md')),
        'period': CLOSE_SPLITS['diagnosis'], 'models': FIRST_MODELS, 'margin_bps': 0,
        'new_models_fitted': False, 'existing_models_reproduced': True, 'trading_returns_evaluated': False,
        'whole_system_periods_already_observed': True})
    print(f'포지션당 최초 청산 선택 진단: {out}', flush=True)
    try:
        reproduced = reproduce_economic_diagnosis(reference, reproduction_root(out))
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_outputs_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        frame = pd.read_parquet(reproduced/'diagnosis_used.parquet')
        predictions = pd.read_parquet(reproduced/'predictions.parquet')
        for column in ['decision_time', 'position_entry_time', 'label_end', 'direction', 'original_intent', 'close_advantage_bps']:
            pd.testing.assert_series_equal(frame[column], predictions[column], check_exact=True)
        frame['sample_weight'] = predictions.sample_weight
        for name in FIRST_MODELS:
            frame[f'predicted_{name}'] = predictions[f'predicted_{name}']
        frame.to_parquet(out/'opportunity_ledger.parquet', index=False)
        (out/'exclusion_ledger.parquet').write_bytes((reproduced/'exclusion_ledger.parquet').read_bytes())
        positions = {name: first_close_positions(frame, f'predicted_{name}') for name in FIRST_MODELS}
        metrics, details = {}, []
        for name, part in positions.items():
            part.to_parquet(out/f'positions_{name}.parquet', index=False)
            metrics[name] = first_close_metrics(part)
            groups = [('direction', str(k), f) for k, f in part.groupby('direction')]
            groups += [('entry_month', k, f) for k, f in part.groupby(part.position_entry_time.dt.strftime('%Y-%m'))]
            for kind, group, f in groups:
                details.append({'model': name, 'kind': kind, 'group': group, **first_close_metrics(f)})
        blocks, draws, replicates, intervals = paired_week_blocks(positions)
        blocks.to_parquet(out/'blocks.parquet', index=False)
        draws.to_parquet(out/'block_draws.parquet', index=False)
        replicates.to_parquet(out/'block_replicates.parquet', index=False)
        opportunity = {name: close_metrics(frame, frame[f'predicted_{name}']) for name in FIRST_MODELS}
        for name, content in [('metrics', metrics), ('breakdown', details), ('block_intervals', intervals), ('opportunity_metrics', opportunity)]:
            save_json(out/f'{name}.json', content)
        save_json(out/'summary.json', {'complete': True, 'opportunities': len(frame), 'positions': len(positions['economic']),
            'all_original_positions_preserved': True, 'new_models_fitted': False, 'profitability_accepted': False,
            'trading_returns_evaluated': False, 'admission_decision_made': False})
        (out/'REPORT.md').write_text('# 포지션당 최초 청산의 조건부 효과\n\n'
            '고정된 두 모형의 최초 양수 선택과 선택 없는 포지션을 모두 기록했다. '
            '공통 기준은 원래 포지션의 첫 이용 가능 판단 당시 순자산이다. '
            '주간 블록 구간은 짧은 기간과 블록 간 의존성의 한계가 있다. '
            '청산 뒤 재진입과 계좌 복리는 평가하지 않았으며 기존 실패나 수익성 판정을 변경하지 않았다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
