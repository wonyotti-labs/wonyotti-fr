from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .addition_effect import position_weights
from .close_economics import load_economic_inputs
from .close_flow import FLOW_WINDOWS, close_flow_source
from .close_learning_inputs import CLOSE_SPLITS, close_learning_splits, load_close_training
from .close_utility_diagnostics import fit_utility
from .common import new_run, save_json, sha256
from .continuation_inputs import load_pending_inputs
from .histogram_management import HISTOGRAM_SETTINGS
from .minute_close_inputs import attach_minute_close_inputs
from .minute_close_learning import (
    MINUTE_COMPARISONS,
    MINUTE_POLICIES,
    MinuteCloseModel,
    evaluate_minute_close,
)
from .minute_close_reference import verify_minute_baseline
from .weekly_flow_diagnostics import load_full_flow_ledger


def run_minute_close_diagnosis(labels: Path, reference: Path, output: Path) -> Path:
    settings = {'labels': str(labels), 'labels_files_sha256': sha256(labels/'files.json'),
        'reference': str(reference), 'reference_files_sha256': sha256(reference/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V75.md')), 'periods': CLOSE_SPLITS,
        'features': MinuteCloseModel.features, 'settings': HISTOGRAM_SETTINGS, 'flow_windows': FLOW_WINDOWS,
        'new_model_count': 1, 'score_threshold': .5, 'candidate': 'minute_1m', 'policies': MINUTE_POLICIES,
        'comparisons': MINUTE_COMPARISONS, 'score_kind': 'cost_weighted_decision_score',
        'cost_weight': 'original_position_weight_times_absolute_effect',
        'zero_effect_policy': 'zero_fit_contribution_preserved_in_evaluation',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True}
    out = new_run(output, 'minute-close-diagnosis', settings)
    print(f'분별 청산의 판단 빈도·학습 비교: {out}', flush=True)
    try:
        raw, labels_proof = load_close_training(labels, decision_seconds=60)
        economic, economic_proof = load_economic_inputs(labels, raw, decision_seconds=60)
        current, pending_proof = load_pending_inputs(labels, economic, decision_seconds=60)
        bars, market_proof = close_flow_source(reference)
        ledger, linkage = attach_minute_close_inputs(current, bars)
        ledger.to_parquet(out/'minute_ledger.parquet', index=False)
        linkage.to_parquet(out/'feature_linkage.parquet', index=False)
        save_json(out/'input_verification.json', {'labels': labels_proof, 'economics': economic_proof,
            'pending': pending_proof, 'market': market_proof, 'confirmed_feature_age_seconds': [0, 300],
            'upper_age_boundary_excluded': True, 'all_original_rows_preserved': True})
        print(f'분별 입력 {len(ledger)}개 검증 완료', flush=True)
        baseline, baseline_proof = verify_minute_baseline(reference, labels, ledger, out)
        save_json(out/'baseline_model.json', baseline.to_dict())
        print('기존 5분 입력·모델·점수·최초 효과 재현 완료', flush=True)
        rows, assignments = close_learning_splits(ledger)
        assignments.to_parquet(out/'exclusion_ledger.parquet', index=False)
        weights = {}
        for name, frame in rows.items():
            frame.to_parquet(out/f'{name}_used.parquet', index=False)
            weights[name] = frame[['decision_time', 'position_entry_time']].assign(sample_weight=position_weights(frame))
            weights[name].to_parquet(out/f'{name}_weights.parquet', index=False)
        model, support, costs = fit_utility(rows['training'], weights['training'], rows['diagnosis'], model_class=MinuteCloseModel)
        save_json(out/'model.json', model.to_dict())
        save_json(out/'training_support.json', support)
        costs.to_parquet(out/'training_cost_ledger.parquet', index=False)
        frame = rows['diagnosis'].assign(sample_weight=weights['diagnosis'].sample_weight)
        values = frame[model.features].to_numpy(dtype=float)
        result = evaluate_minute_close(frame, baseline.probabilities(values)[:, 0], model.probabilities(values)[:, 0],
            support['training_constant_score'])
        result['predictions'].to_parquet(out/'predictions.parquet', index=False)
        result['draws'].to_parquet(out/'block_draws.parquet', index=False)
        for name, part in result['positions'].items():
            part.to_parquet(out/f'positions_{name}.parquet', index=False)
        for key in MINUTE_COMPARISONS:
            result['blocks'][key].to_parquet(out/f'{key}_blocks.parquet', index=False)
            result['replicates'][key].to_parquet(out/f'{key}_block_replicates.parquet', index=False)
        for name in ['metrics', 'first_metrics', 'probability_metrics', 'breakdown', 'first_breakdown',
            'probability_breakdown', 'intervals', 'decision']:
            save_json(out/f'{name}.json', result[name])
        for folder, seal in [(labels, settings['labels_files_sha256']), (reference, settings['reference_files_sha256'])]:
            if sha256(folder/'files.json') != seal or any(sha256(folder/name) != digest
                for name, digest in json.loads((folder/'files.json').read_text()).items()):
                raise ValueError('분별 청산 진단 중 입력·기존 결과 변경')
        for name, digest in json.loads((out/'baseline_files.json').read_text()).items():
            if sha256(out/'baseline_source'/name) != digest:
                raise ValueError('분별 청산 진단 중 기존 결과 사본 변경')
        legacy, final_source = load_full_flow_ledger(reference)
        if final_source != baseline_proof['original_ledger_source'] or close_flow_source(reference)[1] != market_proof:
            raise ValueError('분별 청산 진단 중 원래 원장·시세 연결 변경')
        grid = ledger.decision_time.astype('datetime64[ns, UTC]').array.asi8 % (300*10**9) == 0
        pd.testing.assert_frame_equal(ledger.loc[grid].reset_index(drop=True), legacy, check_exact=True)
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (set(sources) != {p.name for p in Path(__file__).parent.glob('*.py')}
            or any(sha256(Path(__file__).parent/name) != digest for name, digest in sources.items())
            or sha256(Path('docs/EXPERIMENT_V75.md')) != settings['protocol_sha256']):
            raise ValueError('분별 청산 진단 중 구현·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'all_minute_inputs_and_legacy_reference_verified': True,
            'same_position_population_and_first_equity': True, 'new_models_fitted': 1,
            'training_rows': len(rows['training']), 'diagnosis_rows': len(rows['diagnosis']),
            'training_positions': int(rows['training'].position_entry_time.nunique()),
            'diagnosis_positions': int(rows['diagnosis'].position_entry_time.nunique()),
            'zero_effect_rows_preserved': True, 'profitability_accepted': False, **result['decision']})
        (out/'REPORT.md').write_text('# 분별 청산의 판단 빈도·학습 비교\n\n'
            f'사전 조건 통과: {result["decision"]["minute_close_admitted"]}. '
            '같은 분별 모집단과 최초 순자산에서 기존·새 학습 모델의 5분·1분 정책을 비교했다. '
            '새 1분 정책 하나를 후보로 고정하고 선택 없음과 기존 실패를 보존했다. '
            '개발 적합 경로의 조건부 청산 효과이며 연속 계좌 수익성과 다년·다시장 검증은 별도다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(result['decision'], flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
