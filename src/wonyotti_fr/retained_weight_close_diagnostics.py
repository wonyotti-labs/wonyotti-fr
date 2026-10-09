from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .close_learning_inputs import CLOSE_SPLITS, close_learning_splits
from .common import new_run, save_json, sha256
from .histogram_management import HISTOGRAM_SETTINGS
from .retained_weight_close import (
    RETAINED_WEIGHT_SETTINGS,
    RetainedWeightCloseModel,
    fit_retained_weight_close,
)
from .retained_weight_close_evaluation import (
    RETAINED_COMPARISONS,
    RETAINED_POLICIES,
    evaluate_retained_weight_close,
)
from .retained_weight_close_reference import checked_files, reproduce_visited_reference
from .source_snapshot import copy_snapshot
from .visited_close_evaluation import VISITED_POLICIES

RETAINED_INPUT_FILES = {'minute_ledger.parquet', 'feature_linkage.parquet', 'exclusion_ledger.parquet',
    'training_used.parquet', 'training_weights.parquet', 'diagnosis_used.parquet', 'diagnosis_weights.parquet',
    'teacher_models.json', 'teacher_support.json', 'teacher_training_membership.parquet', 'visitation_ledger.parquet',
    'visited_training_used.parquet', 'visited_training_weights.parquet', 'visited_training_cost_ledger.parquet'}


def load_visited_evaluation(reference):
    values = {name: json.loads((reference/f'{name}.json').read_text()) for name in
        ['metrics', 'first_metrics', 'probability_metrics', 'breakdown', 'first_breakdown', 'probability_breakdown']}
    values['positions'] = {name: pd.read_parquet(reference/f'positions_{name}.parquet') for name in VISITED_POLICIES}
    values['predictions'] = pd.read_parquet(reference/'predictions.parquet')
    values['draws'] = pd.read_parquet(reference/'block_draws.parquet')
    return values


def run_retained_weight_close_diagnosis(reference: Path, output: Path) -> Path:
    settings = {'reference': str(reference), 'reference_files_sha256': sha256(reference/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V77.md')), 'periods': CLOSE_SPLITS,
        'features': RetainedWeightCloseModel.features, 'settings': HISTOGRAM_SETTINGS,
        'retained_weight_settings': RETAINED_WEIGHT_SETTINGS, 'new_model_count': 1,
        'existing_models_reproduced': True, 'candidate': 'retained_1m', 'policies': RETAINED_POLICIES,
        'comparisons': RETAINED_COMPARISONS, 'score_threshold': .5, 'score_kind': 'cost_weighted_decision_score',
        'evaluation_weights': 'original_full_minute_position_weights',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True}
    out = new_run(output, 'retained-weight-close-diagnosis', settings)
    print(f'방문 구간의 기존 행 비중 대조: {out}', flush=True)
    try:
        reproduced = reproduce_visited_reference(reference, out)
        print('기존 보조 모델·방문 구간·학습·전체 평가 재현 완료', flush=True)
        for name in sorted(RETAINED_INPUT_FILES):
            copy_snapshot(reproduced/name, out/name)
        ledger = pd.read_parquet(out/'minute_ledger.parquet')
        rows, assignment = close_learning_splits(ledger)
        pd.testing.assert_frame_equal(assignment, pd.read_parquet(out/'exclusion_ledger.parquet'), check_exact=True)
        for name, frame in rows.items():
            pd.testing.assert_frame_equal(frame, pd.read_parquet(out/f'{name}_used.parquet'), check_exact=True)
        visit = pd.read_parquet(out/'visitation_ledger.parquet')
        np.testing.assert_array_equal(visit.opportunity_index, np.flatnonzero(assignment.split.eq('training')))
        model, support = fit_retained_weight_close(rows['training'], pd.read_parquet(out/'training_weights.parquet'),
            visit, rows['diagnosis'], out)
        pd.testing.assert_frame_equal(pd.read_parquet(out/'retained_training_used.parquet'),
            pd.read_parquet(out/'visited_training_used.parquet'), check_exact=True)
        frame = rows['diagnosis'].assign(sample_weight=pd.read_parquet(out/'diagnosis_weights.parquet').sample_weight)
        score = model.probabilities(frame[model.features].to_numpy(dtype=float))[:, 0]
        result = evaluate_retained_weight_close(frame, load_visited_evaluation(reproduced), score, support['training_constant_score'])
        result['predictions'].to_parquet(out/'predictions.parquet', index=False)
        result['draws'].to_parquet(out/'block_draws.parquet', index=False)
        for name, part in result['positions'].items():
            part.to_parquet(out/f'positions_{name}.parquet', index=False)
        for name in RETAINED_COMPARISONS:
            result['blocks'][name].to_parquet(out/f'{name}_blocks.parquet', index=False)
            result['replicates'][name].to_parquet(out/f'{name}_block_replicates.parquet', index=False)
        for name in ['metrics', 'first_metrics', 'probability_metrics', 'breakdown', 'first_breakdown',
            'probability_breakdown', 'intervals', 'decision']:
            save_json(out/f'{name}.json', result[name])
        baseline = json.loads((out/'baseline_files.json').read_text())
        if (sha256(reference/'files.json') != settings['reference_files_sha256'] or checked_files(reference) != baseline
            or any(sha256(out/'baseline_source'/name) != value for name, value in baseline.items())):
            raise ValueError('원래 행 비중 학습 중 이전 결과·사본 변경')
        proof = json.loads((out/'reference_parity.json').read_text())
        linked = [{'path': str(reproduced), 'files_sha256': proof['reproduction_files_sha256']},
            *proof['previous_reproductions'].values()]
        for item in linked:
            folder = Path(item['path'])
            if sha256(folder/'files.json') != item['files_sha256']:
                raise ValueError('원래 행 비중 학습 중 이전 재현 봉인 변경')
            checked_files(folder)
        if any(sha256(out/name) != sha256(reproduced/name) for name in RETAINED_INPUT_FILES):
            raise ValueError('원래 행 비중 학습 중 보존 입력 변경')
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (set(sources) != {p.name for p in Path(__file__).parent.glob('*.py')}
            or any(sha256(Path(__file__).parent/name) != value for name, value in sources.items())
            or sha256(Path('docs/EXPERIMENT_V77.md')) != settings['protocol_sha256']):
            raise ValueError('원래 행 비중 학습 중 구현·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'original_full_training_and_diagnosis_preserved': True, 'same_teacher_scores_and_visitation_prefix': True,
            'same_position_population_and_first_equity': True, 'new_models_fitted': 1,
            'original_training_rows': len(rows['training']), 'retained_training_rows': support['retained_rows'],
            'retained_positions': support['retained_positions'], 'diagnosis_rows': len(frame),
            'diagnosis_positions': int(frame.position_entry_time.nunique()), 'profitability_accepted': False, **result['decision']})
        (out/'REPORT.md').write_text('# 방문 구간의 기존 행 비중 대조\n\n'
            f'사전 조건 통과: {result["decision"]["retained_weight_admitted"]}. '
            '같은 보조 점수·방문 구간·원래 정답에서 유지한 행의 상대 비중을 보존했다. '
            '기존 모든 결과와 전체 진단 모집단을 유지했다. 개발 경로의 조건부 청산 효과이며 '
            '연속 계좌 수익성과 다년·다시장 검증은 별도다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(result['decision'], flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
