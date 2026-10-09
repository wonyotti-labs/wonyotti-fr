from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .close_learning_inputs import CLOSE_SPLITS, close_learning_splits
from .common import new_run, save_json, sha256
from .histogram_management import HISTOGRAM_SETTINGS
from .minute_close_learning import MINUTE_POLICIES
from .source_snapshot import copy_snapshot
from .visited_close import VISITATION_SETTINGS, VisitedCloseModel, fit_visited_close
from .visited_close_evaluation import VISITED_COMPARISONS, VISITED_POLICIES, evaluate_visited_close
from .visited_close_reference import reproduce_minute_reference


def load_minute_evaluation(reference):
    values = {name: json.loads((reference/f'{name}.json').read_text()) for name in
        ['metrics', 'first_metrics', 'probability_metrics', 'breakdown', 'first_breakdown', 'probability_breakdown']}
    values['positions'] = {name: pd.read_parquet(reference/f'positions_{name}.parquet') for name in MINUTE_POLICIES}
    values['predictions'] = pd.read_parquet(reference/'predictions.parquet')
    values['draws'] = pd.read_parquet(reference/'block_draws.parquet')
    return values


def run_visited_close_diagnosis(reference: Path, output: Path) -> Path:
    settings = {'reference': str(reference), 'reference_files_sha256': sha256(reference/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V76.md')), 'periods': CLOSE_SPLITS,
        'features': VisitedCloseModel.features, 'settings': HISTOGRAM_SETTINGS, 'visitation_settings': VISITATION_SETTINGS,
        'teacher_models': 5, 'candidate_models': 1, 'candidate': 'visited_1m', 'policies': VISITED_POLICIES,
        'comparisons': VISITED_COMPARISONS, 'score_threshold': .5, 'score_kind': 'cost_weighted_decision_score',
        'training_target': 'original_natural_close_advantage', 'evaluation_weights': 'original_full_minute_position_weights',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True}
    out = new_run(output, 'visited-close-diagnosis', settings)
    print(f'최초 청산까지의 방문 구간 학습: {out}', flush=True)
    try:
        reproduced = reproduce_minute_reference(reference, out)
        print('기존 분별 원장·모델·정책·실패 재현 완료', flush=True)
        for name in ['minute_ledger', 'feature_linkage', 'training_used', 'training_weights',
            'diagnosis_used', 'diagnosis_weights', 'exclusion_ledger']:
            copy_snapshot(reproduced/f'{name}.parquet', out/f'{name}.parquet')
        ledger = pd.read_parquet(out/'minute_ledger.parquet')
        rows, assignment = close_learning_splits(ledger)
        pd.testing.assert_frame_equal(assignment, pd.read_parquet(out/'exclusion_ledger.parquet'), check_exact=True)
        for name, frame in rows.items():
            pd.testing.assert_frame_equal(frame, pd.read_parquet(out/f'{name}_used.parquet'), check_exact=True)
        indices = np.flatnonzero(assignment.split.eq('training').to_numpy())
        weights = pd.read_parquet(out/'training_weights.parquet')
        model, support = fit_visited_close(rows['training'], weights, rows['diagnosis'], indices, out)
        print(f'방문 구간 {support["visited_rows"]}개와 최종 모델 학습 완료', flush=True)
        frame = rows['diagnosis'].assign(sample_weight=pd.read_parquet(out/'diagnosis_weights.parquet').sample_weight)
        score = model.probabilities(frame[model.features].to_numpy(dtype=float))[:, 0]
        previous = load_minute_evaluation(reproduced)
        result = evaluate_visited_close(frame, previous, score, support['training_constant_score'])
        result['predictions'].to_parquet(out/'predictions.parquet', index=False)
        result['draws'].to_parquet(out/'block_draws.parquet', index=False)
        for name, part in result['positions'].items():
            part.to_parquet(out/f'positions_{name}.parquet', index=False)
        for name in VISITED_COMPARISONS:
            result['blocks'][name].to_parquet(out/f'{name}_blocks.parquet', index=False)
            result['replicates'][name].to_parquet(out/f'{name}_block_replicates.parquet', index=False)
        for name in ['metrics', 'first_metrics', 'probability_metrics', 'breakdown', 'first_breakdown',
            'probability_breakdown', 'intervals', 'decision']:
            save_json(out/f'{name}.json', result[name])
        if (sha256(reference/'files.json') != settings['reference_files_sha256']
            or any(sha256(reference/name) != value or sha256(out/'baseline_source'/name) != value
                for name, value in json.loads((out/'baseline_files.json').read_text()).items())):
            raise ValueError('방문 청산 학습 중 이전 결과·사본 변경')
        proof = json.loads((out/'reference_parity.json').read_text())
        if (sha256(reproduced/'files.json') != proof['reproduction_files_sha256']
            or any(sha256(reproduced/name) != value for name, value in json.loads((reproduced/'files.json').read_text()).items())):
            raise ValueError('방문 청산 학습 중 분별 재현 결과 변경')
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (set(sources) != {p.name for p in Path(__file__).parent.glob('*.py')}
            or any(sha256(Path(__file__).parent/name) != value for name, value in sources.items())
            or sha256(Path('docs/EXPERIMENT_V76.md')) != settings['protocol_sha256']):
            raise ValueError('방문 청산 학습 중 구현·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'original_training_and_full_diagnosis_rows_preserved': True, 'same_full_position_population_and_first_equity': True,
            'teacher_models': 5, 'candidate_models': 1, 'training_rows': len(rows['training']),
            'visited_training_rows': support['visited_rows'], 'training_positions': support['original_positions'],
            'visited_positions': support['visited_positions'], 'diagnosis_rows': len(frame),
            'diagnosis_positions': int(frame.position_entry_time.nunique()), 'profitability_accepted': False, **result['decision']})
        (out/'REPORT.md').write_text('# 최초 청산까지의 방문 구간 학습\n\n'
            f'사전 조건 통과: {result["decision"]["visited_close_admitted"]}. '
            '자기 포지션을 제외한 보조 정책의 최초 청산까지 학습하고 기존 전체 진단 모집단에서 비교했다. '
            '이후 원장·기존 실패·선택 없음은 보존했다. 개발 적합 경로의 조건부 효과이며 '
            '연속 계좌 수익성과 다년·다시장 검증은 별도다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(result['decision'], flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
