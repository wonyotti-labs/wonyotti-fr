from __future__ import annotations

import json
from pathlib import Path

from .close_learning_inputs import CLOSE_SPLITS
from .common import save_json, sha256
from .continuation_diagnostics import same_outputs
from .histogram_management import HISTOGRAM_SETTINGS
from .research_paths import reproduction_root
from .source_snapshot import copy_snapshot
from .visited_close import VISITATION_SETTINGS, VisitedCloseModel
from .visited_close_diagnostics import run_visited_close_diagnosis
from .visited_close_evaluation import VISITED_COMPARISONS, VISITED_POLICIES
from .visited_close_reference import MINUTE_DIAGNOSIS_FILES

VISITED_DIAGNOSIS_FILES = {'manifest.json', 'baseline_files.json', 'reference_parity.json',
    'minute_ledger.parquet', 'feature_linkage.parquet', 'exclusion_ledger.parquet',
    'training_used.parquet', 'training_weights.parquet', 'diagnosis_used.parquet', 'diagnosis_weights.parquet',
    'teacher_models.json', 'teacher_support.json', 'teacher_training_membership.parquet', 'visitation_ledger.parquet',
    'visited_training_used.parquet', 'visited_training_weights.parquet', 'visited_training_cost_ledger.parquet',
    'full_training_contribution.parquet', 'model.json', 'training_support.json', 'predictions.parquet',
    'block_draws.parquet', 'metrics.json', 'first_metrics.json', 'probability_metrics.json', 'breakdown.json',
    'first_breakdown.json', 'probability_breakdown.json', 'intervals.json', 'decision.json', 'summary.json', 'REPORT.md',
    *[f'positions_{name}.parquet' for name in VISITED_POLICIES],
    *[f'{name}_{suffix}.parquet' for name in VISITED_COMPARISONS for suffix in ['blocks', 'block_replicates']]}


def checked_files(folder):
    files = json.loads((folder/'files.json').read_text())
    if (folder.is_symlink() or (folder/'files.json').is_symlink()
        or any(Path(name).name != name or (folder/name).is_symlink() or sha256(folder/name) != value
            for name, value in files.items())):
        raise ValueError('방문 구간 비중 대조의 원본 파일·지문 오류')
    return files


def checked_minute_reproduction(owner, parent):
    proof = json.loads((owner/'reference_parity.json').read_text())
    child = Path(proof['reproduction'])
    if (proof['complete'] is not True or proof['reference'] != str(parent)
        or proof['reference_files_sha256'] != sha256(parent/'files.json')
        or proof['reproduction_files_sha256'] != sha256(child/'files.json')
        or proof['all_minute_inputs_models_costs_scores_and_evaluations_exact'] is not True
        or proof['original_decision'] != json.loads((parent/'decision.json').read_text())
        or set(checked_files(child)) != MINUTE_DIAGNOSIS_FILES):
        raise ValueError('방문 구간 비중 대조의 이전 분별 재현 연결 오류')
    same_outputs(parent, child, MINUTE_DIAGNOSIS_FILES-{'manifest.json'})
    return {'path': str(child), 'files_sha256': sha256(child/'files.json')}


def reproduce_visited_reference(reference, output):
    files = checked_files(reference)
    if set(files) != VISITED_DIAGNOSIS_FILES:
        raise ValueError('방문 구간 비중 대조의 기준 파일 목록 오류')
    before = sha256(reference/'files.json')
    manifest = json.loads((reference/'manifest.json').read_text())
    settings = manifest['settings']
    parent = Path(settings['reference'])
    expected = {'reference': str(parent), 'reference_files_sha256': sha256(parent/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V76.md')), 'periods': CLOSE_SPLITS,
        'features': VisitedCloseModel.features, 'settings': HISTOGRAM_SETTINGS, 'visitation_settings': VISITATION_SETTINGS,
        'teacher_models': 5, 'candidate_models': 1, 'candidate': 'visited_1m', 'policies': VISITED_POLICIES,
        'comparisons': VISITED_COMPARISONS, 'score_threshold': .5, 'score_kind': 'cost_weighted_decision_score',
        'training_target': 'original_natural_close_advantage', 'evaluation_weights': 'original_full_minute_position_weights',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True}
    summary = json.loads((reference/'summary.json').read_text())
    if (settings != expected or any(type(settings[key]) is not int for key in ['teacher_models', 'candidate_models'])
        or settings['trading_returns_evaluated'] is not False or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['profitability_accepted'] is not False
        or any(summary[key] is not True for key in ['all_previous_outputs_reproduced',
            'original_training_and_full_diagnosis_rows_preserved', 'same_full_position_population_and_first_equity'])
        or type(summary['teacher_models']) is not int or summary['teacher_models'] != 5
        or type(summary['candidate_models']) is not int or summary['candidate_models'] != 1):
        raise ValueError('방문 구간 비중 대조의 설정·계획·완료 오류')
    sources = manifest['source_sha256']
    snapshot = reference/'code_snapshot'
    if (set(sources) != {p.name for p in snapshot.glob('*.py')}
        or any(Path(name).name != name or not name.endswith('.py') or (snapshot/name).is_symlink()
            or sha256(snapshot/name) != value for name, value in sources.items())):
        raise ValueError('방문 구간 비중 대조의 기존 코드 사본 오류')
    baseline = json.loads((reference/'baseline_files.json').read_text())
    if (baseline != checked_files(parent) or set(baseline) != MINUTE_DIAGNOSIS_FILES
        or any((reference/'baseline_source'/name).is_symlink() or sha256(reference/'baseline_source'/name) != value
            for name, value in baseline.items())):
        raise ValueError('방문 구간 비중 대조의 기존 분별 사본 오류')
    original_child = checked_minute_reproduction(reference, parent)
    reproduced = run_visited_close_diagnosis(parent, reproduction_root(output))
    same_outputs(reference, reproduced, VISITED_DIAGNOSIS_FILES-{'manifest.json', 'reference_parity.json'})
    reproduced_child = checked_minute_reproduction(reproduced, parent)
    target = output/'baseline_source'
    target.mkdir(mode=0o700)
    for name, value in files.items():
        if sha256(reference/name) != value:
            raise ValueError('방문 구간 비중 대조 중 원본 변경')
        copy_snapshot(reference/name, target/name)
        if sha256(target/name) != value:
            raise ValueError('방문 구간 비중 대조의 사본 불일치')
    if sha256(reference/'files.json') != before:
        raise ValueError('방문 구간 비중 대조 중 원본 봉인 변경')
    save_json(output/'baseline_files.json', files)
    save_json(output/'reference_parity.json', {'complete': True, 'reference': str(reference),
        'reference_files_sha256': before, 'reproduction': str(reproduced),
        'reproduction_files_sha256': sha256(reproduced/'files.json'), 'all_previous_outputs_exact': True,
        'previous_reproductions': {'source': original_child, 'reproduced': reproduced_child},
        'original_decision': json.loads((reference/'decision.json').read_text())})
    return reproduced
