from __future__ import annotations

import json
from pathlib import Path

from .close_learning_inputs import CLOSE_SPLITS
from .common import save_json, sha256
from .continuation_diagnostics import same_outputs
from .histogram_management import HISTOGRAM_SETTINGS
from .research_paths import reproduction_root
from .retained_weight_close import RETAINED_WEIGHT_SETTINGS, RetainedWeightCloseModel
from .retained_weight_close_diagnostics import run_retained_weight_close_diagnosis
from .retained_weight_close_evaluation import RETAINED_COMPARISONS, RETAINED_POLICIES
from .retained_weight_close_reference import (
    VISITED_DIAGNOSIS_FILES,
    checked_files,
    checked_minute_reproduction,
)
from .source_snapshot import copy_snapshot

RETAINED_DIAGNOSIS_FILES = (VISITED_DIAGNOSIS_FILES
    - {f'{name}_{suffix}.parquet' for name in ['visited_constant'] for suffix in ['blocks', 'block_replicates']}
    | {'retained_training_used.parquet', 'retained_training_weights.parquet', 'retained_training_cost_ledger.parquet'}
    | {f'positions_{name}.parquet' for name in RETAINED_POLICIES}
    | {f'{name}_{suffix}.parquet' for name in RETAINED_COMPARISONS for suffix in ['blocks', 'block_replicates']})


def checked_visited_reproduction(owner, parent):
    proof = json.loads((owner/'reference_parity.json').read_text())
    child = Path(proof['reproduction'])
    if (proof['complete'] is not True or proof['reference'] != str(parent)
        or proof['reference_files_sha256'] != sha256(parent/'files.json')
        or proof['reproduction_files_sha256'] != sha256(child/'files.json')
        or proof['all_previous_outputs_exact'] is not True
        or proof['original_decision'] != json.loads((parent/'decision.json').read_text())
        or set(checked_files(child)) != VISITED_DIAGNOSIS_FILES):
        raise ValueError('문턱 보정의 이전 방문 구간 재현 연결 오류')
    same_outputs(parent, child, VISITED_DIAGNOSIS_FILES-{'manifest.json', 'reference_parity.json'})
    minute = Path(json.loads((parent/'manifest.json').read_text())['settings']['reference'])
    expected = {'source': checked_minute_reproduction(parent, minute), 'reproduced': checked_minute_reproduction(child, minute)}
    if proof['previous_reproductions'] != expected:
        raise ValueError('문턱 보정의 중첩 분별 재현 연결 오류')
    return {'path': str(child), 'files_sha256': sha256(child/'files.json'), 'minute_reproductions': expected}


def reproduce_retained_reference(reference, output):
    files = checked_files(reference)
    if set(files) != RETAINED_DIAGNOSIS_FILES:
        raise ValueError('문턱 보정의 기준 파일 목록 오류')
    before = sha256(reference/'files.json')
    manifest = json.loads((reference/'manifest.json').read_text())
    settings = manifest['settings']
    parent = Path(settings['reference'])
    expected = {'reference': str(parent), 'reference_files_sha256': sha256(parent/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V77.md')), 'periods': CLOSE_SPLITS,
        'features': RetainedWeightCloseModel.features, 'settings': HISTOGRAM_SETTINGS,
        'retained_weight_settings': RETAINED_WEIGHT_SETTINGS, 'new_model_count': 1,
        'existing_models_reproduced': True, 'candidate': 'retained_1m', 'policies': RETAINED_POLICIES,
        'comparisons': RETAINED_COMPARISONS, 'score_threshold': .5, 'score_kind': 'cost_weighted_decision_score',
        'evaluation_weights': 'original_full_minute_position_weights',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True}
    summary = json.loads((reference/'summary.json').read_text())
    if (settings != expected or type(settings['new_model_count']) is not int
        or settings['existing_models_reproduced'] is not True or settings['trading_returns_evaluated'] is not False
        or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['profitability_accepted'] is not False
        or any(summary[key] is not True for key in ['all_previous_outputs_reproduced',
            'original_full_training_and_diagnosis_preserved', 'same_teacher_scores_and_visitation_prefix',
            'same_position_population_and_first_equity'])
        or type(summary['new_models_fitted']) is not int or summary['new_models_fitted'] != 1):
        raise ValueError('문턱 보정의 기준 설정·완료 오류')
    sources, snapshot = manifest['source_sha256'], reference/'code_snapshot'
    if (set(sources) != {p.name for p in snapshot.glob('*.py')}
        or any(Path(name).name != name or not name.endswith('.py') or (snapshot/name).is_symlink()
            or sha256(snapshot/name) != value for name, value in sources.items())):
        raise ValueError('문턱 보정의 기준 코드 사본 오류')
    baseline = json.loads((reference/'baseline_files.json').read_text())
    if (baseline != checked_files(parent) or set(baseline) != VISITED_DIAGNOSIS_FILES
        or any((reference/'baseline_source'/name).is_symlink() or sha256(reference/'baseline_source'/name) != value
            for name, value in baseline.items())):
        raise ValueError('문턱 보정의 기존 방문 구간 사본 오류')
    original_child = checked_visited_reproduction(reference, parent)
    reproduced = run_retained_weight_close_diagnosis(parent, reproduction_root(output))
    same_outputs(reference, reproduced, RETAINED_DIAGNOSIS_FILES-{'manifest.json', 'reference_parity.json'})
    reproduced_child = checked_visited_reproduction(reproduced, parent)
    target = output/'baseline_source'
    target.mkdir(mode=0o700)
    for name, value in files.items():
        if sha256(reference/name) != value:
            raise ValueError('문턱 보정 중 기준 변경')
        copy_snapshot(reference/name, target/name)
        if sha256(target/name) != value:
            raise ValueError('문턱 보정의 기준 사본 불일치')
    if sha256(reference/'files.json') != before:
        raise ValueError('문턱 보정 중 기준 봉인 변경')
    save_json(output/'baseline_files.json', files)
    save_json(output/'reference_parity.json', {'complete': True, 'reference': str(reference),
        'reference_files_sha256': before, 'reproduction': str(reproduced),
        'reproduction_files_sha256': sha256(reproduced/'files.json'), 'all_previous_outputs_exact': True,
        'previous_reproductions': {'source': original_child, 'reproduced': reproduced_child},
        'original_decision': json.loads((reference/'decision.json').read_text())})
    return reproduced
