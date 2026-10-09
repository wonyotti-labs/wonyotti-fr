from __future__ import annotations

import json
from pathlib import Path

from .close_threshold import THRESHOLD_SPLITS
from .common import save_json, sha256
from .continuation_diagnostics import same_outputs
from .early_stopping_close import EARLY_STOPPING_SETTINGS, EarlyStoppingCloseModel
from .early_stopping_diagnostics import EARLY_STOPPING_INPUT_FILES, run_early_stopping_diagnosis
from .early_stopping_evaluation import (
    EARLY_STOPPING_COMPARISONS,
    EARLY_STOPPING_NEW_POLICIES,
    EARLY_STOPPING_POLICIES,
)
from .early_stopping_reference import (
    THRESHOLD_FAILURE_FILES,
    checked_calibration,
    checked_retained_reproduction,
)
from .histogram_management import HISTOGRAM_SETTINGS
from .research_paths import reproduction_root
from .retained_weight_close_reference import checked_files
from .source_snapshot import copy_snapshot

STOPPING_FAILURE_FILES = EARLY_STOPPING_INPUT_FILES | {'manifest.json', 'baseline_files.json', 'reference_parity.json',
    'early_teacher_models.json', 'early_teacher_support.json', 'early_teacher_training_membership.parquet',
    'early_stopping_targets.parquet', 'stopping_training_cost_ledger.parquet', 'model.json', 'training_support.json',
    'calibration_decision.json', 'decision.json', 'summary.json', 'REPORT.md'}
STOPPING_CALIBRATION_FILES = {'predictions.parquet', 'metrics.json', 'first_metrics.json', 'probability_metrics.json',
    'breakdown.json', 'first_breakdown.json', 'probability_breakdown.json',
    *[f'positions_{name}.parquet' for name in EARLY_STOPPING_NEW_POLICIES]}


def checked_threshold_reproduction(owner, parent):
    proof = json.loads((owner/'reference_parity.json').read_text())
    child = Path(proof['reproduction'])
    if (proof['complete'] is not True or proof['reference'] != str(parent)
        or proof['reference_files_sha256'] != sha256(parent/'files.json')
        or proof['reproduction_files_sha256'] != sha256(child/'files.json')
        or proof['all_previous_outputs_exact'] is not True
        or proof['original_decision'] != json.loads((parent/'decision.json').read_text())
        or set(checked_files(child)) != THRESHOLD_FAILURE_FILES):
        raise ValueError('이후 청산 회귀의 기존 문턱 실패 재현 연결 오류')
    same_outputs(parent, child, THRESHOLD_FAILURE_FILES-{'manifest.json', 'reference_parity.json'})
    calibration = checked_calibration(parent)
    if (calibration != checked_calibration(child) or calibration != checked_files(owner/'baseline_calibration')
        or sha256(owner/'baseline_calibration/files.json') != proof['calibration_files_sha256']
        or sha256(parent/'calibration/files.json') != proof['calibration_files_sha256']):
        raise ValueError('이후 청산 회귀의 이전 문턱 보정 사본 오류')
    retained = Path(json.loads((parent/'manifest.json').read_text())['settings']['reference'])
    expected = {'source': checked_retained_reproduction(parent, retained), 'reproduced': checked_retained_reproduction(child, retained)}
    if proof['previous_reproductions'] != expected:
        raise ValueError('이후 청산 회귀의 중첩 원래 비중 재현 오류')
    return {'path': str(child), 'files_sha256': sha256(child/'files.json'), 'retained_reproductions': expected}


def checked_stopping_calibration(reference):
    files = checked_files(reference/'calibration')
    if (set(files) != STOPPING_CALIBRATION_FILES
        or sha256(reference/'calibration/files.json') != json.loads((reference/'summary.json').read_text())['calibration_files_sha256']):
        raise ValueError('이후 청산 회귀의 기존 다섯 보정 정책·봉인 오류')
    return files


def reproduce_stopping_reference(reference, output):
    files = checked_files(reference)
    if set(files) != STOPPING_FAILURE_FILES:
        raise ValueError('이후 청산 회귀의 기존 내부 실패 파일 목록 오류')
    before = sha256(reference/'files.json')
    manifest = json.loads((reference/'manifest.json').read_text())
    parent = Path(manifest['settings']['reference'])
    expected = {'reference': str(parent), 'reference_files_sha256': sha256(parent/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V79.md')), 'periods': THRESHOLD_SPLITS,
        'features': EarlyStoppingCloseModel.features, 'settings': HISTOGRAM_SETTINGS,
        'early_stopping_settings': EARLY_STOPPING_SETTINGS, 'teacher_models': 5, 'candidate_models': 1,
        'candidate': 'early_stopping', 'policies': EARLY_STOPPING_POLICIES, 'comparisons': EARLY_STOPPING_COMPARISONS,
        'score_kind': 'cost_weighted_current_vs_future_policy_close', 'score_threshold': .5,
        'evaluation_target': 'original_natural_close_advantage', 'evaluation_weights': 'original_full_minute_position_weights',
        'original_target_probability_metrics_are_descriptive_only': True,
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True}
    summary, decision = (json.loads((reference/name).read_text()) for name in ['summary.json', 'calibration_decision.json'])
    if (manifest['settings'] != expected or type(manifest['settings']['candidate_models']) is not int
        or summary['complete'] is not True or summary['profitability_accepted'] is not False
        or any(summary[key] is not True for key in ['all_previous_outputs_reproduced', 'original_rows_weights_and_targets_preserved', 'whole_positions_disjoint'])
        or summary['teacher_models'] != 5 or type(summary['candidate_models']) is not int or summary['candidate_models'] != 1
        or summary['policy_iterations'] != 1 or summary['threshold_search'] is not False
        or summary['refit_after_calibration'] is not False or summary['diagnosis_evaluated'] is not False
        or decision['calibration_passed'] is not False or decision['fallback_used'] is not False
        or decision['threshold'] != .5 or decision['threshold_search'] is not False
        or (reference/'predictions.parquet').exists() or (reference/'intervals.json').exists()):
        raise ValueError('이후 청산 회귀의 기존 계획·내부 탈락·진단 차단 오류')
    sources, snapshot = manifest['source_sha256'], reference/'code_snapshot'
    if (set(sources) != {p.name for p in snapshot.glob('*.py')}
        or any(Path(name).name != name or not name.endswith('.py') or (snapshot/name).is_symlink()
            or sha256(snapshot/name) != value for name, value in sources.items())):
        raise ValueError('이후 청산 회귀의 기존 코드 사본 오류')
    baseline = json.loads((reference/'baseline_files.json').read_text())
    if (baseline != checked_files(parent) or set(baseline) != THRESHOLD_FAILURE_FILES
        or any((reference/'baseline_source'/name).is_symlink() or sha256(reference/'baseline_source'/name) != value for name, value in baseline.items())):
        raise ValueError('이후 청산 회귀의 기존 문턱 실패 사본 오류')
    calibration = checked_stopping_calibration(reference)
    original_child = checked_threshold_reproduction(reference, parent)
    reproduced = run_early_stopping_diagnosis(parent, reproduction_root(output))
    same_outputs(reference, reproduced, STOPPING_FAILURE_FILES-{'manifest.json', 'reference_parity.json'})
    if checked_stopping_calibration(reproduced) != calibration:
        raise ValueError('이후 청산 회귀의 기존 다섯 보정 정책 재현 오류')
    same_outputs(reference/'calibration', reproduced/'calibration', STOPPING_CALIBRATION_FILES)
    reproduced_child = checked_threshold_reproduction(reproduced, parent)
    for directory, origin, mapping in [('baseline_source', reference, files), ('baseline_calibration', reference/'calibration', calibration)]:
        target = output/directory
        target.mkdir(mode=0o700)
        for name, value in mapping.items():
            if sha256(origin/name) != value:
                raise ValueError('이후 청산 회귀 중 기존 자료 변경')
            copy_snapshot(origin/name, target/name)
            if sha256(target/name) != value:
                raise ValueError('이후 청산 회귀의 사본 불일치')
        if directory == 'baseline_calibration':
            copy_snapshot(origin/'files.json', target/'files.json')
    if sha256(reference/'files.json') != before:
        raise ValueError('이후 청산 회귀 중 기준 봉인 변경')
    save_json(output/'baseline_files.json', files)
    save_json(output/'reference_parity.json', {'complete': True, 'reference': str(reference),
        'reference_files_sha256': before, 'reproduction': str(reproduced),
        'reproduction_files_sha256': sha256(reproduced/'files.json'), 'all_previous_outputs_exact': True,
        'previous_reproductions': {'source': original_child, 'reproduced': reproduced_child},
        'calibration_files_sha256': sha256(reference/'calibration/files.json'),
        'original_decision': json.loads((reference/'decision.json').read_text())})
    return reproduced
