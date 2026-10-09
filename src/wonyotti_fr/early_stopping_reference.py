from __future__ import annotations

import json
from pathlib import Path

from .close_threshold import THRESHOLD_SETTINGS, THRESHOLD_SPLITS, ThresholdCloseModel
from .close_threshold_diagnostics import THRESHOLD_INPUT_FILES, run_close_threshold_diagnosis
from .close_threshold_evaluation import THRESHOLD_COMPARISONS, THRESHOLD_POLICIES
from .close_threshold_reference import RETAINED_DIAGNOSIS_FILES, checked_visited_reproduction
from .common import save_json, sha256
from .continuation_diagnostics import same_outputs
from .histogram_management import HISTOGRAM_SETTINGS
from .research_paths import reproduction_root
from .retained_weight_close_reference import checked_files
from .source_snapshot import copy_snapshot

THRESHOLD_FAILURE_FILES = THRESHOLD_INPUT_FILES | {'manifest.json', 'baseline_files.json', 'reference_parity.json',
    'threshold_exclusion_ledger.parquet', 'early_training_used.parquet', 'early_training_weights.parquet',
    'early_training_cost_ledger.parquet', 'calibration_used.parquet', 'calibration_weights.parquet',
    'model.json', 'training_support.json', 'selection.json', 'decision.json', 'summary.json', 'REPORT.md'}
THRESHOLD_CALIBRATION_FILES = {'predictions.parquet', 'metrics.json', 'first_metrics.json',
    'positions_always_first.parquet', 'positions_never_extra.parquet',
    *[f'positions_threshold_{q:.1f}.parquet' for q in THRESHOLD_SETTINGS['thresholds']]}


def checked_retained_reproduction(owner, parent):
    proof = json.loads((owner/'reference_parity.json').read_text())
    child = Path(proof['reproduction'])
    if (proof['complete'] is not True or proof['reference'] != str(parent)
        or proof['reference_files_sha256'] != sha256(parent/'files.json')
        or proof['reproduction_files_sha256'] != sha256(child/'files.json')
        or proof['all_previous_outputs_exact'] is not True
        or proof['original_decision'] != json.loads((parent/'decision.json').read_text())
        or set(checked_files(child)) != RETAINED_DIAGNOSIS_FILES):
        raise ValueError('분별 이후 청산의 기존 원래 비중 재현 연결 오류')
    same_outputs(parent, child, RETAINED_DIAGNOSIS_FILES-{'manifest.json', 'reference_parity.json'})
    visited = Path(json.loads((parent/'manifest.json').read_text())['settings']['reference'])
    expected = {'source': checked_visited_reproduction(parent, visited), 'reproduced': checked_visited_reproduction(child, visited)}
    if proof['previous_reproductions'] != expected:
        raise ValueError('분별 이후 청산의 중첩 방문 구간 재현 연결 오류')
    return {'path': str(child), 'files_sha256': sha256(child/'files.json'), 'visited_reproductions': expected}


def checked_calibration(reference):
    files = checked_files(reference/'calibration')
    if (set(files) != THRESHOLD_CALIBRATION_FILES
        or sha256(reference/'calibration/files.json') != json.loads((reference/'summary.json').read_text())['calibration_files_sha256']):
        raise ValueError('분별 이후 청산의 기존 보정 파일·봉인 오류')
    return files


def reproduce_threshold_reference(reference, output):
    files = checked_files(reference)
    if set(files) != THRESHOLD_FAILURE_FILES:
        raise ValueError('분별 이후 청산의 기존 내부 선택 실패 파일 목록 오류')
    before = sha256(reference/'files.json')
    manifest = json.loads((reference/'manifest.json').read_text())
    settings = manifest['settings']
    parent = Path(settings['reference'])
    expected = {'reference': str(parent), 'reference_files_sha256': sha256(parent/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V78.md')), 'periods': THRESHOLD_SPLITS,
        'features': ThresholdCloseModel.features, 'settings': HISTOGRAM_SETTINGS,
        'threshold_settings': THRESHOLD_SETTINGS, 'new_model_count': 1,
        'existing_models_reproduced': True, 'candidate': 'calibrated', 'policies': THRESHOLD_POLICIES,
        'comparisons': THRESHOLD_COMPARISONS, 'score_kind': 'cost_weighted_decision_score',
        'evaluation_weights': 'original_full_minute_position_weights',
        'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True}
    summary = json.loads((reference/'summary.json').read_text())
    selection = json.loads((reference/'selection.json').read_text())
    if (settings != expected or type(settings['new_model_count']) is not int
        or settings['existing_models_reproduced'] is not True or settings['trading_returns_evaluated'] is not False
        or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['profitability_accepted'] is not False
        or any(summary[key] is not True for key in ['all_previous_outputs_reproduced',
            'original_full_training_and_diagnosis_preserved', 'whole_positions_disjoint'])
        or type(summary['new_models_fitted']) is not int or summary['new_models_fitted'] != 1
        or summary['refit_after_selection'] is not False or summary['diagnosis_evaluated'] is not False
        or summary['selected_threshold'] is not None or selection['selection_passed'] is not False
        or selection['diagnosis_allowed'] is not False or selection['fallback_used'] is not False
        or (reference/'predictions.parquet').exists() or (reference/'intervals.json').exists()):
        raise ValueError('분별 이후 청산의 기준 계획·내부 실패·진단 차단 오류')
    sources, snapshot = manifest['source_sha256'], reference/'code_snapshot'
    if (set(sources) != {p.name for p in snapshot.glob('*.py')}
        or any(Path(name).name != name or not name.endswith('.py') or (snapshot/name).is_symlink()
            or sha256(snapshot/name) != value for name, value in sources.items())):
        raise ValueError('분별 이후 청산의 기준 코드 사본 오류')
    baseline = json.loads((reference/'baseline_files.json').read_text())
    if (baseline != checked_files(parent) or set(baseline) != RETAINED_DIAGNOSIS_FILES
        or any((reference/'baseline_source'/name).is_symlink() or sha256(reference/'baseline_source'/name) != value
            for name, value in baseline.items())):
        raise ValueError('분별 이후 청산의 기존 아홉 정책 사본 오류')
    calibration = checked_calibration(reference)
    original_child = checked_retained_reproduction(reference, parent)
    reproduced = run_close_threshold_diagnosis(parent, reproduction_root(output))
    same_outputs(reference, reproduced, THRESHOLD_FAILURE_FILES-{'manifest.json', 'reference_parity.json'})
    if checked_calibration(reproduced) != calibration:
        raise ValueError('분별 이후 청산의 이전 일곱 보정 결과 재현 오류')
    same_outputs(reference/'calibration', reproduced/'calibration', THRESHOLD_CALIBRATION_FILES)
    reproduced_child = checked_retained_reproduction(reproduced, parent)
    for directory, origin, mapping in [('baseline_source', reference, files), ('baseline_calibration', reference/'calibration', calibration)]:
        target = output/directory
        target.mkdir(mode=0o700)
        for name, value in mapping.items():
            if sha256(origin/name) != value:
                raise ValueError('분별 이후 청산 중 이전 자료 변경')
            copy_snapshot(origin/name, target/name)
            if sha256(target/name) != value:
                raise ValueError('분별 이후 청산의 사본 불일치')
        if directory == 'baseline_calibration':
            copy_snapshot(origin/'files.json', target/'files.json')
    if sha256(reference/'files.json') != before:
        raise ValueError('분별 이후 청산 중 기준 봉인 변경')
    save_json(output/'baseline_files.json', files)
    save_json(output/'reference_parity.json', {'complete': True, 'reference': str(reference),
        'reference_files_sha256': before, 'reproduction': str(reproduced),
        'reproduction_files_sha256': sha256(reproduced/'files.json'), 'all_previous_outputs_exact': True,
        'previous_reproductions': {'source': original_child, 'reproduced': reproduced_child},
        'calibration_files_sha256': sha256(reference/'calibration/files.json'),
        'original_decision': json.loads((reference/'decision.json').read_text())})
    return reproduced
