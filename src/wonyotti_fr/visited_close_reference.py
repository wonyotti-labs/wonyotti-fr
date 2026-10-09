from __future__ import annotations

import json
from pathlib import Path

from .close_flow import FLOW_WINDOWS
from .close_learning_inputs import CLOSE_SPLITS
from .common import save_json, sha256
from .continuation_diagnostics import same_outputs
from .histogram_management import HISTOGRAM_SETTINGS
from .minute_close_diagnostics import run_minute_close_diagnosis
from .minute_close_learning import MINUTE_COMPARISONS, MINUTE_POLICIES, MinuteCloseModel
from .research_paths import reproduction_root
from .source_snapshot import copy_snapshot

MINUTE_DIAGNOSIS_FILES = {'manifest.json', 'baseline_files.json', 'baseline_model.json', 'baseline_verification.json',
    'minute_ledger.parquet', 'feature_linkage.parquet', 'input_verification.json', 'exclusion_ledger.parquet',
    'training_used.parquet', 'training_weights.parquet', 'diagnosis_used.parquet', 'diagnosis_weights.parquet',
    'model.json', 'training_support.json', 'training_cost_ledger.parquet', 'predictions.parquet', 'block_draws.parquet',
    'metrics.json', 'first_metrics.json', 'probability_metrics.json', 'breakdown.json', 'first_breakdown.json',
    'probability_breakdown.json', 'intervals.json', 'decision.json', 'summary.json', 'REPORT.md',
    *[f'positions_{name}.parquet' for name in MINUTE_POLICIES],
    *[f'{name}_{suffix}.parquet' for name in MINUTE_COMPARISONS for suffix in ['blocks', 'block_replicates']]}


def reproduce_minute_reference(reference, output):
    files = json.loads((reference/'files.json').read_text())
    if (reference.is_symlink() or set(files) != MINUTE_DIAGNOSIS_FILES or (reference/'files.json').is_symlink()
        or any((reference/name).is_symlink() or sha256(reference/name) != value for name, value in files.items())):
        raise ValueError('방문 청산 기준의 분별 원본 파일·지문 오류')
    before = sha256(reference/'files.json')
    manifest = json.loads((reference/'manifest.json').read_text())
    settings = manifest['settings']
    labels, parent = Path(settings['labels']), Path(settings['reference'])
    summary = json.loads((reference/'summary.json').read_text())
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V75.md'))
        or settings['labels_files_sha256'] != sha256(labels/'files.json')
        or settings['reference_files_sha256'] != sha256(parent/'files.json')
        or settings['periods'] != CLOSE_SPLITS or settings['features'] != MinuteCloseModel.features
        or settings['settings'] != HISTOGRAM_SETTINGS or settings['flow_windows'] != FLOW_WINDOWS
        or type(settings['new_model_count']) is not int or settings['new_model_count'] != 1
        or settings['score_threshold'] != .5 or settings['candidate'] != 'minute_1m'
        or settings['policies'] != MINUTE_POLICIES or settings['comparisons'] != MINUTE_COMPARISONS
        or settings['score_kind'] != 'cost_weighted_decision_score'
        or settings['cost_weight'] != 'original_position_weight_times_absolute_effect'
        or settings['zero_effect_policy'] != 'zero_fit_contribution_preserved_in_evaluation'
        or settings['trading_returns_evaluated'] is not False or settings['whole_system_periods_already_observed'] is not True
        or summary['complete'] is not True or summary['all_minute_inputs_and_legacy_reference_verified'] is not True
        or summary['same_position_population_and_first_equity'] is not True or summary['zero_effect_rows_preserved'] is not True
        or type(summary['new_models_fitted']) is not int or summary['new_models_fitted'] != 1
        or summary['profitability_accepted'] is not False):
        raise ValueError('방문 청산 기준의 분별 설정·계획·완료 오류')
    sources = manifest['source_sha256']
    snapshot = reference/'code_snapshot'
    if (set(sources) != {p.name for p in snapshot.glob('*.py')}
        or any(Path(name).name != name or not name.endswith('.py') or (snapshot/name).is_symlink()
            or sha256(snapshot/name) != value for name, value in sources.items())):
        raise ValueError('방문 청산 기준의 분별 구현 사본 오류')
    baseline_files = json.loads((reference/'baseline_files.json').read_text())
    if (baseline_files != json.loads((parent/'files.json').read_text())
        or any(Path(name).name != name or (reference/'baseline_source'/name).is_symlink()
            or sha256(reference/'baseline_source'/name) != value for name, value in baseline_files.items())):
        raise ValueError('방문 청산 기준의 이전 모델 사본 오류')
    reproduced = run_minute_close_diagnosis(labels, parent, reproduction_root(output))
    same_outputs(reference, reproduced, MINUTE_DIAGNOSIS_FILES-{'manifest.json'})
    target = output/'baseline_source'
    target.mkdir(mode=0o700)
    for name, value in files.items():
        if sha256(reference/name) != value:
            raise ValueError('방문 청산 기준 재현 중 원본 변경')
        copy_snapshot(reference/name, target/name)
        if sha256(target/name) != value:
            raise ValueError('방문 청산 기준 사본 지문 불일치')
    if sha256(reference/'files.json') != before:
        raise ValueError('방문 청산 기준 재현 중 원본 봉인 변경')
    save_json(output/'baseline_files.json', files)
    proof = {'complete': True, 'reference': str(reference), 'reference_files_sha256': before,
        'reproduction': str(reproduced), 'reproduction_files_sha256': sha256(reproduced/'files.json'),
        'all_minute_inputs_models_costs_scores_and_evaluations_exact': True,
        'original_decision': json.loads((reference/'decision.json').read_text())}
    save_json(output/'reference_parity.json', proof)
    return reproduced
