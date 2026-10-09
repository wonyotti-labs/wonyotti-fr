from __future__ import annotations

import json
from pathlib import Path

from .common import sha256
from .event_research import load_selection
from .expanded_first_reference import trusted_proof
from .retained_weight_close_reference import checked_files

FIRST_EXIT_PERIODS = {'2021': ['2021-01-01', '2022-01-01'], '2022': ['2022-01-01', '2023-01-01']}
FIRST_EXIT_BASELINES = {'2021': 'candidate-00', '2022': 'confirmation-2022'}
FIRST_EXIT_RUN_FILES = ['config.json', 'equity.parquet', 'fills.parquet', 'trades.parquet', 'metrics.json', 'final_state.json']
FIRST_EXIT_REVIEW_CHECKS = ['complete', 'previous_verified_79_inputs_and_seventeen_policies_reused',
    'all_first_rows_all_positions_and_equal_weights_verified', 'all13_regularized_models_time_memberships_and_selection_verified',
    'all_eighteen_policies_fifteen_gates_and_no_later_scores_verified', 'diagnosis_predictions_absent']


def checked_first_exit_reference(reference, verification, expected):
    proof = trusted_proof(verification, expected)
    if (any(proof.get(key) is not True for key in FIRST_EXIT_REVIEW_CHECKS)
        or any(proof.get(key) is not False for key in ['synthetic_only', 'calibration_passed', 'profitability_accepted'])
        or proof.get('source_files_sha256') != sha256(reference/'files.json')):
        raise ValueError('첫 청산 계좌 대조의 이전 검산·실패 근거 오류')
    files = checked_files(reference)
    required = {'manifest.json', 'summary.json', 'decision.json', 'generation_manifest.json', 'manager_parent_selection.json'}
    if not required <= set(files):
        raise ValueError('첫 청산 계좌 대조의 출처 파일 누락')
    manifest, summary, decision, generation = [json.loads((reference/name).read_text())
        for name in ['manifest.json', 'summary.json', 'decision.json', 'generation_manifest.json']]
    if (manifest['settings']['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V90.md'))
        or summary.get('complete') is not True or summary.get('profitability_accepted') is not False
        or summary.get('diagnosis_evaluated') is not False or decision.get('calibration_passed') is not False
        or decision.get('candidate') != 'regularized_first_linear'
        or any((reference/name).exists() for name in ['predictions.parquet', 'diagnosis_used.parquet', 'intervals.json'])):
        raise ValueError('첫 청산 계좌 대조의 이전 계획·진단 차단 오류')
    for name, field in [('calibration', 'calibration_files_sha256'), ('selection', 'selection_files_sha256')]:
        checked_files(reference/name)
        if sha256(reference/name/'files.json') != summary[field]:
            raise ValueError('첫 청산 계좌 대조의 이전 하위 봉인 변경')
    folds = json.loads((reference/'selection/folds.json').read_text())
    if set(folds) != {'fold-00', 'fold-01', 'fold-02'}:
        raise ValueError('첫 청산 계좌 대조의 이전 내부 회차 오류')
    for name, value in folds.items():
        checked_files(reference/'selection'/name)
        if sha256(reference/'selection'/name/'files.json') != value:
            raise ValueError('첫 청산 계좌 대조의 이전 회차 지문 오류')
    context = generation['settings']
    parent = Path(context['reference'])
    if (sha256(parent/'frozen_selection.json') != context['reference_sha256']
        or (reference/'manager_parent_selection.json').read_bytes() != (parent/'frozen_selection.json').read_bytes()):
        raise ValueError('첫 청산 계좌 대조의 고정 부모 연결 오류')
    frozen, _ = load_selection(parent)
    if frozen['protocol'] != 'exit_move_v54' or frozen['risk']['bar_seconds'] != 60 or frozen['risk']['signal_delay_bars'] != 0:
        raise ValueError('첫 청산 계좌 대조의 부모 정책·실행 간격 오류')
    parent_settings = json.loads((parent/'manifest.json').read_text())['settings']
    manifests = parent_settings['input_manifests']
    for name, value in manifests.items():
        if Path(name).is_symlink() or sha256(Path(name)) != value:
            raise ValueError('첫 청산 계좌 대조의 원래 시세 지문 변경')
    first = {'market': str(Path(context['market'])), 'features': str(Path(context['features']))}
    paths = {str(Path(first['market'])/'manifest-1m.json'), str(Path(first['features'])/'manifest-5m.json')}
    if not paths <= set(manifests):
        raise ValueError('첫 청산 계좌 대조의 개발 시세 연결 오류')
    remaining = [Path(name) for name in manifests if name not in paths]
    if len(remaining) != 2 or {p.name for p in remaining} != {'manifest-1m.json', 'manifest-5m.json'}:
        raise ValueError('첫 청산 계좌 대조의 확인 시세 연결 오류')
    second = {('market' if p.name == 'manifest-1m.json' else 'features'): str(p.parent) for p in remaining}
    parent_files = {p.name: sha256(p) for p in parent.iterdir() if p.is_file()}
    if parent.is_symlink() or any((parent/name).is_symlink() for name in parent_files):
        raise ValueError('첫 청산 계좌 대조의 부모 파일 링크 오류')
    baseline = {year: {name: sha256(parent/folder/name) for name in FIRST_EXIT_RUN_FILES}
        for year, folder in FIRST_EXIT_BASELINES.items()}
    for name, value in context['reference_outputs_sha256'].items():
        if baseline['2021'][name] != value:
            raise ValueError('첫 청산 계좌 대조의 원래 적합 경로 변경')
    return {'reference': str(reference), 'reference_files_sha256': sha256(reference/'files.json'),
        'verification': str(verification), 'verification_sha256': expected, 'parent': str(parent),
        'parent_files_sha256': parent_files, 'baseline_sha256': baseline, 'input_manifests': manifests,
        'inputs': {'2021': first, '2022': second}, 'risk': frozen['risk']}
