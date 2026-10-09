from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .close_threshold import THRESHOLD_SPLITS
from .common import new_run, save_json, sha256
from .early_stopping_diagnostics import save_policy_results
from .first_linear_close import FIRST_LINEAR_SETTINGS
from .first_opportunity_diagnostics import checked_first_inputs
from .minute_first_evaluation import MINUTE_FIRST_POLICIES, evaluate_minute_first
from .minute_first_fit import fit_minute_first
from .minute_first_inputs import (
    MinuteFirstLinearModel,
    checked_minute_first_parent,
    load_first_minute_source,
)
from .minute_first_reference import (
    MINUTE_FIRST_INPUT_FILES,
    checked_minute_first_reference,
    load_minute_first_baseline,
)
from .source_snapshot import copy_snapshot


def run_minute_first_diagnosis(reference: Path, verification: Path, verification_sha256: str, output: Path) -> Path:
    proof = checked_minute_first_reference(reference, verification, verification_sha256)
    manager, parent = checked_minute_first_parent(reference)
    bars, minute_source = load_first_minute_source(reference)
    settings = {'reference': str(reference), 'reference_files_sha256': proof['reference_files_sha256'],
        'verification': str(verification), 'verification_sha256': verification_sha256,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V89.md')), 'features': MinuteFirstLinearModel.features,
        'model_settings': FIRST_LINEAR_SETTINGS, 'periods': {name: THRESHOLD_SPLITS[name] for name in ['early_training', 'calibration']},
        'added_period': ['2020-04-01', '2021-01-01'], 'added_label_cutoff': '2020-12-31',
        'policies': MINUTE_FIRST_POLICIES, 'threshold': .5, 'threshold_search': False, 'new_models': 1,
        'score_rows': 'first_eligible_only', 'target': 'unchanged_v85_expanded_first_natural_close_common_bps',
        'diagnosis_evaluated': False, 'whole_policy_historically_available_claimed': False,
        'whole_system_periods_already_observed': True, 'previous_verified_evidence_reused': True, 'parent': parent['parent'], 'parent_sha256': parent['parent_sha256'], 'minute_source': minute_source}
    out = new_run(output, 'minute-first-linear-diagnosis', settings)
    print(f'확정 분봉 체결 방향의 첫 판단 비교: {out}', flush=True)
    try:
        save_json(out/'reference_evidence.json', proof)
        save_json(out/'manager_evidence.json', parent)
        save_json(out/'minute_source.json', minute_source)
        copy_snapshot(Path(minute_source['market'])/'manifest-1m.json', out/'minute_market_manifest.json')
        for name, item in parent['metadata'].items():
            copy_snapshot(Path(item['path']), out/name)
        copy_snapshot(Path(parent['parent'])/'frozen_selection.json', out/'manager_parent_selection.json')
        for destination, origin in [('previous_verification.json', verification), ('previous_files.json', reference/'files.json'),
            ('previous_calibration_decision.json', reference/'decision.json'), ('previous_model.json', reference/'model.json'),
            ('previous_training_support.json', reference/'training_support.json')]:
            copy_snapshot(origin, out/destination)
        for name in MINUTE_FIRST_INPUT_FILES:
            copy_snapshot(reference/name, out/name)
        baseline_folder = out/'baseline_calibration'
        baseline_folder.mkdir(mode=0o700)
        for name in [*proof['calibration_files'], 'files.json']:
            copy_snapshot(reference/'calibration'/name, baseline_folder/name)
        model, support = fit_minute_first(reference, manager, bars, out)
        parts, weights = checked_first_inputs(out)
        calibration = parts['calibration']
        first = pd.read_parquet(out/'minute_first_calibration.parquet')
        scores = model.probabilities(first[model.features].to_numpy())[:, 0]
        result = evaluate_minute_first(calibration.assign(sample_weight=weights['calibration'].sample_weight),
            scores, load_minute_first_baseline(reference))
        folder = out/'calibration'
        folder.mkdir(mode=0o700)
        save_policy_results(folder, result)
        result['first_membership'].to_parquet(folder/'first_membership.parquet', index=False)
        save_json(folder/'first_probability_metrics.json', result['first_probability_metrics'])
        save_json(folder/'files.json', {p.name: sha256(p) for p in folder.iterdir() if p.is_file()})
        decision = {**result['calibration_decision'], 'candidate': 'minute_first_linear', 'diagnosis_evaluated': False,
            'trading_returns_evaluated': False, 'profitability_accepted': False,
            'next_stage_required': 'fixed_last_diagnosis_protocol' if result['calibration_decision']['calibration_passed'] else 'new_hypothesis'}
        save_json(out/'decision.json', decision)
        if load_first_minute_source(reference)[1] != minute_source or sha256(out/'minute_market_manifest.json') != minute_source['market_manifest_sha256']:
            raise ValueError('확정 분봉 첫 선형 실행 중 시세 근거 변경')
        if checked_minute_first_parent(reference)[1] != parent:
            raise ValueError('확정 분봉 첫 선형 실행 중 부모·메타데이터 변경')
        if any(sha256(out/name) != item['sha256'] for name, item in parent['metadata'].items()):
            raise ValueError('확정 분봉 첫 선형 실행 중 부모 연결 사본 변경')
        if sha256(out/'manager_parent_selection.json') != parent['parent_sha256']:
            raise ValueError('확정 분봉 첫 선형 실행 중 고정 부모 사본 변경')
        if checked_minute_first_reference(reference, verification, verification_sha256) != proof:
            raise ValueError('확정 분봉 첫 선형 실행 중 이전 검산 근거 변경')
        if any(sha256(out/name) != proof['files'][name] for name in MINUTE_FIRST_INPUT_FILES):
            raise ValueError('확정 분봉 첫 선형 실행 중 기존 원장 사본 변경')
        if any(sha256(baseline_folder/name) != value for name, value in proof['calibration_files'].items()):
            raise ValueError('확정 분봉 첫 선형 실행 중 기존 열여섯 정책 사본 변경')
        sources = json.loads((out/'manifest.json').read_text())['source_sha256']
        if (any(sha256(Path(__file__).parent/name) != value for name, value in sources.items())
            or sha256(Path('docs/EXPERIMENT_V89.md')) != settings['protocol_sha256']):
            raise ValueError('확정 분봉 첫 선형 실행 중 코드·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'new_models_fitted': 1,
            'combined_first_positions': support['combined_first_positions'], 'phase_first_positions': support['phase_first_positions'],
            'calibration_files_sha256': sha256(folder/'files.json'), 'previous_models_refitted': False,
            'generation_engine_rerun': False, 'external_nested_runs_reverified': False, 'refit_after_calibration': False, **decision})
        (out/'REPORT.md').write_text('# 확정 분봉 체결 방향의 첫 판단 비교\n\n'
            f'내부 보정 통과: {decision["calibration_passed"]}. '
            '같은 첫 원장과 77개 입력을 보존하고 직전 확정 분봉의 체결 불균형 두 값으로 같은 선형 모델을 비교했다. '
            '기존 열여섯 정책의 저장된 점수로 산식을 다시 대조하고 실패와 전체 원장을 보존했다. '
            '과거 생성 정책의 당시 가용 성과나 연속 매매 수익성으로 해석하지 않는다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
