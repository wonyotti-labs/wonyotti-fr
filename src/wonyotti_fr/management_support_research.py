from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .calibration_diagnostics import PERIODS
from .common import new_run, save_json, sha256
from .event_research import load_selection
from .first_exit_verification import verify_execution
from .holding_stress import seal
from .management_support import (
    FEATURES,
    GROUPS,
    QUANTILES,
    ManagementInputTrace,
    SupportCounts,
    training_profile,
)
from .minute_data import prepare_minute_period
from .streaming_backtest import ParquetRows

CASES = {'btc_2022': ('BTCUSDT', ['2022-01-01', '2023-01-01']),
         'eth_2023_2025': ('ETHUSDT', ['2023-01-01', '2026-01-01'])}


def reference_counts(frame, profile):
    values = frame[FEATURES]
    valid = np.isfinite(values).all(axis=1)
    good = values.loc[valid]
    bounds = np.asarray(profile['bounds'])
    outside, central = {}, {}
    feature_counts = {}
    for i, name in enumerate(FEATURES):
        outside[name] = good[name].lt(bounds[0, i]) | good[name].gt(bounds[8, i])
        central[name] = good[name].lt(bounds[1, i]) | good[name].gt(bounds[7, i])
        feature_counts[name] = {'nonfinite': int((~np.isfinite(values[name])).sum()),
            'outside_training_range': int(outside[name].sum()),
            'outside_training_central98': int(central[name].sum())}
    # 별도 열 연산으로 조합과 범위를 계산해 묶음 누적 결과와 대조한다.
    codes = (good.direction.eq(1).astype(int)*48 + good.adds_capped.astype(int)*8
             + good.remaining_fraction.eq(1).astype(int)*4
             + good.past_increase_exists.astype(int)*2 + good.past_reduce_exists.astype(int))
    counts = codes.value_counts().reindex(range(96), fill_value=0)
    if int(counts.sum()) != len(good):
        raise ValueError('관리 입력 공동 상태 검산의 범위 오류')
    groups = {'all': (0, 50), **GROUPS}
    return {'rows': len(frame), 'finite_rows': len(good), 'nonfinite_rows': int((~valid).sum()),
        'feature_counts': feature_counts,
        'group_outside_training_range': {k: int(pd.DataFrame(outside)[FEATURES[a:b]].any(axis=1).sum())
                                        for k, (a, b) in groups.items()},
        'group_outside_training_central98': {k: int(pd.DataFrame(central)[FEATURES[a:b]].any(axis=1).sum())
                                            for k, (a, b) in groups.items()},
        'unseen_joint_rows': int(counts[np.asarray(profile['joint_counts']) == 0].sum()),
        'joint_counts': counts.tolist(), 'range_denominator': 'finite_rows',
        'independent_trade_samples': False}


def verify_profile(frame, profile):
    expected = []
    ordered = np.sort(frame[FEATURES].to_numpy(float), axis=0)
    for fraction in QUANTILES:
        index = fraction*(len(ordered)-1)
        low, high = int(np.floor(index)), int(np.ceil(index))
        expected.append(ordered[low] + (ordered[high]-ordered[low])*(index-low))
    if not np.allclose(expected, profile['bounds'], rtol=0, atol=1e-12):
        raise ValueError('학습 분위수의 정렬·보간 검산 불일치')
    counts = reference_counts(frame, profile)
    if counts['joint_counts'] != profile['joint_counts'] or counts['rows'] != profile['rows']:
        raise ValueError('학습 공동 상태의 별도 집계 불일치')


def checked_reference(path, expected):
    if path.is_symlink() or path.stat().st_size > 2**20 or sha256(path) != expected:
        raise ValueError('관리 입력 진단의 출처 지문 오류')
    spec = json.loads(path.read_text())
    if (spec.get('profitability_accepted') is not False or spec.get('all_periods_already_observed') is not True
        or set(spec['cases']) != set(CASES)):
        raise ValueError('관리 입력 진단의 고정 비교 오류')
    required = set()
    for phase in ['training', 'calibration', 'diagnosis']:
        required.add(str(Path(spec['teacher'])/f'{phase}_used.parquet'))
    teacher = Path(spec['teacher'])
    required.update(str(teacher/file) for file in ['files.json', 'manifest.json', 'models.json', 'offsets.json'])
    required.add(str(Path(spec['selection'])/'frozen_selection.json'))
    for name, (symbol, bounds) in CASES.items():
        case = spec['cases'][name]
        if case['symbol'] != symbol or case['bounds'] != bounds:
            raise ValueError('관리 입력 진단의 기간·시장 변경')
        required.update(str(Path(case['baseline'])/file) for file in
            ['equity.parquet', 'trades.parquet', 'fills.parquet', 'metrics.json', 'config.json', 'final_state.json'])
        required.update(str(Path(case[key])/file) for key, file in
                        [('market', 'manifest-1m.json'), ('features', 'manifest-5m.json')])
    if not required <= set(spec['sealed_files']):
        raise ValueError('관리 입력 진단의 필수 출처 지문 누락')
    for name, digest in spec['sealed_files'].items():
        item = Path(name)
        if item.is_symlink() or not item.is_file() or sha256(item) != digest:
            raise ValueError('관리 입력 진단의 입력 변경: '+name)
    frozen, _ = load_selection(Path(spec['selection']))
    if frozen['protocol'] != 'exit_state_v48':
        raise ValueError('관리 입력 진단의 고정 정책 오류')
    teacher_files = json.loads((teacher/'files.json').read_text())
    teacher_settings = json.loads((teacher/'manifest.json').read_text())['settings']
    selection = Path(spec['selection'])
    admission = json.loads((selection/'history_admission.json').read_text())
    if (teacher_settings['periods'] != {k: list(v) for k, v in PERIODS.items()}
        or admission['files_sha256'] != sha256(teacher/'files.json')
        or json.loads((selection/'history_manager.json').read_text()) != json.loads((teacher/'models.json').read_text())['histogram']
        or json.loads((selection/'history_offset.json').read_text()) != json.loads((teacher/'offsets.json').read_text())['histogram']
        or any(teacher_files[f'{phase}_used.parquet'] != spec['sealed_files'][str(teacher/f'{phase}_used.parquet')]
               for phase in PERIODS)):
        raise ValueError('관리 입력 진단의 원래 학습 행·모델 연결 오류')
    for case in spec['cases'].values():
        if json.loads((Path(case['baseline'])/'config.json').read_text()) != frozen['risk']:
            raise ValueError('관리 입력 진단의 기존 위험 설정 불일치')
    return spec


def summarize_parquet(path, profile):
    results = {'all': SupportCounts(profile)}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        frame = batch.to_pandas()
        results['all'].update(frame[FEATURES])
        if 'intent' in frame:
            for intent, part in frame.groupby('intent'):
                results.setdefault(str(intent), SupportCounts(profile)).update(part[FEATURES])
    # 저장 파일을 다시 읽고 같은 누적 함수를 쓰지 않는 계산과 비교한다.
    frame = pd.read_parquet(path)
    for key, value in results.items():
        part = frame if key == 'all' else frame.loc[frame.intent == key]
        if value.result() != reference_counts(part, profile):
            raise ValueError('관리 입력 범위의 저장 후 별도 집계 불일치: '+key)
    return {key: value.result() for key, value in results.items()}


def run_management_support(reference, expected, output):
    spec = checked_reference(reference, expected)
    protocol = Path('docs/EXPERIMENT_V113.md')
    out = new_run(output, 'management-input-support', {'reference': str(reference),
        'reference_sha256': expected, 'spec': spec, 'protocol_sha256': sha256(protocol),
        'new_models_fitted': False, 'policy_changed': False, 'profitability_accepted': False})
    print(out, flush=True)
    try:
        teacher = Path(spec['teacher'])
        train = pd.read_parquet(teacher/'training_used.parquet', columns=FEATURES)
        profile = training_profile(train)
        verify_profile(train, profile)
        save_json(out/'training_profile.json', profile)
        del train
        summaries = {}
        for phase in ['training', 'calibration', 'diagnosis']:
            summaries[phase] = summarize_parquet(teacher/f'{phase}_used.parquet', profile)
            save_json(out/'summaries.json', summaries)
            print(phase+': 원본 입력 범위 검산 완료', flush=True)
        for name, case in spec['cases'].items():
            bars, checks = prepare_minute_period(Path(case['market']), Path(case['features']),
                case['symbol'], *case['bounds'], minute_inputs=True)
            folder = out/name
            folder.mkdir(mode=0o700)
            save_json(folder/'input_verification.json', checks)
            _, parent = load_selection(Path(spec['selection']))
            writer = ParquetRows(folder/'management_inputs.parquet', 4096)
            trace = ManagementInputTrace(parent, writer.append)
            print(name+': 기존 실행과 입력 관찰 재생 시작', flush=True)
            try:
                verification = verify_execution(Path(case['baseline']), bars, trace)
            finally:
                writer.close()
            save_json(folder/'execution_verification.json', verification)
            save_json(folder/'coverage.json', trace.coverage())
            del bars, trace, parent
            summaries[name] = summarize_parquet(folder/'management_inputs.parquet', profile)
            save_json(out/'summaries.json', summaries)
            seal(folder)
            print(name+': 계좌 불변·입력 수식·저장 후 집계 검산 완료', flush=True)
        if checked_reference(reference, expected) != spec:
            raise ValueError('관리 입력 진단의 실행 중 출처 변경')
        manifest = json.loads((out/'manifest.json').read_text())
        if (sha256(protocol) != manifest['settings']['protocol_sha256']
            or any(sha256(Path(__file__).parent/name) != value for name, value in manifest['source_sha256'].items())):
            raise ValueError('관리 입력 진단의 실행 중 코드·계획 변경')
        save_json(out/'summary.json', {'complete': True, 'original_cohorts': 3, 'replayed_cases': 2,
            'all_summaries_independently_verified': True, 'all_inputs_formula_verified': True,
            'all_existing_account_outputs_exact': True, 'new_models_fitted': False,
            'policy_changed': False, 'profitability_accepted': False})
        save_json(out/'runs.json', {name: sha256(out/name/'files.json') for name in CASES})
        seal(out)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description='고정 정책의 원본·자체 관리 입력 범위 대조')
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    run_management_support(args.reference, args.sha256, args.output)


if __name__ == '__main__':
    main()
