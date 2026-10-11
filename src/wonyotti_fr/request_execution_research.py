from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .collateral_diagnostics import checked_reference
from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .recorded_change_reference import checked_files
from .request_execution import (
    link_terminations,
    order_observations,
    summarize_execution,
    window_comparison,
)
from .request_execution_reference import (
    compare_summary,
    independent_summary,
    verify_orders,
    verify_windows,
)


def checked_parent(marker, expected, frozen):
    if marker.is_symlink() or sha256(marker) != expected:
        raise ValueError('청산 대응 선행 검증 지문 오류')
    proof = read_json(marker)
    root = Path(proof['run'])
    if (proof.get('complete') is not True or proof.get('profitability_accepted') is not False
        or root.is_symlink() or sha256(root/'files.json') != proof['files_sha256']
        or sha256(root/'summary.json') != proof['summary_sha256']):
        raise ValueError('청산 대응 선행 완료·산출물 오류')
    checked_files(root)
    frozen[str(marker)] = expected
    frozen[str(root/'files.json')] = proof['files_sha256']
    frozen.update({str(root/name): digest for name, digest in read_json(root/'files.json').items()})
    return root


def run(reference, reference_sha256, requests_proof, requests_proof_sha256,
        paired_proof, paired_proof_sha256, protocol, output, progress=print):
    spec = checked_reference(reference, reference_sha256)
    frozen = {**spec['sha256'], str(reference): reference_sha256, str(protocol): sha256(protocol)}
    prior = checked_parent(requests_proof, requests_proof_sha256, frozen)
    paired_root = checked_parent(paired_proof, paired_proof_sha256, frozen)
    public = {str(p): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    frozen.update(public)
    root = new_run(output, 'request-execution-correspondence', {
        'requests': str(prior), 'paired': str(paired_root), 'protocol_sha256': frozen[str(protocol)],
        'model_fitted': False, 'thresholds_changed': False, 'profitability_accepted': False,
        'all_periods_already_observed': True})
    save_json(root/'frozen_inputs.json', frozen)
    (root/'protocol.md').write_bytes(protocol.read_bytes())
    progress(str(root))
    try:
        raw = pd.read_parquet(Path(spec['audit'])/'executions.parquet')
        windows, filled, requested = [pd.read_parquet(prior/f'{name}.parquet')
            for name in ('windows', 'filled_actions', 'requested_actions')]
        paired = pd.read_parquet(paired_root/'all_windows.parquet')
        predictions = pd.read_parquet(paired_root/'predictions.parquet')
        observations = order_observations(raw, filled, requested)
        requests, ended = link_terminations(filled, observations)
        comparison = window_comparison(windows, requested, filled, paired, predictions)
        progress('전체 주문·종료·창 대응 작성 완료')
        verified = verify_orders(raw, filled, requests, ended)
        verified.update(verify_windows(windows, requested, filled, comparison))
        progress('전체 주문·종료·창 독립 순회 검산 완료')
        result = summarize_execution(requests, ended, comparison)
        compare_summary(independent_summary(requests, ended, comparison), result)
        progress('기간·연도·월별 전체 집계와 고정 예측 대조 완료')
        for name, frame in [('orders', requests), ('terminations', ended), ('windows', comparison)]:
            frame.to_parquet(root/f'{name}.parquet', index=False)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(root/f'{name}.parquet'), check_exact=True)
        for name, digest in frozen.items():
            if Path(name).is_symlink() or sha256(Path(name)) != digest:
                raise ValueError('청산 대응 실행 중 입력 변경: '+name)
        checked_reference(reference, reference_sha256)
        summary = {'complete': True, 'results': result, 'verification': {**verified,
            'all_summaries_verified': True, 'roundtrip_exact': True, 'originals_unchanged': 5,
            'production_sources_unchanged': len(public), 'frozen_inputs_unchanged': len(frozen)},
            'model_fitted': False, 'thresholds_changed': False, 'profitability_accepted': False}
        save_json(root/'summary.json', summary)
        seal(root)
        return root
    except Exception as error:
        save_json(root/'failure.json', {'complete': False, 'type': type(error).__name__, 'error': str(error)})
        seal(root)
        raise


def main():
    parser = argparse.ArgumentParser(description='고정 예측과 원본 청산 체결의 오프라인 대응 검산')
    for name in ('reference', 'requests-proof', 'paired-proof'):
        parser.add_argument('--'+name, required=True, type=Path)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', default=Path('artifacts'), type=Path)
    args = vars(parser.parse_args())
    root = run(**args)
    print('완료:', root, flush=True)


if __name__ == '__main__':
    main()
