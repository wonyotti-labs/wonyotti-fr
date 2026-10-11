from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .collateral_diagnostics import checked_reference
from .common import new_run, save_json, sha256
from .holding_stress import seal
from .management_account_reference import verify_account_state, verify_account_summary
from .management_account_state import account_state_frame, summarize_account_state
from .request_execution_research import checked_parent


def run(reference, reference_sha256, paired_proof, paired_proof_sha256,
        valuation_proof, valuation_proof_sha256, protocol, output, progress=print):
    spec = checked_reference(reference, reference_sha256)
    frozen = {**spec['sha256'], str(reference): reference_sha256, str(protocol): sha256(protocol)}
    parent = checked_parent(paired_proof, paired_proof_sha256, frozen)
    valuation = checked_parent(valuation_proof, valuation_proof_sha256, frozen)
    public = {str(p): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    frozen.update(public)
    root = new_run(output, 'management-account-state', {'parent': str(parent), 'valuation': str(valuation),
        'protocol_sha256': frozen[str(protocol)], 'model_fitted': False, 'profitability_accepted': False,
        'all_periods_already_observed': True})
    save_json(root/'frozen_inputs.json', frozen)
    (root/'protocol.md').write_bytes(protocol.read_bytes())
    progress(str(root))
    try:
        original = pd.read_parquet(parent/'all_windows.parquet')
        ledger = pd.read_parquet(valuation/'ledger_1.parquet')
        activity = pd.read_parquet(valuation/'boundary_activity.parquet')
        prices = pd.read_parquet(valuation/'all_prices.parquet')
        frame = account_state_frame(original, ledger, activity, prices)
        progress('전체 창의 같은 시각 계좌 연결 완료')
        verified = verify_account_state(original, ledger, activity, prices, frame)
        results = summarize_account_state(frame)
        verify_account_summary(frame, results)
        progress('전체 행 Decimal 환산·지원 사유·기간별 집계 독립 검산 완료')
        frame.to_parquet(root/'all_windows.parquet', index=False)
        pd.testing.assert_frame_equal(frame, pd.read_parquet(root/'all_windows.parquet'), check_exact=True)
        for name, digest in frozen.items():
            if Path(name).is_symlink() or sha256(Path(name)) != digest:
                raise ValueError('계좌 관리 진단 중 입력 변경: '+name)
        checked_reference(reference, reference_sha256)
        verified.update(all_summaries_verified=True, roundtrip_exact=True, originals_unchanged=5,
            production_sources_unchanged=len(public), frozen_inputs_unchanged=len(frozen))
        summary = {'complete': True, 'results': results, 'verification': verified,
            'model_fitted': False, 'profitability_accepted': False, 'trading_returns_evaluated': False}
        save_json(root/'summary.json', summary)
        seal(root)
        return root
    except Exception as error:
        save_json(root/'failure.json', {'complete': False, 'type': type(error).__name__, 'error': str(error)})
        seal(root)
        raise


def main():
    parser = argparse.ArgumentParser(description='원본 관리 상태와 계좌 대비 포지션 크기의 지원 진단')
    for name in ['reference', 'paired-proof', 'valuation-proof']:
        parser.add_argument('--'+name, required=True, type=Path)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', default=Path('artifacts'), type=Path)
    print('완료:', run(**vars(parser.parse_args())), flush=True)


if __name__ == '__main__':
    main()
