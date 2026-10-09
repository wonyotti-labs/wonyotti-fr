from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .close_threshold import THRESHOLD_SPLITS
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .event_research import load_selection
from .first_opportunity_close import first_opportunity_rows
from .first_opportunity_collection import (
    EXPANDED_FIRST_FEATURES,
    collect_first_opportunities,
    first_context_lookup,
)
from .journal import canonical
from .minute_close_effects import preserve_generation_source
from .minute_data import prepare_minute_period
from .outcome_journal import OutcomeJournal
from .research import load_market
from .streaming_backtest import streaming_backtest
from .weekly_first_linear_reference import checked_weekly_first_reference

EXPANSION_PERIODS = {'parity_2021': ['2021-01-01', '2022-01-01'], 'expansion_2020': ['2020-04-01', '2021-01-01']}
BASELINE_FILES = ['equity.parquet', 'trades.parquet', 'fills.parquet', 'final_state.json']


def time_frame(frame, names):
    result = frame.copy()
    for name in names:
        if name in result:
            result[name] = pd.to_datetime(result[name], utc=True).astype('datetime64[ns, UTC]')
    return result


def verify_first_expansion_parity(labels, positions, reference):
    times = ['decision_time', 'position_entry_time', 'label_end', 'first_available_time', 'first_eligible_time', 'natural_exit_time']
    labels, positions = time_frame(labels, times), time_frame(positions, times)
    if labels.position_entry_time.duplicated().any() or positions.position_entry_time.duplicated().any():
        raise ValueError('과거 첫 기회 대조의 중복 포지션')
    ends = positions.natural_exit_time+pd.to_timedelta((~positions.natural_exit_reason.isin(['intrabar_stop', 'end_of_test'])).astype('int64'), unit='min')
    proof = {}
    for split, bounds in THRESHOLD_SPLITS.items():
        if split == 'diagnosis':
            continue
        original = pd.read_parquet(reference/f'{split}_used.parquet')
        first, expected_positions = first_opportunity_rows(original)
        start, end = [pd.Timestamp(value, tz='UTC') for value in bounds]
        population = positions[positions.position_entry_time.ge(start) & ends.lt(end)
            & positions.available_rows.gt(0) & positions.natural_exit_reason.ne('end_of_test')].reset_index(drop=True)
        fields = ['position_entry_time', 'direction', 'first_available_time', 'reference_equity', 'has_eligible_opportunity', 'first_eligible_time', 'first_target_common_bps']
        expected_positions = time_frame(expected_positions, times)
        pd.testing.assert_frame_equal(population[fields], expected_positions[fields], check_exact=True, check_dtype=False)
        chosen = labels[labels.position_entry_time.isin(population.position_entry_time) & labels.label_status.eq('closed')].reset_index(drop=True)
        fields = ['decision_time', 'position_entry_time', 'label_end', 'original_intent', 'reference_equity',
            'decision_equity', 'close_cash', 'continue_cash', 'close_advantage_pnl', 'close_advantage_bps', 'first_target_common_bps', *EXPANDED_FIRST_FEATURES]
        fields = list(dict.fromkeys(fields))
        pd.testing.assert_frame_equal(chosen[fields], time_frame(first, times)[fields], check_exact=True)
        proof[split] = {'all_positions': len(population), 'first_eligible': len(chosen), 'first_rows_features_equity_cash_and_population_exact': True}
    return {'complete': True, 'previous_reference_files_sha256': sha256(reference/'files.json'), 'splits': proof, 'diagnosis_model_predictions_generated': False}


def sealed_phase(folder):
    mapping = json.loads((folder/'files.json').read_text())
    if (folder.is_symlink() or (folder/'files.json').is_symlink() or any(Path(name).name != name or (folder/name).is_symlink()
        or sha256(folder/name) != value for name, value in mapping.items())):
        raise ValueError('과거 첫 기회 단계의 파일 봉인 오류')
    summary = json.loads((folder/'summary.json').read_text())
    if summary['complete'] is not True:
        raise ValueError('과거 첫 기회 단계의 미완료 재사용')
    replay = Path(summary['replay'])
    if (replay.is_symlink() or not replay.resolve().is_relative_to(folder.resolve())
        or any(sha256(replay/name) != value for name, value in summary['replay_sha256'].items())):
        raise ValueError('과거 첫 기회 단계의 완료 재생 변경')
    baseline = Path(summary['baseline'])
    if (baseline.is_symlink() or set(summary['baseline_sha256']) != set(BASELINE_FILES)
        or any((baseline/name).is_symlink() or sha256(baseline/name) != value for name, value in summary['baseline_sha256'].items())):
        raise ValueError('과거 첫 기회 단계의 완료 기준 경로 변경')
    return summary


def run_first_opportunity_expansion(selection: Path, diagnosis: Path, verification: Path, verification_sha256: str,
                                    market: Path, features: Path, output: Path, *, resume: Path | None = None,
                                    max_opportunities: int | None = None) -> Path:
    proof = checked_weekly_first_reference(diagnosis, verification, verification_sha256)
    frozen, _ = load_selection(selection)
    if frozen['protocol'] != 'exit_move_v54' or (max_opportunities is not None and (type(max_opportunities) is not int or max_opportunities < 1)):
        raise ValueError('과거 첫 기회 확장의 고정 부모·처리 한도 오류')
    settings = {'reference': str(selection), 'reference_sha256': sha256(selection/'frozen_selection.json'),
        'reference_outputs_sha256': {name: sha256(selection/'candidate-00'/name) for name in BASELINE_FILES},
        'diagnosis': str(diagnosis), 'diagnosis_files_sha256': proof['reference_files_sha256'],
        'verification': str(verification), 'verification_sha256': verification_sha256,
        'market': str(market), 'features': str(features), 'market_manifest_sha256': sha256(market/'manifest-1m.json'),
        'feature_manifest_sha256': sha256(features/'manifest-5m.json'), 'periods': EXPANSION_PERIODS,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V84.md')), 'new_models_fitted': False,
        'whole_policy_historically_available_claimed': False, 'features_used': EXPANDED_FIRST_FEATURES,
        'implementation_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    out = resume if resume is not None else new_run(output, 'first-opportunity-expansion-labels', settings)
    if resume is not None and (out.is_symlink() or json.loads((out/'manifest.json').read_text())['settings'] != settings
        or ((out/'summary.json').exists() and json.loads((out/'summary.json').read_text())['complete'])):
        raise ValueError('과거 첫 기회 확장의 변경 입력·완료 실행 재개')
    print(f'고정 정책의 과거 첫 기회 자료 확장: {out}', flush=True)
    try:
        preserve_generation_source(out, settings['implementation_sha256'])
        save_json(out/'reference_evidence.json', proof)
        results = {}
        for phase, bounds in EXPANSION_PERIODS.items():
            folder = out/phase
            if folder.is_symlink():
                raise ValueError('과거 첫 기회 단계의 링크 경로 오류')
            folder.mkdir(exist_ok=True, mode=0o700)
            if (folder/'files.json').exists() and json.loads((folder/'summary.json').read_text())['complete']:
                results[phase] = sealed_phase(folder)
                continue
            if phase == 'expansion_2020':
                prior = sealed_phase(out/'parity_2021')
                if prior.get('reference_parity', {}).get('complete') is not True:
                    raise ValueError('2020년 확장 전 2021년 첫 기회 동일성 누락')
            bars, checks = prepare_minute_period(market, features, 'BTCUSDT', *bounds, minute_inputs=True)
            if (folder/'input_verification.json').exists() and canonical(checks) != canonical(json.loads((folder/'input_verification.json').read_text())):
                raise ValueError('과거 첫 기회 확장의 시세 재개 변경')
            save_json(folder/'input_verification.json', checks)
            five, _ = load_market(features, 'BTCUSDT', '5m')
            context = first_context_lookup(five[five.end.le(bars.end.max())].reset_index(drop=True))
            _, policy = load_selection(selection)
            config = EngineConfig(**frozen['risk'])
            if phase == 'parity_2021':
                baseline = selection/'candidate-00'
            elif (folder/'baseline_source.json').exists():
                baseline_proof = json.loads((folder/'baseline_source.json').read_text())
                baseline = Path(baseline_proof['baseline'])
                if (baseline.is_symlink() or not baseline.resolve().is_relative_to(folder.resolve())
                    or any(sha256(baseline/name) != value for name, value in baseline_proof['sha256'].items())):
                    raise ValueError('과거 첫 기회의 2020년 기준 재생 변경')
            else:
                baseline = folder/f'baseline-{uuid4().hex[:12]}'
                print(f'{phase}: 독립 기준 경로 재생', flush=True)
                streaming_backtest(bars, policy, config, baseline, 8192)
                save_json(folder/'baseline_source.json', {'baseline': str(baseline), 'sha256': {name: sha256(baseline/name) for name in BASELINE_FILES}})
                _, policy = load_selection(selection)
            identity = {'settings': settings, 'phase': phase, 'baseline_sha256': {name: sha256(baseline/name) for name in BASELINE_FILES}}
            replay = folder/f'replay-{uuid4().hex[:12]}'
            cutoff = pd.Timestamp(bounds[1], tz='UTC')-pd.Timedelta(days=1)
            print(f'{phase}: 첫 기회 수집과 전체 경로 대조', flush=True)
            with OutcomeJournal(folder/'outcomes.sqlite', identity) as journal:
                rows, population, counts, complete = collect_first_opportunities(bars, policy, config, baseline, replay, journal,
                    cutoff, context, max_new=max_opportunities)
                journal.verify()
            label_columns = ['decision_time', 'position_entry_time', 'start', 'trade_index', 'original_intent', 'first_available_time',
                'reference_equity', 'opportunity_index', *EXPANDED_FIRST_FEATURES, 'label_status', 'label_end', 'continue_end',
                'continue_exit_reason', 'decision_equity', 'close_cash', 'continue_cash', 'close_advantage_pnl', 'close_advantage_bps', 'first_target_common_bps']
            labels = time_frame(pd.DataFrame(rows).reindex(columns=label_columns), ['decision_time', 'position_entry_time', 'first_available_time', 'label_end', 'continue_end'])
            positions = time_frame(pd.DataFrame(population), ['position_entry_time', 'first_available_time', 'first_eligible_time', 'natural_exit_time'])
            labels.to_parquet(folder/'first_opportunity_ledger.parquet', index=False)
            positions.to_parquet(folder/'all_positions.parquet', index=False)
            labels[labels.label_status.eq('closed')].reset_index(drop=True).to_parquet(folder/'training_labels.parquet', index=False)
            parity = verify_first_expansion_parity(labels, positions, diagnosis) if phase == 'parity_2021' and complete else None
            summary = {'complete': complete, 'phase': phase, 'selected_first': len(labels), 'positions': len(positions),
                'closed': int(labels.label_status.eq('closed').sum()), 'statuses': labels.label_status.value_counts().to_dict(),
                'counts': counts, 'replay': str(replay), 'baseline': str(baseline), 'reference_parity': parity,
                'replay_sha256': {name: sha256(replay/name) for name in [*BASELINE_FILES, 'management_membership.parquet', 'parity.json']},
                'baseline_sha256': identity['baseline_sha256'], 'new_models_fitted': False, 'profitability_accepted': False}
            save_json(folder/'summary.json', summary)
            save_json(folder/'files.json', {p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p.name != 'files.json'})
            results[phase] = summary
            if not complete:
                break
        if (checked_weekly_first_reference(diagnosis, verification, verification_sha256) != proof
            or sha256(selection/'frozen_selection.json') != settings['reference_sha256']
            or any(sha256(selection/'candidate-00'/name) != value for name, value in settings['reference_outputs_sha256'].items())
            or sha256(market/'manifest-1m.json') != settings['market_manifest_sha256']
            or sha256(features/'manifest-5m.json') != settings['feature_manifest_sha256']
            or sha256(Path('docs/EXPERIMENT_V84.md')) != settings['protocol_sha256']
            or any(sha256(Path(__file__).parent/name) != value for name, value in settings['implementation_sha256'].items())):
            raise ValueError('과거 첫 기회 확장 중 입력·정책·근거·코드·계획 변경')
        summary = {'complete': len(results) == 2 and all(row['complete'] for row in results.values()),
            'phases': {phase: {'complete': row['complete'], 'selected_first': row['selected_first'], 'closed': row['closed'],
                'files_sha256': sha256(out/phase/'files.json')} for phase, row in results.items()},
            'new_models_fitted': False, 'forced_boundary_closes_as_labels': False, 'profitability_accepted': False}
        save_json(out/'summary.json', summary)
        save_json(out/'files.json', {name: sha256(out/name) for name in ['manifest.json', 'reference_evidence.json', 'summary.json']})
        print(summary, flush=True)
    except Exception as error:
        save_json(out/f'failure-{uuid4().hex[:12]}.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
