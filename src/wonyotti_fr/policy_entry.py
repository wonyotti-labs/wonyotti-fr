from __future__ import annotations

import json
import re
import sqlite3

import numpy as np
import pandas as pd

from .common import save_json, sha256
from .exit_move_state import EXIT_MOVE_FILES, copy_net_parent, load_exit_move_selection
from .journal import canonical, digest
from .label_weighting import WEIGHTING, lifecycle_weights
from .lifecycle_edge import LifecycleNetPolicy
from .net_edge_model import NetEdgeModel
from .policy_outcomes import OUTCOME_PERIOD, outcome_frame, validate_outcome

ENTRY_FILES = ['entry_net_model.json', 'entry_training_used.parquet',
               'entry_training_weights.parquet', 'entry_weighting.json',
               'entry_training_support.json', 'entry_evidence.json']
LABEL_FILES = ['manifest.json', 'input_verification.json', 'potential_entries.parquet',
               'opportunity_ledger.parquet', 'training_labels.parquet',
               'label_intervals.parquet', 'support.json', 'summary.json', 'outcomes.sqlite']


def load_policy_outcome_training(reference, labels, market, features):
    from pathlib import Path

    files = json.loads((labels/'files.json').read_text())
    if set(files) != set(LABEL_FILES) or any(
        (labels/n).is_symlink() or sha256(labels/n) != files[n] for n in LABEL_FILES
    ):
        raise ValueError('현재 정책 정답의 파일·지문 오류')
    settings = json.loads((labels/'manifest.json').read_text())['settings']
    summary = json.loads((labels/'summary.json').read_text())
    code_hashes = settings.get('implementation_sha256', {})
    if (settings['reference_sha256'] != sha256(reference/'frozen_selection.json')
        or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V55.md'))
        or settings['period'] != OUTCOME_PERIOD or settings['all_opportunities'] is not True
        or settings['forced_boundary_closes'] is not False or settings['new_models_fitted'] is not False
        or settings['market_manifest_sha256'] != sha256(market/'manifest-1m.json')
        or settings['feature_manifest_sha256'] != sha256(features/'manifest-5m.json')
        or not {'engine.py', 'lifecycle_edge.py', 'policy_outcomes.py'} <= set(code_hashes)
        or any(not re.fullmatch(r'[a-z_]+\.py', n) or not re.fullmatch(r'[0-9a-f]{64}', h)
               for n, h in code_hashes.items())
        or summary['complete'] is not True or summary['losing_labels_removed'] is not False
        or summary['forced_boundary_closes'] != 0 or summary['profitability_accepted'] is not False):
        raise ValueError('현재 정책 정답의 완료·부모·시세·생성 코드 오류')
    opportunities = pd.read_parquet(labels/'potential_entries.parquet')
    ledger = pd.read_parquet(labels/'opportunity_ledger.parquet')
    if (len(opportunities) != len(ledger) or summary['processed'] != len(ledger)
        or summary['opportunities'] != len(ledger) or opportunities.empty
        or opportunities.decision_time.duplicated().any()
        or not opportunities.decision_time.is_monotonic_increasing):
        raise ValueError('현재 정책 정답의 전체 기회·원장 수 오류')
    cutoff = pd.Timestamp(OUTCOME_PERIOD[1], tz='UTC')-pd.Timedelta(days=1)
    identity = canonical({'format': 'offline_outcome_journal_v1', 'identity': {
        **settings, 'opportunities_sha256': files['potential_entries.parquet']}})
    previous = digest(identity)
    records = []
    # 완료 원장을 읽기 전용으로 열어 생성 당시의 파일 지문을 보존한다.
    with sqlite3.connect((labels/'outcomes.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as connection:
        connection.execute('PRAGMA trusted_schema=OFF')
        if (connection.execute('PRAGMA quick_check').fetchone() != ('ok',)
            or connection.execute('SELECT value FROM metadata WHERE id=1').fetchone() != (identity,)
            or connection.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0] != len(opportunities)):
            raise ValueError('현재 정책 정답의 저장 원장 무결성 오류')
        cursor = connection.execute('SELECT * FROM outcomes ORDER BY sequence')
        for number, (opportunity, row) in enumerate(zip(opportunities.to_dict('records'), cursor, strict=True)):
            sequence, source, payload, prior, chained = row
            if (sequence != number or source != digest(canonical(opportunity)) or prior != previous
                or chained != digest(canonical([sequence, source, payload, previous]))):
                raise ValueError('현재 정책 정답의 기회·해시 연결 오류')
            record = json.loads(payload)
            validate_outcome(opportunity, record, cutoff)
            records.append({**opportunity, **record['outcome']})
            previous = chained
    pd.testing.assert_frame_equal(outcome_frame(records), ledger, check_exact=True)
    train = ledger.loc[ledger.label_status.eq('closed')].reset_index(drop=True)
    pd.testing.assert_frame_equal(train, pd.read_parquet(labels/'training_labels.parquet'), check_exact=True)
    if (summary['closed'] != len(train) or summary['statuses'] != ledger.label_status.value_counts().to_dict()
        or summary['losing_labels'] != int(train.net_bps.lt(0).sum())
        or summary['positive_labels'] != int(train.net_bps.gt(0).sum())
        or ledger.decision_time.lt(pd.Timestamp(OUTCOME_PERIOD[0], tz='UTC')).any()
        or ledger.decision_time.ge(pd.Timestamp(OUTCOME_PERIOD[1], tz='UTC')).any()):
        raise ValueError('현재 정책 정답의 손실·기간·상태 요약 오류')
    weights, intervals, weighting = lifecycle_weights(train)
    pd.testing.assert_frame_equal(intervals, pd.read_parquet(labels/'label_intervals.parquet'), check_exact=True)
    support = json.loads((labels/'support.json').read_text())
    if canonical(support['overlap']) != canonical(weighting) or support['weights_used_for_fitting'] is not False:
        raise ValueError('현재 정책 정답의 중첩 지원 오류')
    evidence = {'format': 'current_policy_entry_evidence_v1', 'labels_files_sha256': sha256(labels/'files.json'),
        'labels_manifest_sha256': files['manifest.json'], 'training_labels_sha256': files['training_labels.parquet'],
        'outcome_journal_sha256': files['outcomes.sqlite'], 'reference_sha256': settings['reference_sha256'],
        'generation_implementation_sha256': code_hashes, 'training_period': OUTCOME_PERIOD,
        'market_manifest_sha256': settings['market_manifest_sha256'],
        'feature_manifest_sha256': settings['feature_manifest_sha256'],
        'opportunities': len(opportunities), 'statuses': summary['statuses'], 'closed': len(train),
        'losing_labels': summary['losing_labels'], 'positive_labels': summary['positive_labels'],
        'losing_labels_removed': False, 'forced_boundary_closes': 0,
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V56.md'))}
    return train, weights, intervals, weighting, evidence


def prepare_policy_entry(reference, labels, market, features, out):
    train, weights, intervals, weighting, evidence = load_policy_outcome_training(reference, labels, market, features)
    model, support = NetEdgeModel.fit(train, 100, sample_weight=weights)
    (out/ENTRY_FILES[1]).write_bytes((labels/'training_labels.parquet').read_bytes())
    intervals.to_parquet(out/ENTRY_FILES[2], index=False)
    save_json(out/ENTRY_FILES[0], model.to_dict())
    save_json(out/ENTRY_FILES[3], weighting)
    save_json(out/ENTRY_FILES[4], support)
    save_json(out/ENTRY_FILES[5], evidence)


def copy_exit_move_parent(reference, out):
    copy_net_parent(reference, out)
    for name in ['net_exit_selection.json', *EXIT_MOVE_FILES]:
        (out/name).write_bytes((reference/name).read_bytes())
    (out/'exit_move_selection.json').write_bytes((reference/'frozen_selection.json').read_bytes())


def load_policy_entry_parent(selection, frozen):
    path = selection/'exit_move_selection.json'
    if path.is_symlink() or path.stat().st_size > 1024**2 or sha256(path) != frozen['exit_move_selection_sha256']:
        raise ValueError('현재 관리 진입 필터의 부모 지문 오류')
    return load_exit_move_selection(selection, json.loads(path.read_text()))


def load_policy_entry_selection(selection, frozen):
    from pathlib import Path

    parent, manager = load_policy_entry_parent(selection, frozen)
    if (frozen.get('protocol') != 'policy_entry_v56' or parent['protocol'] != 'exit_move_v54'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or frozen['policy_entry_alpha'] != 100 or frozen['policy_entry_margin_bps'] != 8
        or frozen['policy_entry_training_period'] != OUTCOME_PERIOD or frozen['policy_entry_sample_weighting'] != WEIGHTING
        or set(frozen['policy_entry_files_sha256']) != set(ENTRY_FILES)
        or any((selection/n).is_symlink() or (selection/n).stat().st_size > 32*1024**2
               or sha256(selection/n) != frozen['policy_entry_files_sha256'][n] for n in ENTRY_FILES)):
        raise ValueError('현재 관리 진입 필터의 모델·문턱·기반 변경')
    train = pd.read_parquet(selection/ENTRY_FILES[1])
    _, intervals, weighting = lifecycle_weights(train)
    pd.testing.assert_frame_equal(intervals, pd.read_parquet(selection/ENTRY_FILES[2]), check_exact=True)
    evidence = json.loads((selection/ENTRY_FILES[5]).read_text())
    support = json.loads((selection/ENTRY_FILES[4]).read_text())
    if (canonical(weighting) != canonical(json.loads((selection/ENTRY_FILES[3]).read_text()))
        or evidence['format'] != 'current_policy_entry_evidence_v1' or evidence['training_period'] != OUTCOME_PERIOD
        or evidence['reference_sha256'] != frozen['exit_move_selection_sha256']
        or evidence['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V56.md'))
        or evidence['training_labels_sha256'] != sha256(selection/ENTRY_FILES[1])
        or evidence['losing_labels_removed'] is not False or evidence['forced_boundary_closes'] != 0
        or evidence['closed'] != len(train) or support['rows'] != len(train)
        or evidence['losing_labels'] != int(train.net_bps.lt(0).sum())
        or evidence['positive_labels'] != int(train.net_bps.gt(0).sum())
        or not train.label_status.eq('closed').all() or train.entry_notional.le(0).any()
        or not np.isfinite(train[['net_bps', 'net_pnl', 'entry_notional']]).all().all()
        or not np.allclose(train.net_bps, train.net_pnl/train.entry_notional*10000, rtol=0, atol=1e-7)):
        raise ValueError('현재 관리 진입 필터의 학습 정답·손실·가중치 오류')
    model = NetEdgeModel.from_dict(json.loads((selection/ENTRY_FILES[0]).read_text()))
    return frozen, LifecycleNetPolicy(manager, model)
