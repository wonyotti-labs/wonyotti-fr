from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .addition_effect import position_weights
from .close_calibration import reproduce_close_diagnosis
from .close_effect import CLOSE_FEATURES
from .close_learning import close_metrics
from .close_learning_inputs import CLOSE_FILES, CLOSE_SPLITS, close_learning_splits
from .common import new_run, save_json, sha256
from .engine import EngineConfig
from .entry_regression import REGRESSION_SETTINGS, EntryRegressionModel
from .journal import canonical, digest

ECONOMIC_FEATURES = ['current_gross_exposure', 'current_exit_net_bps']


class EconomicCloseModel(EntryRegressionModel):
    features = CLOSE_FEATURES+ECONOMIC_FEATURES
    format = 'close_economic_histogram_v1'


def current_economic_values(state, risk):
    trade = state['active_trade']
    q, price, entry, cash = (state[k] for k in ['quantity', 'last_close', 'entry_price', 'cash'])
    if (trade is None or not np.isfinite([q, price, entry, cash, trade['gross_realized'], trade['fees'], trade['funding_cost']]).all()
        or q == 0 or min(price, entry) <= 0 or trade['fees'] < 0):
        raise ValueError('현재 경제적 입력의 보유·가격·숫자 오류')
    equity = cash+q*price
    if not np.isfinite(equity) or equity <= 0:
        raise ValueError('현재 경제적 입력의 비양수 순자산')
    execution = price*(1-np.sign(q)*risk.slippage_bps/10000)
    # 실제 다음 체결 대신 현재 확정 종가의 비용을 적용한다.
    net = trade['gross_realized']-trade['fees']-trade['funding_cost']+q*(execution-entry)-abs(q)*execution*risk.fee_bps/10000
    values = np.array([abs(q)*price/equity, net/equity*10000])
    if not np.isfinite(values).all():
        raise ValueError('현재 경제적 입력의 계산 범위 오류')
    return values


def load_economic_inputs(labels, ledger):
    hashes = json.loads((labels/'files.json').read_text())
    if (set(hashes) != CLOSE_FILES or (labels/'files.json').is_symlink()
        or any((labels/n).is_symlink() or sha256(labels/n) != h for n, h in hashes.items())
        or any((labels/('outcomes.sqlite'+s)).exists() for s in ['-wal', '-shm'])):
        raise ValueError('현재 경제적 입력의 원장 파일·지문 오류')
    pd.testing.assert_frame_equal(ledger, pd.read_parquet(labels/'opportunity_ledger.parquet'), check_exact=True)
    settings = json.loads((labels/'manifest.json').read_text())['settings']
    parent = Path(settings['reference'])/'frozen_selection.json'
    if sha256(parent) != settings['reference_sha256'] or json.loads((labels/'summary.json').read_text())['complete'] is not True:
        raise ValueError('현재 경제적 입력의 미완료·부모 연결 오류')
    risk = EngineConfig(**json.loads(parent.read_text())['risk'])
    identity = canonical({'format': 'offline_outcome_journal_v1', 'identity': settings})
    previous, number = digest(identity), 0
    values = np.full((len(ledger), len(ECONOMIC_FEATURES)), np.nan)
    times = ledger.decision_time.astype('datetime64[ns, UTC]').array.asi8
    entries = ledger.position_entry_time.astype('datetime64[ns, UTC]').array.asi8
    connection = sqlite3.connect((labels/'outcomes.sqlite').resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
    try:
        connection.execute('PRAGMA trusted_schema=OFF')
        if (connection.execute('PRAGMA quick_check').fetchone() != ('ok',)
            or connection.execute('SELECT value FROM metadata').fetchall() != [(identity,)]
            or connection.execute('SELECT COUNT(*) FROM outcomes').fetchone() != (len(ledger),)
            or connection.execute('SELECT COALESCE(MAX(length(payload)),0) FROM outcomes').fetchone()[0] > 1024**2):
            raise ValueError('현재 경제적 입력의 원자 크기·무결성 오류')
        for index, source, payload, prior, chained in connection.execute('SELECT * FROM outcomes ORDER BY sequence'):
            op = json.loads(payload)['opportunity']
            if (index != number or prior != previous or source != digest(canonical(op))
                or chained != digest(canonical([index, source, payload, previous]))):
                raise ValueError('현재 경제적 입력의 원자 순서·입력·해시 연결 오류')
            state = op['state']
            if (pd.Timestamp(op['decision_time']).value != times[index]
                or pd.Timestamp(op['position_entry_time']).value != entries[index]
                or pd.Timestamp(state['last_end']).value != times[index]
                or pd.Timestamp(state['active_trade']['entry_time']).value != entries[index]):
                raise ValueError('현재 경제적 입력의 판단·보유 시각 연결 오류')
            values[index] = current_economic_values(state, risk)
            previous, number = chained, number+1
    finally:
        connection.close()
    if number != len(ledger) or any(sha256(labels/n) != h for n, h in hashes.items()):
        raise ValueError('현재 경제적 입력의 전체 원장·읽기 전용 지문 오류')
    result = ledger.copy()
    result[ECONOMIC_FEATURES] = values
    return result, {'rows': number, 'journal_read_only': True, 'labels_files_sha256': sha256(labels/'files.json'),
        'all_original_rows_and_features_preserved': True, 'future_fill_or_outcome_used': False}


def economic_admission(metrics):
    candidate = metrics['economic']
    if set(metrics) != {'economic', 'boosted', 'ridge', 'constant'} or len({(m['rows'], m['positions']) for m in metrics.values()}) != 1:
        raise ValueError('경제적 입력 진단의 대조·행·포지션 수 오류')
    checks = {f'weighted_mse_vs_{k}': candidate['weighted_mse'] < metrics[k]['weighted_mse']*.99 for k in ['boosted', 'ridge', 'constant']}
    checks.update({f'unweighted_mse_vs_{k}': candidate['mse'] <= metrics[k]['mse']+1e-9 for k in ['boosted', 'ridge']})
    checks.update(at_least_100_selected=candidate['selected'] >= 100, at_least_30_selected_positions=candidate['selected_positions'] >= 30,
        positive_selected_weighted_mean=candidate['selected_weighted_mean_bps'] is not None and candidate['selected_weighted_mean_bps'] > 0,
        positive_selected_mean=candidate['selected_mean_bps'] is not None and candidate['selected_mean_bps'] > 0)
    return {'checks': checks, 'economic_inputs_admitted': all(checks.values()), 'trading_returns_evaluated': False}


def run_close_economic_diagnosis(reference: Path, output: Path) -> Path:
    out = new_run(output, 'close-economic-diagnosis', {'reference': str(reference),
        'reference_files_sha256': sha256(reference/'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V62.md')),
        'periods': CLOSE_SPLITS, 'features': EconomicCloseModel.features, 'settings': REGRESSION_SETTINGS,
        'margin_bps': 0, 'new_model_count': 1, 'trading_returns_evaluated': False, 'whole_system_periods_already_observed': True})
    print(f'현재 경제적 상태의 청산 순효과 진단: {out}', flush=True)
    try:
        reproduced = reproduce_close_diagnosis(reference, out/'reference-reproduction')
        save_json(out/'reference_parity.json', {'complete': True, 'reproduction': str(reproduced),
            'all_previous_models_predictions_rows_weights_exact': True, 'reproduced_files_sha256': sha256(reproduced/'files.json')})
        labels = Path(json.loads((reference/'manifest.json').read_text())['settings']['labels'])
        ledger = pd.read_parquet(reproduced/'exclusion_ledger.parquet').drop(columns='split')
        augmented, verification = load_economic_inputs(labels, ledger)
        augmented.to_parquet(out/'economic_ledger.parquet', index=False)
        save_json(out/'input_verification.json', verification)
        rows, assignments = close_learning_splits(augmented)
        assignments.to_parquet(out/'exclusion_ledger.parquet', index=False)
        pd.testing.assert_frame_equal(assignments.drop(columns=ECONOMIC_FEATURES), pd.read_parquet(reference/'exclusion_ledger.parquet'), check_exact=True)
        values, weights = {}, {}
        for name, frame in rows.items():
            pd.testing.assert_frame_equal(frame.drop(columns=ECONOMIC_FEATURES), pd.read_parquet(reference/f'{name}_used.parquet'), check_exact=True)
            weights[name] = position_weights(frame)
            stored = pd.read_parquet(reference/f'{name}_weights.parquet')
            pd.testing.assert_frame_equal(frame[['decision_time', 'position_entry_time']].assign(sample_weight=weights[name]), stored, check_exact=True)
            frame.to_parquet(out/f'{name}_used.parquet', index=False)
            stored.to_parquet(out/f'{name}_weights.parquet', index=False)
            values[name] = frame[EconomicCloseModel.features].to_numpy(dtype=float)
        model, support = EconomicCloseModel.fit(values['training'], rows['training'].close_advantage_bps,
            weights['training'], values['diagnosis'])
        save_json(out/'model.json', model.to_dict())
        save_json(out/'training_support.json', support)
        (out/'previous_models.json').write_bytes((reference/'models.json').read_bytes())
        frame = pd.read_parquet(reference/'predictions.parquet')
        frame['predicted_economic'] = model.predict(values['diagnosis'])
        frame.to_parquet(out/'predictions.parquet', index=False)
        metrics = json.loads((reference/'metrics.json').read_text())
        metrics['economic'] = close_metrics(frame, frame.predicted_economic)
        decision, details = economic_admission(metrics), []
        groups = [('direction', str(k), part) for k, part in frame.groupby('direction')]
        groups += [('month', k, part) for k, part in frame.groupby(frame.decision_time.dt.strftime('%Y-%m'))]
        for kind, group, part in groups:
            for name in metrics:
                details.append({'kind': kind, 'group': group, 'model': name, **close_metrics(part, part[f'predicted_{name}'])})
        for name, content in [('metrics', metrics), ('breakdown', details), ('decision', decision)]:
            save_json(out/f'{name}.json', content)
        save_json(out/'summary.json', {'complete': True, 'all_previous_outputs_reproduced': True,
            'all_original_rows_and_weights_preserved': True, 'profitability_accepted': False, **decision})
        (out/'REPORT.md').write_text('# 현재 경제적 상태의 청산 순효과 진단\n\n'
            +f'사전 진단 조건 통과: {decision["economic_inputs_admitted"]}. '
            '같은 원장·시간·포지션 비중·원래 50개 입력에 현재 노출 비율과 비용 포함 추정 청산 손익만 추가했다. '
            '다음 체결과 향후 정답은 입력에 쓰지 않았으며 매매 수익성은 별도 검증이다.\n')
        save_json(out/'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(decision, flush=True)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
