from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .action_model import select_threshold
from .calibration_diagnostics import PERIODS
from .common import new_run, save_json, sha256
from .episode_balance import load_episode_reference, verified_files
from .management_calibration import ManagementOffset
from .management_diagnostics import management_metrics
from .minute_inventory import load_minute_inventory_labels
from .minute_management import purged_window
from .order_history import HISTORY_FEATURES
from .order_history_boost import OrderHistoryBoostModels
from .reports import table

HORIZON_MINUTES = 15
HORIZON = pd.Timedelta(minutes=HORIZON_MINUTES)


class ExitHorizonModel(OrderHistoryBoostModels):
    format = 'exit_horizon_histogram_v1'
    actions = ['exit']


def attach_exit_horizon(frame, last_source_time):
    last = pd.Timestamp(last_source_time)
    if (last.tzinfo is None or last.utcoffset().total_seconds() or pd.isna(last)
        or frame.end.isna().any() or frame.end.duplicated().any() or not frame.end.is_monotonic_increasing
        or not isinstance(frame.end.dtype, pd.DatetimeTZDtype) or str(frame.end.dt.tz) != 'UTC'
        or not frame.end.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all()
        or not frame.label_end.eq(frame.end + pd.Timedelta(minutes=1)).all()
        or not frame.y_exit.isin([0, 1]).all()):
        raise ValueError('청산 범위 정답의 시각·연속성·기존 정답 오류')
    events = frame.loc[frame.y_exit.eq(1), ['end', 'episode_id']].copy()
    if (events.episode_id.le(0).any() or not np.isfinite(events.episode_id).all()
        or events.episode_id.ne(np.floor(events.episode_id)).any()):
        raise ValueError('청산 범위 정답의 포지션 오류')
    result = frame.copy()
    result['original_y_exit'] = frame.y_exit
    result['original_label_end'] = frame.label_end
    result['original_usable'] = frame.usable
    result['label_end'] = frame.end + HORIZON
    result['usable'] &= result.label_end.le(last)
    result['y_exit'] = 0
    links = np.full(len(frame), pd.NaT.value, dtype=np.int64)
    times = frame.end.astype('datetime64[ns, UTC]').array.asi8
    if (times % pd.Timedelta(minutes=1).value).any():
        raise ValueError('청산 범위 정답의 분 경계 오류')
    groups = frame.groupby('episode_id', sort=False).indices
    for episode, part in events.groupby('episode_id', sort=False):
        indices = groups[episode]
        candidates = part.end.astype('datetime64[ns, UTC]').array.asi8
        positions = np.searchsorted(candidates, times[indices], side='left')
        present = positions < len(candidates)
        targets = indices[present]
        next_event = candidates[positions[present]]
        matched = next_event < times[targets] + HORIZON.value
        links[targets[matched]] = next_event[matched]
    result['horizon_event_end'] = pd.to_datetime(links, utc=True)
    result['y_exit'] = result.horizon_event_end.notna().astype(int)
    return result


def horizon_splits(frame, reference):
    rows, support, ledgers = {}, {}, []
    for name, period in PERIODS.items():
        part = purged_window(frame, *period).reset_index(drop=True)
        old = reference[name].set_index('end')
        if not part.end.isin(old.index).all():
            raise ValueError('청산 범위 정답의 기존 지원 밖 행')
        restored = part[reference[name].columns].copy()
        for current, original in [('y_exit', 'original_y_exit'), ('label_end', 'original_label_end'), ('usable', 'original_usable')]:
            restored[current] = part[original]
        pd.testing.assert_frame_equal(restored, old.loc[part.end].reset_index()[restored.columns], check_exact=True)
        y, ids = part.y_exit.to_numpy(), part.episode_id.to_numpy()
        if (len(part) < (1000 if name == 'training' else 100) or min(y.sum(), len(y)-y.sum()) < 20
            or part.original_y_exit.sum() < 20
            or not np.isfinite(part[ExitHorizonModel.features]).all().all()
            or part[['end', 'entry_time', 'label_end', 'episode_id']].isna().any().any()
            or not np.isfinite(ids).all() or ((ids <= 0) | (ids >= 2**53) | (ids != np.floor(ids))).any()
            or part.entry_time.ge(part.end).any() or not part.label_end.eq(part.end + HORIZON).all()
            or not part.y_exit.eq(part.horizon_event_end.notna().astype(int)).all()
            or part.loc[part.y_exit.eq(1), 'horizon_event_end'].lt(part.loc[part.y_exit.eq(1), 'end']).any()
            or part.loc[part.y_exit.eq(1), 'horizon_event_end'].ge(part.loc[part.y_exit.eq(1), 'label_end']).any()):
            raise ValueError('청산 범위 학습의 지원·특징·연결 오류')
        rows[name] = part
        support[name] = {'period': list(period), 'rows': len(part), 'positive_minutes': int(y.sum()),
            'original_positive_minutes': int(part.original_y_exit.sum()), 'episodes': int(part.episode_id.nunique()),
            'positive_episodes': int(part.loc[part.y_exit.eq(1), 'episode_id'].nunique()),
            'removed_original_rows': len(old)-len(part)}
        ledger = reference[name][['end', 'episode_id', 'label_end', 'usable']].copy()
        ledger['horizon_label_end'] = ledger.end + HORIZON
        ledger['included'] = ledger.end.isin(part.end)
        end_limit = pd.Timestamp(period[1], tz='UTC') - pd.Timedelta(days=1)
        ledger['reason'] = np.where(ledger.included, 'included', np.where(
            ledger.horizon_label_end.ge(end_limit), 'extended_end_embargo', 'observation_end'))
        ledger['split'] = name
        ledgers.append(ledger)
    for a, b in [('training', 'calibration'), ('training', 'diagnosis'), ('calibration', 'diagnosis')]:
        if set(rows[a].episode_id) & set(rows[b].episode_id) or rows[a].label_end.max() >= rows[b].end.min():
            raise ValueError('청산 범위 학습의 포지션·시간 중첩')
    return rows, support, pd.concat(ledgers, ignore_index=True)


def fit_exit_offset(scores, labels):
    # 기존 절편 구현을 같은 이진 문제로 복제해 경계·수렴 조건을 공유한다.
    p, y = np.asarray(scores), np.asarray(labels)
    if p.ndim != 1 or y.shape != p.shape:
        raise ValueError('청산 범위 절편의 차원 오류')
    return ManagementOffset.fit(np.repeat(p[:, None], 3, axis=1), np.repeat(y[:, None], 3, axis=1))


def exit_offset_predict(offset, scores):
    if not np.equal(offset.offsets, offset.offsets[0]).all():
        raise ValueError('청산 범위 절편의 단일 목표 불일치')
    p = np.asarray(scores)
    if p.ndim != 1:
        raise ValueError('청산 범위 절편의 예측 차원 오류')
    return offset.predict(np.repeat(p[:, None], 3, axis=1))[:, 0]


def fit_exit_horizon(rows, baseline):
    train, calibration, diagnosis = (rows[n] for n in PERIODS)
    x, cx, vx = (v[ExitHorizonModel.features].to_numpy(dtype=float) for v in [train, calibration, diagnosis])
    model, fitted = ExitHorizonModel.fit(x, train[['y_exit']].to_numpy(), cx)
    raw_cal = model.probabilities(cx)[:, 0]
    original_cal = baseline.probabilities(cx)[:, 0]
    offset, calibrated = fit_exit_offset(raw_cal, calibration.y_exit.to_numpy())
    original_offset, original_support = fit_exit_offset(original_cal, calibration.y_exit.to_numpy())
    raw = model.probabilities(vx)[:, 0]
    scores = {'candidate_raw': raw, 'candidate': exit_offset_predict(offset, raw),
        'original': exit_offset_predict(original_offset, baseline.probabilities(vx)[:, 0]),
        'constant': np.full(len(vx), calibration.y_exit.mean())}
    thresholds, threshold_support = {}, {}
    for name, values in [('candidate', exit_offset_predict(offset, raw_cal)),
                         ('original', exit_offset_predict(original_offset, original_cal))]:
        thresholds[name], threshold_support[name] = select_threshold(calibration.y_exit.to_numpy(), values, 2.)
    return model, {'candidate': offset, 'original': original_offset}, scores, thresholds, {
        'model': fitted, 'offset': calibrated, 'original_offset': original_support, 'thresholds': threshold_support}


def horizon_requests(frame, scores, threshold):
    p, y = np.asarray(scores), frame.y_exit.to_numpy()
    if (p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
        or type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('청산 범위 요청의 점수·문턱 오류')
    selected = p >= threshold
    positive, predicted, hits = int(y.sum()), int(selected.sum()), int((selected & (y == 1)).sum())
    events = frame.loc[frame.original_y_exit.eq(1), ['end', 'episode_id']].copy()
    requests = frame.loc[selected, ['end', 'episode_id']]
    detected = np.zeros(len(events), dtype=bool)
    for episode, indices in events.groupby('episode_id', sort=False).indices.items():
        ends = events.iloc[indices].end.astype('datetime64[ns, UTC]').array.asi8
        times = requests.loc[requests.episode_id.eq(episode), 'end'].astype('datetime64[ns, UTC]').array.asi8
        detected[indices] = np.searchsorted(times, ends, side='right') > np.searchsorted(times, ends-HORIZON.value, side='right')
    events['detected'] = detected
    if not len(events):
        raise ValueError('청산 범위 요청의 기존 사건 지원 부족')
    return {'positive_minutes': positive, 'requested_minutes': predicted, 'true_positive_minutes': hits,
        'precision': hits/predicted if predicted else 0., 'recall': hits/positive if positive else 0.,
        'f2': 5*hits/(4*positive+predicted) if 4*positive+predicted else 0.,
        'original_events': len(events), 'detected_events': int(detected.sum()),
        'event_recall': float(detected.mean()), 'original_event_episodes': int(events.episode_id.nunique())}, events


def exit_horizon_admission(metrics, requests):
    old, new, raw, constant = (metrics[k] for k in ['original', 'candidate', 'candidate_raw', 'constant'])
    losses = [m['log_loss'] for m in [old, new, raw, constant]]
    aps = [m['average_precision'] for m in [old, new]]
    rates = [requests[n][k] for n in ['original', 'candidate'] for k in ['f2', 'event_recall']]
    if not np.isfinite(losses+aps+rates).all() or min(losses) < 0 or not all(0 <= v <= 1 for v in aps+rates):
        raise ValueError('청산 범위 진단의 지표 범위 오류')
    gain = 1-new['log_loss']/old['log_loss'] if old['log_loss'] else 0.
    checks = {'loss_gain': bool(gain > .01 and not np.isclose(gain, .01, rtol=0, atol=1e-12)),
        'average_precision_preserved': new['average_precision'] >= old['average_precision'],
        'constant_preserved': new['log_loss'] <= constant['log_loss'], 'raw_preserved': new['log_loss'] <= raw['log_loss'],
        'f2_preserved': requests['candidate']['f2'] >= requests['original']['f2'],
        'event_recall_preserved': requests['candidate']['event_recall'] >= requests['original']['event_recall']}
    return {'exit_horizon_admitted': all(checks.values()), 'checks': checks, 'relative_log_loss_gain': gain,
        'profitability_accepted': False, 'trading_returns_evaluated': False, 'all_source_periods_already_observed': True}


def load_exit_horizon_inputs(source):
    reference, _ = load_episode_reference(source)
    settings = json.loads((source / 'manifest.json').read_text())['settings']
    history, labels = Path(settings['history']), Path(settings['labels'])
    verified_files(history)
    meta = json.loads((history / 'manifest.json').read_text())['settings']
    if (sha256(history / 'files.json') != settings['history_files_sha256']
        or sha256(labels / 'files.json') != settings['labels_files_sha256']
        or meta['labels_files_sha256'] != settings['labels_files_sha256']
        or meta['selection_sha256'] != settings['selection_sha256'] or meta['new_features'] != HISTORY_FEATURES
        or any(sha256(Path(meta['audit']) / n) != h for n, h in meta['audit_sha256'].items())):
        raise ValueError('청산 범위 진단의 원본·기반 정답 연결 오류')
    frame, _ = load_minute_inventory_labels(labels)
    features = pd.read_parquet(history / 'history_features.parquet')
    if features.columns.tolist() != ['end', *HISTORY_FEATURES] or not features.end.equals(frame.end):
        raise ValueError('청산 범위 진단의 과거 특징 시각 오류')
    frame[HISTORY_FEATURES] = features[HISTORY_FEATURES].to_numpy()
    for name, period in PERIODS.items():
        pd.testing.assert_frame_equal(purged_window(frame, *period).reset_index(drop=True), reference[name], check_exact=True)
    last = pd.read_parquet(Path(meta['audit']) / 'actions.parquet', columns=['time']).time.max()
    targets = attach_exit_horizon(frame, last)
    rows, support, ledger = horizon_splits(targets, reference)
    model = OrderHistoryBoostModels.from_dict(json.loads((source / 'models.json').read_text())['histogram'])
    original_offset = ManagementOffset.from_dict(json.loads((source / 'offsets.json').read_text())['histogram'])
    from .history_state import CalibratedHistoryModels
    return rows, support, ledger, targets[['end', 'episode_id', 'original_y_exit', 'y_exit', 'label_end', 'usable', 'horizon_event_end']], CalibratedHistoryModels(model, original_offset)


def run_exit_horizon_diagnosis(source: Path, output: Path) -> Path:
    out = new_run(output, 'exit-horizon-diagnosis', {'source': str(source),
        'source_files_sha256': sha256(source / 'files.json'), 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V51.md')),
        'horizon_minutes': HORIZON_MINUTES, 'periods': PERIODS, 'new_model_families': 1, 'trading_returns_evaluated': False})
    print(f'청산 시간 범위의 예측 진단: {out}', flush=True)
    try:
        rows, splits, ledger, targets, baseline = load_exit_horizon_inputs(source)
        for name, frame in rows.items():
            frame.to_parquet(out / f'{name}_used.parquet', index=False)
        ledger.to_parquet(out / 'inclusion.parquet', index=False)
        targets.to_parquet(out / 'horizon_labels.parquet', index=False)
        model, offsets, scores, thresholds, support = fit_exit_horizon(rows, baseline)
        validation = rows['diagnosis']
        predictions = validation[['end', 'episode_id', 'original_y_exit', 'y_exit']].copy()
        metrics, requests = {}, {}
        for name, values in scores.items():
            predictions[name] = values
            metrics[name] = management_metrics(validation.y_exit, values)
            if name in thresholds:
                requests[name], events = horizon_requests(validation, values, thresholds[name])
                events.to_parquet(out / f'{name}_event_detection.parquet', index=False)
        decision = exit_horizon_admission(metrics, requests)
        predictions.to_parquet(out / 'predictions.parquet', index=False)
        for name, value in [('model', model.to_dict()), ('offsets', {k: v.to_dict() for k, v in offsets.items()}),
            ('thresholds', thresholds), ('training_support', {**support, 'splits': splits}),
            ('metrics', metrics), ('requests', requests), ('decision', decision)]:
            save_json(out / f'{name}.json', value)
        save_json(out / 'summary.json', {'complete': True, 'original_features_and_other_targets_exact': True,
            'episode_intersection': 0, **decision})
        (out / 'REPORT.md').write_text('# 청산 시간 범위 학습 진단\n\n'
            + table(pd.DataFrame([{'model': k, **v} for k, v in metrics.items()]))
            + '\n\n' + table(pd.DataFrame([{'model': k, **v} for k, v in requests.items()]))
            + '\n\n동일 포지션의 15분 청산 목표를 직접 학습했다. 기존 점수도 같은 상반기 목표에 절편을 보정했다. '
            '양성 분과 사건·포지션 수를 구분하며 기존 특징과 시간 배제를 보존했다. 이미 관찰한 원본 상태의 진단이고 매매 수익성 결과는 아니다.\n')
        save_json(out / 'files.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file()})
        print(f'청산 범위 모형의 후속 연구 허용: {decision["exit_horizon_admitted"]}', flush=True)
    except Exception as error:
        save_json(out / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
