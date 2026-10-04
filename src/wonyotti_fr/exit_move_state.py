from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import save_json, sha256
from .episode_balance import verified_files
from .exit_horizon import load_exit_horizon_inputs
from .minute_management import purged_window
from .net_exit_state import NetExitPolicy, copy_horizon_parent, load_net_exit_selection

EXIT_MOVE_PERIOD = ['2020-01-01', '2021-01-01']
EXIT_MOVE_FILES = ['exit_move_floor.json', 'exit_move_events.parquet', 'exit_move_episodes.parquet']
EVENT_COLUMNS = ['end', 'entry_time', 'episode_id', 'favorable_move']


class ExitMovePolicy(NetExitPolicy):
    def __init__(self, parent, minimum_move):
        super().__init__(parent)
        if type(minimum_move) not in (int, float) or not np.isfinite(minimum_move) or minimum_move <= 0:
            raise ValueError('청산 가격 이동 하한의 숫자 오류')
        self.minimum_move = minimum_move

    def action_threshold(self, action, state):
        threshold = super().action_threshold(action, state)
        if action == 'exit':
            move = state.get('favorable_move')
            if type(move) not in (int, float) or not np.isfinite(move):
                raise ValueError('청산 가격 이동의 현재 값 오류')
            return threshold if move >= self.minimum_move else self.thresholds['exit']
        return threshold


def summarize_exit_moves(events):
    if (events.columns.tolist() != EVENT_COLUMNS or len(events) < 20
        or events.isna().any().any() or events.end.duplicated().any() or not events.end.is_monotonic_increasing
        or not isinstance(events.end.dtype, pd.DatetimeTZDtype) or str(events.end.dt.tz) != 'UTC'
        or not isinstance(events.entry_time.dtype, pd.DatetimeTZDtype) or str(events.entry_time.dt.tz) != 'UTC'
        or not np.isfinite(events[['episode_id', 'favorable_move']]).all().all()
        or events.episode_id.le(0).any() or events.episode_id.ge(2**53).any()
        or events.episode_id.ne(np.floor(events.episode_id)).any()
        or events.favorable_move.le(0).any() or events.entry_time.ge(events.end).any()
        or events.end.lt(pd.Timestamp(EXIT_MOVE_PERIOD[0], tz='UTC')+pd.Timedelta(days=1)).any()
        or events.end.ge(pd.Timestamp(EXIT_MOVE_PERIOD[1], tz='UTC')-pd.Timedelta(days=1)).any()):
        raise ValueError('청산 가격 이동 사건의 기간·지원·숫자 오류')
    episodes = events.groupby('episode_id', sort=True).favorable_move.agg(['median', 'size']).reset_index()
    if len(episodes) < 20 or events.groupby('episode_id').entry_time.nunique().gt(1).any():
        raise ValueError('청산 가격 이동의 독립 포지션 지원 오류')
    return {'format': 'positive_exit_move_median_v1', 'training_period': EXIT_MOVE_PERIOD,
        'method': 'median_of_episode_medians', 'threshold': float(episodes['median'].median()),
        'positive_events': len(events), 'positive_episodes': len(episodes)}, episodes


def prepare_exit_move(reference, diagnosis, original, out):
    files = verified_files(diagnosis)
    settings = json.loads((diagnosis/'manifest.json').read_text())['settings']
    previous = json.loads((reference/'horizon_exit_evidence.json').read_text())
    source = Path(settings['source'])
    if (settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V51.md'))
        or settings['horizon_minutes'] != 15 or settings['trading_returns_evaluated'] is not False
        or previous['diagnosis_files_sha256'] != sha256(diagnosis/'files.json')
        or previous['source_files_sha256'] != sha256(source/'files.json')):
        raise ValueError('청산 가격 이동의 기존 원본 연결 오류')
    rows, _, _, _, baseline = load_exit_horizon_inputs(source)
    train = rows['training']
    pd.testing.assert_frame_equal(train, pd.read_parquet(diagnosis/'training_used.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(train, purged_window(train, *EXIT_MOVE_PERIOD).reset_index(drop=True), check_exact=True)
    if (original.manager.model.to_dict() != baseline.model.to_dict()
        or original.manager.offset.to_dict() != baseline.offset.to_dict()
        or not train.original_y_exit.isin([0, 1]).all()
        or not np.isfinite(train.favorable_move).all()):
        raise ValueError('청산 가격 이동의 기존 모델·정답 오류')
    # 현재 양수 이동의 원본 청산 사건만 하한 요약에 쓰고 모델과 손실 사건은 보존한다.
    events = train.loc[train.original_y_exit.eq(1) & train.favorable_move.gt(0), EVENT_COLUMNS].reset_index(drop=True)
    choices, episodes = summarize_exit_moves(events)
    events.to_parquet(out/EXIT_MOVE_FILES[1], index=False)
    episodes.to_parquet(out/EXIT_MOVE_FILES[2], index=False)
    save_json(out/EXIT_MOVE_FILES[0], {**choices, 'training_used_sha256': files['training_used.parquet'],
        'source_files_sha256': sha256(source/'files.json'), 'diagnosis_files_sha256': sha256(diagnosis/'files.json'),
        'protocol_sha256': sha256(Path('docs/EXPERIMENT_V54.md')), 'new_models_fitted': False,
        'selected_by_profit': False, 'future_price_used': False})


def copy_net_parent(reference, out):
    copy_horizon_parent(reference, out)
    (out/'horizon_exit_selection.json').write_bytes((reference/'horizon_exit_selection.json').read_bytes())
    (out/'net_exit_rule.json').write_bytes((reference/'net_exit_rule.json').read_bytes())
    (out/'net_exit_selection.json').write_bytes((reference/'frozen_selection.json').read_bytes())


def load_exit_move_parent(selection, frozen):
    path = selection/'net_exit_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['net_exit_selection_sha256']:
        raise ValueError('청산 가격 이동의 이전 선택 지문 오류')
    return load_net_exit_selection(selection, json.loads(path.read_text()))


def load_exit_move_selection(selection, frozen):
    parent, original = load_exit_move_parent(selection, frozen)
    if (frozen.get('protocol') != 'exit_move_v54' or parent['protocol'] != 'net_exit_v53'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or set(frozen['exit_move_files_sha256']) != set(EXIT_MOVE_FILES)
        or any((selection/n).stat().st_size > 10*1024**2 or sha256(selection/n) != frozen['exit_move_files_sha256'][n]
               for n in EXIT_MOVE_FILES)):
        raise ValueError('청산 가격 이동의 고정 모델·위험·파일 오류')
    choices = json.loads((selection/EXIT_MOVE_FILES[0]).read_text())
    evidence = json.loads((selection/'horizon_exit_evidence.json').read_text())
    expected, episodes = summarize_exit_moves(pd.read_parquet(selection/EXIT_MOVE_FILES[1]))
    pd.testing.assert_frame_equal(episodes, pd.read_parquet(selection/EXIT_MOVE_FILES[2]), check_exact=True)
    if (any(choices.get(k) != v for k, v in expected.items())
        or choices['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V54.md'))
        or choices['source_files_sha256'] != evidence['source_files_sha256']
        or choices['diagnosis_files_sha256'] != evidence['diagnosis_files_sha256']
        or any(choices[k] is not False for k in ['new_models_fitted', 'selected_by_profit', 'future_price_used'])):
        raise ValueError('청산 가격 이동 하한의 원본·요약·계획 오류')
    return frozen, ExitMovePolicy(original, choices['threshold'])
