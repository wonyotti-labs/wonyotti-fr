from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import sha256
from .expansion_model import BinaryModel
from .position_prior import PRIOR_FILES, copy_position_parent, load_position_prior_selection
from .recent_entry import RecentEntryPolicy

DIRECTION_PERIOD = ['2018-01-01', '2022-01-01']
DIRECTION_FILES = ['new_position_direction.json', 'new_direction_training.json']


def position_direction_training(data, episodes):
    if (episodes.episode_id.duplicated().any()
        or not data.loc[data.active.eq(1), 'target_episode_id'].isin(episodes.episode_id).all()):
        raise ValueError('신규 방향 목표의 에피소드 원장 누락·중복')
    first, last = (pd.Timestamp(t, tz='UTC') for t in DIRECTION_PERIOD)
    crossing = [episodes.loc[episodes.entry_time.lt(t) & (episodes.exit_time.isna() | episodes.exit_time.ge(t)),
                             'episode_id'] for t in [first, last]]
    reason = np.select([
        data.end.lt(first + pd.Timedelta(days=1)), data.label_end.ge(last - pd.Timedelta(days=1)),
        data.episode_id.isin(crossing[0]) | data.target_episode_id.isin(crossing[0]),
        data.episode_id.isin(crossing[1]) | data.target_episode_id.isin(crossing[1]), ~data.usable, data.active.ne(1),
    ], ['before_training_or_left_embargo', 'after_training_or_right_embargo', 'left_episode_boundary',
        'right_episode_boundary', 'unusable_original_event', 'no_new_position'], default='included')
    ledger = data[['end', 'label_end', 'episode_id', 'target_episode_id', 'active']].assign(reason=reason)
    train = data[ledger.reason.eq('included')].copy()
    if (train.empty or not train.end.is_monotonic_increasing or train.end.duplicated().any()
        or not np.isin(train.buy, [0, 1]).all() or train.target_time.isna().any()
        or train.target_time.lt(train.end).any() or train.target_time.ge(train.label_end).any()):
        raise ValueError('실제 신규 방향 학습의 순서·정답·시각 오류')
    return train, ledger


def copy_prior_parent(reference, out):
    copy_position_parent(reference, out)
    for name in ['position_selection.json', *PRIOR_FILES]:
        (out / name).write_bytes((reference / name).read_bytes())
    (out / 'prior_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())


def load_position_direction_selection(selection, frozen):
    path = selection / 'prior_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['prior_selection_sha256']:
        raise ValueError('신규 방향 모형의 이전 선택 지문 오류')
    parent = json.loads(path.read_text())
    if (frozen.get('protocol') != 'position_direction_v27' or parent.get('protocol') != 'position_prior_v26'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or set(frozen['direction_files_sha256']) != set(DIRECTION_FILES)
        or any((selection / n).stat().st_size > 1024**2 or sha256(selection / n) != frozen['direction_files_sha256'][n]
               for n in DIRECTION_FILES)):
        raise ValueError('신규 방향 모형의 고정 설정·모델 지문 오류')
    _, original = load_position_prior_selection(selection, parent)
    metadata = json.loads((selection / 'new_direction_training.json').read_text())
    direction = BinaryModel.from_dict(json.loads((selection / 'new_position_direction.json').read_text()))
    if (metadata['training_period'] != DIRECTION_PERIOD or metadata['prior_offset_applied'] is not False
        or metadata['direction_threshold'] != .65 or direction.data['kind'] != 'logistic'
        or metadata['model']['rows'] < 1000 or min(metadata['model']['positive'], metadata['model']['negative']) < 20):
        raise ValueError('신규 방향 모형의 학습 범위·지원·직접 점수 설정 오류')
    return frozen, RecentEntryPolicy(original, original.base.activity, direction, original.base.activity_threshold)
