from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import new_run, records, save_json, sha256
from .event_features import purged_train
from .event_model import EventModel
from .event_study import evaluate_imitation
from .reports import table


def run_event_walkforward(study: Path, audit: Path, output: Path) -> Path:
    hashes = json.loads((study / 'files.json').read_text())
    data_path = study / 'training_events.parquet'
    study_manifest = json.loads((study / 'manifest.json').read_text())
    episodes_path = audit / 'episodes.parquet'
    if (sha256(data_path) != hashes[data_path.name]
        or sha256(episodes_path) != study_manifest['settings']['audit_sha256']['episodes.parquet']):
        raise ValueError('워크포워드 입력이 원래 학습 자료와 다릅니다.')
    destination = new_run(output, 'event-walkforward', {
        'study_files_sha256': sha256(study / 'files.json'), 'episodes_sha256': sha256(episodes_path),
        'folds': [2019, 2020, 2021], 'model': 'balanced logistic C=0.1, no search',
        'scope': '이미 관찰한 원본 기간의 확장 학습 모사 진단. 새 수익성 검증이 아님.',
        'purge': '학습 말일 24시간과 경계를 가로지르는 전체 포지션 제외',
    })
    data, episodes = pd.read_parquet(data_path), pd.read_parquet(episodes_path)
    rows, details = [], {}
    for year in [2019, 2020, 2021]:
        train = purged_train(data, episodes, f'{year}-01-01')
        test = data[data.usable & (data.end >= f'{year}-01-01') & (data.label_end < f'{year+1}-01-01')]
        for name, management in [('entry', False), ('management', True)]:
            training = train[train.direction.ne(0) if management else train.direction.eq(0)]
            evaluation = test[test.direction.ne(0) if management else test.direction.eq(0)]
            model = EventModel.fit(training, management)
            save_json(destination / f'{year}_{name}_model.json', model.to_dict())
            metrics = evaluate_imitation(model, evaluation)
            key = f'{year}/{name}'
            details[key] = metrics
            rows.append({'evaluation_year': year, 'model': name, 'training_rows': len(training),
                         'evaluation_rows': len(evaluation), 'balanced_accuracy': metrics['balanced_accuracy'],
                         'action_targets': int(evaluation.target.ne('hold').sum()),
                         'insufficient_class_support': metrics.get('insufficient_class_support', True)})
        print(f'{year} 확장 학습 모사 검증 완료', flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(destination / 'folds.csv', index=False)
    save_json(destination / 'metrics.json', details)
    save_json(destination / 'summary.json', {'folds': records(summary), 'completed': True,
                                             'retuned_after_evaluation': False})
    (destination / 'REPORT.md').write_text(
        '# 사건별 분류기의 확장 학습 워크포워드\n\n' + table(summary) + '\n\n'
        '2018→2019, 2018~2019→2020, 2018~2020→2021 순서로 모델을 새로 학습했다. '
        '각 경계에 걸친 포지션 에피소드와 학습 말일을 제외했다. 클래스 가중치와 정규화 방식·C 값은 고정했다. '
        '상세 클래스별 정밀도·재현율과 지원 표본은 metrics.json에 보존했다.\n\n'
        '이는 원자료 기간의 모사 안정성 진단이다. 해당 기간을 이미 관찰했으므로 새로운 미사용 평가나 '
        '수익률 워크포워드로 부르지 않는다. 정책 위험 설정을 매년 재선택하거나 2022년 이후 행동 정답을 만들지 않았다. '
        '진입 표본이 적거나 정답 클래스가 없는 해의 점수로 일반화를 주장하지 않는다.\n', encoding='utf-8')
    print(f'워크포워드 기록: {destination / "REPORT.md"}', flush=True)
    return destination
