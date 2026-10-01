from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .common import new_run, save_json, sha256
from .event_features import purged_train, training_events
from .event_model import EventModel


def evaluate_imitation(model: EventModel, data: pd.DataFrame) -> dict:
    if data.empty:
        return {'rows': 0, 'balanced_accuracy': None, 'status': 'no_samples',
                'reason': '해당 포지션 상태의 평가 표본 없음'}
    predicted = model.predict(data)
    labels = sorted(set(model.classes) | set(data.target))
    report = classification_report(data.target, predicted, labels=labels, output_dict=True, zero_division=0)
    observed = [label for label in labels if report[label]['support'] > 0]
    return {'rows': len(data), 'balanced_accuracy': sum(report[label]['recall'] for label in observed) / len(observed),
            'labels': labels, 'missing_target_classes': [label for label in labels if label not in observed],
            'insufficient_class_support': any(report[label]['support'] < 20 for label in labels),
            'confusion_matrix': confusion_matrix(data.target, predicted, labels=labels).tolist(),
            'classification': report}


def run_event_study(audit_run: Path, history: Path, output: Path) -> Path:
    manifest = json.loads((history / 'manifest.json').read_text())
    path = (history / manifest['file']).resolve()
    if not path.is_relative_to(history.resolve()) or sha256(path) != manifest['sha256']:
        raise ValueError('원거래소 시세 경로 또는 무결성 오류')
    names = ['actions.parquet', 'xbtusd_events.parquet', 'episodes.parquet', 'executions.parquet']
    destination = new_run(output, 'event-study', {
        'audit_sha256': {name: sha256(audit_run / name) for name in names},
        'history_manifest_sha256': sha256(history / 'manifest.json'),
        'train_end_exclusive': '2020-01-01', 'purge': '경계에 걸친 전체 에피소드와 마지막 24시간 제외',
        'temporal_check': '2018 학습, 2019 평가. 설정 재선택 없이 C=0.1 고정.',
        'protocol': 'docs/EXPERIMENT_V2.md', 'next_step': '2020에서 정책 선택 후 2021과 후속 기간 평가',
    })
    print(f'사건별 연구 시작: {destination}', flush=True)
    try:
        frames = {name: pd.read_parquet(audit_run / name) for name in names}
        data, diagnostics = training_events(pd.read_parquet(path), frames['actions.parquet'],
                                             frames['xbtusd_events.parquet'], frames['executions.parquet'])
        data.to_parquet(destination / 'training_events.parquet', index=False)
        training = purged_train(data, frames['episodes.parquet'], '2020-01-01')
        early = purged_train(data, frames['episodes.parquet'], '2019-01-01')
        # 검증 첫날도 비워 경계를 가로지르는 관측의 영향을 줄인다.
        check = training[training.end >= '2019-01-02']
        temporal = {}
        for name, management in [('entry', False), ('management', True)]:
            def choose(frame, active=management):
                return frame[frame.direction.ne(0) if active else frame.direction.eq(0)]
            selected = choose(training)
            model = EventModel.fit(selected, management)
            save_json(destination / f'{name}_model.json', model.to_dict())
            early_model = EventModel.fit(choose(early), management)
            temporal[name] = evaluate_imitation(early_model, choose(check))
            diagnostics[f'{name}_train_rows'] = len(selected)
            diagnostics[f'{name}_class_counts'] = {str(k): int(v) for k, v in selected.target.value_counts().items()}
            pd.DataFrame(model.coefficients.T, index=model.features,
                         columns=model.classes if len(model.classes) > 2 else ['binary_logit']).to_csv(destination / f'{name}_coefficients.csv')
        save_json(destination / 'diagnostics.json', diagnostics)
        save_json(destination / 'temporal_2019.json', temporal)
        save_json(destination / 'files.json', {p.name: sha256(p) for p in destination.iterdir() if p.is_file()})
        (destination / 'REPORT.md').write_text(
            '# 독립 주문 기반 사건별 학습\n\n'
            f'전체 {diagnostics["bars"]:,}개 봉 중 {diagnostics["usable_bars"]:,}개를 연결했다. '
            f'독립 주문은 {diagnostics["independent_orders"]:,}개다.\n\n'
            f'한 창에 여러 독립 주문이 있는 경우는 {diagnostics["multiple_order_windows"]:,}개다. '
            '첫 주문만 정답으로 사용하며 5분보다 빠른 후속 판단은 이 모델이 설명하지 못한다. '
            f'관측한 직전 포지션과 맞지 않는 창 {diagnostics["incompatible_state_windows"]}개는 제외했다.\n\n'
            '부분 체결은 새 주문으로 중복 학습하지 않았다. 최초 체결 시각을 사용하므로 주문 제출·취소 시각은 복원하지 못한다. '
            '2018~2019년의 학습과 2018→2019년 시간순 검증을 구분했다. 원거래소 시세는 특징에만 사용했다.\n\n'
            '소수의 주문을 학습하기 위해 클래스 가중치를 적용했다. 출력 확률을 수익 확률로 해석하면 안 된다. '
            '손실 거래도 보존했으며 최종 손익으로 입력·정답을 수정하지 않았다. '
            '매매 성과와 새 평가 자료의 채택 여부는 다음 단계에서 별도로 판정한다.\n', encoding='utf-8')
    except Exception as error:
        save_json(destination / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    print(f'사건별 자료와 모델 저장: {destination}', flush=True)
    return destination
