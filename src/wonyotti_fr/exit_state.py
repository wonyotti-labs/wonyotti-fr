from __future__ import annotations

import json
from pathlib import Path

from .activity_ablation import copy_holding_parent, load_activity_ablation_selection
from .common import save_json, sha256
from .exit_direct import MODEL_FILES, exit_direct_admission
from .first_state import FirstStatePolicy
from .inventory_labels import verify_files

EXIT_POLICY_FILES = ['exit_threshold.json', 'exit_admission.json']


class ExitStatePolicy(FirstStatePolicy):
    def __init__(self, parent):
        super().__init__(parent, parent.first_thresholds)

    def action_threshold(self, action, state):
        if action == 'exit':
            return self.thresholds['exit']
        return super().action_threshold(action, state)


def validate_exit_admission(evidence):
    decision = evidence['decision']
    expected = exit_direct_admission(evidence['metrics'], decision['unchanged_masks'])
    settings, summary = evidence['settings'], evidence['summary']
    if (not expected['exit_direct_admitted'] or decision != expected
        or any(summary.get(k) != v for k, v in expected.items()) or summary.get('complete') is not True
        or summary.get('episode_intersection') != 0 or summary.get('new_models_fitted') is not False
        or settings['protocol_sha256'] != sha256(Path('docs/EXPERIMENT_V47.md'))
        or settings.get('exit_multiplier') != 1. or settings.get('original_multiplier') != 1.5
        or settings.get('new_models_fitted') is not False
        or settings.get('calibration_period') != ['2021-01-01', '2021-07-01']
        or settings.get('diagnosis_period') != ['2021-07-01', '2022-01-01']):
        raise ValueError('직접 청산 실행의 원본 진단 통과 근거 오류')
    for action, pairs in evidence['metrics'].items():
        if set(pairs) != {'original', 'direct'}:
            raise ValueError('직접 청산 진단의 비교 오류')
        counts = set()
        for metric in pairs.values():
            n, positive = metric['rows'], metric['positive']
            if (type(n) is not int or type(positive) is not int or min(positive, n-positive) < 20
                or n != summary['diagnosis_rows']):
                raise ValueError('직접 청산 진단의 지원 부족')
            counts.add((n, positive))
        if len(counts) != 1 or (action != 'exit' and pairs['original'] != pairs['direct']):
            raise ValueError('직접 청산 진단의 다른 행동 변경')
    if set(evidence['metrics']) != {'exit', 'reduce', 'increase'}:
        raise ValueError('직접 청산 진단의 행동 누락')


def copy_activity_parent(reference, out):
    copy_holding_parent(reference, out)
    (out / 'holding_selection.json').write_bytes((reference / 'holding_selection.json').read_bytes())
    (out / 'activity_selection.json').write_bytes((reference / 'frozen_selection.json').read_bytes())


def prepare_exit_evidence(reference, diagnosis, out):
    files = json.loads((diagnosis / 'files.json').read_text())
    required = {'manifest.json', 'summary.json', 'decision.json', 'metrics.json', 'exit_threshold.json', *MODEL_FILES}
    if (not required <= set(files)
        or any(not (diagnosis / n).resolve().is_relative_to(diagnosis.resolve()) for n in files)):
        raise ValueError('직접 청산 실행의 진단 파일·경로 오류')
    verify_files(diagnosis, list(files))
    evidence = {name: json.loads((diagnosis / f'{name}.json').read_text())
                for name in ['summary', 'decision', 'metrics']}
    evidence.update(settings=json.loads((diagnosis / 'manifest.json').read_text())['settings'],
        files_sha256=sha256(diagnosis / 'files.json'), model_files_sha256={n: files[n] for n in MODEL_FILES})
    validate_exit_admission(evidence)
    if (evidence['settings']['selection_sha256'] != sha256(reference / 'frozen_selection.json')
        or any((diagnosis / n).read_bytes() != (reference / n).read_bytes() for n in MODEL_FILES)):
        raise ValueError('직접 청산 실행의 모델·부모 연결 오류')
    (out / 'exit_threshold.json').write_bytes((diagnosis / 'exit_threshold.json').read_bytes())
    save_json(out / 'exit_admission.json', evidence)


def load_exit_parent(selection, frozen):
    path = selection / 'activity_selection.json'
    if path.stat().st_size > 1024**2 or sha256(path) != frozen['activity_selection_sha256']:
        raise ValueError('직접 청산 실행의 이전 선택 지문 오류')
    return load_activity_ablation_selection(selection, json.loads(path.read_text()))


def load_exit_state_selection(selection, frozen):
    parent, original = load_exit_parent(selection, frozen)
    if (frozen.get('protocol') != 'exit_state_v48' or parent['protocol'] != 'activity_ablation_v46'
        or any(frozen.get(k) != v for k, v in parent.items() if k not in {'protocol', 'development_metrics'})
        or frozen.get('exit_protocol_sha256') != sha256(Path('docs/EXPERIMENT_V48.md'))
        or set(frozen['exit_files_sha256']) != set(EXIT_POLICY_FILES)
        or any((selection / n).stat().st_size > 1024**2 or sha256(selection / n) != frozen['exit_files_sha256'][n]
               for n in EXIT_POLICY_FILES)):
        raise ValueError('직접 청산 실행의 고정 모델·문턱·위험 오류')
    evidence = json.loads((selection / 'exit_admission.json').read_text())
    validate_exit_admission(evidence)
    value = json.loads((selection / 'exit_threshold.json').read_text())
    old = json.loads((selection / 'history_thresholds.json').read_text())
    if (evidence['settings']['selection_sha256'] != frozen['activity_selection_sha256']
        or set(evidence['model_files_sha256']) != set(MODEL_FILES)
        or any(sha256(selection / n) != evidence['model_files_sha256'][n] for n in MODEL_FILES)
        or value['exit_multiplier'] != 1. or value['original_multiplier'] != 1.5 or original.multiplier != 1.5
        or value['threshold'] != original.thresholds['exit'] or value['support'] != old['support']['exit']
        or value['support']['actual_positive'] < 20 or value['support']['predicted_positive'] < 20
        or value['minimum_predicted_positive'] != 20 or value['beta'] != 2.
        or value['calibration_period'] != ['2021-01-01', '2021-07-01']):
        raise ValueError('직접 청산 실행의 기존 모델·상반기 문턱 변경')
    return frozen, ExitStatePolicy(original)
