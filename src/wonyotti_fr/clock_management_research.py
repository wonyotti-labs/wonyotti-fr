from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .clock_management import (
    ClockManagementLinear,
    ClockManagementModel,
    clock_admission,
    extend_clock_frame,
)
from .clock_management_reference import (
    independent_admission,
    independent_clock_models,
    verify_clock_frame,
)
from .collateral_diagnostics import checked_reference
from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .minute_management import ACTIONS, FEATURES
from .paired_management import PairedManagementModel, split_support
from .paired_management_evaluation import choose_thresholds, evaluate
from .paired_management_reference import manual_metrics, manual_threshold
from .position_boost_verification import compare_statistics
from .request_execution_research import checked_parent


def run(reference, reference_sha256, paired_proof, paired_proof_sha256, protocol, output, progress=print):
    spec = checked_reference(reference, reference_sha256)
    frozen = {**spec['sha256'], str(reference): reference_sha256, str(protocol): sha256(protocol)}
    parent = checked_parent(paired_proof, paired_proof_sha256, frozen)
    public = {str(p): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    frozen.update(public)
    root = new_run(output, 'clock-management', {'parent': str(parent), 'protocol_sha256': frozen[str(protocol)],
        'features': ClockManagementModel.features, 'model_fitted': True, 'profitability_accepted': False,
        'all_periods_already_observed': True})
    save_json(root/'frozen_inputs.json', frozen)
    (root/'protocol.md').write_bytes(protocol.read_bytes())
    progress(str(root))
    try:
        original = pd.read_parquet(parent/'all_windows.parquet')
        frame = extend_clock_frame(original)
        verified = verify_clock_frame(original, frame)
        rows, support = split_support(frame)
        if not support['supported'] or support != read_json(parent/'support.json'):
            raise ValueError('주기 관리의 원래 학습 지원 불일치')
        for part, values in rows.items():
            pd.testing.assert_frame_equal(values[original.columns], pd.read_parquet(parent/f'{part}_used.parquet'), check_exact=True)
        frame.to_parquet(root/'all_windows.parquet', index=False)
        pd.testing.assert_frame_equal(frame, pd.read_parquet(root/'all_windows.parquet'), check_exact=True)
        save_json(root/'support.json', support)
        train = rows['training'].copy()
        cal = rows['calibration'].copy()
        validation = pd.concat([cal, rows['diagnosis']], ignore_index=True)
        for action in ACTIONS:
            train['y_'+action] = train['all_y_'+action]
            cal['y_'+action] = cal['all_y_'+action]
        features = ClockManagementModel.features
        x = np.ascontiguousarray(train[features].to_numpy(dtype=float))
        y = train[['y_'+a for a in ACTIONS]].to_numpy(dtype=int)
        vx = np.ascontiguousarray(validation[features].to_numpy(dtype=float))
        vy = validation[['all_y_'+a for a in ACTIONS]].to_numpy(dtype=int)
        cy = vy[:len(cal)]
        progress('전체 주기 입력·기존 행·지원 검산 완료')
        clock, clock_fit = ClockManagementModel.fit(x, y, vx)
        linear, _, linear_fit = ClockManagementLinear.fit(train, cal)
        model_data = {'clock': clock.to_dict(), 'linear': linear.to_dict()}
        save_json(root/'models.json', model_data)
        save_json(root/'fit_support.json', {'clock': clock_fit, 'linear': linear_fit})
        reference_scores = independent_clock_models(model_data['clock'], model_data['linear'], x, y, vx)
        predictions = {'clock': clock.probabilities(vx), 'linear': linear.probabilities(vx)}
        for name, model in [('clock', clock), ('linear', linear)]:
            np.testing.assert_allclose(predictions[name], reference_scores[name], atol=1e-12, rtol=0)
            scalar = np.vstack([model.probabilities(row[None, :]) for row in vx])
            np.testing.assert_allclose(scalar, predictions[name], atol=1e-12, rtol=0)
        progress('192개 트리·선형 계수·전체 점수와 단일 상태 추론 검산 완료')
        prior_predictions = pd.read_parquet(parent/'predictions.parquet')
        pd.testing.assert_frame_equal(validation[['end', 'state_episode_id', 'paired_part']],
            prior_predictions[['end', 'state_episode_id', 'paired_part']], check_exact=True)
        baseline = PairedManagementModel.from_dict(read_json(parent/'models.json')['all'])
        predictions['baseline'] = baseline.probabilities(validation[FEATURES].to_numpy(dtype=float))
        predictions['constant'] = np.tile(y.mean(axis=0), (len(vx), 1))
        for name, source in [('baseline', 'all'), ('constant', 'constant')]:
            np.testing.assert_array_equal(predictions[name], prior_predictions[[f'{source}_{a}_score' for a in ACTIONS]])
            reference_scores[name] = predictions[name].copy()
        thresholds, details = choose_thresholds(cy, {k: v[:len(cal)] for k, v in predictions.items()})
        old_thresholds = read_json(parent/'thresholds.json')['thresholds']
        if thresholds['baseline'] != old_thresholds['all'] or thresholds['constant'] != old_thresholds['constant']:
            raise ValueError('주기 관리의 고정 대조 문턱 불일치')
        for name, scores in reference_scores.items():
            for i, action in enumerate(ACTIONS):
                threshold, detail = manual_threshold(cy[:, i], scores[:len(cal), i], (2., 1., .5)[i])
                assert threshold == thresholds[name][action]
                compare_statistics(details[name][action], detail)
        save_json(root/'thresholds.json', {'thresholds': thresholds, 'selection': details})
        saved = validation[['end', 'state_episode_id', 'paired_part', *['all_y_'+a for a in ACTIONS]]].copy()
        for name, scores in predictions.items():
            for i, action in enumerate(ACTIONS):
                saved[f'{name}_{action}_score'] = scores[:, i]
                saved[f'{name}_{action}_request'] = False
            saved[name+'_chosen'] = 'hold'
        metrics, monthly = {}, []
        def checked_metrics(mask):
            result, decisions = evaluate(vy[mask], {k: v[mask] for k, v in predictions.items()}, thresholds)
            for name, reference_ in reference_scores.items():
                for i, action in enumerate(ACTIONS):
                    compare_statistics(result[action][name], manual_metrics(vy[mask, i], reference_[mask, i], thresholds[name][action], (2., 1., .5)[i]))
                    np.testing.assert_array_equal(decisions[name]['requests'][:, i], reference_[mask, i] >= thresholds[name][action])
                chosen = [next((a for i, a in enumerate(ACTIONS) if row[i] >= thresholds[name][a]), 'hold') for row in reference_[mask]]
                np.testing.assert_array_equal(decisions[name]['chosen'], chosen)
            return result, decisions
        for part in ('calibration', 'diagnosis'):
            mask = validation.paired_part.eq(part).to_numpy()
            result, decisions = checked_metrics(mask)
            metrics[part] = result
            for name in predictions:
                for i, action in enumerate(ACTIONS):
                    saved.loc[mask, f'{name}_{action}_request'] = decisions[name]['requests'][:, i]
                saved.loc[mask, name+'_chosen'] = decisions[name]['chosen']
        month_keys = validation.end.dt.strftime('%Y-%m')
        for month in sorted(month_keys.unique()):
            result, _ = checked_metrics(month_keys.eq(month).to_numpy())
            monthly.append({'month': month, 'metrics': result})
        decision = clock_admission(metrics['diagnosis'])
        assert decision == independent_admission(metrics['diagnosis'])
        saved.to_parquet(root/'predictions.parquet', index=False)
        pd.testing.assert_frame_equal(saved, pd.read_parquet(root/'predictions.parquet'), check_exact=True)
        for filename, result in [('metrics', metrics), ('monthly', monthly), ('decision', decision)]:
            save_json(root/f'{filename}.json', result)
        verified.update(independent_trees_refitted=192, linear_parameters_exact=True,
            prediction_rows_verified=len(validation), single_state_predictions_verified=len(validation)*2,
            thresholds_metrics_months_and_decision_verified=True, prior_predictions_and_thresholds_exact=True,
            originals_unchanged=5, production_sources_unchanged=len(public))
        for name, digest in frozen.items():
            if Path(name).is_symlink() or sha256(Path(name)) != digest:
                raise ValueError('주기 관리 실행 중 입력 변경: '+name)
        checked_reference(reference, reference_sha256)
        save_json(root/'verification.json', verified)
        save_json(root/'summary.json', {'complete': True, 'support': support, 'metrics': metrics,
            'decision': decision, 'verification': verified, 'model_fitted': True, 'profitability_accepted': False})
        seal(root)
        progress('전체 기간·월별·고정 대조·판정 검산 완료')
        return root
    except Exception as error:
        save_json(root/'failure.json', {'complete': False, 'type': type(error).__name__, 'error': str(error)})
        seal(root)
        raise


def main():
    parser = argparse.ArgumentParser(description='같은 관리 정답의 UTC 하루·주간 주기 학습 비교')
    for name in ('reference', 'paired-proof'):
        parser.add_argument('--'+name, required=True, type=Path)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', default=Path('artifacts'), type=Path)
    print('완료:', run(**vars(parser.parse_args())), flush=True)


if __name__ == '__main__':
    main()
