from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .account_management import (
    ACCOUNT_FEATURES,
    AccountManagementLinear,
    AccountManagementModel,
    account_admission,
    account_split,
    evaluation_masks,
    subset_metrics,
)
from .account_management_reference import (
    independent_account_models,
    independent_admission,
    verify_account_split,
    verify_subset,
)
from .collateral_diagnostics import checked_reference
from .common import new_run, save_json, sha256
from .holding_stress import read_json, seal
from .minute_management import ACTIONS, FEATURES
from .paired_management import PairedManagementModel
from .paired_management_evaluation import choose_thresholds
from .paired_management_reference import manual_threshold
from .position_boost_verification import compare_statistics
from .request_execution_research import checked_parent


def run(reference, reference_sha256, account_proof, account_proof_sha256, protocol, output, progress=print):
    spec = checked_reference(reference, reference_sha256)
    frozen = {**spec['sha256'], str(reference): reference_sha256, str(protocol): sha256(protocol)}
    parent = checked_parent(account_proof, account_proof_sha256, frozen)
    public = {str(p): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    frozen.update(public)
    root = new_run(output, 'account-management-model', {'parent': str(parent),
        'protocol_sha256': frozen[str(protocol)], 'features': ACCOUNT_FEATURES,
        'model_fitted': True, 'profitability_accepted': False, 'all_periods_already_observed': True})
    save_json(root/'frozen_inputs.json', frozen)
    (root/'protocol.md').write_bytes(protocol.read_bytes())
    progress(str(root))
    try:
        source = pd.read_parquet(parent/'all_windows.parquet')
        rows, support = account_split(source)
        verified = verify_account_split(source, rows, support, read_json(parent/'summary.json'))
        if not support['supported']:
            raise ValueError('계좌 크기 관리의 학습·보정·진단 표본 부족')
        for part, frame in rows.items():
            frame.to_parquet(root/f'{part}_used.parquet', index=False)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(root/f'{part}_used.parquet'), check_exact=True)
        save_json(root/'support.json', support)
        train, cal = rows['training'].copy(), rows['calibration'].copy()
        validation = pd.concat([cal, rows['diagnosis']], ignore_index=True)
        for action in ACTIONS:
            train['y_'+action], cal['y_'+action] = train['all_y_'+action], cal['all_y_'+action]
        x = np.ascontiguousarray(train[ACCOUNT_FEATURES].to_numpy(dtype=float))
        vx = np.ascontiguousarray(validation[ACCOUNT_FEATURES].to_numpy(dtype=float))
        y = train[['y_'+a for a in ACTIONS]].to_numpy(dtype=int)
        vy = validation[['all_y_'+a for a in ACTIONS]].to_numpy(dtype=int)
        progress('공동 지원·기간 분리·모든 기존 열 검산 완료')
        account, account_fit = AccountManagementModel.fit(x, y, vx)
        baseline, baseline_fit = PairedManagementModel.fit(np.ascontiguousarray(x[:, :len(FEATURES)]), y,
            np.ascontiguousarray(vx[:, :len(FEATURES)]))
        linear, _, linear_fit = AccountManagementLinear.fit(train, cal)
        models = {'account': account, 'baseline': baseline, 'linear': linear}
        data = {name: model.to_dict() for name, model in models.items()}
        save_json(root/'models.json', data)
        saved_models = read_json(root/'models.json')
        for name, model in models.items():
            assert saved_models[name] == model.to_dict()
            models[name] = type(model).from_dict(saved_models[name])
        save_json(root/'fit_support.json', {'account': account_fit, 'baseline': baseline_fit, 'linear': linear_fit})
        reference_scores = independent_account_models(data, x, y, vx)
        predictions = {}
        for name, model in models.items():
            values = np.ascontiguousarray(vx[:, :len(model.features)])
            predictions[name] = model.probabilities(values)
            np.testing.assert_allclose(predictions[name], reference_scores[name], atol=1e-12, rtol=0)
            scalars = np.vstack([model.probabilities(row[None, :]) for row in values])
            np.testing.assert_allclose(scalars, predictions[name], atol=1e-12, rtol=0)
            progress(name+' 전체·단일 상태 추론 검산 완료')
        predictions['constant'] = np.tile(y.mean(axis=0), (len(vx), 1))
        np.testing.assert_array_equal(predictions['constant'], reference_scores['constant'])
        thresholds, details = choose_thresholds(vy[:len(cal)], {k: v[:len(cal)] for k, v in predictions.items()})
        for name, scores in reference_scores.items():
            for i, action in enumerate(ACTIONS):
                threshold, detail = manual_threshold(vy[:len(cal), i], scores[:len(cal), i], (2., 1., .5)[i])
                assert threshold == thresholds[name][action]
                compare_statistics(details[name][action], detail)
        save_json(root/'thresholds.json', {'thresholds': thresholds, 'selection': details})
        saved = validation[['end', 'state_episode_id', 'paired_part', 'account_band', 'futures_leverage',
            'log_futures_leverage', *['all_y_'+a for a in ACTIONS]]].copy()
        for name, scores in predictions.items():
            for i, action in enumerate(ACTIONS):
                saved[f'{name}_{action}_score'] = scores[:, i]
                saved[f'{name}_{action}_request'] = False
            saved[name+'_chosen'] = 'hold'
        groups, metrics = [], {}
        for key, mask in evaluation_masks(validation):
            result, decisions = subset_metrics(validation, mask, vy, predictions, thresholds)
            verify_subset(validation, key, mask, result, decisions, vy, reference_scores, thresholds)
            groups.append({**key, **result})
            if key['kind'] == 'period' and key['band'] == 'all':
                metrics[key['group']] = result['metrics']
                for name in predictions:
                    for i, action in enumerate(ACTIONS):
                        saved.loc[mask, f'{name}_{action}_request'] = decisions[name]['requests'][:, i]
                    saved.loc[mask, name+'_chosen'] = decisions[name]['chosen']
        decision = account_admission(metrics['diagnosis'])
        assert decision == independent_admission(metrics['diagnosis'])
        saved.to_parquet(root/'predictions.parquet', index=False)
        pd.testing.assert_frame_equal(saved, pd.read_parquet(root/'predictions.parquet'), check_exact=True)
        for filename, result in [('metrics', metrics), ('groups', groups), ('decision', decision)]:
            save_json(root/f'{filename}.json', result)
        progress('월·크기 구간별 전체 지표·선택·17개 판정 검산 완료')
        verified.update(independent_trees_refitted=384, linear_parameters_exact=True,
            prediction_rows_verified=len(validation), single_state_predictions_verified=len(validation)*3,
            groups_verified=len(groups), thresholds_metrics_decisions_verified=True, roundtrip_exact=True,
            originals_unchanged=5, production_sources_unchanged=len(public), frozen_inputs_unchanged=len(frozen))
        for name, digest in frozen.items():
            if Path(name).is_symlink() or sha256(Path(name)) != digest:
                raise ValueError('계좌 크기 관리 실행 중 입력 변경: '+name)
        checked_reference(reference, reference_sha256)
        save_json(root/'verification.json', verified)
        save_json(root/'summary.json', {'complete': True, 'support': support, 'metrics': metrics,
            'decision': decision, 'verification': verified, 'model_fitted': True, 'profitability_accepted': False,
            'trading_returns_evaluated': False})
        seal(root)
        return root
    except Exception as error:
        save_json(root/'failure.json', {'complete': False, 'type': type(error).__name__, 'error': str(error)})
        seal(root)
        raise


def main():
    parser = argparse.ArgumentParser(description='동일 관리 표본의 계좌 크기 입력 학습 비교')
    for name in ('reference', 'account-proof'):
        parser.add_argument('--'+name, required=True, type=Path)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', default=Path('artifacts'), type=Path)
    print('완료:', run(**vars(parser.parse_args())), flush=True)


if __name__ == '__main__':
    main()
