from __future__ import annotations

from pathlib import Path

from wonyotti_fr.common import new_run, save_json, sha256
from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.holding_stress import seal
from wonyotti_fr.position_boost import PositionBoostModel
from wonyotti_fr.position_boost_evaluation import position_boost_admission
from wonyotti_fr.position_boost_reference import (
    checked_position_reference,
    checked_runtime_sources,
    position_baseline,
    position_comparisons,
    read,
)
from wonyotti_fr.source_snapshot import copy_snapshot


def run_position_boost(reference, expected, output):
    spec, files = checked_position_reference(reference, expected)
    source = Path(spec['source'])
    out = new_run(output, 'position-boost-diagnosis', {'reference': spec, 'reference_path': str(reference),
        'reference_sha256': expected, 'protocol_sha256': sha256(Path('docs/EXPERIMENT_V94.md')),
        'new_model_count': 1, 'request_threshold': .65, 'trading_returns_evaluated': False})
    print(out, flush=True)
    try:
        baseline = out/'baseline_source'
        baseline.mkdir()
        for name, digest in files.items():
            copy_snapshot(source/name, baseline/name)
            if sha256(baseline/name) != digest:
                raise ValueError('보유 방향 부스팅의 기존 사본 불일치')
        save_json(out/'baseline_files.json', files)
        save_json(out/'reference_review.json', read(Path(spec['review'])/'verification.json'))
        train, validation, scores = position_baseline(source)
        x, y, vx = train[MARKET_FEATURES].to_numpy(), train.position_target.to_numpy(), validation[MARKET_FEATURES].to_numpy()
        model, support = PositionBoostModel.fit(x, y, vx)
        scores['boosted'] = model.probabilities(vx)
        predictions, metrics, monthly = position_comparisons(validation, scores)
        decision = position_boost_admission(metrics, monthly)
        for name, value in [('model', model.to_dict()), ('training_support', support), ('metrics', metrics),
            ('monthly', monthly), ('decision', decision)]:
            save_json(out/f'{name}.json', value)
        predictions.to_parquet(out/'predictions.parquet', index=False)
        save_json(out/'summary.json', {'complete': True, 'new_models_fitted': 1,
            'training_rows': len(train), 'diagnosis_rows': len(validation), 'prior_states_and_ledgers_verification_reused': True,
            'original_baseline_scores_and_metrics_recomputed': True, **decision})
        current, after_files = checked_position_reference(reference, expected)
        if current != spec or after_files != files:
            raise ValueError('보유 방향 부스팅 실행 중 기존 출처 변경')
        checked_runtime_sources(out)
        seal(out)
    except Exception as error:
        save_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    return out
