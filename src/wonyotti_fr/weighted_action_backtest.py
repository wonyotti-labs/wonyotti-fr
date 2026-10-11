from wonyotti_fr.market_change_policy import MarketChangePolicy
from wonyotti_fr.recorded_change_reference import check_snapshot, checked_files
from wonyotti_fr.sparse_exposure_backtest import backtest as sparse_backtest
from wonyotti_fr.sparse_exposure_backtest import verify_path as sparse_verify
from wonyotti_fr.weighted_action_model import WeightedActionModel
from wonyotti_fr.weighted_action_reference import ReferenceWeightedPolicy


def backtest(data, model, config, output, batch_size=8192, progress=None):
    if type(model) is not WeightedActionModel or config.allocation != .5:
        raise ValueError('행동 가중 실행의 모형·노출 한도 오류')
    policy = MarketChangePolicy(model)
    policy.kind = 'weighted_action_'+model.condition
    return sparse_backtest(data, policy, config, output, batch_size, progress)


def verify_path(data, raw, config, output, review, progress=None):
    model = WeightedActionModel.from_dict(raw)
    checked_files(output)
    if config.allocation != .5:
        raise ValueError('행동 가중 검산의 고정 노출 한도 오류')
    return sparse_verify(data, 'weighted_action_'+model.condition, raw, config, output, review, progress,
        snapshot_validator=check_snapshot, reference_factory=ReferenceWeightedPolicy)
