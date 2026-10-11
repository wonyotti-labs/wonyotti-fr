from wonyotti_fr.market_change_model import MarketChangeModel
from wonyotti_fr.market_change_policy import MarketChangePolicy
from wonyotti_fr.recorded_change_reference import check_snapshot, checked_files
from wonyotti_fr.sparse_exposure_backtest import backtest as sparse_backtest
from wonyotti_fr.sparse_exposure_backtest import verify_path as sparse_verify


def backtest(data, model, config, output, batch_size=8192, progress=None):
    if config.allocation != .5:
        raise ValueError('시장 행동 실행의 고정 노출 한도 오류')
    return sparse_backtest(data, MarketChangePolicy(model), config, output, batch_size, progress)


def verify_path(data, raw, config, output, review, progress=None):
    model = MarketChangeModel.from_dict(raw)
    checked_files(output)
    if config.allocation != .5:
        raise ValueError('시장 행동 검산의 고정 노출 한도 오류')
    # 검산기의 상태 값은 마지막 열에만 붙으며 모든 트리는 시장 열 0~13만 참조한다.
    return sparse_verify(data, 'market_only_execution_'+model.timing, raw, config, output, review,
        progress, snapshot_validator=check_snapshot)
