from wonyotti_fr.event_evaluation import evaluate_gate


def rows():
    return [{'symbol': symbol, 'strategy': 'fixed_policy', 'closed_trades': 30,
             'total_return': 0.1, 'max_drawdown': -0.1, 'permanent_halt': False}
            for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']]


def test_profitability_gate_requires_development_and_every_market():
    data = rows()
    assert evaluate_gate({'total_return': 0.1}, data)['profitability_review_candidate']
    assert not evaluate_gate({'total_return': -0.1}, data)['profitability_review_candidate']
    data[-1]['total_return'] = -0.01
    assert not evaluate_gate({'total_return': 0.1}, data)['profitability_review_candidate']
    data = rows()
    data[0]['closed_trades'] = 0
    assert not evaluate_gate({'total_return': 0.1}, data)['profitability_review_candidate']
    assert not evaluate_gate({'total_return': 0.1}, rows()[:2])['profitability_review_candidate']
    assert not evaluate_gate({'total_return': 0.1}, rows())['live_trading_approved']
