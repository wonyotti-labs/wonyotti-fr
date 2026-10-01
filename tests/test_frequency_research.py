from wonyotti_fr.frequency_research import frequency_gate


def test_new_data_gate_requires_all_predeclared_conditions():
    development = {'total_return': 0.1}
    validation = {'total_return': 0.1, 'closed_trades': 20, 'permanent_halt': False}
    assert frequency_gate(development, validation)['may_open_new_period']
    assert not frequency_gate({'total_return': 0}, validation)['may_open_new_period']
    for changed in [{'total_return': 0}, {'closed_trades': 19}, {'permanent_halt': True}]:
        assert not frequency_gate(development, {**validation, **changed})['may_open_new_period']
    assert not frequency_gate(development, validation)['live_trading_approved']
