import json

import pandas as pd
import pytest

from wonyotti_fr.event_diagnostics import decompose_run


def test_saved_accounting_is_reconciled_independently(tmp_path):
    pd.DataFrame({'equity': [10000, 10008], 'halted': [False, False], 'quantity': [1, 0],
                  'time': ['2020-01-01T00:05:00+00:00', '2020-01-01T00:10:00+00:00']}).to_parquet(tmp_path / 'equity.parquet')
    pd.DataFrame({'net_pnl': [8], 'gross_realized': [10], 'fees': [2], 'funding_cost': [0],
                  'hold_bars': [1], 'adds': [0]}).to_parquet(tmp_path / 'trades.parquet')
    pd.DataFrame({'delta_quantity': [1, -1], 'price': [100, 110],
                  'reason': ['entry', 'signal_exit']}).to_parquet(tmp_path / 'fills.parquet')
    (tmp_path / 'metrics.json').write_text(json.dumps({'fees': 2}))
    (tmp_path / 'final_state.json').write_text(json.dumps({'completed': True, 'quantity': 0}))
    result = decompose_run(tmp_path, 10000)
    assert result['net_pnl'] == 8 and result['turnover_over_initial_equity'] == 0.021
    (tmp_path / 'metrics.json').write_text(json.dumps({'fees': 3}))
    with pytest.raises(ValueError, match='일치하지'):
        decompose_run(tmp_path, 10000)
