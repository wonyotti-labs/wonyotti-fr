import json

import pandas as pd
import pytest
from test_portfolio_prices import response, spec

from wonyotti_fr.portfolio_prices import parse_price_response
from wonyotti_fr.portfolio_prices_review import independent_price_page


def test_independent_response_values_gaps_and_nan_match():
    request = spec(count=5)
    value = response(request)
    value[0]['close'] = None
    value[2]['close'] = -4
    del value[3]
    content = json.dumps(value).encode()
    got = pd.DataFrame(independent_price_page(content, request), columns=['symbol', 'end', 'close', 'support'])
    got['end'] = got.end.astype('datetime64[ns, UTC]')
    pd.testing.assert_frame_equal(got, parse_price_response(content, request), check_exact=True)


@pytest.mark.parametrize('kind', ['future', 'duplicate', 'naive_time', 'boolean_price', 'wrong_symbol'])
def test_independent_response_rejects_unsupported_values(kind):
    request = spec()
    value = response(request)
    if kind == 'future':
        value[-1]['timestamp'] = '2020-01-01T00:20:00+00:00'
    elif kind == 'duplicate':
        value[1] = value[0]
    elif kind == 'naive_time':
        value[0]['timestamp'] = '2020-01-01T00:05:00'
    elif kind == 'boolean_price':
        value[0]['close'] = True
    else:
        value[0]['symbol'] = 'OTHER'
    with pytest.raises(ValueError):
        independent_price_page(json.dumps(value).encode(), request)
