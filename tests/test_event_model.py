import copy
import json

import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.event_features import MARKET_FEATURES
from wonyotti_fr.event_model import EventModel


def test_portable_model_probabilities_and_invalid_input_hold():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(3000, len(MARKET_FEATURES)))
    data = pd.DataFrame(values, columns=MARKET_FEATURES)
    data['target'] = np.where(values[:, 0] > 0.7, 'enter_long',
                             np.where(values[:, 0] < -0.7, 'enter_short', 'hold'))
    model = EventModel.fit(data)
    restored = EventModel.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_array_equal(model.probabilities(values), restored.probabilities(values))
    assert (restored.predict(data) == data.target).mean() > 0.9
    np.testing.assert_allclose(model.probabilities(values).sum(axis=1), 1)
    data.loc[0, MARKET_FEATURES[0]] = np.nan
    assert model.predict(data)[0] == 'hold'
    corrupt = copy.deepcopy(model.to_dict())
    corrupt['scale'][0] = 0
    with pytest.raises(ValueError, match='수치 배열'):
        EventModel.from_dict(corrupt)
    corrupt = copy.deepcopy(model.to_dict())
    corrupt['features'][0] = 'future_profit'
    with pytest.raises(ValueError, match='모델 형식'):
        EventModel.from_dict(corrupt)
