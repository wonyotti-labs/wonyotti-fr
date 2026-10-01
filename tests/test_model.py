import numpy as np
import pandas as pd
import pytest

from wonyotti_fr.features import FEATURES
from wonyotti_fr.model import DirectionModel


def training_data():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(1500, len(FEATURES))), columns=FEATURES)
    frame["label"] = np.where(frame.return_1h > 0.3, 1, np.where(frame.return_1h < -0.3, -1, 0))
    return frame


def test_portable_model_preserves_predictions():
    frame = training_data()
    model = DirectionModel.fit(frame)
    restored = DirectionModel.from_dict(model.to_dict())
    np.testing.assert_array_equal(model.signals(frame, 0.6), restored.signals(frame, 0.6))
    assert (model.signals(frame, 0) == frame.label).mean() > 0.9


def test_missing_features_never_create_orders():
    frame = training_data()
    model = DirectionModel.fit(frame)
    frame[FEATURES] = np.nan
    assert not model.signals(frame, 0).any()


def test_invalid_model_rejected():
    value = DirectionModel.fit(training_data()).to_dict()
    value["scale"][0] = 0
    with pytest.raises(ValueError):
        DirectionModel.from_dict(value)
