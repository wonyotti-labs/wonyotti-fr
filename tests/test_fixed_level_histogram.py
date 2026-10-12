import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from threadpoolctl import threadpool_limits

from wonyotti_fr.first_state_models import FEATURES
from wonyotti_fr.fixed_level_histogram import FixedLevelHistogram
from wonyotti_fr.forward_final_reference import ReferenceFixedLevelHistogram
from wonyotti_fr.quiet_management import CLASSES


def learner(kind, **options):
    result = kind(**(CLASSES["increase"].fixed_settings | options))
    result._bin_mapper = _BinMapper(n_bins=256, random_state=0, n_threads=1)
    return result


def test_weighted_resampling_loses_rare_level_above_200000_rows():
    x = np.zeros((200001, len(FEATURES)))
    x[:-1, -1] = np.arange(len(x) - 1) % 9
    x[-1, -1] = 9
    w = np.ones(len(x))
    w[-1] = 0.1
    w *= len(w) / w.sum()
    regular = learner(HistGradientBoostingClassifier)
    fixed, reference = learner(FixedLevelHistogram), learner(ReferenceFixedLevelHistogram)
    with threadpool_limits(limits=1):
        bad = regular._bin_data(x, w, True)
        assert bad[-1, -1] == 8 and x[-1, -1] == 9
        assert (bad != x).sum() == 1
        np.testing.assert_array_equal(fixed._bin_data(x, w, True), x)
        np.testing.assert_array_equal(reference._bin_data(x, w, True), x)
        assert fixed._bin_mapper.subsample is reference._bin_mapper.subsample is None
        assert regular._bin_mapper.subsample == 200000
        np.testing.assert_array_equal(fixed._bin_data(x[-1:], None, False), x[-1:])
    assert _BinMapper().subsample == 200000


@pytest.mark.parametrize(
    "damage",
    [
        "nan",
        "negative",
        "fraction",
        "overflow",
        "missing_level",
        "shape",
        "zero_weight",
        "nonfinite_weight",
        "early",
        "max_bins",
        "categorical",
    ],
)
def test_invalid_encoded_levels_and_settings_rejected(damage):
    x, w, options = np.zeros((10, len(FEATURES))), np.ones(10), {}
    if damage == "nan":
        x[0, 0] = np.nan
    elif damage == "negative":
        x[0, 0] = -1
    elif damage == "fraction":
        x[0, 0] = 0.5
    elif damage == "overflow":
        x[0, 0] = 255
    elif damage == "missing_level":
        x[0, 0] = 2
    elif damage == "shape":
        x = x[:, :-1]
    elif damage == "zero_weight":
        w[0] = 0
    elif damage == "nonfinite_weight":
        w[0] = np.inf
    elif damage == "early":
        options["early_stopping"] = True
    elif damage == "max_bins":
        options["max_bins"] = 254
    else:
        options["categorical_features"] = [0]
    with pytest.raises(ValueError):
        learner(FixedLevelHistogram, **options)._bin_data(x, w, True)
