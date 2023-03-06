import pytest
from riks_ds_utils.main import aggregate_mean

@pytest.mark.parametrize("column, expected", [("feature_1", {0: 3, 1: 4})])
def test_aggregate_mean_feature_1(data, column, expected):
    assert expected == aggregate_mean(data, column)
