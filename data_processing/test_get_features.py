import pandas as pd
import pytest
from get_features import get_data


@pytest.fixture
def train_df():
    return pd.DataFrame({"generated": [0, 1, 0, 1]})


@pytest.fixture
def test_df():
    return pd.DataFrame({"generated": [1, 0, 1, 0]})


def test_get_data(train_df, test_df):
    train_df_idf, test_df_idf, y_train, y_test = get_data(train_df, test_df)

    assert isinstance(train_df_idf, pd.DataFrame)
    assert isinstance(test_df_idf, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    assert train_df_idf.shape[0] == train_df.shape[0]
    assert test_df_idf.shape[0] == test_df.shape[0]
    assert y_train.shape[0] == train_df.shape[0]
    assert y_test.shape[0] == test_df.shape[0]
