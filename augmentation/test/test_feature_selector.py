import pytest

from random import choices, shuffle
import numpy as np
import pandas as pd

from augmentation.feature_selector import FeatureSelector


@pytest.fixture()
def feature_selector():
    return FeatureSelector()


@pytest.fixture()
def external_table_dict():
    categoric_values = ["a", "b", "c", "d"]
    external_table_dict = {}
    for table in range(6):
        query_column = np.hstack([np.arange(15), np.arange(10)])
        shuffle(query_column)
        categorical_column = np.hstack([choices(categoric_values, k=10), [None] * 15])
        float_column_1 = np.hstack([np.random.rand(10), [None] * 15])
        float_column_2 = np.hstack([np.random.randn(10), [None] * 15])
        int_column = np.hstack([np.random.randint(10, size=(10,)), [None] * 15])
        df_dict = {
            1: query_column,
            2: categorical_column,
            3: float_column_1,
            4: float_column_2,
            5: int_column,
        }
        df = pd.DataFrame(df_dict)
        table_dict = {row: list(df.loc[row]) for row in df.index}
        external_table_dict[table] = table_dict
    return external_table_dict


@pytest.fixture()
def table_id():
    return 3


@pytest.fixture()
def col_id_dict():
    return {i: 0 for i in range(6)}


@pytest.fixture()
def data():
    return pd.DataFrame({"target": np.random.randn(10), "query": np.arange(10)})


@pytest.fixture()
def query_column():
    return "query"


@pytest.fixture()
def target_column():
    return "target"


# Test prepare_join
def test_prepare_join(
    table_id, external_table_dict, col_id_dict, data, query_column, feature_selector
):
    df_to_join = feature_selector.prepare_join(
        table_id, external_table_dict, col_id_dict, data, query_column
    )
    df_join = pd.merge(
        data,
        df_to_join.reset_index(),
        how="left",
        left_on=query_column,
        right_on=query_column,
    )
    df_to_join = df_to_join.reset_index(drop=True)
    df_append = data
    for c in df_to_join.columns:
        if c != col_id_dict[table_id]:
            df_append[c] = df_to_join[c]
    assert df_join.equals(df_append)


# Test get_type_dict
def test_get_type_dict(
    table_id, external_table_dict, col_id_dict, data, query_column, feature_selector
):
    type_dict = feature_selector.get_type_dict(
        external_table_dict, col_id_dict, data, query_column
    )
    type_dict_table = type_dict[table_id]
    assert type_dict_table == {1: "categoric", 2: "numeric", 3: "numeric", 4: "numeric"}


# Test stat_numeric_numeric
def test_pearson(
    table_id, external_table_dict, col_id_dict, data, query_column, target_column,
):
    feature_selector = FeatureSelector(numeric_stat="pearson")
    df = feature_selector.prepare_join(
        table_id, external_table_dict, col_id_dict, data, query_column
    )
    numeric_column = df[2]
    stat = feature_selector.stat_numeric_numeric(numeric_column, data[target_column])
    assert stat <= 1.0
