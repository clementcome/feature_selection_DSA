from augmentation.framework import Framework
import pytest
import pandas as pd
import os


@pytest.fixture()
def data_path():
    return "../datasets/universities.csv"


@pytest.fixture()
def framework():
    framework_instance = Framework()
    return framework_instance

# Test get_dataset


def test_get_dataset(framework, data_path):
    framework.get_dataset(data_path, use_default_path=False)


@pytest.fixture()
def data(framework, data_path):
    return framework.get_dataset(data_path, use_default_path=False)


@pytest.fixture()
def query_column():
    return "name"


@pytest.fixture()
def target_column():
    return "target"

# Test connect_to_database


def test_connect_to_database(framework, ):
    connection = framework.connect_to_database()
    connection.close()


# Test get_overlappings
def test_overlappings(framework, data, query_column):
    overlappings = framework.get_overlappings(3, data, query_column)
    assert len(overlappings) < 4

# Test extract_table_and_col_id
@pytest.fixture()
def table_and_col_id(framework, ):
    table_id_list, col_id_dict = framework.extract_table_and_col_id(
        [" 123 _ 4 ", "567_8"])
    return table_id_list, col_id_dict


def test_table_id_split(table_and_col_id):
    table_id_list, col_id_dict = table_and_col_id
    assert table_id_list[1] == 567


def test_col_id_split(table_and_col_id):
    table_id_list, col_id_dict = table_and_col_id
    assert col_id_dict[567] == 8


def test_table_id_strip(table_and_col_id):
    table_id_list, col_id_dict = table_and_col_id
    assert table_id_list[0] == 123


@pytest.fixture()
def connection(framework, ):
    return framework.connect_to_database()


@pytest.fixture()
def table_id1():
    return 62738948


@pytest.fixture()
def col_id1():
    return 1


@pytest.fixture()
def table_id2():
    return 61332440

# Test table_max_column


def test_table_max_column(framework, connection, table_id1):
    dict_max = framework.table_max_column([table_id1], connection)
    assert dict_max[table_id1] == 2

# Test get_external_table_dict
@pytest.fixture()
def external_table_dict(framework, table_id1, table_id2):
    return framework.get_external_table_dict([table_id1, table_id2], )

# Test get_external_table_dict


def test_dataframe_from_external_table(framework, table_id2, external_table_dict):
    table = external_table_dict[table_id2]
    df = pd.DataFrame.from_dict(table, orient="index")
    assert len(df.columns) > 1

# Test get_external_table_cleaned


def test_join_external_table(framework, external_table_dict, table_id1, col_id1, data, query_column):
    mock_col_id_dict = {table_id1: col_id1}
    df_to_join = framework.get_external_table_cleaned(
        table_id1, external_table_dict, mock_col_id_dict, data, query_column)
    df_join = pd.merge(data, df_to_join.reset_index(), how="left",
                       left_on=query_column, right_on=query_column)
    df_to_join = df_to_join.reset_index(drop=True)
    df_append = data
    for c in df_to_join.columns:
        if c != col_id1:
            df_append[c] = df_to_join[c]
    assert df_join.equals(df_append)
