from augmentation.framework import get_dataset, extract_table_and_col_id, connect_to_database, get_overlappings, \
    table_max_column, get_external_tables
import pytest
import pandas as pd
import os


@pytest.fixture()
def data_path():
    return "../datasets/universities.csv"

# Test get_dataset


def test_get_dataset(data_path):
    get_dataset(data_path, use_default_path=False)


@pytest.fixture()
def data(data_path):
    return get_dataset(data_path, use_default_path=False)


@pytest.fixture()
def query_column():
    return "name"


@pytest.fixture()
def target_column():
    return "target"

# Test connect_to_database


def test_connect_to_database():
    connection = connect_to_database()
    connection.close()


# Test get_overlappings
def test_overlappings(data, query_column):
    overlappings = get_overlappings(3, data, query_column)
    assert len(overlappings) < 4

# Test extract_table_and_col_id
@pytest.fixture()
def table_and_col_list():
    table_id_list, col_id_list = extract_table_and_col_id(
        [" 123 _ 4 ", "567_8"])
    return table_id_list, col_id_list


def test_table_id_split(table_and_col_list):
    table_id_list, col_id_list = table_and_col_list
    assert table_id_list[1] == 567


def test_col_id_split(table_and_col_list):
    table_id_list, col_id_list = table_and_col_list
    assert col_id_list[1] == 8


def test_table_id_strip(table_and_col_list):
    table_id_list, col_id_list = table_and_col_list
    assert table_id_list[0] == 123


@pytest.fixture()
def connection():
    return connect_to_database()


@pytest.fixture()
def table_id1():
    return 62738948


@pytest.fixture()
def table_id2():
    return 61332440

# Test table_max_column


def test_table_max_column(connection, table_id1):
    dict_max = table_max_column([table_id1], connection)
    assert dict_max[table_id1] == 2

# Test get_external_tables
@pytest.fixture()
def external_table_dict(table_id2):
    return 0


def test_dataframe_from_external_table(table_id2):
    table_dict = get_external_tables([table_id2])
    table = table_dict[table_id2]
    df = pd.DataFrame.from_dict(table, orient="index")
    assert len(df.columns) > 1
