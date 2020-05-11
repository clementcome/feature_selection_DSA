from augmentation.framework import extract_table_and_col_id, connect_to_database
import pytest
import os


def test_connect_to_database():
    connection = connect_to_database()
    connection.close()


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
