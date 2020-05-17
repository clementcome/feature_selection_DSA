from typing import Tuple, List, Dict
from timer.timer import timer

import numpy as np
import pandas as pd
import vertica_python
import re
import os

DBConnection = vertica_python.vertica.connection.Connection


class Framework:
    @timer
    def get_dataset(self, file_name: str, use_default_path: bool = True, separ: str = ",") -> pd.DataFrame:
        """
        Gets the dataset located at file_name
        If use_default_path then it adds a base_url before the file name
        Converts all the data into string data

        Parameters
        ----------
        file_name : str
            Name of the file to retrieve the dataset from
        use_default_path : bool, optional
            Boolean indicating whether we will be using "../datasets/" as a base_url or not, by default True
        separ : str, optional
            Separator for importing the dataset, by default ","

        Returns
        -------
        pd.DataFrame
            Dataset contained in the file designated by file_name
        """
        base_url = '../datasets/'
        if use_default_path:
            file = pd.read_csv(base_url+file_name+'.csv', sep=separ)
        else:
            file = pd.read_csv(file_name, sep=separ)
        # file = file.replace("'", "''")
        file = file.apply(lambda x: x.astype(str).str.lower())
        return file

    def get_cleaned_text(self, text: str) -> str:
        """Removes non-word characters and stopwords."""
        if text is None:
            return ''
        stopwords = ['a', 'the', 'of', 'on', 'in', 'an', 'and', 'is', 'at', 'are', 'as', 'be', 'but', 'by', 'for', 'it', 'no',
                     'not', 'or', 'such', 'that', 'their', 'there', 'these', 'to', 'was', 'with', 'they', 'will',  'v', 've', 'd']  # , 's']

        cleaned = re.sub(
            '[\W_]+', ' ', text.encode('ascii', 'ignore').decode('ascii'))
        # feature_one = re.sub(' +', '', cleaned).strip()
        feature_one = re.sub(' +', ' ', cleaned).strip()
        # feature_one = feature_one.replace(" s ", "''s  ")

        for x in stopwords:
            feature_one = feature_one.replace(' {} '.format(x), ' ')
            if feature_one.startswith('{} '.format(x)):
                feature_one = feature_one[len('{} '.format(x)):]
            if feature_one.endswith(' {}'.format(x)):
                feature_one = feature_one[:-len(' {}'.format(x))]
        return feature_one

    @timer
    def connect_to_database(self) -> DBConnection:
        """
        Example
        -------
        Standard use is :\n
        connection = connect_to_database()\n
        cur = connection.cursor()\n
        cur.execute(query1)\n
        cur.fetchall()\n
        cur.execute(query1)\n
        ...\n
        connection.close()\n

        Returns
        -------
        vertica_python.vertica.connnection.Connection
            connection from vertica_python
        """
        conn_info = {
            'host': '127.0.0.1',
            'port': 5433,
            'user': 'olib92',
            'password': 'Transformer2',
            'database': 'xformer',
            # autogenerated session label by default,
            'session_label': 'some_label',
            # 10 minutes timeout on queries
            'read_timeout': 6000,
            # default throw error on invalid UTF-8 results
            'unicode_error': 'strict',
            # SSL is disabled by default
            'ssl': False,
            # using server-side prepared statements is disabled by default
            'use_prepared_statements': False,
            # connection timeout is not enabled by default
            # 'connection_timeout': 5
        }
        connection = vertica_python.connect(**conn_info)
        return connection

    @timer
    def get_overlappings(self, k: int, data: pd.DataFrame, query_column: str, connection: DBConnection = None) -> List[str]:
        """
        Choose k table to perform join with the base dataset.
        The order of the tables is determined by the number of appearances of every value
        in the query column of the base dataset.

        Parameters
        ----------
        k : int
            Max number of table to join the base dataset to
        data : pd.DataFrame
            Base dataset
        query_column : str
            Column of the base dataset to use as join key

        Returns
        -------
        List[str]
            List of "tableid_colid" where\n
            - tableid is the table to join\n
            - colid is the name of the column to use as join key
        """

        if connection == None:
            connection = self.connect_to_database()
        cur = connection.cursor()

        query_data = data[query_column].apply(self.get_cleaned_text)

        distinct_clean_values = query_data.unique()
        joint_distinct_values = '\',\''.join(
            distinct_clean_values).encode('utf-8')

        query = 'SELECT SUBQ.ids FROM (SELECT table_col_id AS ids,' \
                'CONCAT(table_col_id,CONCAT(\'_\',REGEXP_REPLACE(REGEXP_REPLACE(' \
                'tokenized, \'\W+\', \' \'), \' +\', \' \'))) AS COL_ELEM from cbi_inverted_index_2 WHERE REGEXP_REPLACE(' \
                'REGEXP_REPLACE(tokenized, \'\W+\', \' \'), \' +\', \' \') IN (\'{}\') ' \
                'GROUP BY table_col_id,CONCAT(table_col_id,CONCAT(\'_\',' \
                'REGEXP_REPLACE(REGEXP_REPLACE(tokenized, \'\W+\', \' \'), \' +\', \' \'))) ) AS SUBQ GROUP BY SUBQ.ids ' \
                'HAVING COUNT(COL_ELEM) > {} ' \
                'ORDER BY COUNT(COL_ELEM) DESC LIMIT {};'.format(
                    joint_distinct_values, 3, k)
        cur.execute(query)

        result = [item for sublist in cur.fetchall()
                  for item in sublist]
        return result

    def extract_table_and_col_id(self, overlappings: List[str]) -> Tuple[List[int], Dict[int, int]]:
        """
        Return a list of table_id and a dictionary giving query columns as values and table ids as keys
        extracted from the result of overlappings.
        """
        table_id_list = []
        column_id_dict = {}
        for o in overlappings:
            table_id = int(o.split('_')[0].strip())
            column_id = int(o.split('_')[1].strip())
            table_id_list.append(table_id)
            column_id_dict[table_id] = column_id
        return table_id_list, column_id_dict

    @timer
    def table_max_column(self, table_id_list: List[int], connection: DBConnection = None) -> Dict[int, int]:
        """
        Return a dictionary where key is the id of a table in table_id_list
        and value is the max column of this table 
        """

        if connection == None:
            connection = self.connect_to_database()
        cur = connection.cursor()

        # Transforming table_ids from integers to strings
        s = [str(i) for i in table_id_list]
        cur.execute('SELECT tableid, maxcol from cbi_inverted_index_2 WHERE tableid IN (\'{}\');'.format(
            '\',\''.join(s)))
        result = pd.DataFrame(cur.fetchall(), columns=[
            'tableid', 'max_col_id'])
        max_column_dict = result.set_index(
            'tableid').to_dict()['max_col_id']
        return max_column_dict

    def get_clean_dataset(self, data_path: str, query_column: str, target_column: str) -> pd.DataFrame:
        """
        Returns the dataset contain in the file found at data_path
        The dataset will only contain :\n
        - the query_colum with values cleaned by get_cleaned_text function\n
        - the target column with values as float
        """
        data = self.get_dataset(data_path, use_default_path=False)[
            [query_column, target_column]]
        data[query_column] = data[query_column].apply(
            self.get_cleaned_text)
        # Only supports regression until now
        data[target_column] = data[target_column].astype(float)
        return data

    @timer
    def get_external_table_dict(self, table_id_list: List[int], connection: DBConnection = None) -> Dict[int, Dict[int, List[str]]]:
        """
        Gathers in a dictionary all the tables listed by table_id_list.
        To obtain the table designated by the table_id as a DataFrame, you can do :\n
        table = pd.DataFrame.from_dict(result[table_id], orient="index")

        Parameters
        ----------
        table_id_list : List[int]
            List of the table ids to retrieve
        connection : DBConnection, optional
            Connection to the database. If not provided, the connection will be initialized, by default None

        Returns
        -------
        Dict[int, Dict[int, List[str]]]
            Dictionary with table_id as key.
            result[table_id] is also a dictionary with the row_id as key.
            result[table_id][row_id] is the list of the values contained in the row in the table designated by column_id and table_id
        """

        if connection == None:
            connection = self.connect_to_database()
        cur = connection.cursor()

        s = [str(table_id) for table_id in table_id_list]
        tables_fetch_query = 'SELECT tableid, colid, rowid, table_row_id, tokenized FROM cbi_inverted_index_2 WHERE tableid IN (\'{}\') order by tableid, colid, rowid;'.format(
            '\',\''.join(s))
        cur.execute(tables_fetch_query)
        external_tables = pd.DataFrame(cur.fetchall(), columns=[
            'tableid', 'colid', 'rowid', 'table_row_id', 'tokenized'])

        temp = external_tables.sort_values(by=['tableid', 'rowid', 'colid']).groupby(
            ['tableid', 'rowid']).tokenized.apply(list).reset_index()

        external_table_dict = {
            table_id: table_df[["rowid", "tokenized"]]
            .set_index("rowid")
            .to_dict()["tokenized"]
            for table_id, table_df in temp.groupby("tableid")}

        return external_table_dict

    @timer
    def get_external_table_cleaned(self, table_id: int, external_dict: Dict[int, Dict[int, List[str]]],
                                   col_id_dict: Dict[int, int], data: pd.DataFrame, query_column: str) -> pd.DataFrame:
        """
        Generate the table corresponding to table_id without duplicates and 
        ordered in the same way as data along the query_column and col_id_dict[table_id]
        Query column is not anymore in columns but is an index.
        To append the column to an other dataset, you should reset the index and drop it.

        Parameters
        ----------
        table_id : int
            Identifier of the table to retrieve
        external_dict : Dict[int, Dict[int, List[str]]]
            Dictionary containing the external tables (output of get_external_table_dict)
        col_id_dict : Dict[int, int]
            Dictionary of the column ids to perform join on with table ids as keys
        data : pd.DataFrame
            Base dataset
        query_column : str
            Column to join external datasets on

        Returns
        -------
        pd.DataFrame
            DataFrame without duplicates and ordered like the base dataset
        """
        query_column_external = col_id_dict[table_id]
        df_table = pd.DataFrame.from_dict(
            external_dict[table_id], orient="index")
        # Here a choice is made according to the strategy for joining where multiple rows could be corresponding
        df_table = df_table.drop_duplicates(
            query_column_external, keep="last")
        df_table = df_table.set_index(
            query_column_external).reindex(data[query_column])
        return df_table

    @timer
    def perform_join(self, table_and_col_to_keep: Dict[int, List[int]], col_id_dict: Dict[int, int],
                     external_dict: Dict[int, Dict[int, List[str]]], data: pd.DataFrame, query_column: str,
                     data_path: str, k: int):
        for table_id, column_list in table_and_col_to_keep.items():
            if len(column_list) > 0:
                query_column_external = col_id_dict[table_id]
                df_table = self.get_external_table_cleaned(
                    table_id, external_dict, col_id_dict, data, query_column)
                df_table = df_table.reset_index(drop=True)
                for column_id in column_list:
                    data[f"{table_id}_{column_id}"] = df_table[column_id]
        dataset_name = os.path.basename(data_path).split(".")[0]
        data.to_csv(
            f"../enriched/{dataset_name}_enriched_{k}.csv", index=False)
        return data

    def run(self, data_path: str, query_column: str, target_column: str, k: int):
        data = self.get_clean_dataset(
            data_path, query_column, target_column)
        connection = self.connect_to_database()

        overlappings = self.get_overlappings(
            k, data, query_column, connection)
        table_id_list, col_id_dict = self.extract_table_and_col_id(
            overlappings)
        max_column_dict = self.table_max_column(
            table_id_list, connection)

        external_table_dict = self.get_external_table_dict(
            table_id_list, connection)

        # Iterating on the external_tables to get statistics
        table_and_col_to_keep = {}
        for table_id in table_id_list:
            df_table = self.get_external_table_cleaned(
                table_id, external_table_dict, col_id_dict, data, query_column)
            # To do : integrate a feature selector component
            table_and_col_to_keep[table_id] = list(df_table.columns)

        data = self.perform_join(table_and_col_to_keep, col_id_dict,
                                 external_table_dict, data, query_column, data_path, k)

        connection.close()
        return data
