from typing import Dict, List
from timer.timer import timer
from operator import itemgetter
import pandas as pd
import numpy as np
from .statistics import pearson, anova
from .strategy import k_best_independent
import logging
import os


class FeatureSelector:
    """
    Class to define the strategy of feature selection
    """

    log_file = os.getenv("LOG_FILE", "time.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(message)s",
    )

    def __init__(
        self,
        numeric_threshold: float = 0.5,
        categoric_threshold: float = 0.5,
        numeric_stat: str = "constant",
        categoric_stat: str = "constant",
        select_strategy: str = "threshold",
        k_best: str = 10,
        one_hot_encoding: bool= True,
    ):
        self.numeric_threshold = numeric_threshold
        self.categoric_threshold = categoric_threshold
        self.numeric_stat = numeric_stat
        self.categoric_stat = categoric_stat
        self.select_strategy = select_strategy
        self.k_best = k_best
        self.one_hot_encoding=one_hot_encoding

    def stat_numeric_numeric(
        self, column: pd.Series, target_column: pd.Series
    ) -> float:
        if self.numeric_stat == "pearson":
            return pearson(column, target_column)
        return 1.0

    def stat_numeric_categoric(
        self, column: pd.Series, target_column: pd.Series
    ) -> float:
        if self.categoric_stat == "anova":
            return anova(column, target_column)
        return 1.0

    @timer
    def select_column(
        self,
        stat_dict: Dict[int, Dict[int, float]],
        type_dict: Dict[int, Dict[int, str]],
    ) -> Dict[int, List[int]]:
        """
        Select the column of external_tables to be kept

        Parameters
        ----------
        stat_dict : Dict[int, Dict[int, float]]
            Statistics of each column
        type_dict : Dict[int, Dict[int, str]]
            Type of each column

        Returns
        -------
        Dict[int, List[int]]
            result[table_id] is the list of the columns to keep for a given table
        """
        logging.info(f"Execution of select_column with {self.select_strategy} strategy")

        table_column_to_keep = {}
        if self.select_strategy == "k_best":
            table_column_to_keep = k_best_independent(stat_dict, type_dict, self.k_best)
        if self.select_strategy == "threshold":
            for table_id in stat_dict.keys():
                column_to_keep = []
                stat_table_dict = stat_dict[table_id]
                for column_id in stat_table_dict.keys():
                    if type_dict[table_id][column_id] == "numeric":
                        if stat_table_dict[column_id] > self.numeric_threshold:
                            column_to_keep.append(column_id)
                    else:
                        if stat_table_dict[column_id] > self.categoric_threshold:
                            column_to_keep.append(column_id)
                table_column_to_keep[table_id] = column_to_keep
        return table_column_to_keep

    def get_type_dict(
        self,
        external_table_dict: Dict[int, Dict[int, List[str]]],
        col_id_dict: Dict[int, int],
        data: pd.DataFrame,
        query_column: str,
    ) -> Dict[int, Dict[int, float]]:
        """Return a dictionary indicating whether result[table_id][column_id] is numeric or categoric"""
        type_dict = {}
        for table_id in external_table_dict.keys():
            external_table = self.prepare_join(
                table_id, external_table_dict, col_id_dict, data, query_column
            )
            external_table = external_table.apply(pd.to_numeric, errors="ignore")
            type_table_dict = {}
            for column_id in external_table.columns:
                column = external_table[column_id]
                if column.dtype == float:
                    type_table_dict[column_id] = "numeric"
                else:
                    type_table_dict[column_id] = "categoric"
            type_dict[table_id] = type_table_dict
        return type_dict

    @timer
    def evaluate(
        self,
        external_table_dict: Dict[int, Dict[int, List[str]]],
        col_id_dict: Dict[int, int],
        type_dict: Dict[int, Dict[int, float]],
        data: pd.DataFrame,
        query_column: str,
        target_column: str,
    ) -> Dict[int, Dict[int, float]]:
        """
        Compute the statistic for every table and column in the external_table_dict

        Parameters
        ----------
        external_table_dict : Dict[int, Dict[int, List[str]]]
            Output of Framework().get_external_table_dict
        col_id_dict : Dict[int, int]
            Column ids of the external tables to perform join on
        type_dict : Dict[int, Dict[int, float]]
            Indicates whether a column is to be considered as numeric or categoric
        data : pd.DataFrame
            Base dataset
        query_column : str
            Column of the base dataset to perform join on
        target_column : str
            Column of the base dataset to predict

        Returns
        -------
        Dict[int, Dict[int, float]]
            result[table_id][column_id] is the value of the statistic comparing the column and the target_column
        """
        logging.info(
            f"Execution of evaluate with {self.numeric_stat} {self.categoric_stat} as numeric and categorical stats."
        )
        stat_dict = {}
        for table_id in external_table_dict.keys():
            external_table = self.prepare_join(
                table_id, external_table_dict, col_id_dict, data, query_column
            )
            stat_table_dict = {}
            for column_id in external_table.columns:
                column = external_table[column_id]
                if type_dict[table_id][column_id] == "numeric":
                    column = pd.to_numeric(column)
                    stat = self.stat_numeric_numeric(column, data[target_column])
                else:
                    stat = self.stat_numeric_categoric(column, data[target_column])
                stat_table_dict[column_id] = stat
            stat_dict[table_id] = stat_table_dict
        return stat_dict
    
    def features_to_encode(
        self,
        df : pd.DataFrame,
    ):
        features_encode=[]
        for column in df:
            if column.dtype != float:
                features_encode.append(column)
        return(list(features_encode))

    def encode_and_bind(
        self,
        df : pd.DataFrame, 
        features_encode : list
    ):    
        if self.one_hot_encoding == True:
            features_encode = features_to_encode(df)
            for feature in features_encode:
                
                dummies = pd.get_dummies(df[[feature]])
                df = pd.concat([df, dummies], axis=1)
                df = df.drop([feature], axis=1)
        return(df)



    



    def prepare_join(
        self,
        table_id: int,
        external_dict: Dict[int, Dict[int, List[str]]],
        col_id_dict: Dict[int, int],
        data: pd.DataFrame,
        query_column: str,
    ) -> pd.DataFrame:
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
            Dictionary containing the external tables (output of Framework().get_external_table_dict)
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
        df_table = pd.DataFrame.from_dict(external_dict[table_id], orient="index")
        # Here a choice is made according to the strategy for joining where multiple rows could be corresponding
        df_table = df_table.drop_duplicates(query_column_external, keep="last")
        df_table = df_table.set_index(query_column_external).reindex(data[query_column])
        return df_table
