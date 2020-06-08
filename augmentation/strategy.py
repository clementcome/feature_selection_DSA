import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from operator import itemgetter


def table_col_stat(
    stat_dict: Dict[int, Dict[int, float]],
    type_dict: Dict[int, Dict[int, float]],
    type_: str,
) -> List[Tuple[int, int, float]]:
    """
    Get the statistics for the column of type `type`

    Parameters
    ----------
    stat_dict : Dict[int, Dict[int,float]]
        Dictionary containing the statistics under stat_dict[table_id][col_id]
    type_dict : Dict[int, Dict[int,float]]
        Dictionary containing the types of the columns under type_dict[table_id][col_id]
    type_ : str
        Type to retrieve the statistics of.

    Returns
    -------
    List[Tuple[int, int, float]]
        Returns the list of statistics with their table and columns ids :\n
        result = [(table_id, col_id, score)]
    """
    table_col_stat = [
        (table_id, col_id, score)
        for table_id, dic_table in stat_dict.items()
        for col_id, score in dic_table.items()
        if type_dict[table_id][col_id] == type_
    ]


def k_best_independent(
    stat_dict: Dict[int, Dict[int, float]],
    type_dict: Dict[int, Dict[int, float]],
    k_best: int,
) -> Dict[int, List[int]]:
    """
    Peforms the k-best strategy on numeric data and categorical data independently

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
    table_column_to_keep = {}
    table_col_stat_numeric = table_col_stat(stat_dict, type_dict, "numeric")
    table_col_stat_categoric = table_col_stat(stat_dict, type_dict, "categoric")
    table_col_stat_numeric_to_keep = sorted(
        table_col_stat_numeric, key=itemgetter(2), reverse=True
    )[:k_best]
    table_col_stat_categoric_to_keep = sorted(
        table_col_stat_categoric, key=itemgetter(2), reverse=True
    )[:k_best]
    for table_id, col_id, _ in table_col_stat_numeric_to_keep:
        if table_id in table_column_to_keep.keys():
            table_column_to_keep[table_id].append(col_id)
        else:
            table_column_to_keep[table_id] = [col_id]
    for table_id, col_id, _ in table_col_stat_categoric_to_keep:
        if table_id in table_column_to_keep.keys():
            table_column_to_keep[table_id].append(col_id)
        else:
            table_column_to_keep[table_id] = [col_id]
    return table_column_to_keep
