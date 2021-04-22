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
        if not (np.isnan(score))
    ]
    return table_col_stat


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
    )[: k_best - k_best // 2]
    table_col_stat_categoric_to_keep = sorted(
        table_col_stat_categoric, key=itemgetter(2), reverse=True
    )[: k_best // 2]
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


def k_best_min_max(
    stat_dict: Dict[int, Dict[int, float]],
    type_dict: Dict[int, Dict[int, float]],
    k_best: int,
) -> Dict[int, List[int]]:
    """
    Peforms the k-best strategy on numeric data and categorical data comparing the statistics with min-max normalization

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
    stat_numeric = np.array([line[2] for line in table_col_stat_numeric])
    stat_numeric_norm = (stat_numeric - np.min(stat_numeric)) / (
        np.max(stat_numeric) - np.min(stat_numeric)
    )
    table_col_stat_numeric_norm = [
        (table, col, stat_norm)
        for (table, col, _), stat_norm in zip(table_col_stat_numeric, stat_numeric_norm)
    ]
    stat_categoric = np.array([line[2] for line in table_col_stat_categoric])
    stat_categoric_norm = (stat_categoric - np.min(stat_categoric)) / (
        np.max(stat_categoric) - np.min(stat_categoric)
    )
    table_col_stat_categoric_norm = [
        (table, col, stat_norm)
        for (table, col, _), stat_norm in zip(
            table_col_stat_categoric, stat_categoric_norm
        )
    ]
    table_col_stat_norm = table_col_stat_numeric_norm + table_col_stat_categoric_norm
    table_col_stat_to_keep = sorted(
        table_col_stat_norm, key=itemgetter(2), reverse=True
    )[:k_best]
    for table_id, col_id, _ in table_col_stat_to_keep:
        if table_id in table_column_to_keep.keys():
            table_column_to_keep[table_id].append(col_id)
        else:
            table_column_to_keep[table_id] = [col_id]
    return table_column_to_keep


def solve_2nd(alpha: float, x_i: float) -> Tuple[float, float]:
    """
    Solves the equation for the 2nd order normalization on min-max normalized data

    Parameters
    ----------
    alpha : float
        New value of the ith percentile of the normalized data
    x_i : float
        Value of the ith percentile of the original data

    Returns
    -------
    Tuple [float, float]
        a, b are the solution of the 2nd order normalization (with c=0)
    """
    a = (x_i - alpha) / (x_i - x_i ** 2)
    b = (alpha - x_i ** 2) / (x_i - x_i ** 2)
    return a, b


def k_best_2nd_order(
    stat_dict: Dict[int, Dict[int, float]],
    type_dict: Dict[int, Dict[int, float]],
    k_best: int,
    percentile: int = 50,
) -> Dict[int, List[int]]:
    """
    Peforms the k-best strategy on numeric data and categorical data comparing the statistics with 2nd order normalization with fixed percentile

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
    stat_numeric = np.array([line[2] for line in table_col_stat_numeric])
    stat_numeric_min_max = (stat_numeric - np.min(stat_numeric)) / (
        np.max(stat_numeric) - np.min(stat_numeric)
    )
    alpha = percentile / 100.0
    a, b = solve_2nd(alpha, np.percentile(stat_numeric_min_max, percentile))
    stat_numeric_norm = a * stat_numeric_min_max ** 2 + b * stat_numeric_min_max
    stat_categoric = np.array([line[2] for line in table_col_stat_categoric])
    stat_categoric_min_max = (stat_categoric - np.min(stat_categoric)) / (
        np.max(stat_categoric) - np.min(stat_categoric)
    )
    a, b = solve_2nd(alpha, np.percentile(stat_categoric_min_max, percentile))
    stat_categoric_norm = a * stat_categoric_min_max ** 2 + b * stat_categoric_min_max
    table_col_stat_numeric_norm = [
        (table, col, stat_norm)
        for (table, col, _), stat_norm in zip(table_col_stat_numeric, stat_numeric_norm)
    ]
    table_col_stat_categoric_norm = [
        (table, col, stat_norm)
        for (table, col, _), stat_norm in zip(
            table_col_stat_categoric, stat_categoric_norm
        )
    ]
    table_col_stat_norm = table_col_stat_numeric_norm + table_col_stat_categoric_norm
    table_col_stat_to_keep = sorted(
        table_col_stat_norm, key=itemgetter(2), reverse=True
    )[:k_best]
    for table_id, col_id, _ in table_col_stat_to_keep:
        if table_id in table_column_to_keep.keys():
            table_column_to_keep[table_id].append(col_id)
        else:
            table_column_to_keep[table_id] = [col_id]
    return table_column_to_keep


def quantile_normalization(
    stat_numeric: np.array, stat_categoric: np.array
) -> Tuple[np.array, np.array]:
    """
    Performs quantile normalization on data of different length used for statistics normalization

    Parameters
    ----------
    stat_numeric : np.array
        Original numeric statistics
    stat_categoric : np.array
        Original categoric statistics

    Returns
    -------
    Tuple[np.array, np.array]
        stat_numeric_norm, stat_categoric_norm the resulting data after quantile normalization
    """
    stat_numeric_quantile = np.copy(stat_numeric)
    stat_categoric_quantile = np.copy(stat_categoric)
    n_split = min(stat_numeric.shape[0], stat_categoric.shape[0])
    sorted_numeric_idx = np.argsort(stat_numeric)
    sorted_categoric_idx = np.argsort(stat_categoric)
    split_idx_numeric = np.array_split(sorted_numeric_idx, n_split)
    split_idx_categoric = np.array_split(sorted_categoric_idx, n_split)
    for idx_numeric, idx_categoric in zip(split_idx_numeric, split_idx_categoric):
        mean_numeric = np.mean(stat_numeric[idx_numeric])
        mean_categoric = np.mean(stat_categoric[idx_categoric])
        for i_numeric in idx_numeric:
            stat_numeric_quantile[i_numeric] = np.mean(
                [stat_numeric[i_numeric], mean_categoric]
            )
        for i_categoric in idx_categoric:
            stat_categoric_quantile[i_categoric] = np.mean(
                [stat_categoric[i_categoric], mean_numeric]
            )
    return stat_numeric_quantile, stat_categoric_quantile


def k_best_quantile(
    stat_dict: Dict[int, Dict[int, float]],
    type_dict: Dict[int, Dict[int, float]],
    k_best: int,
) -> Dict[int, List[int]]:
    """
    Peforms the k-best strategy on numeric data and categorical data comparing the statistics with quantile normalization

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
    stat_numeric = np.array([line[2] for line in table_col_stat_numeric])
    stat_numeric_min_max = (stat_numeric - np.min(stat_numeric)) / (
        np.max(stat_numeric) - np.min(stat_numeric)
    )
    stat_categoric = np.array([line[2] for line in table_col_stat_categoric])
    stat_categoric_min_max = (stat_categoric - np.min(stat_categoric)) / (
        np.max(stat_categoric) - np.min(stat_categoric)
    )
    # Perform quantile normalization
    stat_numeric_norm, stat_categoric_norm = quantile_normalization(
        stat_numeric_min_max, stat_categoric_min_max
    )

    table_col_stat_numeric_norm = [
        (table, col, stat_norm)
        for (table, col, _), stat_norm in zip(table_col_stat_numeric, stat_numeric_norm)
    ]
    table_col_stat_categoric_norm = [
        (table, col, stat_norm)
        for (table, col, _), stat_norm in zip(
            table_col_stat_categoric, stat_categoric_norm
        )
    ]
    table_col_stat_norm = table_col_stat_numeric_norm + table_col_stat_categoric_norm
    table_col_stat_to_keep = sorted(
        table_col_stat_norm, key=itemgetter(2), reverse=True
    )[:k_best]
    for table_id, col_id, _ in table_col_stat_to_keep:
        if table_id in table_column_to_keep.keys():
            table_column_to_keep[table_id].append(col_id)
        else:
            table_column_to_keep[table_id] = [col_id]
    return table_column_to_keep
