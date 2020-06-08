from scipy.stats import pearsonr, f_oneway
import pandas as pd
import numpy as np

def pearson(column: pd.Series, target_column: pd.Series) -> float:
    replace_value = 0
    if pd.isna(column.mean()) == False:
        if column.mean() < np.inf:
            replace_value = column.mean()
    column = column.fillna(replace_value)
    inf_mask = column == np.inf
    column[inf_mask] = replace_value
    correlation = pearsonr(column, target_column)[0]
    return abs(correlation)

def anova(column: pd.Series, target_column: pd.Series) -> float:
    column = column.reset_index(drop=True).fillna("None value")
    target_column = target_column.reset_index(drop=True)
    df = pd.concat(
        [column, target_column], axis=1, keys=["column_to_evaluate", "target"]
    )
    data_grouped = df.groupby("column_to_evaluate")["target"].agg(list).values
    f_test_stat = f_oneway(*list(data_grouped))[0]
    return f_test_stat