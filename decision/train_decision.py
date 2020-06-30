import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from joblib import dump, load

from augmentation.feature_selector import FeatureSelector
from augmentation.strategy import (
    k_best_independent,
    k_best_min_max,
    k_best_2nd_order,
    k_best_quantile,
)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier


## Data generation
def create_all_num_var(nb_num, max_sigma, target):
    # sigma = max_sigma*(np.random.rand(nb_num)) # for randomized sigma
    n = target.shape[0]
    sigma = max_sigma * (np.linspace(0, 1, nb_num))  # for deterministic sigma
    epsilon = np.random.randn(n, nb_num)
    N = target + epsilon * sigma
    return N


def create_cat_var(shuffle_proportion, num_var, target):
    n = target.shape[0]
    idx = np.argsort(target, axis=0)
    cat_var = np.zeros(n)
    split_idx = np.array_split(idx, num_var)
    for i, indice_array in enumerate(split_idx):
        indice_array.reshape((-1))
        for indice in indice_array:
            if np.random.rand() < shuffle_proportion:
                cat_var[indice] = np.random.randint(num_var)
            else:
                cat_var[indice] = i
    cat_var = cat_var.reshape((-1, 1))
    return cat_var


def create_all_cat_var(nb_cat, min_cat, max_cat, target):
    # shuffle_proportion_array = np.random.rand(nb_cat) # for random shuffle proportion
    shuffle_proportion_array = (
        np.linspace(0.1, 1, nb_cat) ** 3
    )  # I chose shuffle proportion non linear so that the distribution of anova is less skewed
    num_cat_array = np.random.randint(min_cat, max_cat, size=(nb_cat))
    C = np.hstack(
        (
            create_cat_var(shuffle_proportion, num_var, target)
            for shuffle_proportion, num_var in zip(
                shuffle_proportion_array, num_cat_array
            )
        )
    )
    return C


def compute_stat(N, C, Y, feature_selector, show=False):
    corr = np.array(
        [
            feature_selector.stat_numeric_numeric(pd.Series(Ni), Y.reshape((-1)))
            for Ni in N.T
        ]
    )
    anova = np.array(
        [
            feature_selector.stat_numeric_categoric(
                pd.Series(Ci), pd.Series(Y.reshape((-1)))
            )
            for Ci in C.T
        ]
    )
    return corr, anova


def find_nearest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def single_uniform_sample(array, stat, nb_unif):
    min_stat = np.min(stat)
    max_stat = np.max(stat)
    uni = np.linspace(0, 1, nb_unif) * (max_stat - min_stat) + min_stat
    idx_sample = [find_nearest_index(stat, value) for value in uni]
    return array[:, idx_sample], stat[idx_sample]


def uniform_sample(N, C, corr, anova, nb_unif_num, nb_unif_cat, show=False):
    new_N, new_corr = single_uniform_sample(N, corr, nb_unif_num)
    new_C, new_anova = single_uniform_sample(C, anova, nb_unif_cat)
    return new_N, new_C, new_corr, new_anova


def sample_distrib(idx, distrib, num_sample, **kwargs):
    """Sample a given distribution `distrib` between min_stat and max_stat from a uniform distribution"""
    n = idx.shape[0]
    if distrib == "uniform":
        new_idx = idx[np.sort(np.random.randint(0, n, size=(num_sample)))]
    elif distrib == "binomial":
        p = kwargs.get("p", 0.5)
        new_idx = idx[np.sort(np.random.binomial(n, p, size=(num_sample)))]
    elif distrib == "geometric":
        p = kwargs.get("p", 0.1)
        geo = np.sort(np.random.geometric(p, size=(num_sample)))
        geo = (geo - geo[0]) / (geo[-1] - geo[0])
        geo = geo * (n - 1)
        new_idx = idx[geo.astype(int)]
    elif distrib == "poisson":
        l = kwargs.get("l", 10)
        poi = np.sort(np.random.poisson(l, size=(num_sample)))
        poi = (poi - poi[0]) / (poi[-1] - poi[0])
        poi = poi * (n - 1)
        new_idx = idx[poi.astype(int)]
    else:
        raise RuntimeError(f"Did not recognise the distribution: {distrib}")
    return new_idx


def sub_sample(
    N,
    C,
    corr,
    anova,
    min_corr,
    max_corr,
    min_anova,
    max_anova,
    distrib_corr,
    distrib_anova,
    num_corr=nb_num,
    num_anova=nb_cat,
    show=False,
    show_old=False,
    **kwargs,
):
    N_idx = np.argwhere(np.logical_and(min_corr < corr, corr < max_corr)).reshape(-1)
    N_idx = sample_distrib(N_idx, distrib_corr, num_corr)
    C_idx = np.argwhere(np.logical_and(min_anova < anova, anova < max_anova)).reshape(
        -1
    )
    C_idx = sample_distrib(C_idx, distrib_anova, num_anova)
    new_corr, new_anova = corr[N_idx], anova[C_idx]
    return N[:, N_idx], C[:, C_idx], new_corr, new_anova


## Feature selection
def df_no_selection(N, C, Y):
    df_N = pd.DataFrame(N)
    df_N.columns = [f"N_{col}" for col in df_N.columns]
    df_C = pd.DataFrame(C)
    df_C.columns = [f"C_{col}" for col in df_C.columns]
    df = pd.concat([df_N, df_C], axis=1)
    df["target"] = Y
    return df


def stat_and_type_dict(fs_corr, fs_anova, nb_num, nb_cat):
    stat_dict = {
        0: {i: corr for i, corr in enumerate(fs_corr)},
        1: {i: anova for i, anova in enumerate(fs_anova)},
    }
    type_dict = {
        0: {i: "numeric" for i in range(nb_num)},
        1: {i: "categoric" for i in range(nb_cat)},
    }
    return stat_dict, type_dict


def select_independent(N, C, Y, stat_dict, type_dict, k_best=10, show=False):
    col_independent = k_best_independent(stat_dict, type_dict, 10)
    if show:
        print("Independently chosen columns are: ", col_independent)
    dict_independent = {
        "target": Y.reshape((-1)),
        **{f"N_{j}": N[:, j] for j in col_independent[0]},
        **{f"C_{j}": C[:, j] for j in col_independent[1]},
    }
    df_independent = pd.DataFrame(dict_independent)
    col_categoric = list(filter(lambda col: col[0] == "C", df_independent.columns))
    dummies = pd.get_dummies(df_independent[col_categoric].astype(int).astype(str))
    df_independent = df_independent.drop(columns=col_categoric)
    df_independent = pd.concat([df_independent, dummies], axis=1)
    return df_independent


def select_min_max_norm(N, C, Y, stat_dict, type_dict, k_best=20, show=False):
    col_normalized = k_best_min_max(stat_dict, type_dict, 20)
    if show:
        print("Min max normalized chosen columns are: ", col_normalized)
    dict_normalized = {
        "target": Y.reshape((-1)),
    }
    if 0 in col_normalized.keys():
        dict_normalized = {
            **dict_normalized,
            **{f"N_{j}": N[:, j] for j in col_normalized[0]},
        }
    if 1 in col_normalized.keys():
        dict_normalized = {
            **dict_normalized,
            **{f"C_{j}": C[:, j] for j in col_normalized[1]},
        }
    df_normalized = pd.DataFrame(dict_normalized)
    if 1 in col_normalized.keys():
        col_categoric = list(filter(lambda col: col[0] == "C", df_normalized.columns))
        dummies = pd.get_dummies(df_normalized[col_categoric].astype(int).astype(str))
        df_normalized = df_normalized.drop(columns=col_categoric)
        df_normalized = pd.concat([df_normalized, dummies], axis=1)
    return df_normalized


def select_2nd_order_norm(N, C, Y, stat_dict, type_dict, k_best=20, show=False):
    col_normalized = k_best_2nd_order(stat_dict, type_dict, 20, percentile=90)
    if show:
        print("2nd order normalized chosen columns are: ", col_normalized)
    dict_normalized = {
        "target": Y.reshape((-1)),
    }
    if 0 in col_normalized.keys():
        dict_normalized = {
            **dict_normalized,
            **{f"N_{j}": N[:, j] for j in col_normalized[0]},
        }
    if 1 in col_normalized.keys():
        dict_normalized = {
            **dict_normalized,
            **{f"C_{j}": C[:, j] for j in col_normalized[1]},
        }
    df_normalized = pd.DataFrame(dict_normalized)
    if 1 in col_normalized.keys():
        col_categoric = list(filter(lambda col: col[0] == "C", df_normalized.columns))
        dummies = pd.get_dummies(df_normalized[col_categoric].astype(int).astype(str))
        df_normalized = df_normalized.drop(columns=col_categoric)
        df_normalized = pd.concat([df_normalized, dummies], axis=1)
    return df_normalized


def select_quantile_norm(N, C, Y, stat_dict, type_dict, k_best=20, show=False):
    col_normalized = k_best_quantile(stat_dict, type_dict, 20)
    if show:
        print("Quantile normalized chosen columns are: ", col_normalized)
    dict_normalized = {
        "target": Y.reshape((-1)),
    }
    if 0 in col_normalized.keys():
        dict_normalized = {
            **dict_normalized,
            **{f"N_{j}": N[:, j] for j in col_normalized[0]},
        }
    if 1 in col_normalized.keys():
        dict_normalized = {
            **dict_normalized,
            **{f"C_{j}": C[:, j] for j in col_normalized[1]},
        }
    df_normalized = pd.DataFrame(dict_normalized)
    if 1 in col_normalized.keys():
        col_categoric = list(filter(lambda col: col[0] == "C", df_normalized.columns))
        dummies = pd.get_dummies(df_normalized[col_categoric].astype(int).astype(str))
        df_normalized = df_normalized.drop(columns=col_categoric)
        df_normalized = pd.concat([df_normalized, dummies], axis=1)
    return df_normalized


## Model training


def train_linear(df):
    prediction_col = df.columns.drop("target")
    X_train, X_test, y_train, y_test = train_test_split(
        df[prediction_col], df["target"]
    )
    lin = LinearRegression().fit(X_train, y_train)
    mse = mean_squared_error(y_test, lin.predict(X_test), squared=False)
    return mse


# Decision model


def get_input_from_stat_and_type(stat_dict, type_dict):
    stat_numeric = np.array(
        [
            stat_dict[table][col]
            for table, stat_table in stat_dict.items()
            for col in stat_table.keys()
            if type_dict[table][col] == "numeric"
        ]
    )
    stat_categoric = np.array(
        [
            stat_dict[table][col]
            for table, stat_table in stat_dict.items()
            for col in stat_table.keys()
            if type_dict[table][col] == "categoric"
        ]
    )
    characteristic_func = {
        "min": np.min,
        "max": np.max,
        "mean": np.mean,
        "median": np.median,
        "stdev": np.std,
        "count": lambda x: x.shape[0],
    }
    input_data = {}
    for func_name, func in characteristic_func.items():
        input_data[f"numeric_{func_name}"] = func(stat_numeric)
        input_data[f"categoric_{func_name}"] = func(stat_categoric)
    return input_data


def decision_training(
    N,
    C,
    Y,
    corr,
    anova,
    min_corr_list,
    max_corr_list,
    min_anova_list,
    max_anova_list,
    num_corr_list,
    num_anova_list,
    corr_distribution_list,
    anova_distribution_list,
    strategy_dict,
):
    input_list = []
    best_strategy_list = []
    nb_num, nb_cat = corr.shape[0], anova.shape[0]
    t = tqdm(
        product(
            min_corr_list,
            max_corr_list,
            min_anova_list,
            max_anova_list,
            num_corr_list,
            num_anova_list,
            corr_distribution_list,
            anova_distribution_list,
        ),
        total=len(min_corr_list)
        * len(max_corr_list)
        * len(min_anova_list)
        * len(max_anova_list)
        * len(num_corr_list)
        * len(num_anova_list)
        * len(corr_distribution_list)
        * len(anova_distribution_list),
    )
    for (
        min_corr,
        max_corr,
        min_anova,
        max_anova,
        num_corr,
        num_anova,
        distrib_corr,
        distrib_anova,
    ) in t:
        # t.set_description(f"Correlation range: [{min_corr}, {max_corr}], Anova range: [{min_anova}, {max_anova}]")
        # t.refresh()
        sample_N, sample_C, sample_corr, sample_anova = sub_sample(
            N,
            C,
            corr,
            anova,
            min_corr,
            max_corr,
            min_anova,
            max_anova,
            distrib_corr,
            distrib_anova,
            num_corr,
            num_anova,
        )

        # Testing feature selection
        sample_stat_dict, sample_type_dict = stat_and_type_dict(
            sample_corr, sample_anova, nb_num, nb_cat
        )
        input_list.append(
            get_input_from_stat_and_type(sample_stat_dict, sample_type_dict)
        )

        best_mse = 100

        for strategy_name, strategy_func in strategy_dict.items():
            df_normalized = strategy_func(
                sample_N, sample_C, Y, sample_stat_dict, sample_type_dict
            )
            mse = train_linear(df_normalized)
            if mse < best_mse:
                best_mse = mse
                best_strategy = strategy_name
        best_strategy_list.append(best_strategy)
    return input_list, best_strategy_list


def main():
    # Define the target array
    n = 1000
    Y = np.random.randn(n, 1)

    # Define the numeric data
    nb_num = 300
    nb_unif_num = 300
    max_sigma = 4

    N = create_all_num_var(nb_num, max_sigma, Y)

    # Define the categorical data
    # number of categorical variables : nb_cat
    nb_cat = 1000
    nb_unif_cat = 1000

    # min and max number of categorical values for a categorical column : min_cat, max_cat
    min_cat = 10
    max_cat = 15

    C = create_all_cat_var(nb_cat, min_cat, max_cat, Y)

    # Compute the statistics
    fs = FeatureSelector(numeric_stat="pearson", categoric_stat="anova")
    corr, anova = compute_stat(N, C, Y, fs)

    # Get a uniform distribution for the data
    N, C, corr, anova = uniform_sample(N, C, corr, anova, nb_unif_num, nb_unif_cat)

    # Parameter definition
    min_corr_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    max_corr_list = [0.6, 0.7, 0.8, 0.9]
    min_anova_list = [100, 200, 300, 500]
    max_anova_list = [1000, 2000, 3000, 4000]
    num_corr_list = [50, 150, 250]
    num_anova_list = [100, 250, 1000]
    distribution = ["uniform", "binomial", "geometric", "poisson"]
    strategy_dict = {
        "independent": select_independent,
        "min_max": select_min_max_norm,
        "2nd_order": select_2nd_order_norm,
        "quantile": select_quantile_norm,
    }

    # Run experiments to get data
    input_list, best_strategy_list = decision_training(
        N,
        C,
        Y,
        corr,
        anova,
        min_corr_list,
        max_corr_list,
        min_anova_list,
        max_anova_list,
        num_corr_list,
        num_anova_list,
        distribution,
        distribution,
    )

    # Process the data in a pandas DataFrame
    decision_df = pd.DataFrame(input_list)
    decision_df["best_strategy"] = best_strategy_list

    decision_df.to_csv("decision.csv")

    print(
        f"Chosen strategies have been : {decision_df['best_strategy'].value_counts()}"
    )

    target = "best_strategy"
    X, y = decision_df.drop(columns=[target]), pd.get_dummies(decision_df[target])

    # Train decision model
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    decision_model = MLPClassifier(hidden_layer_sizes=(20,)).fit(X_train, y_train)

    dump(decision_model, "decision_model.joblib")

    # Evaluate decision model
    decision_model.score(X_test, y_test)
