import pandas as pd
import numpy as np
from joblib import dump, load
from scipy.stats import beta, gamma

# Decision model


def find_index_split(array, slices):
    """Returns the indices of the ordered value of slices in the sorted array"""
    indices = np.zeros_like(slices)
    i_array = 0
    for i_slice, value in enumerate(slices):
        while (i_array < len(array)) and (array[i_array] < value):
            i_array += 1
        indices[i_slice] = i_array
    return indices


def stat_distance(stat, law):
    stat_norm = (stat - np.min(stat)) / (np.max(stat) - np.min(stat))
    ordered_stat_norm = np.sort(stat_norm)
    nb_split = 10
    slices = np.linspace(0, 1, nb_split + 1)
    indice_slices = find_index_split(ordered_stat_norm, slices)
    indice_slices = indice_slices.astype(int)
    nb_slices = np.array(
        [
            indice_slices[i] - indice_slices[i - 1]
            for i in range(1, indice_slices.shape[0])
        ]
    )
    prob_slices = nb_slices / np.sum(nb_slices)
    mid_slices = (stat[indice_slices[:-1]] + stat[indice_slices[1:]]) / 2
    prob = law(mid_slices)
    distance = np.sum(
        np.where(np.logical_not(np.isinf(prob)), abs(prob_slices - law(mid_slices)), 0)
    )
    return distance


def stat_quality(stat, law):
    ordered_stat = np.sort(stat)
    nb_split = 10
    split_array = np.array_split(ordered_stat, nb_split)
    split_means = np.array([np.mean(split) for split in split_array])
    prob = law(split_means)
    quality = np.sum(np.where(np.logical_not(np.isinf(prob)), prob, 0))
    if np.isinf(quality):
        print(split_means)
        print(prob)
        print(np.where(not (np.isinf(prob)), prob, 0))
    return quality


def beta_law(a, b):
    return lambda x: beta.pdf(x, a, b)


def beta_coeff_likelihood(stat):
    stat_norm = (stat - np.min(stat)) / (np.max(stat) - np.min(stat))
    a, b, _, _ = beta.fit(stat_norm)
    law = beta_law(a, b)
    distance = stat_distance(stat_norm, law)
    quality = stat_quality(stat_norm, law)
    return a, b, distance, quality


def gamma_law(k):
    return lambda x: gamma.pdf(x, k)


def gamma_coeff_likelihood(stat):
    k, _, _ = gamma.fit(stat)
    law = gamma_law(k)
    distance = stat_distance(stat, law)
    quality = stat_quality(stat, law)
    return k, distance, quality


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
        "beta_coeff": beta_coeff_likelihood,
        "gamma_coeff": gamma_coeff_likelihood,
    }
    input_data = {}
    for func_name, func in characteristic_func.items():
        if func_name == "beta_coeff":
            a_num, b_num, distance_num, quality_num = func(stat_numeric)
            input_data[f"numeric_beta_a"] = a_num
            input_data[f"numeric_beta_b"] = b_num
            input_data[f"numeric_beta_distance"] = distance_num
            input_data[f"numeric_beta_quality"] = quality_num
            a_cat, b_cat, distance_cat, quality_cat = func(stat_categoric)
            input_data[f"categoric_beta_a"] = a_cat
            input_data[f"categoric_beta_b"] = b_cat
            input_data[f"categoric_beta_distance"] = distance_cat
            input_data[f"categoric_beta_quality"] = quality_cat
            continue
        elif func_name == "gamma_coeff":
            k_num, distance_num, quality_num = func(stat_numeric)
            input_data[f"numeric_gamma_k"] = k_num
            input_data[f"numeric_gamma_distance"] = distance_num
            input_data[f"numeric_gamma_quality"] = quality_num
            k_cat, distance_cat, quality_cat = func(stat_categoric)
            input_data[f"categoric_gamma_k"] = k_cat
            input_data[f"categoric_gamma_distance"] = distance_cat
            input_data[f"categoric_gamma_quality"] = quality_cat
            continue
        input_data[f"numeric_{func_name}"] = func(stat_numeric)
        input_data[f"categoric_{func_name}"] = func(stat_categoric)
    return input_data


def decide(self, stat_dict, type_dict):
    input_data = get_input_from_stat_and_type(stat_dict, type_dict)
    input_scaler = load("input_scaler.joblib")
    decision_model = load("decision_model_reg.joblib")
    input_data_n = input_scaler.transform([input_data])
    y_pred = pd.DataFrame(
        decision_model.predict(input_data_n),
        columns=["independent", "min_max", "2nd_order", "quantile"],
    )
    return y_pred.idxmax(axis=1)[0]
