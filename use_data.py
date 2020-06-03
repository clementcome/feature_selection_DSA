import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import os
import time


model_dict = {
    "Random Forest Regressor": RandomForestRegressor,
    "Linear Regressor": LinearRegression,
    "Support Vector Regressor": SVR,
}
model_kwargs_dict = {
    "Random Forest Regressor": {"max_depth": 10},
    "Linear Regressor": {},
    "Support Vector Regressor": {},
}


def test_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
):
    kwargs_model = model_kwargs_dict[model_name]
    start = time.time()
    reg = model_dict[model_name](**kwargs_model).fit(x_train, y_train)
    end = time.time()
    reg_mse = mean_squared_error(y_test, reg.predict(x_test), squared=False)
    print(
        f"MSE for {model_name} on file {file} is {reg_mse}, took {end-start}s to train"
    )


try:
    os.chdir("/home/ccome/git_files/enriched")
except OSError:
    print("Cannot change directory.")

print("File available here : ", os.listdir())

name_query_target_list = [
    ("universities", "name", "target"),
    ("presidential", "County", "Votes"),
    ("movie", "movie_title", "imdb_score"),
    ("pageviews", "name", "visit"),
    ("worldcitiespop", "City", "Population"),
]

for file in os.listdir():
    df = pd.read_csv(file, nrows=500)
    df_categoric = df[df.columns[df.dtypes == "object"]]
    df_numeric = df.drop(columns=df_categoric.columns)
    for col in df_numeric.columns:
        if not (pd.isna(df_numeric[col].mean())):
            df_numeric[col] = df_numeric[col].fillna(df_numeric[col].mean())
        else:
            df_numeric = df_numeric.drop(columns=[col])
    df_categoric_one_hot = pd.get_dummies(df_categoric)
    df = pd.concat([df_numeric, df_categoric_one_hot], axis=1)
    predicting_columns = list(df.columns)

    for dataset_name, query, target in name_query_target_list:
        if dataset_name in file:
            target_column = target
            if dataset_name != "universitites":
                continue

    predicting_columns.remove(target_column)

    df_train, df_test = train_test_split(df, random_state=0)
    x_train = df_train[predicting_columns]
    y_train = df_train[target_column]
    x_test = df_test[predicting_columns]
    y_test = df_test[target_column]

    for model_name in model_dict.keys():
        test_model(x_train, y_train, x_test, y_test, model_name)
    print("---")

    # Performing wrapper methods for feature selection
    if "constant_constant" in file:

        # Forward selection
        sfs = SFS(
            LinearRegression(),
            k_features=10,
            forward=True,
            scoring="neg_mean_squared_error",
            verbose=1
        )
        start_fs = time.time()
        sfs.fit(df[predicting_columns], df[target_column])
        stop_fs = time.time()
        print(f"Forward selection on {file} took {stop_fs-start_fs}s")
        # Run the experiments
        print(f"Results of the models after forward selection : ")
        new_features = list(sfs.k_feature_names_)
        print(f"Features selected are : {new_features}")
        for model_name in model_dict.keys():
            test_model(x_train[new_features], y_train, x_test[new_features], y_test, model_name)
        print("---")

        # # Backward elimination
        # sfs = SFS(
        #     LinearRegression(),
        #     k_features=10,
        #     forward=False,
        #     scoring="neg_mean_squared_error",
        #     verbose=1
        # )
        # start_fs = time.time()
        # sfs.fit(df[predicting_columns], df[target_column])
        # stop_fs = time.time()
        # print(f"Backward_elimination on {file} took {stop_fs-start_fs}s")
        # # Run the experiments
        # print(f"Results of the models after backward elimination : ")
        # new_features = list(sfs.k_feature_names_)
        # print(f"Features selected are : {new_features}")
        # for model_name in model_dict.keys():
        #     test_model(x_train[new_features], y_train, x_test[new_features], y_test, model_name)
        # print("---")

