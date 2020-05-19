import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import os

try:
    os.chdir("/home/ccome/git_files/enriched")
except OSError:
    print("Cannot change directory.")

print("File available here : ", os.listdir())

for file in os.listdir():
    if "universities" in file:
        df = pd.read_csv(file)
        df = df.drop(columns=df.columns[df.dtypes == "object"])
        df = df.fillna(0.0)
        df_train, df_test = train_test_split(df)
        predicting_columns = list(df.columns)
        predicting_columns.remove("target")

        # Training with random forest
        rf = RandomForestRegressor(max_depth=10).fit(
            df_train[predicting_columns], df_train["target"]
        )
        rf_mse = mean_squared_error(
            df_test["target"], rf.predict(df_test[predicting_columns])
        )
        print(f"MSE for random forest on file {file} is {rf_mse}")

        # Training with support vector machine
        svr = SVR().fit(df_train[predicting_columns], df_train["target"])
        svr_mse = mean_squared_error(
            df_test["target"], svr.predict(df_test[predicting_columns])
        )
        print(f"MSE for Support Vector Regressor on file {file} is {svr_mse}")

        # Training with linear regression
        lr = LinearRegression().fit(df_train[predicting_columns], df_train["target"])
        lr_mse = mean_squared_error(
            df_test["target"], lr.predict(df_test[predicting_columns])
        )
        print(f"MSE for Linear Regression on file {file} is {lr_mse}")