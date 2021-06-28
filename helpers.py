import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

def outlier_threshold(data, col_name, w1=0.05, w2=0.95):
    q1 = data[col_name].quantile(w1)
    q3 = data[col_name].quantile(w2)
    IQR = q3 - q1
    up = q3 + 1.5 * IQR
    low = q1 - 1.5 * IQR
    return up,low



def check_outlier(data, col_name, w1=0.05, w2=0.95):
    up, low = outlier_threshold(data, col_name, w1, w2)
    if data[(data[col_name]<low) | (data[col_name]>up)][col_name].any(axis=None):
        return True
    else:
        return False

def grab_outliers(data, col_name, index=False, w1=0.05, w2=0.95):
    up, low = outlier_threshold(data, col_name, w1, w2)
    if data[(data[col_name] < low) | (data[col_name] > up)][col_name].shape[0]>10:
        print(data[(data[col_name] < low) | (data[col_name] > up)][col_name].shape[0])
    else:
        print(data[(data[col_name] < low) | (data[col_name] > up)][col_name])
    if index:
        outlier_index = data[(data[col_name] < low) | (data[col_name] > up)][col_name].index
        return outlier_index

def remove_outlier(data, col_name, w1, w2):
    up, low = outlier_threshold(data, col_name, w1, w2)
    df_without_outliers = data[~(data[col_name]<low)|(data[col_name]>up)]
    return df_without_outliers

def replace_with_thresholds(data, col_name, w1, w2):
    up, low = outlier_threshold(data, col_name, w1, w2)
    data[(data.col_name>up)] = up
    data[data.col_name<low] = low

def missing_values_table(data, na_name=False):
    na_columns = [col for col in data.columns if data[col].isnull().sum() > 0]
    n_miss = data[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (data[na_columns].isnull().sum() / data.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def missing_vs_target(data, target, na_columns): #eksik değerlerin target değişken durumu
    temp_df = data.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(data, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(data, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype not in ["int", "float"]]
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}))


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car