import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from mrmr import mrmr_classif


def normalize_df(X_train, X_test):
    scaler = StandardScaler()
    normalized_train = scaler.fit_transform(X_train)
    normalized_test = scaler.transform(X_test)
    return pd.DataFrame(normalized_train), pd.DataFrame(normalized_test)


#####################################################
#           Original method                         #
#####################################################

def mw_test(row):
    """
    Mann-Whitney significance test
    """
    length = len(row)
    index = np.argmax(np.diff(sorted(row.values)))+1

    if 1 < index < (length-2):
        return mannwhitneyu(list(row)[0:index], list(row)[(index+1):length]).pvalue < 0.07
    else:
        return False


def get_significant_cpg_index(df):
    dataframe = df
    mw_results = dataframe.apply(lambda row: mw_test(row), axis=1)
    significant_cpg_coords = [idx for idx, x in enumerate(mw_results.values) if x]
    df_significant_cpg = df.iloc[significant_cpg_coords]
    return df_significant_cpg


#####################################################
#           Custom methods                          #
#####################################################

def imput_missing_values(X, y):
    df_cancer = X[y == 1]
    df_control = X[y == 0]

    df_cancer_imp = pd.DataFrame(SimpleImputer(strategy='mean').fit(df_cancer).transform(df_cancer))
    df_control_imp = pd.DataFrame(SimpleImputer(strategy='mean').fit(df_control).transform(df_control))
    df_imputed = pd.concat([df_cancer_imp, df_control_imp], axis=0).dropna(axis=1).reset_index(drop=True)
    # dataframes = [df_imputed, df_cancer_imp, df_control_imp]
    return df_imputed


def use_mrmr(X, y, k):
    disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    X_disc = pd.DataFrame(disc.fit_transform(pd.DataFrame(X.T))).T
    y_disc = disc.transform(y.array.reshape(1, -1)).T

    selected_features = mrmr_classif(X=X_disc, y=y_disc, K=k, n_jobs=-1)
    return selected_features


