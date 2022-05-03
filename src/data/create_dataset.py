import glob
import pandas as pd
from pathlib import Path
import os

import numpy as np


#####################################################
#           Original method                         #
#####################################################
def create_dataset_original(data_folder):
    path = Path(data_folder)

    df_cancer = pd.concat(
        [pd.read_csv(f, sep='\t', header=None, index_col=0) for f in glob.glob(rf'{path}/cancer/*.breakage')], axis=1
    ).dropna()
    df_control = pd.concat(
        [pd.read_csv(f, sep='\t', header=None, index_col=0) for f in glob.glob(rf'{path}/control/*.breakage')], axis=1
    ).dropna()

    df = pd.concat([df_cancer, df_control], axis=1).dropna()
    # df.columns = [i for i in range(df.shape[1])]
    dataframes = [df, df_cancer, df_control]

    return dataframes


#####################################################
#           Custom methods                          #
#####################################################
def create_dataset_all_cpgs(data_folder):
    path = Path(data_folder)

    df_cancer = pd.concat(
        [pd.read_csv(f, sep='\t', header=None, index_col=0) for f in glob.glob(rf'{path}/cancer/*.breakage')], axis=1
    )
    df_control = pd.concat(
        [pd.read_csv(f, sep='\t', header=None, index_col=0) for f in glob.glob(rf'{path}/control/*.breakage')], axis=1
    )

    df = pd.concat([df_cancer, df_control], axis=1)
    dataframes = [df, df_cancer, df_control]
    return dataframes


def filter_nans(dataframe):
    df = dataframe
    nans = df.isnull().sum(axis=1).tolist()
    index = [index for index, value in enumerate(nans) if value <= int(dataframe.shape[1]*0.3)]
    filtered_df = df.iloc[index]
    return filtered_df


def create_dataset_filtered(data_folder):
    path = Path(data_folder)

    df_cancer = pd.concat(
        [pd.read_csv(f, sep='\t', header=None, index_col=0) for f in glob.glob(rf'{path}/cancer/*.breakage')], axis=1
    )
    df_control = pd.concat(
        [pd.read_csv(f, sep='\t', header=None, index_col=0) for f in glob.glob(rf'{path}/control/*.breakage')], axis=1
    )

    df_cancer_filtered = filter_nans(df_cancer)
    df_control_filtered = filter_nans(df_control)
    df_filtered = pd.concat([df_cancer_filtered, df_control_filtered], axis=1)

    dataframes = [df_filtered, df_cancer_filtered, df_control_filtered]
    return dataframes


def create_dataset_all_cancer(data_folder):
    cancer_types = ["Breast_Cancer", "Hepatocarcinoma", "Lymphoma", "Meduloblastoma", "Prostate_Cancer"]

    dataframes = []
    list_target = []
    list_cancer_type = []
    for cancer in cancer_types:
        df, df_control, df_cancer = create_dataset_original(data_folder + cancer)
        dataframes.append(df)
        list_target.append([0] * df_control.shape[1])
        list_target.append([1] * df_cancer.shape[1])
        list_cancer_type.append([cancer] * (df_control.shape[1] + df_cancer.shape[1]))

    df_full = pd.concat(dataframes, axis=1).dropna()
    df_final = df_full.T

    target = [t for l in list_target for t in l]
    cancer_type = [c for l in list_cancer_type for c in l]

    df_final["cancer_type"] = cancer_type
    df_final["Target"] = target

    return df_final


