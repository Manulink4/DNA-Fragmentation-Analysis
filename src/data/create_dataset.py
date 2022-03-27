import glob
import pandas as pd
from pathlib import Path

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




