import glob
import pandas as pd
from pathlib import Path


def create_dataset(data_folder):

    path = Path(data_folder)

    df_control = pd.concat(
        [pd.read_csv(f, sep='\t', header=None, index_col=0) for f in glob.glob(rf'{path}/control/*.breakage')], axis=1
    ).dropna()

    df_cancer = pd.concat(
        [pd.read_csv(f, sep='\t', header=None, index_col=0) for f in glob.glob(rf'{path}/cancer/*.breakage')], axis=1
    ).dropna()

    df = pd.concat([df_control, df_cancer], axis=1).dropna()
    dataframes = [df, df_control, df_cancer]

    return dataframes
