import pandas as pd

from features.feature_selection import get_significant_cpg_index
from features.feature_selection import *
from classification_model import original_classification_pipeline


def original_analysis_pipeline(df, df_cancer, df_control):
    # Dimensionality reduction
    print("Applying dimensionality reduction...")
    df_significant_cpg = get_significant_cpg_index(df)
    print(df.shape)
    print(df_significant_cpg.shape)

    # Dataset post-processing
    df_significant_cpg = df_significant_cpg.T
    df_significant_cpg.index = list(range(0, df_significant_cpg.shape[0]))
    df_final = df_significant_cpg.copy()

    df_final["Target"] = [0] * df_cancer.shape[1] + [1] * df_control.shape[1]

    print("Fitting classification model...")
    confusion_matrix = original_classification_pipeline(df_final)

    return confusion_matrix


def train_test_split(dataframe, df_cancer, df_control):
    df = dataframe.T
    df["Target"] = [0] * df_cancer.shape[1] + [1] * df_control.shape[1]



    return df.T

def custom_analysis_pipeline(df, df_cancer, df_control):
    normalized_df = normalize_df(df)
    transposed_df = normalized_df.T

    # Feature selection
    df_feature_sel = use_umap(transposed_df)
    df_final = pd.DataFrame(df_feature_sel)

    # Dataset post-processing
    # Control = 0, Cancer = 1
    df_final["Target"] = [0] * df_cancer.shape[1] + [1] * df_control.shape[1]

    # Classification model
    print("Fitting classification model...")
    confusion_matrix = original_classification_pipeline(df_final)

    return confusion_matrix
