# Paths and directories
from pathlib import Path
import glob

# dataset
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# ML
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# models
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

# utils
from data.dataset import create_dataset
from features.feature_engineering import get_significant_cpg_index
from features.feature_engineering import *
from classification_model import original_classification_pipeline


def use_original_analysis_pipeline(df, df_control, df_cancer):

    # Dimensionality reduction
    print("Applying dimensionality reduction...")
    df_significant_cpg = get_significant_cpg_index(df)

    # Dataset postprocessing
    df_significant_cpg = df_significant_cpg.T
    df_significant_cpg.index = list(range(0, df_significant_cpg.shape[0]))
    df_final = df_significant_cpg.copy()

    df_final["Target"] = [0] * df_control.shape[1] + [1] * df_cancer.shape[1]

    print("Fitting classification model...")
    confusion_matrix = original_classification_pipeline(df_final)

    return confusion_matrix


def analysis_pipeline(norm_df, df_control, df_cancer):
    # Dimensionality reduction
    transposed_df = norm_df.T
    components = 40
    df_pca = use_kpca(transposed_df, components)
    df_final = pd.DataFrame(df_pca)

    # Control = 0, Cancer = 1
    df_final["Target"] = [0] * df_control.shape[1] + [1] * df_cancer.shape[1]

    # Classification model
    print("Fitting classification model...")
    confusion_matrix = original_classification_pipeline(df_final)

    return confusion_matrix


def main(original=False, cancer_type='Breast_Cancer'):
    # Dataset creation
    print("Creating dataset...")
    folder = '../data/'
    df, df_control, df_cancer = create_dataset(folder + cancer_type)

    if original:
        confusion_matrix = use_original_analysis_pipeline(df, df_control, df_cancer)
    else:
        normalized_df = normalize_df(df)
        confusion_matrix = analysis_pipeline(normalized_df, df_control, df_cancer)

    print(confusion_matrix)
    tn, fp, fn, tp = confusion_matrix.ravel()
    print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))


if __name__ == '__main__':
    use_original_pipeline = False

    main(original=use_original_pipeline, cancer_type='Breast_Cancer')
