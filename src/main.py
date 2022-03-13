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
from analysis_pipeline import *


def main(original=False, cancer_type='Breast_Cancer'):
    # Dataset creation
    print("Creating dataset...")
    folder = '../data/'
    df, df_control, df_cancer = create_dataset(folder + cancer_type)

    if original:
        confusion_matrix = original_analysis_pipeline(df, df_control, df_cancer)
    else:
        confusion_matrix = custom_analysis_pipeline(df, df_control, df_cancer)

    print(confusion_matrix)
    tn, fp, fn, tp = confusion_matrix.ravel()
    print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))


if __name__ == '__main__':
    use_original_pipeline = False

    main(original=use_original_pipeline, cancer_type='Breast_Cancer')
