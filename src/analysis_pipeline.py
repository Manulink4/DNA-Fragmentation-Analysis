import pandas as pd
from sklearn.model_selection import train_test_split
from features.feature_selection import get_significant_cpg_index
from sklearn.model_selection import LeaveOneOut
from features.feature_selection import *
from classification_model import *


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


#####################################################
#           Custom methods                          #
#####################################################

def create_X_y(dataframe, df_cancer, df_control):
    df = dataframe.T
    df["Target"] = [1] * df_cancer.shape[1] + [0] * df_control.shape[1]
    X = df.loc[:, df.columns != 'Target']
    y = df["Target"]
    return X, y


def train_test_splitter(df, df_cancer, df_control):
    X, y = create_X_y(df, df_cancer, df_control)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y
    )
    return X_train, X_test, y_train, y_test


def loop_classifier_pipeline(df, df_cancer, df_control, cancer_type, iterations=15):
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(iterations):
        print("Iter", i)
        # Train-Test split
        X_train, X_test, y_train, y_test = train_test_splitter(df, df_cancer, df_control)

        # Imput missing values
        if cancer_type == "Breast_Cancer" or cancer_type == "Hepatocarcinoma":
            X_train = imput_missing_values(X_train, y_train)
            X_test = imput_missing_values(X_test, y_test)

        X_merged = pd.concat([X_train, X_test], axis=0).dropna(axis=1).reset_index(drop=True)
        X_train = X_merged[:X_train.shape[0]]
        X_test = X_merged[X_train.shape[0]:]

        # Normalize data
        X_train = normalize_df(X_train)
        X_test = normalize_df(X_test)

        # Feature Selection / Dimensionality Reduction
        features = use_mrmr(X_train, y_train, 15)
        selected_cpgs = [df_control.index[i] for i in features]
        X_train_redux = X_train[features]
        X_test_redux = X_test[features]

        # Classification model
        y_pred = use_svm(X_train_redux, X_test_redux, y_train)
        confusion_matrix = metrics.confusion_matrix(y_pred, y_test)
        print(confusion_matrix)

        tn_, fp_, fn_, tp_ = confusion_matrix.ravel()
        tn += tn_
        fp += fp_
        fn += fn_
        tp += tp_
    return tp, fp, fn, tn


def loocv_pipeline(df, df_cancer, df_control, cancer_type):
    X, y = create_X_y(df, df_cancer, df_control)
    cv = LeaveOneOut()

    y_pred_list = []
    y_test_list = []
    for train_idx, test_idx in cv.split(X):
        print("Iter", test_idx[0])
        # Train-Test split
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Normalize data
        # X_train = normalize_df(X_train)
        # X_test = normalize_df(X_test)
        X_train, X_test = normalize_df(X_train, X_test)

        # Feature Selection / Dimensionality Reduction
        features = use_mrmr(X_train, y_train, 5)
        selected_cpgs = [df_control.index[i] for i in features]
        print(selected_cpgs)
        X_train_redux = X_train[features]
        X_test_redux = X_test[features]


        # Classification model
        y_pred = use_svm(X_train_redux, X_test_redux, y_train)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

    confusion_matrix = metrics.confusion_matrix(y_pred_list, y_test_list)
    print(confusion_matrix)
    tn, fp, fn, tp = confusion_matrix.ravel()

    return tp, fp, fn, tn
