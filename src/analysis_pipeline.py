import pandas as pd
from sklearn.model_selection import train_test_split
from features.feature_selection import get_significant_cpg_index
from sklearn.model_selection import LeaveOneOut
from features.feature_selection import *
from classification_model import *
import random

def original_analysis_pipeline(df, df_cancer, df_control):
    # Dimensionality reduction
    print("Applying dimensionality reduction...")

    df_significant_cpg = get_significant_cpg_index(df)

    # print(df.shape)
    # print(df_significant_cpg.shape)

    # Dataset post-processing
    df_significant_cpg = df_significant_cpg.T

    df_significant_cpg.index = list(range(0, df_significant_cpg.shape[0]))
    df_final = df_significant_cpg.copy()

    df_final["Target"] = [0] * df_cancer.shape[1] + [1] * df_control.shape[1]

    X = df_final.loc[:, df_final.columns != 'Target']
    y = df_final["Target"]

    print("Fitting classification model...")
    confusion_matrix = original_classification_pipeline(df_final)
    # print(confusion_matrix)
    tn, fp, fn, tp = confusion_matrix.ravel()

    return tp, fp, fn, tn


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


def holdout_pipeline(df, df_cancer, df_control, cancer_type, iterations=15):
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(iterations):
        print("Iter", i)
        # Train-Test split
        X_train, X_test, y_train, y_test = train_test_splitter(df, df_cancer, df_control)

        # # Imput missing values
        # if cancer_type == "Breast_Cancer" or cancer_type == "Hepatocarcinoma":
        #     X_train = imput_missing_values(X_train, y_train)
        #     X_test = imput_missing_values(X_test, y_test)

        X_merged = pd.concat([X_train, X_test], axis=0).dropna(axis=1).reset_index(drop=True)
        X_train = X_merged[:X_train.shape[0]]
        X_test = X_merged[X_train.shape[0]:]

        # Normalize data
        X_train, X_test= normalize_df(X_train, X_test)

        # Feature Selection / Dimensionality Reduction
        # features = use_mrmr(X_train, y_train, 15)
        # selected_cpgs = [df_control.index[i] for i in features]
        # X_train_redux = X_train[features]
        # X_test_redux = X_test[features]
        X_train_redux, X_test_redux = use_kbest(X_train, X_test, y_train, 50)


        # Classification model
        y_pred = use_svm(X_train_redux, X_test_redux, y_train)
        confusion_matrix = metrics.confusion_matrix(y_pred, y_test)
        # print(confusion_matrix)

        tn_, fp_, fn_, tp_ = confusion_matrix.ravel()
        tn += tn_
        fp += fp_
        fn += fn_
        tp += tp_
    return tp, fp, fn, tn


def loocv_pipeline(df, df_cancer, df_control, cancer_type, k):
    X, y = create_X_y(df, df_cancer, df_control)
    cv = LeaveOneOut()

    y_pred_list = []
    y_proba_list = []
    y_test_list = []
    for train_idx, test_idx in cv.split(X):
        # print("Iter", test_idx[0])
        # Train-Test split
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Normalize data
        X_train, X_test = normalize_df(X_train, X_test)

        # Feature Selection / Dimensionality Reduction
        # selected_cpgs, X_train_redux, X_test_redux = use_mrmr(X.columns, X_train, X_test, y_train, k)
        X_train_redux, X_test_redux = use_kbest(X_train, X_test, y_train, k)
        # X_train_redux, X_test_redux = use_SelectFromModel(X_train, X_test, y_train, k)
        print(X_train_redux.shape)

        # train_selected_cpgs, significant_cpg_coords = get_significant_cpg_index(X_train.T)
        # X_train_redux = train_selected_cpgs.T
        # print(X_train_redux.shape)
        # X_test_redux = pd.DataFrame(X_test).T.iloc[significant_cpg_coords].T

        # Classification model
        y_pred = use_svm(X_train_redux, X_test_redux, y_train)
        y_pred_list.append(y_pred)
        # y_proba_list.append(y_proba)
        y_test_list.append(y_test)
        # print(y_proba.flatten())

    confusion_matrix = metrics.confusion_matrix(y_test_list, y_pred_list)
    # fpr, tpr, thresh = metrics.roc_curve(y_test_list, y_proba_list)
    # auc = metrics.roc_auc_score(y_test_list, y_proba_list)
    # print("Confusion matrix:\n", confusion_matrix)
    tp, fp, fn, tn = confusion_matrix.ravel()

    # return [confusion_matrix, fpr, tpr, thresh, auc, y_test_list, y_proba_list, y_pred_list]
    return tp, fp, fn, tn

def skfcv_pipeline(df, df_cancer, df_control, cancer_type,k):
    X, y = create_X_y(df, df_cancer, df_control)
    skfCV = StratifiedKFold(n_splits=5, shuffle=True)

    y_pred_list, y_test_list = [], []
    for train_index, test_index in skfCV.split(X, y):
        # Train-Test split
        X_train, X_test, y_train, y_test = X.iloc[list(train_index)], X.iloc[list(test_index)],\
                                           y.iloc[list(train_index)], y.iloc[list(test_index)]

        # Normalize data
        X_train, X_test = normalize_df(X_train, X_test)

        # Feature Selection / Dimensionality Reduction
        selected_cpgs, X_train_redux, X_test_redux = use_mrmr(X.columns, X_train, X_test, y_train, k)
        # X_train_redux, X_test_redux = use_kbest(X_train, X_test, y_train, 50)
        # X_train_redux, X_test_redux = use_SelectFromModel(X_train, X_test, y_train, 40)

        # Classification model
        y_pred = use_automl(X_train_redux, X_test_redux, y_train, time=3)
        # y_pred = use_svm(X_train_redux, X_test_redux, y_train)

        y_pred_list.extend(y_pred)
        y_test_list.extend(list(y_test))

    confusion_matrix = metrics.confusion_matrix(y_pred_list, y_test_list)
    print("Confusion matrix:\n", confusion_matrix)
    tn, fp, fn, tp = confusion_matrix.ravel()
    return tn, fp, fn, tp


def all_cancer_pipeline(df):
    X, y = df.iloc[:, :-2], df["Target"]
    cv = LeaveOneOut()

    y_pred_list, y_test_list = [], []
    for train_idx, test_idx in cv.split(X):
        # Train-Test split
        X_train, X_test, y_train, y_test = X.iloc[list(train_idx)], X.iloc[list(test_idx)], \
                                           y.iloc[list(train_idx)], y.iloc[list(test_idx)]

        # Normalize data
        X_train, X_test = normalize_df(X_train, X_test)

        # Feature Selection / Dimensionality Reduction
        # selected_cpgs, X_train_redux, X_test_redux = use_mrmr(X.columns, X_train, X_test, y_train, 40)
        X_train_redux, X_test_redux = use_kbest(X_train, X_test, y_train, 50)
        # X_train_redux, X_test_redux = use_SelectFromModel(X_train, X_test, y_train, 100)

        # Classification model
        # y_pred = use_automl(X_train_redux, X_test_redux, y_train, time=1)
        y_pred = use_svm(X_train_redux, X_test_redux, y_train)

        y_pred_list.extend(y_pred)
        y_test_list.extend(list(y_test))

    confusion_matrix = metrics.confusion_matrix(y_pred_list, y_test_list)
    print("Confusion matrix:\n", confusion_matrix)
    tn, fp, fn, tp = confusion_matrix.ravel()
    return tn, fp, fn, tp


def all_cancer_LOOCV(df):
    X, y = df.iloc[:, :-2], df["Target"]
    cv = LeaveOneOut()

    y_pred_list = []
    y_test_list = []
    for train_idx, test_idx in cv.split(X):
        print("Iter", test_idx[0])
        # Train-Test split
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Normalize data
        X_train, X_test = normalize_df(X_train, X_test)

        # Feature Selection / Dimensionality Reduction
        # selected_cpgs, X_train_redux, X_test_redux = use_mrmr(X.columns, X_train, X_test, y_train, 40)
        X_train_redux, X_test_redux = use_kbest(X_train, X_test, y_train, 50)
        # X_train_redux, X_test_redux = use_SelectFromModel(X_train, X_test, y_train, 40)

        # Classification model
        y_pred = use_svm(X_train_redux, X_test_redux, y_train)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

    confusion_matrix = metrics.confusion_matrix(y_pred_list, y_test_list)
    print("Confusion matrix:\n", confusion_matrix)
    tn, fp, fn, tp = confusion_matrix.ravel()

    return tp, fp, fn, tn


