# ML
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# models
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import autosklearn.classification
import pandas as pd
from pprint import pprint
import pickle


def original_classification_pipeline(df_final):

    X = df_final.loc[:, df_final.columns != 'Target']
    y = df_final["Target"]

    clf = make_pipeline(svm.SVC(kernel='linear'))
    # clf = MLPClassifier(alpha=0.033, beta_1=0.99, beta_2=0.9, learning_rate_init=0.1)
    iterations = 250
    all_y_true, all_y_pred = [], []

    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(2/X.shape[0]), stratify=y
        )

        all_y_true.extend(y_test)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        all_y_pred.append(y_pred)

        print(y_pred)
        print(y_test)
    all_y_pred = [list(p) for p in all_y_pred]
    filtered_y_pred = [pred for sample in all_y_pred for pred in sample]

    confusion_matrix = metrics.confusion_matrix(all_y_true, filtered_y_pred)
    return confusion_matrix


def use_svm(X_train, X_test, y_train):
    clf = make_pipeline(svm.SVC(kernel='linear'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def use_mlp(X_train, X_test, y_train):
    clf = MLPClassifier(alpha=0.033, beta_1=0.99, beta_2=0.9, learning_rate_init=0.1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def get_metric_result(cv_results):
    results = pd.DataFrame.from_dict(cv_results)
    results = results[results['status'] == "Success"]
    cols = ['rank_test_scores', 'param_classifier:__choice__', 'mean_test_score']
    cols.extend([key for key in cv_results.keys() if key.startswith('metric_')])
    return results[cols]


def use_automl(X_train, X_test, y_train, time=5):

    print("AutoML...")
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60*time,
        memory_limit=None,  # n_jobs=-1,
        # include={
        #     'feature_preprocessor': ["no_preprocessing"]
        # },
        ensemble_size=1
    )
    print("Fitting...")
    automl.fit(X_train, y_train)

    print("Final Ensemble:")
    # print(automl.leaderboard())
    pprint(automl.show_models(), indent=4)
    with open('models/automl_test.pkl', 'wb') as f:
        pickle.dump(automl, f)

    print("Results:")
    print(get_metric_result(automl.cv_results_).to_string(index=False))

    y_pred = automl.predict(X_test)
    return list(y_pred)

