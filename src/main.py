import pandas as pd

from data.create_dataset import *
from analysis_pipeline import *
from classification_model import *
from math import sqrt


def main(pipeline="custom", cancer_type='Breast_Cancer'):

    folder = '../data/'

    if pipeline == "original":
        df, df_cancer, df_control = create_dataset_original(folder + cancer_type)
        tn, fp, fn, tp = original_analysis_pipeline(df, df_cancer, df_control)

    elif pipeline == "custom":
        df, df_cancer, df_control = create_dataset_original(folder + cancer_type)
        tp, fp, fn, tn = loocv_pipeline(df, df_cancer, df_control, cancer_type)

    else:
        df = create_dataset_all_cancer(folder)
        tp, fp, fn, tn = all_cancer_pipeline(df)

    print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))
    print("Phi coefficient:", (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))


if __name__ == '__main__':

    # Pipeline modes: original, custom, all_cancer
    pipeline_mode = "custom"

    # Cancer types: Breast_Cancer, Hepatocarcinoma, Lymphoma, Meduloblastoma, Prostate_Cancer
    main(pipeline=pipeline_mode, cancer_type='Breast_Cancer')

