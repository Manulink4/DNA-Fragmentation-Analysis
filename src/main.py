
from data.dataset import create_dataset
from analysis_pipeline import *


def main(original=False, cancer_type='Breast_Cancer'):
    # Dataset creation
    print("Creating dataset...")
    folder = '../data/'
    df, df_control, df_cancer = create_dataset(folder + cancer_type)

    # Original or custom analysis
    if original:
        confusion_matrix = original_analysis_pipeline(df, df_control, df_cancer)
    else:
        confusion_matrix = custom_analysis_pipeline(df, df_control, df_cancer)

    print(confusion_matrix)
    tn, fp, fn, tp = confusion_matrix.ravel()
    print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))


if __name__ == '__main__':
    use_original_pipeline = False

    # Cancer types: Breast_Cancer, Hepatocarcinoma, Lymphoma, Meduloblastoma, Prostate_Cancer
    main(original=use_original_pipeline, cancer_type='Breast_Cancer')
