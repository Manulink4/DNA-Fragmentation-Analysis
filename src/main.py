
from data.dataset import *
from analysis_pipeline import *


def main(original=False, cancer_type='Breast_Cancer'):
    # Dataset creation
    print("Creating datasets...")
    folder = '../data/'
    df, df_cancer, df_control = create_dataset_original(folder + cancer_type)

    # Train-Test split
    x = train_test_split(df, df_cancer, df_control)
    print(x)
    # Feature Selection / Dimensionality Reduction
    # Classification model
    # Confusion Matrix / Results

    #
    # # Original or custom analysis
    # if original:
    #     confusion_matrix = original_analysis_pipeline(df, df_cancer, df_control)
    # else:
    #     confusion_matrix = custom_analysis_pipeline(df, df_cancer, df_control)
    #
    # print(confusion_matrix)
    # tn, fp, fn, tp = confusion_matrix.ravel()
    # print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))


if __name__ == '__main__':
    use_original_pipeline = True

    # Cancer types: Breast_Cancer, Hepatocarcinoma, Lymphoma, Meduloblastoma, Prostate_Cancer
    main(original=use_original_pipeline, cancer_type='Breast_Cancer')
