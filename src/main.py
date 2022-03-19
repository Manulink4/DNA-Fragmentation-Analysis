
from data.dataset import *
from analysis_pipeline import *


def main(original=False, cancer_type='Breast_Cancer'):
    # Dataset creation
    print("Creating dataset...")
    folder = '../data/'
    df, df_control, df_cancer = create_dataset_augmented(folder + cancer_type)
    print(df.shape, df_control.shape, df_cancer.shape)

    nans = df.isnull().sum(axis=1).tolist()
    print(len([x for x in nans if x < 2]))

    # # Original or custom analysis
    # if original:
    #     confusion_matrix = original_analysis_pipeline(df, df_control, df_cancer)
    # else:
    #     confusion_matrix = custom_analysis_pipeline(df, df_control, df_cancer)
    #
    # print(confusion_matrix)
    # tn, fp, fn, tp = confusion_matrix.ravel()
    # print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))


if __name__ == '__main__':
    use_original_pipeline = True

    # Cancer types: Breast_Cancer, Hepatocarcinoma, Lymphoma, Meduloblastoma, Prostate_Cancer
    main(original=use_original_pipeline, cancer_type='Breast_Cancer')
