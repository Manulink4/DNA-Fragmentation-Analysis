
from data.create_dataset import *
from analysis_pipeline import *
from classification_model import *


def main(original=False, cancer_type='Breast_Cancer'):

    folder = '../data/'

    # Original or custom analysis
    if original:
        df, df_cancer, df_control = create_dataset_original(folder + cancer_type)
        confusion_matrix = original_analysis_pipeline(df, df_cancer, df_control)
        print(confusion_matrix)
        tn, fp, fn, tp = confusion_matrix.ravel()
        print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))

    else:
        df, df_cancer, df_control = create_dataset_original(folder + cancer_type)
        print("Original read shape:", df.shape, df_cancer.shape, df_control.shape)

        tp, fp, fn, tn = automl_pipeline(df, df_cancer, df_control, cancer_type)
        print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))


if __name__ == '__main__':
    use_original_pipeline = False

    # Cancer types: Breast_Cancer, Hepatocarcinoma, Lymphoma, Meduloblastoma, Prostate_Cancer
    main(original=use_original_pipeline, cancer_type='Breast_Cancer')
