
from data.create_dataset import *
from analysis_pipeline import *


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
        df, df_cancer, df_control = create_dataset_filtered(folder + cancer_type)
        print("Original read shape:", df.shape, df_cancer.shape, df_control.shape)

        # Train-Test split
        X_train, X_test, y_train, y_test = train_test_splitter(df, df_cancer, df_control)

        # Imput missing values
        if cancer_type == "Breast_Cancer" or cancer_type == "Hepatocarcinoma":
            X_train_imp = imput_missing_values(X_train, y_train)
            X_test_imp = imput_missing_values(X_test, y_test)

        # Feature Selection / Dimensionality Reduction
        features = use_mrmr(X_train_imp, y_train)
        print(features)


        # Classification model
        # Confusion Matrix / Results
        #
        # confusion_matrix = custom_analysis_pipeline(df, df_cancer, df_control)
        # print(confusion_matrix)
        # tn, fp, fn, tp = confusion_matrix.ravel()
        # print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))



if __name__ == '__main__':
    use_original_pipeline = False

    # Cancer types: Breast_Cancer, Hepatocarcinoma, Lymphoma, Meduloblastoma, Prostate_Cancer
    main(original=use_original_pipeline, cancer_type='Breast_Cancer')
