import numpy as np
from scipy.stats import mannwhitneyu

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler


def mw_test(row):
    """
    Mann-Whitney significance test
    """
    length = len(row)
    index = np.argmax(np.diff(sorted(row.values)))+1

    if 1 < index < (length-2):
        return mannwhitneyu(list(row)[0:index], list(row)[(index+1):length]).pvalue < 0.07
    else:
        return False


def get_significant_cpg_index(df):
    dataframe = df
    mw_results = dataframe.apply(lambda row: mw_test(row), axis=1)
    significant_cpg_coords = [idx for idx, x in enumerate(mw_results.values) if x]
    df_significant_cpg = df.iloc[significant_cpg_coords]

    return df_significant_cpg


def use_pca(df, components=40):
    pca = PCA(n_components=components)
    df_pca = pca.fit_transform(df)
    return df_pca

def use_kpca(df, components):
    kpca = KernelPCA(n_components=7, kernel='linear')
    df_pca = kpca.fit_transform(df)
    return df_pca

def normalize_df(df):
    scaler = StandardScaler()
    normalized_df = scaler.fit_transform(df)
    return normalized_df



