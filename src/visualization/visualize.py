
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from umap import UMAP


def use_pca(df):
    pca = PCA(n_components=10)
    df_pca = pca.fit_transform(df)
    return df_pca


def use_kpca(df):
    kpca = KernelPCA(n_components=10, kernel='linear')
    df_pca = kpca.fit_transform(df)
    return df_pca


def use_umap(df):
    reducer = UMAP()
    embedding = reducer.fit_transform(df)
    return embedding


def use_tsne(df):
    tsne_embedding = TSNE(n_components=2, learning_rate='auto').fit_transform(df)
    return tsne_embedding
