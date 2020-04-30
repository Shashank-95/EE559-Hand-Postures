#Feature Selection and Dimenstionality Reduction
import utils

def PrincipalComponentAnalysis(PreProcessedData):
    pca = utils.PCA(n_components = 3, copy = True, whiten=True)
    pca.fit(PreProcessedData)
    PCA_Data = pca.fit_transform(PreProcessedData)
    return PCA_Data



