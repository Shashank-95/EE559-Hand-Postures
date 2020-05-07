#Feature Selection and Dimenstionality Reduction
import utils

def PrincipalComponentAnalysis(TrainPreProcessedData, TestPreProcessedData):
    pca = utils.PCA(n_components=3)
    PCATrain = pca.fit_transform(TrainPreProcessedData)
    PCATest = pca.transform(TestPreProcessedData)
    return PCATrain, PCATest



