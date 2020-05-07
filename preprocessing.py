#Read the raw data and preprocess it. Standardise/Normalize
#Arguments: Feature Extracted Dataset
#return value: Zero Mean, Std one Standardized data
import utils
def Standardization(TrainData, TestData):
    standardize = utils.preprocessing.StandardScaler()
    standardized_Train = standardize.fit_transform(TrainData)
    standardized_Test = standardize.transform(TestData)
    return standardized_Train, standardized_Test

def Normalization(TrainData, TestData):
    normalize = utils.preprocessing.Normalizer()
    normalized_Train = normalize.fit_transform(TrainData)
    normalized_Test = normalize.transform(TestData)
    return normalized_Train, normalized_Test

