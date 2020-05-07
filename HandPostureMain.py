#This is main file where all other files and functions are called
from FeatureExtraction import FeatureExtraction
from preprocessing import Standardization
from preprocessing import Normalization
from FeatureDimenstionAdjust import PrincipalComponentAnalysis
from CrossValidation import CrossValidation
from Training import SupportVectorMachine
from Training import NaiveBayes
from Training import RandomForestClassifierWithXGBoost
from Training import KNearestNeighbors
import utils


#Feature Extraction and Preprocessing
def ExtractFeatures():
    trainFeatureDataSet, trainlabels = FeatureExtraction('/home/shashankns/Desktop/Masters/2nd_sem/EE559-MPR/559_final_project/Code/EE559-Hand-Postures/dataset/D_train.csv')
    testFeatureDataSet, testLabels = FeatureExtraction('/home/shashankns/Desktop/Masters/2nd_sem/EE559-MPR/559_final_project/Code/EE559-Hand-Postures/dataset/D_test.csv')
    TrainPreProcessedData, TestPreProcessedData = Standardization(trainFeatureDataSet, testFeatureDataSet)
    return TrainPreProcessedData, trainlabels, TestPreProcessedData, testLabels 

#Feature Dimensionality
def DimensionalityReduction(TrainPreProcessedData, TestPreProcessedData):
    PCA_train, PCA_test = PrincipalComponentAnalysis(TrainPreProcessedData, TestPreProcessedData)
    return PCA_train, PCA_test


#Plot the 3D feature space
def PlotPCAData(PCA_train, labels,PCA_test):
    plot = utils.plt.figure()
    plot_3d = utils.plt.axes(projection = '3d')
    x_points = PCA_test[:,0]
    y_points = PCA_test[:,1]
    z_points = PCA_test[:,2]
    plot_3d.scatter(x_points, y_points, z_points, c='r', marker='.')
    utils.plt.title('Reduced Feature Space After PCA')
    utils.plt.show()

#Training
def Training(PCA_train, labels,PCA_test,testLabels):
    #NaiveBayes(PCA_train,labels[:,0], PCA_test,testLabels[:,0])
    #SupportVectorMachine(PCA_train,labels[:,0], PCA_test,testLabels[:,0], 4.9341, 0.3404)
    #RandomForestClassifierWithXGBoost(PCA_train,labels[:,0], PCA_test,testLabels[:,0])
    KNearestNeighbors(PCA_train,labels[:,0], PCA_test,testLabels[:,0])

TrainPreProcessedData, trainlabels, TestPreProcessedData, testLabels = ExtractFeatures()
#PCA_train, PCA_test = DimensionalityReduction(TrainPreProcessedData, TestPreProcessedData)
#PlotPCAData(PCA_train, labels,PCA_test)
Training(PCA_train, trainlabels,PCA_test,testLabels)
