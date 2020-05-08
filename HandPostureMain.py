#This is main file where all other files and functions are called
from FeatureExtraction import FeatureExtraction
from preprocessing import Standardization
from preprocessing import Normalization
from FeatureDimenstionAdjust import PrincipalComponentAnalysis
from FeatureDimenstionAdjust import LinearDiscriminantAnalysis
from FeatureDimenstionAdjust import FeatureSelection
from CrossValidation import CrossValidation
from Training import SupportVectorMachine
from Training import NaiveBayes
from Training import RandomForestClassifierWithXGBoost
from Training import KNearestNeighbors
from Training import DiscriminantAnalysisLinear
from Training import DiscriminantAnalysisQuadratic
from Training import Logistic_Regression
import utils

#Feature Extraction and Preprocessing
def ExtractFeatures(selection):
    trainFeatureDataSet, trainlabels = FeatureExtraction('/home/shashankns/Desktop/Masters/2nd_sem/EE559-MPR/559_final_project/Code/EE559-Hand-Postures/dataset/D_train.csv')
    testFeatureDataSet, testLabels = FeatureExtraction('/home/shashankns/Desktop/Masters/2nd_sem/EE559-MPR/559_final_project/Code/EE559-Hand-Postures/dataset/D_test.csv')
    
    if(selection == 0):
        TrainPreProcessedData, TestPreProcessedData = Standardization(trainFeatureDataSet, testFeatureDataSet)
    elif(selection ==1):
        TrainPreProcessedData, TestPreProcessedData = Normalization(trainFeatureDataSet, testFeatureDataSet)
    
    return TrainPreProcessedData, trainlabels, TestPreProcessedData, testLabels 

#Feature Dimensionality
def DimensionalityReduction(selection, TrainPreProcessedData, trainLabels, TestPreProcessedData, testLabels, num_components):
    if(selection == 0):
        TrainRed_Dimension, TestRed_Dimension = PrincipalComponentAnalysis(TrainPreProcessedData, TestPreProcessedData, num_components)
    elif(selection ==1):
        TrainRed_Dimension, TestRed_Dimension = LinearDiscriminantAnalysis(TrainPreProcessedData, trainLabels, TestPreProcessedData, testLabels, num_components)
    elif(selection == 2):
        TrainRed_Dimension, TestRed_Dimension = FeatureSelection(TrainPreProcessedData, trainLabels, TestPreProcessedData, testLabels, num_components)
    
    #Mutual Info

    return TrainRed_Dimension, TestRed_Dimension


#Plot the 3D feature space
def PlotPCAData(PCA_train, labels,PCA_test):
    #Collecting Indexes of the classes
    class1_idx = utils.np.where(labels ==1)
    class2_idx = utils.np.where(labels ==2)
    class3_idx = utils.np.where(labels ==3)
    class4_idx = utils.np.where(labels ==4)
    class5_idx = utils.np.where(labels ==5)

    x = PCA_train[:,0]
    y = PCA_train[:,1]
    z = PCA_train[:,2]

    x1 = x[class1_idx]
    x2 = x[class2_idx]
    x3 = x[class3_idx]
    x4 = x[class4_idx]
    x5 = x[class5_idx]

    y1 = y[class1_idx]
    y2 = y[class2_idx]
    y3 = y[class3_idx]
    y4 = y[class4_idx]
    y5 = y[class5_idx]

    z1 = z[class1_idx]
    z2 = z[class2_idx]
    z3 = z[class3_idx]
    z4 = z[class4_idx]
    z5 = z[class5_idx]

    plot = utils.plt.figure()
    plot_3d = utils.plt.axes(projection = '3d')

    c1 = plot_3d.scatter(x1, y1, z1, c='r', marker='.', label='Class 1')
    c2 = plot_3d.scatter(x2, y2, z2, c='g', marker='.', label='Class 2')
    c3 = plot_3d.scatter(x3, y3, z3, c='b', marker='.', label='Class 3')
    c4 = plot_3d.scatter(x4, y4, z4, c='m', marker='.', label='Class 4')
    c5 = plot_3d.scatter(x5, y5, z5, c='c', marker='.', label='Class 5')


    utils.plt.title('Reduced Feature Space After PCA')
    utils.plt.legend()
    utils.plt.show()

#Training
def Training(selection, TrainRed_Dimension, labels,TestRed_Dimension,testLabels):
    resultList = []
    if(selection==1 or selection==0):
        resultList.append(NaiveBayes(TrainRed_Dimension,labels[:,0], TestRed_Dimension,testLabels[:,0]))
    if(selection==2 or selection ==0):
        resultList.append(SupportVectorMachine(TrainRed_Dimension,labels[:,0], TestRed_Dimension,testLabels[:,0], 4.9341, 0.3404))
    if(selection == 3 or selection ==0):
        resultList.append(RandomForestClassifierWithXGBoost(TrainRed_Dimension,labels[:,0], TestRed_Dimension,testLabels[:,0]))
    if(selection ==4 or selection ==0):
        resultList.append(KNearestNeighbors(TrainRed_Dimension,labels[:,0], TestRed_Dimension,testLabels[:,0]))
    if(selection == 5 or selection ==0):
        resultList.append(DiscriminantAnalysisLinear(TrainRed_Dimension,labels[:,0], TestRed_Dimension,testLabels[:,0]))
    if(selection ==6 or selection ==0):
        resultList.append(DiscriminantAnalysisQuadratic(TrainRed_Dimension,labels[:,0], TestRed_Dimension,testLabels[:,0]))
    if(selection ==7 or selection ==0):
        resultList.append(Logistic_Regression(TrainRed_Dimension,labels[:,0], TestRed_Dimension,testLabels[:,0]))
    return resultList

def MainFunction(dict):
    TrainPreProcessedData, trainlabels, TestPreProcessedData, testLabels = ExtractFeatures(dict['preprocessing'])
    PCA_train, PCA_test = DimensionalityReduction(dict['dimensionality_reduction'], TrainPreProcessedData, trainlabels[:,0], TestPreProcessedData,testLabels[:,0],3)
    #PlotPCAData(PCA_train, trainlabels[:,0],PCA_test)
    resultList = Training(dict['classifier'], PCA_train, trainlabels,PCA_test,testLabels)
    return resultList