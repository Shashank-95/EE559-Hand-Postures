#This is main file where all other files and functions are called
from FeatureExtraction import FeatureExtraction
from preprocessing import PreProcessing
from FeatureDimenstionAdjust import PrincipalComponentAnalysis
from CrossValidation import CrossValidation
import utils

#Feature Extraction
featureDataSet, labels = FeatureExtraction('/home/shashankns/Desktop/Masters/2nd_sem/EE559-MPR/559_final_project/Code/EE559-Hand-Postures/dataset/D_train.csv')
PreProcessedData = PreProcessing(featureDataSet)
#testFeatureDataSet, testLabels = FeatureExtraction('/home/shashankns/Desktop/Masters/2nd_sem/EE559-MPR/559_final_project/Code/EE559-Hand-Postures/dataset/D_test.csv')
#TestPreProcessedData = PreProcessing(testFeatureDataSet)
#Save the preprocessed data into a csv file
#utils.np.savetxt("PreProcessedData.csv", PreProcessedData, delimiter=",")
#utils.np.savetxt("TestPreProcessedData.csv", TestPreProcessedData, delimiter=",")

#Feature Dimensionality
PCA_Data = PrincipalComponentAnalysis(PreProcessedData)
print(PCA_Data.shape)

#Plot the 3D feature space
plot = utils.plt.figure()
plot_3d = utils.plt.axes(projection = '3d')
x_points = PCA_Data[:,0]
y_points = PCA_Data[:,1]
z_points = PCA_Data[:,2]
plot_3d.scatter(x_points, y_points, z_points, c='r', marker='.')
utils.plt.title('Reduced Feature Space After PCA')
utils.plt.show()

gamma, c = CrossValidation(PCA_Data, labels)
print("Gamma: ",gamma, "C: ", c)