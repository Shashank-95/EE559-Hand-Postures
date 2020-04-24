#This is main file where all other files and functions are called
from FeatureExtraction import FeatureExtraction
from preprocessing import PreProcessing

featureDataSet, labels = FeatureExtraction('/home/shashankns/Desktop/Masters/2nd_sem/EE559-MPR/559_final_project/Code/EE559-Hand-Postures/dataset/D_train.csv')
PreProcessedData = PreProcessing(featureDataSet)