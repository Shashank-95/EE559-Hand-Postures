#Features are defined and extracted from the raw data and a new feature space is created and stored.

#Arguments: file name of the csv file (D_train.csv)
#return values: Feature Matrix and Labels containing classes and userID(FratureDataSet and labels)

import utils

def FeatureExtraction(fileName):
    rawData = utils.np.genfromtxt(fileName, delimiter=',')
    rawData = utils.np.delete(rawData, 0,0)
    rawData = utils.np.delete(rawData, 0,1)

    
    #Create 13 features from the raw data
    FeatureDataSet = utils.np.empty([13500, 13])

    #Creating labels (userID and Classes)
    labels = utils.np.empty([13500,2])
    labels[:,0] = rawData[:,0]
    labels[:,1] = rawData[:,1]


    #Feature 1: Number of recorded markers
    def feature_num(row_data):
        nan_array = utils.np.isnan(row_data)
        not_nan_array = ~ nan_array
        temp = row_data[not_nan_array]
        return len(temp)-2
    
    FeatureDataSet[:,0] =utils.np.apply_along_axis( feature_num, axis=1, arr=rawData )


    #Feature 2,3,4: Mean of X, Y, Z marker coordinates 
    def feature_mean(row_data):
        xVals = utils.np.array([row_data[2], row_data[5], row_data[8], row_data[11], row_data[14], row_data[17], row_data[20], row_data[23], row_data[26], row_data[29], row_data[32], row_data[35]])
        yVals = utils.np.array([row_data[3], row_data[6], row_data[9], row_data[12], row_data[15], row_data[18], row_data[21], row_data[24], row_data[27], row_data[30], row_data[33], row_data[36]])
        zVals = utils.np.array([row_data[4], row_data[7], row_data[10], row_data[13], row_data[16], row_data[19], row_data[22], row_data[25], row_data[28], row_data[31], row_data[34], row_data[37]])

        xMean = utils.np.nanmean(xVals , axis=0)
        yMean = utils.np.nanmean(yVals , axis=0)
        zMean = utils.np.nanmean(zVals , axis=0)
        return xMean, yMean, zMean

    temp =utils.np.apply_along_axis( feature_mean, axis=1, arr=rawData )
    FeatureDataSet[:,1] = temp[:,0]
    FeatureDataSet[:,2] = temp[:,1]
    FeatureDataSet[:,3] = temp[:,2]

    #Feature 5,6,7: Standard Deviation of X, Y, Z marker coordinates 
    def feature_std(row_data):
        xVals = utils.np.array([row_data[2], row_data[5], row_data[8], row_data[11], row_data[14], row_data[17], row_data[20], row_data[23], row_data[26], row_data[29], row_data[32], row_data[35]])
        yVals = utils.np.array([row_data[3], row_data[6], row_data[9], row_data[12], row_data[15], row_data[18], row_data[21], row_data[24], row_data[27], row_data[30], row_data[33], row_data[36]])
        zVals = utils.np.array([row_data[4], row_data[7], row_data[10], row_data[13], row_data[16], row_data[19], row_data[22], row_data[25], row_data[28], row_data[31], row_data[34], row_data[37]])

        xStd = utils.np.nanstd(xVals , axis=0)
        yStd = utils.np.nanstd(yVals , axis=0)
        zStd = utils.np.nanstd(zVals , axis=0)
        return xStd, yStd, zStd

    temp =utils.np.apply_along_axis(feature_std, axis=1, arr=rawData)
    FeatureDataSet[:,4] = temp[:,0]
    FeatureDataSet[:,5] = temp[:,1]
    FeatureDataSet[:,6] = temp[:,2]

    #Feature 8,9,10: Maximum of X, Y, Z marker coordinates 
    def feature_max(row_data):
        xVals = utils.np.array([row_data[2], row_data[5], row_data[8], row_data[11], row_data[14], row_data[17], row_data[20], row_data[23], row_data[26], row_data[29], row_data[32], row_data[35]])
        yVals = utils.np.array([row_data[3], row_data[6], row_data[9], row_data[12], row_data[15], row_data[18], row_data[21], row_data[24], row_data[27], row_data[30], row_data[33], row_data[36]])
        zVals = utils.np.array([row_data[4], row_data[7], row_data[10], row_data[13], row_data[16], row_data[19], row_data[22], row_data[25], row_data[28], row_data[31], row_data[34], row_data[37]])

        xMax = utils.np.nanmax(xVals , axis=0)
        yMax = utils.np.nanmax(yVals , axis=0)
        zMax = utils.np.nanmax(zVals , axis=0)
        return xMax, yMax, zMax

    temp =utils.np.apply_along_axis(feature_max, axis=1, arr=rawData )
    FeatureDataSet[:,7] = temp[:,0]
    FeatureDataSet[:,8] = temp[:,1]
    FeatureDataSet[:,9] = temp[:,2]


    #Feature 11,12,13: Standard Deviation of X, Y, Z marker coordinates 
    def feature_min(row_data):
        xVals = utils.np.array([row_data[2], row_data[5], row_data[8], row_data[11], row_data[14], row_data[17], row_data[20], row_data[23], row_data[26], row_data[29], row_data[32], row_data[35]])
        yVals = utils.np.array([row_data[3], row_data[6], row_data[9], row_data[12], row_data[15], row_data[18], row_data[21], row_data[24], row_data[27], row_data[30], row_data[33], row_data[36]])
        zVals = utils.np.array([row_data[4], row_data[7], row_data[10], row_data[13], row_data[16], row_data[19], row_data[22], row_data[25], row_data[28], row_data[31], row_data[34], row_data[37]])

        xMin = utils.np.nanstd(xVals , axis=0)
        yMin = utils.np.nanstd(yVals , axis=0)
        zMin = utils.np.nanstd(zVals , axis=0)
        return xMin, yMin, zMin

    temp =utils.np.apply_along_axis(feature_min, axis=1, arr=rawData )
    FeatureDataSet[:,10] = temp[:,0]
    FeatureDataSet[:,11] = temp[:,1]
    FeatureDataSet[:,12] = temp[:,2]

    #Feature 14, 15, 15 Median Value of X,Y,Z coordinates
    #Feature 16, 17, 18 Nearer to origin for each coordinate.

    return FeatureDataSet, labels