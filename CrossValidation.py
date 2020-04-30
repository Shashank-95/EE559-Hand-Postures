#Write a black box code which performs K-Fold cross validation.
#This file/method will be called for each training/classification algorithm
import utils

def CrossValidation(PCAData, labels):
    classes = labels[:,0]
    users = labels[:,1]
    loou = utils.LeaveOneOut()
    print(loou.get_n_splits(PCAData, classes, users))

    #Defining intervals for C and Gamma
    intervals = utils.np.logspace(-3, 3, 60 ,endpoint = True)
    accuracyMat = utils.np.ndarray([60, 60])
    deviationMat = utils.np.ndarray([60,60])

    for i in range(0,60):
        for j in range(0,60):
            d = intervals[i]
            c = intervals[j]
            acc = utils.np.array([])

            userID = [0,1,2,5,6,8,9,10,11]
            for id in userID:
                index_user = utils.np.where(users==id)
                index_rest = utils.np.where(users !=id)

                #Defining The validation set
                validation = PCAData[index_user]
                labels_validation = classes[index_user]

                #Defining the train set
                train = PCAData[index_rest]
                labels_train = classes[index_rest]

                #SVM on train and validation set
                svmPred = utils.SVC(C=c, kernel = 'rbf', gamma = d)
                svmPred.fit(train, labels_train)
                accuracy= svmPred.score(validation, labels_validation)
                acc = utils.np.append(acc, accuracy)

            accuracyMat[i,j] = utils.np.mean(acc)
            deviationMat[i,j] = utils.np.std(acc)

    maxAccuracy = accuracyMat[0,0]
    for i in range(0,60):
        for j in range(0,60):
            if(accuracyMat[i,j] > maxAccuracy):
                maxAccuracy = accuracyMat[i,j]
                indices = [i,j]
    
    print("Maximum Accuracy ", maxAccuracy)
    gamma = intervals[indices[0]]
    c = intervals[indices[1]]
    
    return gamma, c
