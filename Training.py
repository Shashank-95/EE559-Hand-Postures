#This file will contain different functions for different training methods.
#We can write one common function for for finding error rate and accuracy.
import utils

def SupportVectorMachine(Train, TrainLabels, Test, TestLabels, d, c):
    #Thanks to sklearn for SVC interface
    svmPred = utils.SVC(gamma='auto',kernel='rbf')
    svmPred.fit(Train, TrainLabels)
    accuracy_train= svmPred.score(Train, TrainLabels)
    predictions = svmPred.predict(Test)
    accuracy = utils.metrics.accuracy_score(TestLabels, predictions)
    fscore = utils.f1_score(TestLabels, predictions, average='weighted')
    print("SVM with RBF:  Train Accuracy- ", accuracy_train)
    print("SVM with RBF:  Test Accuracy - ",  accuracy, '  F1-Score-  ', fscore)    
    print("====================================================================\n")
    return ('Support Vector Machine (RBF Kernel)',accuracy_train, accuracy, fscore)

def NaiveBayes(Train, TrainLabels, Test, TestLabels):
    #Thanks to sklearn for GaussianNB interface
    naive_bayes = utils.GaussianNB()
    predictions = naive_bayes.fit(Train, TrainLabels).predict(Test)
    accuracy_train= naive_bayes.score(Train, TrainLabels)
    accuracy = utils.metrics.accuracy_score(TestLabels, predictions)
    fscore = utils.f1_score(TestLabels, predictions, average='weighted')
    print("Naive Bayes Classifier: Train Accuracy- ", accuracy_train)
    print("Naive Bayes Classifier: Test Accuracy - ",  accuracy, '  F1-Score-  ', fscore)
    print("====================================================================\n")
    return ('Naive Bayes (Baseline)',accuracy_train, accuracy, fscore)

def RandomForestClassifierWithXGBoost(Train, TrainLabels, Test, TestLabels):
    #Thanks to sklearn for RandomForest interface
    RFModel = utils.RandomForestClassifier(n_estimators = 80, max_depth=100, random_state = 0)
    RFModel.fit(Train, TrainLabels)
    predictions = RFModel.predict(Test)
    predictions = RFModel.predict(Test)
    accuracy_train= RFModel.score(Train, TrainLabels)
    accuracy = utils.metrics.accuracy_score(TestLabels, predictions)
    fscore = utils.f1_score(TestLabels, predictions, average='weighted')
    print("Random Forest:  Train Accuracy- ", accuracy_train)
    print("Random Forest:  Test Accuracy - ",  accuracy, '  F1-Score-  ', fscore)    
    print("====================================================================\n")
    return ('Random Forest', accuracy_train, accuracy, fscore)

def KNearestNeighbors(Train, TrainLabels, Test, TestLabels):
    #K_vals = range(1,26)
    #accuracy = {}
    #accuracy_list =[]
    # for i in K_vals:
    #     K_neighbor = utils.KNeighborsClassifier(n_neighbors =i)
    #     K_neighbor.fit(Train, TrainLabels)
    #     predictions = K_neighbor.predict(Test)
    #     accuracy[i] = utils.metrics.accuracy_score(TestLabels, predictions)
    #     accuracy_list.append(utils.metrics.accuracy_score(TestLabels, predictions))

    #Accuracy with auto values
    #Thanks to sklearn for KNeighborsClassifier interface
    K_neighbor_new = utils.KNeighborsClassifier(algorithm='auto')
    K_neighbor_new.fit(Train, TrainLabels)
    predictions_new = K_neighbor_new.predict(Test)
    accuracy_train= K_neighbor_new.score(Train, TrainLabels)
    accuracy = utils.metrics.accuracy_score(TestLabels, predictions_new)
    fscore = utils.f1_score(TestLabels, predictions_new, average='weighted')
    print("K-nearest Neighbor:  Train Accuracy- ", accuracy_train)
    print("K-nearest Neighbor:  Test Accuracy - ",  accuracy, '  F1-Score-  ', fscore)  
    print("====================================================================\n")
    return ('K-Nearest Neigbors', accuracy_train, accuracy, fscore)
    #Plot the Test Accuracy vs K values
    # utils.plt.plot(K_vals, accuracy_list)
    # utils.plt.xlabel('Value of K for K-Nearest Neighbor')
    # utils.plt.ylabel('Testing Accuracy')
    # utils.plt.title('KNN Accuracy vs K value with Standardization and PCA (3-Components)')
    # utils.plt.savefig("KNN_accuracyVSK's.png")


def DiscriminantAnalysisLinear(Train, TrainLabels, Test, TestLabels):
    linear_fit = utils.LDA()
    linear_fit.fit(Train, TrainLabels)
    predictions = linear_fit.predict(Test)
    accuracy_train= linear_fit.score(Train, TrainLabels)
    accuracy = utils.metrics.accuracy_score(TestLabels, predictions)
    fscore = utils.f1_score(TestLabels, predictions, average='weighted')
    print("Linear Discriminant Analysis:  Train Accuracy- ", accuracy_train)
    print("Linear Discriminant Analysis:  Test Accuracy - ",  accuracy, '  F1-Score-  ', fscore)  
    print("====================================================================\n")
    return ('Linear Discriminant Analysis',accuracy_train, accuracy, fscore)


def DiscriminantAnalysisQuadratic(Train, TrainLabels, Test, TestLabels):
    quadratic_fit = utils.QDA()
    quadratic_fit.fit(Train, TrainLabels)
    predictions = quadratic_fit.predict(Test)
    accuracy_train= quadratic_fit.score(Train, TrainLabels)
    accuracy = utils.metrics.accuracy_score(TestLabels, predictions)
    fscore = utils.f1_score(TestLabels, predictions, average='weighted')
    print("Quadratic Discriminant Analysis:  Train Accuracy- ", accuracy_train)
    print("Quadratic Discriminant Analysis:  Test Accuracy - ",  accuracy, '  F1-Score-  ', fscore)  
    print("====================================================================\n")
    return ('Quadratic Discriminant Analysis', accuracy_train, accuracy, fscore)


def Logistic_Regression(Train, TrainLabels, Test, TestLabels):
    #Thanks to sklearn for Logistic Regression interface
    logistic_reg = utils.LogisticRegression(random_state=0, max_iter= 100, multi_class='ovr', solver='lbfgs').fit(Train, TrainLabels)
    predictions = logistic_reg.predict(Test)
    accuracy_train= logistic_reg.score(Train, TrainLabels)
    accuracy = utils.metrics.accuracy_score(TestLabels, predictions)
    fscore = utils.f1_score(TestLabels, predictions, average='weighted')
    print("Logistic Regression: Train Accuracy - ", accuracy_train)
    print("Logistic Regression: Test Accuracy  - ",  accuracy, '  F1-Score-  ', fscore)  
    print("====================================================================\n")
    return ('Logistic Regression',accuracy_train, accuracy, fscore)

