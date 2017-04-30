from pymongo import MongoClient
from insertIntoDb import insert
from getTweets import lookupTweets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
import matplotlib.pyplot as plt

import random
import numpy as np

import sys

def convert_to_integer_gender(labels):
    labels_int = [0 if x == "MALE" else 1 for x in labels]
    return labels_int

def convert_to_integer_age(labels):
    labels_int = []
    for item in labels:
        if item == "18-24":
            labels_int.append(0)
        elif item == "25-34":
            labels_int.append(1)
        elif item == "35-49":
            labels_int.append(2)
        elif item == "50-64":
            labels_int.append(3)
        elif item == "65-xx":
            labels_int.append(4)
    return labels_int

def classify_gender(vectorizer, vec_name, flag):
    file = open("results-gender-"+vec_name+".txt","w")
    data = [db["Status"] + "-GENDER-" + db["Sex"] for db in db.collection.find()]
    
    random.seed(1234)  # randomizing data

    random.shuffle(data)
    print("Training "+vec_name+":\n")
    
    if flag:
        train_data = data[:int(len(data) * .0016)]
        test_data = data[int(len(data) * .0016):int(len(data) * .0020)]
    else:
        train_data = data[:int(len(data) * .8)]
        test_data = data[int(len(data) * .8):]
    
    train_labels = [data.split("-GENDER-")[1] for data in train_data]
    test_labels = [data.split("-GENDER-")[1] for data in test_data]
    train_data = [data.split("-GENDER-")[0] for data in train_data]
    test_data = [data.split("-GENDER-")[0] for data in test_data]

    # Create the vocabulary and the feature weights from the training data
    train_vectors = vectorizer.fit_transform(train_data)
    # Create the feature weights for the test data
    test_vectors = vectorizer.transform(test_data)

    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000], 'kernel': ['linear']},
        {'C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # Finding the Best Parameters
    classifier_best = svm.SVC()
    classifier_cv = GridSearchCV(classifier_best, param_grid, scoring='accuracy')
    classifier_cv.fit(train_vectors, train_labels)
    
    # Classification with SVM, kernel=linear is the best
    classifier_linear = svm.SVC(C=0.01, kernel='linear')
    classifier_linear.fit(train_vectors, train_labels)
    prediction_linear = classifier_linear.predict(test_vectors)
 
    print("Best Score for given dataset for SVM Classifier - ", classifier_cv.best_score_,"\n")
    print("Best Parameter for given dataset for SVM Classifier - ", classifier_cv.best_params_,"\n")
    file.write("Best Score for given dataset for SVM Classifier - " + str(classifier_cv.best_score_) +"\n")
    file.write("Best Parameter for given dataset for SVM Classifier - " + str(classifier_cv.best_params_) +"\n")
    
    print(classification_report(test_labels, prediction_linear),"\n")
    file.write(classification_report(test_labels, prediction_linear)+"\n")
    
    # Plotting the data for different values of C
    def rmse_cv(model):
        rmse = np.sqrt(cross_val_score(classifier_linear, train_vectors, train_labels, scoring='accuracy', cv=5))
        return (rmse)
    train_labels_int = convert_to_integer_gender(train_labels)
    c_lst = [0.01, 0.1, 1, 10, 50, 100, 500, 1000]
    crossval_ridge = [np.mean(rmse_cv(classifier_linear)) for x in c_lst]
    
    crossval_ridge = pd.Series(crossval_ridge, index=c_lst)
    crossval_ridge.plot(title="Validation")
    #plt.xlabel("c_values")
    #plt.ylabel("accuracy")
    #plt.show()

    # Cross Validation Scores
    # scores = cross_val_score(classifier_linear, train_vectors, train_labels, cv=10)
    # from sklearn.model_selection import cross_val_score
    # print("Accuracy",scores.mean(),scores.std()* 2)

    # Predictions using cross validation
    # from sklearn.model_selection import cross_val_predict
    # import sklearn.metrics as metrics
    # predicted = cross_val_predict(classifier_linear, train_vectors, train_labels, cv=10)
    # print("Prediction Accuracy for SVM linear",metrics.accuracy_score(train_labels, predicted))



    # Classification with Naive Bayes

    classifier_nb = MultinomialNB()
    classifier_nb.fit(train_vectors, train_labels)
    prediction_nb = classifier_nb.predict(test_vectors)
    
    print(classification_report(test_labels, prediction_nb),"\n")
    file.write(classification_report(test_labels, prediction_nb)+"\n")

    # Plotting the data for different values of C
    # def rmse_cv(model):
    #     rmse = np.sqrt(cross_val_score(classifier_nb, train_vectors, train_labels, scoring='accuracy', cv=5))
    #     return (rmse)
    #
    # from sklearn.model_selection import cross_val_score
    # import numpy as np
    # train_labels_int = convert_to_integer_gender(train_labels)
    # crossval_ridge = [np.mean(rmse_cv(classifier_linear)) for x in c_lst]
    # import pandas as pd
    # import matplotlib.pyplot as plt
    #
    # crossval_ridge = pd.Series(crossval_ridge, index=c_lst)
    # crossval_ridge.plot(title="Validation")
    # plt.xlabel("c_values")
    # plt.ylabel("accuracy")
    # plt.show()


    # Finding the best parameters
    k = np.arange(20) + 1
    parameters = {'n_neighbors': k}
    classifier_knearest = KNeighborsClassifier()
    classifier_cv_kn = GridSearchCV(classifier_knearest, parameters, scoring='accuracy', cv=10)
    classifier_cv_kn.fit(train_vectors, train_labels)
    print("Best Score for given dataset for KNearest Classifier - ", classifier_cv_kn.best_score_,"\n")
    print("Best Parameter for given dataset for KNearest Classifier - ", classifier_cv_kn.best_params_,"\n")
    file.write("Best Score for given dataset for KNearest Classifier - "+ str(classifier_cv_kn.best_score_)+"\n")
    file.write("Best Parameter for given dataset for KNearest Classifier - "+ str(classifier_cv_kn.best_params_)+"\n")

    # Classification with KNearest Neighbors with parameter 3
    classifier_knearest = KNeighborsClassifier(classifier_cv_kn.best_params_["n_neighbors"])
    classifier_knearest.fit(train_vectors, train_labels)
    prediction_knearest = classifier_knearest.predict(test_vectors)
    
    print(classification_report(test_labels, prediction_knearest),"\n")
    file.write(classification_report(test_labels, prediction_knearest)+"\n")
    file.close()
    
def classify_age(vectorizer, vec_name, flag):
    
    if flag:
        file = open("results-age-" + vec_name+".txt","w")
        
        data1824 = [db["Status"] + "-AGE-" + db["Age"] for db in db.collection.find( { "Age": "18-24" } ).limit(600)]

        data2534 = [db["Status"] + "-AGE-" + db["Age"] for db in db.collection.find( { "Age": "25-34" } ).limit(600)]

        data3549 = [db["Status"] + "-AGE-" + db["Age"] for db in db.collection.find( { "Age": "35-49" } ).limit(600)]

        data5064 = [db["Status"] + "-AGE-" + db["Age"] for db in db.collection.find( { "Age": "50-64" } ).limit(600)]

        data65xx = [db["Status"] + "-AGE-" + db["Age"] for db in db.collection.find( { "Age": "65-xx" } ).limit(600)]

        data = data1824+data2534+data3549+data5064+data65xx
    
    else:
        data = [db["Status"] + "-AGE-" + db["Age"] for db in db.collection.find()]
    

    random.seed(1234)  # randomizing data

    random.shuffle(data)

    print("Training "+vec_name+":")


    temp_train_data = data[:int(len(data) * .8)]
    temp_test_data = data[int(len(data) * .8):]

    train_labels = []
    train_data = []
    test_labels = []
    test_data = []

    for item in temp_train_data:
        temp = item.split("-AGE-")
        train_data.append(temp[0])
        train_labels.append(temp[1])

    for item in temp_test_data:
        temp = item.split("-AGE-")
        test_data.append(temp[0])
        test_labels.append(temp[1])

    
    # Create the vocabulary and the feature weights from the training data
    train_vectors = vectorizer.fit_transform(train_data)
    # Create the feature weights for the test data
    test_vectors = vectorizer.transform(test_data)

    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000], 'kernel': ['linear']},
        {'C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    mlb = MultiLabelBinarizer()
    train_labels_multi = mlb.fit_transform(train_labels)
    test_labels_multi = mlb.fit_transform(test_labels)

    classifier_linear = OneVsRestClassifier(LinearSVC(random_state=0))
    prediction_linear = classifier_linear.fit(train_vectors, train_labels).predict(test_vectors)

    print("Multilabel OneVsRestClassifier")
    file.write("Multilabel OneVsRestClassifier"+"\n")

    print(classification_report(test_labels, prediction_linear))
    file.write(str(classification_report(test_labels, prediction_linear))+"\n")

    # Plotting the data for different values of C
    def rmse_cv(model):
        rmse = np.sqrt(cross_val_score(classifier_linear, train_vectors, train_labels, scoring='accuracy', cv=5))
        return (rmse)

    train_labels_int = convert_to_integer_age(train_labels)
    c_lst = [0.01, 0.1, 1, 10, 50, 100, 500, 1000]
    crossval_ridge = [np.mean(rmse_cv(classifier_linear)) for x in c_lst]
    
    crossval_ridge = pd.Series(crossval_ridge, index=c_lst)
    crossval_ridge.plot(title="Validation")

    #plt.xlabel("c_values")
    #plt.ylabel("accuracy")
    #plt.show()

    # Classification with Naive Bayes

    classifier_nb = MultinomialNB()
    classifier_nb.fit(train_vectors, train_labels)
    prediction_nb = classifier_nb.predict(test_vectors)

    print("Multinomial Naive Bayes")
    file.write("Multinomial Naive Bayes"+"\n")

    print(classification_report(test_labels, prediction_nb))
    file.write(str(classification_report(test_labels, prediction_nb))+"\n")

    # Finding the best parameters
    k = np.arange(20) + 1
    parameters = {'n_neighbors': k}
    classifier_knearest = KNeighborsClassifier()
    classifier_cv_kn = GridSearchCV(classifier_knearest, parameters, scoring='accuracy', cv=10)
    classifier_cv_kn.fit(train_vectors, train_labels)
    print("Best Score for given dataset for KNearest Classifier - ", classifier_cv_kn.best_score_)
    print("Best Parameter for given dataset for KNearest Classifier - ", classifier_cv_kn.best_params_)

    file.write("Best Score for given dataset for KNearest Classifier - "+ str(classifier_cv_kn.best_score_)+"\n")
    file.write("Best Parameter for given dataset for KNearest Classifier - "+ str(classifier_cv_kn.best_params_)+"\n")

    # Classification with KNearest Neighbors with parameter 3
    classifier_knearest = KNeighborsClassifier(classifier_cv_kn.best_params_["n_neighbors"])
    classifier_knearest.fit(train_vectors, train_labels)
    prediction_knearest = classifier_knearest.predict(test_vectors)
    print(classification_report(test_labels, prediction_knearest))
    file.write(str(classification_report(test_labels, prediction_knearest))+"\n")

    file.close()

def wrapper_call(flag):
    # Tokenizing & Filtering the text
    # min_df=5, discard words appearing in less than 5 documents
    # max_df=0.8, discard words appering in more than 80% of the documents
    # sublinear_tf=True, use sublinear weighting
    # use_idf=True, enable IDF
    tfidfVectorizer = TfidfVectorizer(min_df=8,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    
    countVectorizer = CountVectorizer(min_df=1)
    
    #Baseline
    classify_gender(tfidfVectorizer, "tfidf", flag)
    classify_age(tfidfVectorizer, "tfidf", flag)

    #Count Vectorizer
    classify_gender(countVectorizer, "count", flag)
    classify_age(countVectorizer, "count", flag)

client = MongoClient('localhost:27017', serverSelectionTimeoutMS=1000)

db = client['user-details']
collection = db['status']

flag = False

if len(sys.argv)==2:
    flag = eval(sys.argv[1])  #Sample flag
elif len(sys.argv)>2:
    print("Invalid arguments")
    sys.exit()

if db.collection.count() < lookupTweets():
    db.collection.remove()
    insert()
    wrapper_call(flag)
else:
    wrapper_call(flag)
    