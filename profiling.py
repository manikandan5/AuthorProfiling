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
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

import pandas as pd
import matplotlib.pyplot as plt

import pickle
import random
import numpy as np
import datetime

import sys


# Function to convert the labels(Male/Female) to integer for efficienct classification
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


# Classifcation - Gender
def classify_gender(vectorizer, vec_name, flag):
    file = open("results-gender-" + vec_name + ".txt", "w")
    data = [db["Status"] + "-GENDER-" + db["Sex"] for db in db.collection.find()]

    random.seed(1234)  # randomizing data

    random.shuffle(data)

    print("Training gender " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    if flag:
        temp_train_data = data[:int(len(data) * .0016)]
        temp_test_data = data[int(len(data) * .0016):int(len(data) * .0020)]
    else:
        temp_train_data = data[:int(len(data) * .8)]
        temp_test_data = data[int(len(data) * .8):]

    train_labels = []
    train_data = []
    test_labels = []
    test_data = []

    for item in temp_train_data:
        temp = item.split("-GENDER-")
        train_data.append(temp[0])
        train_labels.append(temp[1])

    for item in temp_test_data:
        temp = item.split("-GENDER-")
        test_data.append(temp[0])
        test_labels.append(temp[1])

    print("Fitting gender data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    # Create the vocabulary and the feature weights from the training data
    train_vectors = vectorizer.fit_transform(train_data)

    print("Transforming gender data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    test_vectors = vectorizer.transform(test_data)

    # Create the feature weights for the test data
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000], 'kernel': ['linear']},
        {'C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    print("Fitting SVM gender data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    n_estimators = 10
    # Linear SVM Classifier
    classifier_linear = OneVsRestClassifier(
        BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'),
                          max_samples=1.0 / n_estimators, n_estimators=n_estimators))
    classifier_linear.fit(train_vectors, train_labels)

    print("Dumping SVM gender data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    with open("svm-gender-" + vec_name + ".pkl", 'wb') as f:
        pickle.dump(classifier_linear, f)

    print("Loading SVM gender data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    with open("svm-gender-" + vec_name + ".pkl", 'rb') as f:
        classifier_linear = pickle.load(f)

    print("Predicting SVM gender data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")
    prediction_linear = classifier_linear.predict(test_vectors)

    print(classification_report(test_labels, prediction_linear), "\n")
    file.write(classification_report(test_labels, prediction_linear) + "\n")

    # Naive Bayes Classifier
    classifier_nb = BaggingClassifier(MultinomialNB(), max_samples=0.5, max_features=0.5)
    classifier_nb.fit(train_vectors, train_labels)

    print("Fitting NB gender data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    classifier_nb.fit(train_vectors, train_labels)

    print("Dumping NB gender data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")
    with open("nb-gender-" + vec_name + ".pkl", 'wb') as f:
        pickle.dump(classifier_nb, f)

    print("Loading NB gender data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")
    with open("nb-gender-" + vec_name + ".pkl", 'rb') as f:
        classifier_nb = pickle.load(f)

    print("Predicting NB gender result " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")
    prediction_nb = classifier_nb.predict(test_vectors)

    print(classification_report(test_labels, prediction_nb), "\n")
    file.write(classification_report(test_labels, prediction_nb) + "\n")

    file.close()


def classify_age(vectorizer, vec_name, flag):
    file = open("results-age-" + vec_name + ".txt", "w")

    if flag:

        data1824 = [db["Status"] + "-AGE-" + db["Age"] for db in db.collection.find({"Age": "18-24"}).limit(1800)]

        data2534 = [db["Status"] + "-AGE-" + db["Age"] for db in db.collection.find({"Age": "25-34"}).limit(1800)]

        data3549 = [db["Status"] + "-AGE-" + db["Age"] for db in db.collection.find({"Age": "35-49"}).limit(1800)]

        data5064 = [db["Status"] + "-AGE-" + db["Age"] for db in db.collection.find({"Age": "50-64"}).limit(1800)]

        data65xx = [db["Status"] + "-AGE-" + db["Age"] for db in db.collection.find({"Age": "65-xx"}).limit(1800)]

        data = data1824 + data2534 + data3549 + data5064 + data65xx

    else:
        data = [db["Status"] + "-AGE-" + db["Age"] for db in db.collection.find()]

    random.seed(1234)  # randomizing data

    random.shuffle(data)

    print("Training age " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

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

    print("Fitting age data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    train_vectors = vectorizer.fit_transform(train_data)

    print("Transforming age data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    test_vectors = vectorizer.transform(test_data)

    # Create the feature weights for the test data
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000], 'kernel': ['linear']},
        {'C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    print("Fitting SVM age data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")
    n_estimators = 10

    # Linear SVC Classifier
    classifier_linear = OneVsRestClassifier(
        BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'),
                          max_samples=1.0 / n_estimators, n_estimators=n_estimators))
    classifier_linear.fit(train_vectors, train_labels)

    print("Dumping SVM age data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    with open("svm-age-" + vec_name + ".pkl", 'wb') as f:
        pickle.dump(classifier_linear, f)

    print("Loading SVM age data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    with open("svm-age-" + vec_name + ".pkl", 'rb') as f:
        classifier_linear = pickle.load(f)

    print("Predicting SVM age result " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    prediction_linear = classifier_linear.predict(test_vectors)

    print("Multilabel OneVsRestClassifier")
    file.write("Multilabel OneVsRestClassifier" + "\n")

    print(classification_report(test_labels, prediction_linear))
    file.write(str(classification_report(test_labels, prediction_linear)) + "\n")

    print("Fitting NB age data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    # Classification with Naive Bayes
    classifier_nb = BaggingClassifier(MultinomialNB(), max_samples=0.5, max_features=0.5)
    classifier_nb.fit(train_vectors, train_labels)

    print("Dumping NB age data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    with open("nb-age-" + vec_name + ".pkl", 'wb') as f:
        pickle.dump(classifier_nb, f)

    print("Loading NB age data " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    with open("nb-age-" + vec_name + ".pkl", 'rb') as f:
        classifier_nb = pickle.load(f)

    print("Predicting NB age result " + vec_name + " started - " + str(datetime.datetime.now().time()) + "\n")

    prediction_nb = classifier_nb.predict(test_vectors)

    print("Multinomial Naive Bayes")
    file.write("Multinomial Naive Bayes" + "\n")

    print(classification_report(test_labels, prediction_nb))
    file.write(str(classification_report(test_labels, prediction_nb)) + "\n")
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

    # Count Vectorizer
    countVectorizer = CountVectorizer(min_df=1)

    # Baseline
    classify_gender(tfidfVectorizer, "tfidf", flag)
    classify_age(tfidfVectorizer, "tfidf", flag)


    classify_gender(countVectorizer, "count", flag)
    classify_age(countVectorizer, "count", flag)


client = MongoClient('localhost:27017', serverSelectionTimeoutMS=1000)

db = client['user-details']
collection = db['status']

flag = False # flag to run the sample data set

if len(sys.argv) == 2:
    flag = eval(sys.argv[1])  # Sample flag
elif len(sys.argv) > 2:
    print("Invalid arguments")
    sys.exit()

if db.collection.count() < lookupTweets():
    db.collection.remove()
    insert()
    wrapper_call(flag)
else:
    wrapper_call(flag)
