# -*- coding:utf-8 -*-

import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')

import re
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from numpy import *
from sklearn import metrics
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
import time
import datetime

from sklearn.neural_network import MLPClassifier

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
class VModel:
    def __init__(self):
        pass
    def getPLData(self):
        pos_file_name= dirname+"/data/lpos.txt"
        neg_file_name = dirname+"/data/lneg.txt"

        pos_file = open(pos_file_name)
        neg_file = open(neg_file_name)
        posD=[]
        negD=[]
        rdata1=[]
        rdata2=[]
        for line in pos_file.readlines():
            line= line.strip(' ')
            line = line.strip('\n')
            lines = line.split(' ')
            #print "pos data line split len",len(lines)
            temp=[]
            for item in lines:
                temp.append(int(item))
            posD.append(temp)
        for line in neg_file.readlines():
            line= line.strip(' ')
            line = line.strip('\n')
            lines = line.split(' ')
            #print "neg data line split len",len(lines)
            temp=[]
            for item in lines:
                temp.append(int(item))
            negD.append(temp)
        print "pos counts:",len(posD)
        print "neg counts:",len(negD)
        posCount=len(posD)
        negCount=len(negD)
        if posCount > 0 :
            shuffleArray = range(len(posD))
            np.random.shuffle(shuffleArray)
            for ii in xrange(posCount):
                rdata1.append(posD[shuffleArray[ii]])
        else:
            rdata1 = posD

        if negCount > 0:
            shuffleArray = range(len(negD))
            np.random.shuffle(shuffleArray)
            for ii in xrange(negCount):
                rdata2.append(negD[shuffleArray[ii]])
        else:
            rdata2 = negD
        return  (rdata1,rdata2)

    def testModel(self,X,y,size=0.2):
        classifiers = [
            KNeighborsClassifier(),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            LogisticRegression(),
            GradientBoostingClassifier(),
            MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
        ]
        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
             "Decision Tree", "Random Forest", "AdaBoost",
             "LogisticRegression", "GradientBoostingClassifier","MLPClassifier"]

        #classifiers = [clf = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(5, 2), random_state=1)  ()]
        #names=["LogisticRegression"]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size)

        for name,clf in zip(names,classifiers):
            d1 = datetime.datetime.now()

            print name
            startT = time.time()
            clf.fit(X_train,y_train)
            endT = time.time()
            y_true,y_pred = y_test,clf.predict(X_test)
            startT = time.time()
            #print (clf.coef_)
            print (classification_report(y_true,y_pred))
            print (metrics.confusion_matrix(y_true,y_pred))

            d2 = datetime.datetime.now()
            interval=d2-d1

            print "===========time============",interval.days*24*3600 + interval.seconds+interval.microseconds/1000000.0

def testVector():
    d1 = datetime.datetime.now()
    model = VModel()
    pos_data,neg_data=model.getPLData()

    y = np.concatenate((np.ones(len(pos_data)), np.zeros(len(neg_data))))
    X = np.concatenate((pos_data, neg_data))
    X_vec = []
    for item in X:
        X_vec.append(tuple(item.tolist()))

    d2 = datetime.datetime.now()
    interval=d2-d1

    print "yuchuli time ,",interval.days*24*3600 + interval.seconds+interval.microseconds/1000000.0

    model.testModel(X_vec,y)

if __name__ == "__main__":
    testVector()



