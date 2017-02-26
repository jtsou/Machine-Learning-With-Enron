#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf=SVC(kernel='rbf', C=10000)
clf.fit(features_train, labels_train)
t0=time()
print 'training time: ', round(time()-t0,2), 's'

t0=time()
pred=clf.predict(features_test)
print 'prediction time: ', round(time()-t0,2), 's'
accuracy=accuracy_score(pred, labels_test)
print 'accuracy={0}'.format(accuracy)

print pred[10]
print pred[26]
print pred[50]
print sum(pred)
#########################################################


