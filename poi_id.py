#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
email_features_list=['from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]

financial_features_list=['bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]

features_list = ['poi']+email_features_list + financial_features_list 
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#Dataset exploration
print ('Exploratory Data Analysis')
data_dict.keys()
print ('Total number of data points= {0}').format(len(data_dict.keys()))

count_poi=0
for name in data_dict.keys():
    if data_dict[name]['poi']==True:
        count_poi+=1

print ('Number of Persons of Interest: {0}').format(count_poi)
print ('Number of Non-Person of Interest: {0}').format(len(data_dict.keys())-count_poi)

##Feature exploration
# Find missing data
all_features=data_dict['BAXTER JOHN C'].keys()
print ('Total Features everyone on the list has:', len(all_features))

missing={}
for feature in all_features:
    missing[feature]=0

for person in data_dict:
    records=0
    for feature in all_features:
        if data_dict[person][feature]=='NaN':
            missing[feature]+=1
        else:
            records+=1

print ('Number of Missing Values for each Feature:')
for feature in all_features:
    print (feature, missing[feature])
    
### Task 2: Remove outliers

#make it into a function based on the multiple variables
def PlotOutlier(data_dict, ax, ay):
    data = featureFormat(data_dict, [ax,ay,'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi=point[2]
        if poi:
            color='blue'
        else:
            color='green'

        plt.scatter( x, y, color=color )
    plt.xlabel(ax)
    plt.ylabel(ay)

PlotOutlier(data_dict, 'from_poi_to_this_person','from_this_person_to_poi')
PlotOutlier(data_dict, 'total_payments', 'total_stock_value')
PlotOutlier(data_dict, 'from_messages','to_messages')
PlotOutlier(data_dict, 'salary','bonus')

##function to remove outliers
def remove_outliers(data_dict, outliers):
    for outlier in outliers:
        data_dict.pop(outlier, 0)
outliers =['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHARD EUGENE E']
remove_outliers(data_dict, outliers)



### Task 3: Create new feature(s) that will help identify POI
### Store to my_dataset for easy export below.
my_dataset = data_dict

##Add new features to dataset


def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if all_messages =='NaN':
        return fraction
    if poi_messages=='NaN':
        return fraction
        
    fraction=float(poi_messages)/float(all_messages)
    return fraction
submit_dict={}
for name in my_dataset:

    data_point = my_dataset[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi


my_feature_list=features_list+['from_poi_to_this_person','to_messages','fraction_from_poi','from_this_person_to_poi',
                               'from_messages','fraction_to_poi']

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation
from sklearn.svm import SVC


def getkbest(data_dict, features_list, k):
    data=featureFormat(my_dataset, features_list)
    labels, features = targetFeatureSplit(data)
    selection=SelectKBest(k=k).fit(features,labels)
    scores=selection.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs=list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    selection_best = dict(sorted_pairs[:k])
    return selection_best
num=10
best_features = getkbest(my_dataset, my_feature_list, num)
print ('Selected features and their scores: ', best_features)
my_feature_list = ['poi'] + best_features.keys()
print ("{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:]))

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn import preprocessing
scaler=preprocessing.MinMaxScaler()
features=scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

clf_p=Pipeline(steps=[
    ('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
    ('classifier', LogisticRegression(penalty='l2', tol=0.001, C=0.0000001, random_state=42))])

from sklearn.cluster import KMeans
clf_k=KMeans(n_clusters=2, random_state=42, tol=0.001)

from sklearn.svm import SVC
clf_s=SVC(kernel='rbf',C = 1000,random_state = 42)

from sklearn.naive_bayes import GaussianNB
clf_g=GaussianNB()

from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42) 

from sklearn.tree import DecisionTreeClassifier
clf_d=DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=2, min_samples_split=7, splitter='best',random_state=42)

from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from statistics import mean
def evaluate(clf, features, labels, num=1000):
    print (clf)
    accuracy=[]
    precision=[]
    recall=[]
    for trial in range(num):
        features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.30, random_state=42)
        clf.fit(features_train, labels_train)
        pred=clf.predict(features_test)
        accuracy.append(clf.score(features_test, labels_test))
        precision.append(precision_score(labels_test, pred))
        recall.append(recall_score(labels_test, pred))
    print ('precision: {}').format(mean(precision))
    print ('recall: {}').format(mean(recall))
    return mean(precision), mean(recall), confusion_matrix(labels_test, pred),classification_report(labels_test, pred) 

evaluate(clf_k, features, labels)       
evaluate(clf_g, features, labels)
evaluate(clf_p, features, labels)
evaluate(clf_rf, features, labels)
evaluate(clf_s, features, labels)
evaluate(clf_d, features, labels)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from sklearn.cross_validation import StratifiedShuffleSplit


rf_clf=RandomForestClassifier()
param_grid={'n_estimators':[5,10,20,40,50,80,160],'criterion':['gini','entropy'], 'max_leaf_nodes':[10],
            'max_depth' :[5,6,7], 'max_features': ['sqrt']}
sss=StratifiedShuffleSplit(labels)
rf_search=GridSearchCV(rf_clf,param_grid, cv=sss)
rf_search.fit(features, labels)
rf_search.best_estimator_


d_clf=DecisionTreeClassifier()
param_grid={'min_samples_leaf':[1,2,3,4,5], 'min_samples_split':[2,3,4,5]}
sss=StratifiedShuffleSplit(labels)
d_search=GridSearchCV(d_clf,param_grid, cv=sss)
d_search.fit(features, labels)
d_search.best_estimator_



p_clf=LogisticRegression()
param_grid={'C':[1, 10, 100, 1000], 'tol':[0.001,0.0001]}
sss=StratifiedShuffleSplit(labels)
p_search=GridSearchCV(p_clf,param_grid, cv=sss)
p_search.fit(features, labels)
p_search.best_estimator_


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
from tester import test_classifier
test_classifier(rf_clf, my_dataset, features_list)
test_classifier(d_clf, my_dataset, features_list)
test_classifier(p_clf, my_dataset, features_list)

