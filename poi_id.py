#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
import pprint
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] 
#copied all features in from Udacity course
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                     'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                     'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
                  'shared_receipt_with_poi']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

all_features = features_list + email_features + financial_features 
all_features.remove('email_address') 


# In[2]:


### Task 2: Remove outliers

'''Removing outlying keys based on manual examination of the dataset above- including The Travel Agency... and 'Total'. 
Additionaly, removing email addresses as they will not add prediction value.'''
email_features.remove('email_address') 
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('TOTAL')


# In[3]:


#Print minimums and maximums to look for outliers
for feature in all_features:
    print feature
    feature = [item[feature] for k, item in 
    data_dict.iteritems() if not item[feature] == "NaN"]
    print ('min is: %d' % min(feature))
    print ('max is: %d' % max(feature))


# In[4]:


### Task 3: Create new feature(s)
def calcluatePercent(messages, allMessages):
    percent = 0
    if (messages == 'NaN' or allMessages == 'NaN'):
        return percent
    percent = messages / float(allMessages)
    return percent


def createNewFeatures(data_dict):
    for poi_name in data_dict:
        new_dict = data_dict[poi_name]
        new_dict['from_poi_to_this_person_ratio'] = calcluatePercent(new_dict['from_poi_to_this_person'],
                                                                   new_dict['to_messages'])
        new_dict['from_this_person_to_poi_ratio'] = calcluatePercent(new_dict['from_this_person_to_poi'],
                                                                   new_dict['from_messages'])
    return new_dict, ['from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio']



for entry in data_dict:

    data_point = data_dict[entry]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    percent_from_poi = calcluatePercent(from_poi_to_this_person, to_messages )
    data_point["percent_from_poi"] = percent_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    percent_to_poi = calcluatePercent( from_this_person_to_poi, from_messages )
    data_point["percent_to_poi"] = percent_to_poi
features_list_n = all_features
features_list_n =  features_list_n + ['percent_from_poi', 'percent_to_poi']
pprint.pprint (features_list_n)


### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_n, sort_keys = True)
labels, features = targetFeatureSplit(data)



# In[5]:


#Accidentally pulled in Email Address again, removing:
#features_list_n.remove('email_address') 
pprint.pprint (features_list_n)


# In[6]:


import pandas as pd
pd.set_option('display.max_rows', 10)
pd.DataFrame.from_dict(data_dict, orient='index',
                       columns=['poi',
 'to_messages',
 'from_poi_to_this_person',
 'from_messages',
 'from_this_person_to_poi',
 'shared_receipt_with_poi',
 'salary',
 'deferral_payments',
 'total_payments',
 'loan_advances',
 'bonus',
 'restricted_stock_deferred',
 'deferred_income',
 'total_stock_value',
 'expenses',
 'exercised_stock_options',
 'other',
 'long_term_incentive',
 'restricted_stock',
 'director_fees',
 'percent_from_poi',
 'percent_to_poi'])


# In[7]:


for feature in features_list_n:
    print feature
    feature = [item[feature] for k, item in 
    data_dict.iteritems() if not item[feature] == "NaN"]
    print ('min is: %d' % min(feature))
    print ('max is: %d' % max(feature))


# In[8]:


def findKbestFeatures(data_dict, features_list_n, k):
    from sklearn.feature_selection import f_classif
    data = featureFormat(data_dict, features_list_n)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(f_classif, k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    print("sorted_pairs", sorted_pairs)
    k_best_features = dict(sorted_pairs[:k])

    return k_best_features
    


# In[9]:



from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# In[10]:


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Example starting point. Try investigating other evaluation techniques!
'''
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from time import time
clf = GaussianNB()

# train
t0 = time()
clf.fit(features_train, labels_train)
print "\ntraining time:", round(time()-t0, 3), "s"

# predict
t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

accuracy = accuracy_score(pred, labels_test)

print '\naccuracy = {0}'.format(accuracy)
'''


# In[11]:


'''
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


from sklearn import tree
from time import time
from sklearn.model_selection import GridSearchCV
clf = tree.DecisionTreeClassifier()
dt_param = {'criterion':('gini', 'entropy'),
'splitter':('best','random')}
dt_grid_search = GridSearchCV(estimator = clf, param_grid = dt_param)
# train
t0 = time()
clf.fit(features_train, labels_train)
print "\ntraining time:", round(time()-t0, 3), "s"

# predict
t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

accuracy = accuracy_score(pred, labels_test)

print '\naccuracy = {0}'.format(accuracy)

'''


# In[12]:


from time import time

def naive_bayes_clf(features_train, features_test, labels_train, labels_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print "\ntraining time:", round(time()-t0, 3), "s"

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time()-t0, 3), "s"
    accuracy = accuracy_score(pred, labels_test)
    print '\naccuracy = {0}'.format(accuracy)

    return clf


def svm_clf(features_train, features_test, labels_train, labels_test):
    from sklearn.svm import SVC
    clf = SVC(kernel="linear", C=1000)
    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print "\ntraining time:", round(time()-t0, 3), "s"

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time()-t0, 3), "s"
    accuracy = accuracy_score(pred, labels_test)
    print '\naccuracy = {0}'.format(accuracy)

    return clf


def decision_tree_clf(features_train, features_test, labels_train, labels_test):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print "\ntraining time:", round(time()-t0, 3), "s"

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time()-t0, 3), "s"
    accuracy = accuracy_score(pred, labels_test)
    print '\naccuracy = {0}'.format(accuracy)

    return clf

def adaboost_clf(features_train, features_test, labels_train, labels_test):
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(learning_rate=1, algorithm='SAMME', n_estimators=23)
    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print "\ntraining time:", round(time()-t0, 3), "s"

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time()-t0, 3), "s"
    accuracy = accuracy_score(pred, labels_test)
    print '\naccuracy = {0}'.format(accuracy)

    return clf


# In[13]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
#!/usr/bin/pickle

#clf = naive_bayes_clf(features_train, features_test, labels_train, labels_test)
#clf = svm_clf(features_train, features_test, labels_train, labels_test)
clf = decision_tree_clf(features_train, features_test, labels_train, labels_test)
#clf = adaboost_clf(features_train, features_test, labels_train, labels_test)


# In[14]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list_n)


# In[ ]:




