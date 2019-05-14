# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics.pairwise import manhattan_distances as m_dist
import seaborn as sns
import math


# Reading data into a pandas dataframe named data
data = pd.read_csv('./sample_data/nursery.csv',index_col=False,names=['parents','has_nurs','form','children','housing','finance','social','health','y'])

# Data Preprocessing:

'''The string values have been mapped to integer values'''
parents_map = {"usual":0,"pretentious":1,"great_pret":2}
has_nurs_map = {'proper':0, 'less_proper':1, 'improper':2, 'critical':3, 'very_crit':4}
form_map = {'complete':0, 'completed':1, 'incomplete':2, 'foster':3}
children_map = { '1':1, '2':2, '3':3, 'more':4}
housing_map = {'convenient':0, 'less_conv':1, 'critical':2}
finance_map = {'convenient':0, 'inconv':1}
social_map  = {'nonprob':0, 'slightly_prob':1, 'problematic':2}
health_map = {'recommended':0, 'priority':1, 'not_recom':2}
y_map = {'not_recom':0,'recommend':1,'very_recom':1,'priority':2,'spec_prior':3}

data['parents'] = data['parents'].map(parents_map)
data['has_nurs'] = data['has_nurs'].map(has_nurs_map)
data['form'] = data['form'].map(form_map)
data['children'] = data['children'].map(children_map)
data['housing'] = data['housing'].map(housing_map)
data['finance'] = data['finance'].map(finance_map)
data['social'] = data['social'].map(social_map)
data['health'] = data['health'].map(health_map)
data['y'] = data['y'].map(y_map)

# Standard Normalization
for col in data.columns:
    if col!='y':
        data[col] = ((data[col]-data[col].mean())/(data[col].std()))

# setting random seed and sampling the entire dataset
random.seed(7)
data = data.sample(frac=1)

# data is split into 70%(train):30%(test) 
train_data = data[0:int(data.shape[0]*0.7)]
test_data = data[int(data.shape[0]*0.7):]

# The target labels have been put into a new dataframe named y_train and y_test 
y_train = train_data['y'] 
y_test = test_data['y']

test_data.drop('y',axis=1,inplace=True)

# data_means:Dataframe that contains means of all the attributes after grouping by the target variable 'y'
#data_means = train_data.groupby('y').mean()
# data_vars:Dataframe that contains variances of all the attributes after grouping by the target variable 'y'
#data_vars = train_data.groupby('y').var()

train_data.drop('y',axis=1,inplace=True)

# 'ny' gives the number of unique classes
ny = len(data['y'].unique())
# 'class_probs' is a list of probabilities of getting each class (prior)
class_probs = [0 for x in range(ny)]
val_counts = y_train.value_counts()
for i in range(ny):
    class_probs[i] = (val_counts[i])/len(y_train)


def likelihood(a,b,x):
    '''
    input: attribute name 'a'(str), class name 'b'(str) and 'x'(test data point)
    Returns: The the conditional probability of the attribute 'a' to have the value given the class as 'b'. ( p(data[a] | class) )
    '''
    tmp = train_data[a][y_train==b]
    res = len(tmp[train_data[a]==x[a]])/len(tmp)
    return res

def naive_bayes(x):
    '''
    input: row/data point 'x'
    Returns: The list of probablities of 'x' belonging to each of the classes
    '''
    probs = [0 for x in range(ny)]
    for i in range(ny):
        post = 1
        for col in train_data.columns:
            if col!='assignments':
                post *= likelihood(col,i,x)
        probs[i] = post*class_probs[i]
        
    return probs
    
# The final predicted values of test data are present in 'test_data['assignments'].
test_data['assignments'] = -1
# This piece of code uses the 'naive_bayes' function to calculate the 
# most probable class of all the data points in 'test_data'.
for i in test_data.index:
    probs = naive_bayes(test_data.loc[i])
    maxi = 0
    max_index = 0
    for j in range(ny):
        if probs[j]>maxi:
            maxi = probs[j]
            max_index = j
    test_data['assignments'][i] = max_index
 
# Function to print the accuracy, precision and recall score
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,test_data['assignments']))
# print(confusion_matrix(y_test,test_data['assignments']))

cm = confusion_matrix(y_test,test_data['assignments']) 

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['not_recom','recommend/very_recom','priority','spec_prior'], 
                     columns = ['not_recom','recommend/very_recom','priority','spec_prior'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
#plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()