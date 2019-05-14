# Importing necessary libraries
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import manhattan_distances as m_dist


# Reading data into a pandas dataframe named data
data = pd.read_csv('./sample_data/nursery.csv',index_col=False,names=['parents','has_nurs','form','children','housing','finance','social','health','y'])

# Data Preprocessing:-

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

# Rows of every column have been mapped 
data['parents'] = data['parents'].map(parents_map)
data['has_nurs'] = data['has_nurs'].map(has_nurs_map)
data['form'] = data['form'].map(form_map)
data['children'] = data['children'].map(children_map)
data['housing'] = data['housing'].map(housing_map)
data['finance'] = data['finance'].map(finance_map)
data['social'] = data['social'].map(social_map)
data['health'] = data['health'].map(health_map)
data['y'] = data['y'].map(y_map)


# Min-max scaling:
for col in data.columns:
    data[col] = ((data[col]-data[col].min())/(data[col].max()-data[col].min()))

# setting random seed and sampling the entire dataset
random.seed(2)
data = data.sample(frac=1)

# data is split into 70%(train):30%(test) 
train_data = data[0:int(data.shape[0]*0.7)]
test_data = data[int(data.shape[0]*0.7):]


# The target labels have been put into new series named y_train and y_test 

y_train = train_data['y'] 
y_test = test_data['y']

# The target labels column is dropped from the original train and test dataframes
test_data.drop('y',axis=1,inplace=True)
train_data.drop('y',axis=1,inplace=True)

# mapping the actual indices of data points with a series from 0 to len(test_data) or len(train_data)
# for ease of access from distance_matrix
test_data_index = dict([])
train_data_index = dict([])
for i in range(len(test_data.index)):
    test_data_index[i] = test_data.index[i]
for i in range(len(train_data.index)):
    train_data_index[i] = train_data.index[i]

# Distance matrix has been calculated using m_dist function 
dist_mat = m_dist(test_data,train_data)

# A new column named 'result' has been added into the test_data dataframe
test_data['result'] = -1.0
test_data['result'] = test_data['result'].astype('float64')

def KNN(k=4):
    '''
    This function finds the k nearest neighbours of a test data point
    and assigns the label that most of its neighbours have.
    '''
    for i in range(dist_mat.shape[0]):
        temp = dist_mat[i].copy()
        nbrs = {}
                
        for j in range(k):
            minindex = 0
            minimum = 100000
            for l in range(dist_mat.shape[1]):
                if temp[l]!=-1 and temp[l]<=minimum:
                    minimum =  dist_mat[i][l]
                    minindex = l
            res = y_train[train_data_index[minindex]]
            if res in nbrs.keys():
                nbrs[y_train[train_data_index[minindex]]]+=1    
            else:
                nbrs[y_train[train_data_index[minindex]]] = 0    
            temp[minindex] = -1
        
        test_data['result'][test_data_index[i]] = max(nbrs.keys(), key=lambda x:nbrs[x])

# Initialisation of 'k':
k = 4

# calling the KNN function with k value passed argument.
KNN(k)

# dictionary to map the class labels.
map_dict = {}
o = 0
for i in test_data['result'].unique():
    map_dict[i] = o
    o+=1

test_data['result'] = test_data['result'].map(map_dict).astype('int64')

y_train = y_train.map(map_dict).astype('int64')
y_test = y_test.map(map_dict).astype('int64')

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,test_data['result']))
#print(confusion_matrix(y_test,test_data['result']))

accuracy = 0
for i in y_test.index:
    if y_test.loc[i]==test_data['result'][i]:
        accuracy+=1
accuracy = accuracy/len(y_test)    
print('Accuracy: '+str(accuracy))


cm = confusion_matrix(y_test,test_data['result']) 

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