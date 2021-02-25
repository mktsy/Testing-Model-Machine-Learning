#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from sklearn.svm import SVC
# Set seed for reproducibility
SEED = 123


# In[2]:


dataset = pd.read_csv('C:/Users/User/data.csv')


# In[3]:


dataset.info()


# In[4]:


dataset.head()


# In[5]:


dataset.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)


# In[6]:


dataset['diagnosis'] = dataset['diagnosis'].map({'M':1, 'B':0})


# In[7]:


dataset.describe()


# In[8]:


dataset['diagnosis'].value_counts()


# In[9]:


sns.countplot(x='diagnosis', data=dataset)
plt.title('Breast Cancer Diagnosis')
plt.show()


# In[10]:


dataset.plot(kind='density', subplots=True, layout=(4,8), sharex=False, legend=False, fontsize=1)
plt.show()


# In[11]:


sns.scatterplot(x = 'area_mean', y = 'radius_mean', hue = 'diagnosis', data = dataset)
plt.show()


# In[12]:


sns.scatterplot(x = 'radius_worst', y = 'radius_mean', hue = 'diagnosis', data = dataset)
plt.show()


# In[13]:


plt.figure(figsize=(20,10)) 
sns.heatmap(dataset.corr(), annot=True)
plt.show()


# In[14]:


new_dataset = dataset.drop(['perimeter_mean', 'area_mean', 
                            'radius_worst', 'perimeter_worst', 'area_worst',
                           'perimeter_se', 'area_se', 'texture_worst',
                           'concave points_worst', 'concavity_mean', 'compactness_worst'], axis=1)


# In[15]:


plt.figure(figsize=(20,10)) 
sns.heatmap(new_dataset.corr(), annot=True)
plt.show()


# In[16]:


X = new_dataset.drop(['diagnosis'], axis=1)
Y = new_dataset['diagnosis']


# In[17]:


X.info()


# In[18]:


# Split dataset into 70% train, 30% test
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.3, random_state=SEED)


# In[19]:


# normalize
# fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# transform training data
X_train_norm = norm.transform(X_train)

# transform testing dataabs
X_test_norm = norm.transform(X_test)


# In[20]:


DataFrame(X_train_norm).describe()


# In[21]:


#Standardize
# fit scaler on training data
stdscale = StandardScaler().fit(X_train)

# transform training data
X_train_std = stdscale.transform(X_train)

# transform testing dataabs
X_test_std = stdscale.transform(X_test)


# In[22]:


DataFrame(X_train_std).describe()


# In[23]:


# Instantiate individual classifiers
lr = LogisticRegression(max_iter = 500, n_jobs=-1, random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)
svc = SVC(kernel='rbf', probability = True, random_state=SEED)
rf = RandomForestClassifier(random_state=SEED)

# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr),
('K Nearest Neighbours', knn),
('SVM', svc),
('Random Forest Classifier', rf),
('Decision Tree', dt)]              


# In[24]:


# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    #fit clf to the training set
    clf.fit(X_train, Y_train)
    # Predict the labels of the test set
    y_pred = clf.predict(X_test)
    # Evaluate the accuracy of clf on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(Y_test, y_pred)))


# In[25]:


# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    #fit clf to the training set
    clf.fit(X_train_norm, Y_train)
    # Predict the labels of the test set
    Y_pred = clf.predict(X_test_norm)
    # Evaluate the accuracy of clf on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(Y_test, Y_pred)))


# In[26]:


# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    #fit clf to the training set
    clf.fit(X_train_std, Y_train)
    # Predict the labels of the test set
    Y_pred = clf.predict(X_test_std)
    # Evaluate the accuracy of clf on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(Y_test, Y_pred)))


# In[27]:


cm = confusion_matrix(Y_test, lr.predict(X_test_std))
sns.heatmap(cm, annot=True, fmt="d")
plt.show()


# In[28]:


cm = confusion_matrix(Y_test, svc.predict(X_test_std))
sns.heatmap(cm, annot=True, fmt="d")
plt.show()


# In[ ]:




