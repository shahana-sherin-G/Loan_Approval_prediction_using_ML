#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


data = pd.read_csv("LoanApprovalPrediction.csv") 


# In[3]:


data.head(10)


# In[4]:


obj=(data.dtypes == 'object')


# In[5]:


print("Categorical varoiables:",len(list(obj[obj].index)))


# In[6]:


data.drop(['Loan_ID'],axis=1,inplace=True)


# In[7]:


obj=(data.dtypes=='object')
obj_cols=list(obj[obj].index)
plt.figure(figsize=(20,40))
index=1

for col in obj_cols:
    y=data[col].value_counts()
    plt.subplot(11,4,index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index),y=y)
    index+=1


# In[8]:


from sklearn import preprocessing  
label_encoder = preprocessing.LabelEncoder() 
obj = (data.dtypes == 'object') 
for col in list(obj[obj].index): 
 data[col] = label_encoder.fit_transform(data[col])


# In[9]:


obj = (data.dtypes == 'object') 
print("Categorical variables:",len(list(obj[obj].index)))


# In[37]:


sns.catplot(x="Gender", y="Married", hue="Loan_Status", kind="bar", data=data)


# In[10]:


for col in data.columns: 
    data[col] = data[col].fillna(data[col].mean())
data.isna().sum()


# In[11]:


from sklearn.model_selection import train_test_split 

X = data.drop(['Loan_Status'],axis=1) 
Y = data['Loan_Status'] 
X.shape,Y.shape 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1) 
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[12]:


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 

from sklearn import metrics 

knn = KNeighborsClassifier(n_neighbors=3) 
rfc = RandomForestClassifier(n_estimators = 7, criterion = 'entropy', random_state =7) 
svc = SVC() 
lc = LogisticRegression() 

for clf in (rfc, knn, svc,lc): 
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_train) 
    print("Accuracy score of ", clf.__class__.__name__, "=",100*metrics.accuracy_score(Y_train, Y_pred))


# In[ ]:




