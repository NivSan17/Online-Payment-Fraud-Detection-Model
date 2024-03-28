#!/usr/bin/env python
# coding: utf-8

# # Online Payments Fraud Detection Model

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("C:\\Users\\dell\\Desktop\\Credit card_log.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.dtypes


# In[7]:


data.isna().sum()


# In[8]:


data.describe()


# In[9]:


data.duplicated().sum()


# In[10]:


# Checking transaction types
print(data.type.value_counts())


# In[11]:


#set variables for transaction and quanity
type = data["type"].value_counts()
transactions = type.index
quantity = type.values


# In[46]:


sns.countplot(x=data.type)


# In[12]:


#Pie chart
# We can demonstarte the above type using a pie chart
plt.figure(figsize=(10,5))
plt.pie(quantity,labels=transaction,autopct="%.2f")
plt.title="Distribution of Transaction type"
plt.show()


# In[13]:


data['isFraud'].value_counts()


# In[14]:


plt.figure(figsize=(6,2))
plt.hist(data['isFraud'],bins=3)
plt.show()


# In[15]:


data=data.drop(["nameDest","nameOrig","isFlaggedFraud"],axis=1)


# In[16]:


data['type'] = data['type'].map({'CASH_OUT':1, 'PAYMENT':2, 'CASH_IN':3, 'TRANSFER':4, 'DEBIT':5})
data['type']


# In[17]:


correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))


# In[18]:


# Heatmap to check the correation
plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(),
            cmap='BrBG',
            fmt='.2f',
            linewidths=2,
            annot=True)
plt.show()


# In[19]:


data.head()


# In[20]:


#we are tranforming to numerical format
from sklearn.preprocessing import LabelEncoder


# In[21]:


le=LabelEncoder()


# In[22]:


data["isFraud"]=le.fit_transform(data["isFraud"])


# # Splitting the data into train and test 

# In[23]:


#Applying train and test model
from sklearn.model_selection import train_test_split


# In[24]:


x=data.drop(["isFraud"],axis=1)


# In[25]:


y=data.isFraud


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=40)


# # Logistic Regression Model

# In[27]:


#Applying Logistic Regression algorithm
from sklearn.linear_model import LogisticRegression


# In[28]:


lr=LogisticRegression()


# In[29]:


lr.fit(x_train,y_train)


# In[30]:


pred=lr.predict(x_test)


# In[31]:


pred


# In[32]:


from sklearn.metrics import classification_report,accuracy_score


# In[33]:


print(accuracy_score(y_test,pred))


# In[34]:


print(classification_report(y_test,pred))


# # Decision Tree Model

# In[35]:


from sklearn.tree import DecisionTreeClassifier


# In[36]:


dt=DecisionTreeClassifier()


# In[37]:


dt.fit(x_train,y_train)


# In[38]:


pred2=dt.predict(x_test)


# In[39]:


print(accuracy_score(y_test,pred2))


# In[40]:


print(classification_report(y_test,pred2))


# # Kneighbour Classifier

# In[41]:


from sklearn.neighbors import KNeighborsClassifier


# In[42]:


knn=KNeighborsClassifier(n_neighbors=3)


# In[43]:


knn.fit(x_train,y_train)


# In[44]:


pred4=knn.predict(x_test)


# In[45]:


print(classification_report(y_test,pred4))


# # After comparing alll three models, we can see Decision tree has highly improved score as compared to Logistic regression and kNeighbours
