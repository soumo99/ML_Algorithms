#!/usr/bin/env python
# coding: utf-8

# # LINEAR REGRESSION

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data  = pd.read_csv("/Users/soumobratamanna/Documents/Data_Science_Projects/tensorflow-test/Stores.csv")


# In[3]:


data.head(10)


# In[4]:


data.tail(10)


# In[5]:


data.shape


# In[6]:


data.info


# In[7]:


data.isnull().sum()


# In[8]:


#DIVIDING THE DATA BETWEEN INDEPENDENT AND DEPENDENT VARIABLES 
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]


# In[9]:


X


# In[10]:


Y


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test ,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


# In[18]:


X_train


# In[23]:


X_train.shape


# In[24]:


Y_train.shape


# In[19]:


X_test


# In[20]:


Y_train


# In[12]:


#LOADING THE MODEL 
from sklearn.linear_model import LinearRegression


# In[13]:


model = LinearRegression()
model.fit(X_train,Y_train)
predictions = model.predict(X_test)


# In[14]:


from sklearn.metrics import r2_score, mean_squared_error , mean_absolute_error
r2 = r2_score(Y_test,predictions)
mse = mean_squared_error(Y_test,predictions)
mae = mean_absolute_error(Y_test,predictions)


# In[15]:


print("The r2 score of the model is :",r2)


# In[16]:


print("The mse of the model is :",mse)


# In[17]:


print("The mae of the model is : ",mae)


# In[33]:


X = data['Daily_Customer_Count']
Y = data['Store_Sales']
plt.xlabel('Daily_Customer_Count')
plt.ylabel('Store_Sales')
plt.scatter(X,Y)
plt.grid()
plt.show()


# In[ ]:




