#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd


# In[13]:


var1=pd.read_csv("xsiri.csv")
y=pd.read_csv("ysiri.csv")


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[15]:


model=LinearRegression()


# In[16]:


model.fit(var1,y)


# In[17]:


print(model.coef_)
print(model.intercept_)


# In[ ]:




