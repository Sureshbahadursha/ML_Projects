#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[59]:


df=pd.read_csv('house_pricing_data.csv')


# In[12]:


df1=df[['area','price']]


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df1.area,df1.price,color='red',marker='*')


# In[30]:


df2 = df1.drop('price',axis='columns')
df2


# In[31]:


df3=df.price
df3


# In[26]:


reg=linear_model.LinearRegression()


# In[32]:


reg.fit(df2,df3)


# In[33]:


reg.predict([[5000]])


# In[36]:


data={'area':[1000,1500,2300,3540,4120,4560,5490,3460,4750,2300,9000,8600,7100]}
area=pd.DataFrame(data)


# In[44]:


df4=area
df4


# In[39]:


Prices_Prediction=reg.predict(area)
Prices_Prediction


# In[46]:


df4['Prices_Prediction']=Prices_Prediction
df4


# In[52]:


plt.scatter(df4.area,df4.Prices_Prediction,color='red',marker='*')


# In[56]:


plt.plot(df4.area,df4.Prices_Prediction,color='blue',marker='*')


# In[61]:


plt.plot(df4.area,df4.Prices_Prediction,color='blue',marker='*')
plt.scatter(df1.area,df1.price,color='red',marker='*')


# In[ ]:




