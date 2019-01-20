#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv("C:\\Users\\Rakesh\\Desktop\\PYTHON PGMS\\dataset.csv")


# In[4]:


data.head()


# In[5]:


data.describe()


# In[6]:


sns.lmplot(x='mmax',y='estperfor',data=data)


# In[7]:


y = data['estperfor']


# In[9]:


data.columns


# In[19]:


X = data[[ 'cycle', 'mmin', 'mmax', 'cach', 'minchan',
       'maxchan', 'pperfor']]


# In[12]:


sns.pairplot(data)


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[22]:


from sklearn.linear_model import LinearRegression


# In[23]:


lm = LinearRegression()


# In[24]:


lm.fit(X_train,y_train)


# In[25]:


print(lm.intercept_)


# In[26]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])


# In[27]:


coeff_df


# In[28]:


predictions = lm.predict(X_test)


# In[29]:


plt.scatter(y_test,predictions)


# In[30]:


from sklearn import metrics


# In[31]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




