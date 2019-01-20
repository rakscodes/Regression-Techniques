#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[64]:


data= pd.read_csv("C:\\Users\\Rakesh\\Desktop\\PYTHON PGMS\\cancer.csv")


# In[65]:


data.replace(['?'],np.NaN,inplace=True)


# In[66]:


data


# In[67]:


data.isna().any()


# In[69]:


data.dropna(how='any',inplace=True)


# In[70]:


data


# In[71]:


from sklearn.model_selection import train_test_split


# In[72]:


X = data[['thick','size','shpe','adhesion','cell size','nuclei','chromatin','nuceloli','mitoses']]
y = data['class']


# In[73]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[74]:


from sklearn.linear_model import LogisticRegression


# In[75]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[76]:


predictions = logmodel.predict(X_test)


# In[77]:


from sklearn.metrics import classification_report


# In[78]:


print(classification_report(y_test,predictions))


# In[ ]:




