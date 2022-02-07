#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import random
import math


# In[6]:


X1 = [0,0,1,1]
X2 = [0,1,0,1]
Y = [0,0,0,1]


# In[13]:


plt.figure()
for i in range(4):
    if Y[i] == 0:
        plt.plot(X1[i], X2[i], "ro")
    else:
        plt.plot(X1[i], X2[i], "bo" )
plt.show()


# In[ ]:




