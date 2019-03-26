
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn


# In[14]:


import pandas as pd
def load_data(file):
    data = pd.read_csv(file,engine="python")
    return data
file = "D:\\机器学习\\zuoye.csv"
data = load_data(file)
print(data)


# In[15]:


X = np.c_[data['x']]
y = np.c_[data['y']]


# In[16]:


data.plot(kind="scatter",x="x",y="y")
plt.show()


# In[17]:


from sklearn import linear_model
lr_model = linear_model.LinearRegression()
lr_model.fit(X,y)
print("斜率:%s,截距:%s"%(lr_model.coef_[0][0],lr_model.intercept_[0]))
print("估计模型为：y=%sx+%sy"%(lr_model.coef_[0][0],lr_model.intercept_[0]))


# In[21]:


data.plot(kind="scatter",x="x",y="y")
plt.plot(X,lr_model.predict(X.reshape(-1,1)),color='red',linewidth=4)
plt.show()


# In[22]:


X_new = [[100]]
print(lr_model.predict(X_new))

