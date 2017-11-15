
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[3]:


df_train.describe()


# In[6]:


columns = df_train.columns
columns


# In[7]:


columns2 = df_test.columns
columns2


# In[8]:


df_data = pd.DataFrame(df_train)
df_data.head()


# In[9]:


df_data2 = pd.DataFrame(df_test)
df_data2.head()


# In[10]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_train)
X_std = scaler.fit_transform(df_train)


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_test)
X_std2 = scaler.fit_transform(df_test)


# In[12]:


from sklearn.decomposition import PCA
pca = PCA(n_components=270)
X_pca = pca.fit_transform(X_std)


# In[13]:


from sklearn.decomposition import PCA
pca2 = PCA(n_components=270)
X_pca2 = pca2.fit_transform(X_std2)


# In[14]:


X_pca[:10]


# In[15]:


X_pca2[:10]


# In[16]:


print(np.cumsum(pca.explained_variance_ratio_))


# In[17]:


ratio = np.cumsum(pca.explained_variance_ratio_)
ratio


# In[18]:


count = 0
for i in range(len(ratio)):
    if ratio[i] < 0.8:
        count += 1
    else:
        print(count)
        break


# In[19]:


X_pca = X_pca[:, :108]


# In[56]:


X_pca2 = X_pca2[:, :108]


# In[23]:


y = df_train.loc[:, ['SalePrice']].values


# In[25]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[164]:


from sklearn.model_selection import train_test_split
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.3, random_state = 0)


# In[165]:


lr.fit(X_pca_train, y_train)


# In[166]:


lr.intercept_


# In[167]:


lr.coef_


# In[183]:


from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=2350)
model_lasso.fit(X_pca_train, y_train)


# In[184]:


print('R^2')
print('train: %.3f' % model_lasso.score(X_pca_train, y_train))
print('test : %.3f' % model_lasso.score(X_pca_test, y_test))


# In[185]:


model_lasso.coef_


# In[186]:


X_pca2.shape


# In[187]:


y_pred = model_lasso.predict(X_pca2)


# In[188]:


y_pred[:10]


# In[189]:


kaggle_df_test = pd.DataFrame({"Id":np.arange(1461, 2920), "SalePrice":y_pred})
kaggle_df_test.head()


# In[190]:


kaggle_df_test.to_csv("housepriceC.csv", index=False)

