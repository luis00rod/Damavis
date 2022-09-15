#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import matplotlib.pyplot as plt
import re
from statsmodels.formula.api import ols
import numpy as np


# In[2]:


def count(S, target):
 
    # if the total is 0, return 1
    if target == 0:
        return 1
 
    # return 0 if total becomes negative
    if target < 0:
        return 0
 
    # initialize the total number of ways to 0
    result = 0
 
    # do for each coin
    for c in S:
        # recur to see if total can be reached by including current coin `c`
        result += count(S, target - c)
 
    # return the total number of ways
    return result


# In[3]:


if __name__ == '__main__':
 
    # `n` coins of given denominations
    S = [2, 3, 4]
 
    # total change required
    target = 45
 
    print('The total number of ways to get the desired change is', count(S, target))


# In[4]:


import pandas as pd
import datetime
import matplotlib.pyplot as plt
import re
from statsmodels.formula.api import ols


# In[5]:


df_r1_c1 = pd.read_csv('datasets/region1_company1.csv')
df_r1_c2 = pd.read_csv('datasets/region1_company2.csv')
df_r2_c1 = pd.read_csv('datasets/region2_company1.csv')
df_r2_c2 = pd.read_csv('datasets/region2_company2.csv')
df_r1_c1.head()


# In[6]:


def string_date(string):
    string = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", string)
    string = string[1:4] + ' ' + (' '.join([string[5:9][i:i+2] for i in range(0, len(string[5:9]), 2)]))
    string = string.replace(' ','-')
    return string


# In[7]:


df_r1_c1.Date = df_r1_c1.Date.apply(lambda x: string_date(x) )
df_r1_c2.Date = df_r1_c1.Date.apply(lambda x: string_date(x) )
df_r2_c1.Date = df_r1_c1.Date.apply(lambda x: string_date(x) )
df_r2_c2.Date = df_r1_c1.Date.apply(lambda x: string_date(x) )


# In[8]:


df_r1_c1.Date = df_r1_c1.Date.astype('datetime64[D]')
df_r1_c2.Date = df_r1_c1.Date.astype('datetime64[D]')
df_r2_c1.Date = df_r1_c1.Date.astype('datetime64[D]')
df_r2_c2.Date = df_r1_c1.Date.astype('datetime64[D]')


# In[9]:


plt.figure(figsize=(13,9))
plt.xlabel("Time", fontsize = 20)
plt.ylabel("Prices", fontsize = 20)
plt.plot(df_r1_c1.Date,df_r1_c1.Sales_USD)
plt.plot(df_r2_c1.Date,df_r2_c1.Sales_USD)
plt.legend(['Region 1','Region 2'])


# In[10]:


plt.figure(figsize=(13,9))
plt.xlabel("Time", fontsize = 20)
plt.ylabel("Prices", fontsize = 20)
plt.plot(df_r1_c2.Date,df_r1_c2.Sales_USD)
plt.plot(df_r2_c2.Date,df_r2_c2.Sales_USD)
plt.legend(['Region 1','Region 2'])


# In[11]:


df_r1_c1.Date.astype('datetime64[D]')


# In[12]:


plt.xlabel("Time", fontsize = 12)
plt.ylabel("Prices", fontsize = 12)
plt.scatter(df_r1_c1.Sales_USD,df_r1_c1.Sales_U)
plt.scatter(df_r2_c1.Sales_USD,df_r1_c1.Sales_U)
plt.legend(['Region 1','Region 2'])


# In[ ]:





# In[13]:


plt.xlabel("Time", fontsize = 12)
plt.ylabel("Prices", fontsize = 12)
plt.scatter(df_r2_c2.Sales_USD,df_r2_c1.Sales_U)
plt.scatter(df_r2_c2.Sales_USD,df_r1_c1.Sales_U)
plt.legend(['Region 1','Region 2'])


# In[14]:


df_r1_c1.columns


# In[15]:


df_r1_c2.columns


# ## Price elasticity of demand for region 1, company 1

# In[16]:


result_r1_c1 = ols("Sales_U ~ Sales_USD", data=df_r1_c1).fit()
print(result_r1_c1.summary())


# In[17]:


intercept_r1_c1, slope_r1_c1 = result_r1_c1.params
mean_price_r1_c1 = np.mean(df_r1_c1.Sales_USD)
mean_quantity_r1_c1 = np.mean(df_r1_c1.Sales_U)


# In[18]:


#Price elasticity Formula
price_elasticity_r1_c1 = (slope_r1_c1)*(mean_price_r1_c1/mean_quantity_r1_c1)  
print('price elasticity of demand for company 1 in region 1:', price_elasticity_r1_c1)


# ## Price elasticity of demand for region 2, company 1

# In[19]:


result_r2_c1 = ols("Sales_U ~ Sales_USD", data=df_r2_c1).fit()
print(result_r2_c1.summary())


# In[20]:


intercept_r2_c1, slope_r2_c1 = result_r2_c1.params
mean_price_r2_c1 = np.mean(df_r2_c1.Sales_USD)
mean_quantity_r2_c1 = np.mean(df_r2_c1.Sales_U)


# In[21]:


#Price elasticity Formula
price_elasticity_r2_c1 = (slope_r2_c1)*(mean_price_r2_c1/mean_quantity_r2_c1)  
print('price elasticity of demand for company 1 in region 2:', price_elasticity_r2_c1)


# ## Price elasticity of demand for region 1, company 2

# In[22]:


result_r1_c2 = ols("Sales_U ~ Sales_USD", data=df_r1_c2).fit()
print(result_r1_c2.summary())


# In[23]:


intercept_r1_c2, slope_r1_c2 = result_r1_c2.params
mean_price_r1_c2 = np.mean(df_r1_c2.Sales_USD)
mean_quantity_r1_c2 = np.mean(df_r1_c2.Sales_U)


# In[24]:


#Price elasticity Formula
price_elasticity_r1_c2 = (slope_r1_c2)*(mean_price_r1_c2/mean_quantity_r1_c2)  
print('price elasticity of demand for company 1 in region 2:', price_elasticity_r1_c2)


# ## Price elasticity of demand for region 2, company 2

# In[25]:


result_r2_c2 = ols("Sales_U ~ Sales_USD", data=df_r2_c2).fit()
print(result_r2_c1.summary())


# In[26]:


intercept_r2_c2, slope_r2_c2 = result_r2_c2.params
mean_price_r2_c2 = np.mean(df_r2_c2.Sales_USD)
mean_quantity_r2_c2 = np.mean(df_r2_c2.Sales_U)


# In[27]:


#Price elasticity Formula
price_elasticity_r2_c2 = (slope_r2_c2)*(mean_price_r2_c2/mean_quantity_r2_c2)  
print('price elasticity of demand for company 1  in region 2:', price_elasticity_r2_c2)


# ## Absolute magnitud of the price elasticity of demand for company 1

# In[28]:


print('price elasticity of demand for company 1:', price_elasticity_r1_c1 + price_elasticity_r2_c1)


# ## Absolute magnitud of the price elasticity of demand for company 2
# 

# In[29]:


print('price elasticity of demand for company 2:', price_elasticity_r1_c2 + price_elasticity_r2_c2)


# ## To calculate  the % change in unit sales we'll use:
# $$\Delta Q=\:\frac{\Delta P.E.Q}{P}$$
# $$\Delta Q: Change, in the sales$$
# $$\Delta P: Change in the price$$
# $$Q: Sales$$
# $$P: Price$$
# $$E: Price elasticity of demand$$
# 
# 
# 

# In[30]:


change_in_quantity = (0.1*1.1422119379726128*mean_quantity_r1_c1)/mean_price_r1_c1
print('the change in unit sales in company 1 for region 1 is:',change_in_quantity )


# In[ ]:




