
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import Imputer


# In[143]:


data = pd.read_csv('/Users/RobertKohler/Desktop/OtherHousing.csv')

Y = data[["SalePrice"]]
X = data.loc[:,data.columns != "SalePrice" + "Fence" + "Misc.Feature" + "Misc.Val" + "Utilities" + "Heating" + "Gr.Liv.Area" + "Electrical" + "Street"]



# In[144]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col_name in X.columns:
    if(X[col_name].dtype == 'object'):
        X[col_name]= X[col_name].astype('category')
        X[col_name] = X[col_name].cat.codes
        


# In[145]:


imp = Imputer(missing_values=float("NaN"), strategy="median")
X = imp.fit_transform(X)


# In[146]:


Xtr, Xt, Ytr, Yt = train_test_split(X, Y, test_size = 0.1, random_state=52)

Y = Ytr
X = Xtr


# In[151]:


Xnorm = preprocessing.scale(Xtr)
Ynorm = preprocessing.scale(Ytr)
Xtnorm = preprocessing.scale(Xt)
Ytnorm = preprocessing.scale(Yt)
Y = Ynorm
X = Xnorm
Xt = Xtnorm
Yt = Ytnorm
X


# In[231]:


inputs = X.shape[1]
hidden = 5
hidden2 = 9
output = Y.shape[0]


wih = np.random.randn(inputs, hidden)
whh = np.random.randn(hidden, hidden2)
who = np.random.randn(hidden2, output)

bih = np.zeros((1, hidden))
bhh = np.zeros((1, hidden2))
bho = np.zeros((1, output))

for i in range(1000):
    
    pre1 = X.dot(wih) + bih        
    act1 = 1/(1+np.exp(pre1))
    pre2 = act1.dot(whh) + bhh
    act2 = 1/(1+np.exp(pre2))
    pre3 = act2.dot(who) + bho
    act3 = 1/(1+np.exp(pre3))

    
    errors = Y - act3
    derrors = errors *(1-np.exp(act3))
    hherror = np.dot(derrors, who.T)
    hhdelta = hherror *(1-np.exp(act2))
    hierror = np.dot(hhdelta, whh.T)
    hidelta = hierror *(1-np.exp(act1))
  
  

    who = who - 0.03 * act2.T.dot(derrors)
    whh = whh - 0.02 * act1.T.dot(hhdelta)
    wih = wih - 0.01 * X.T.dot(hidelta)
    bho = bho - 0.01 * derrors
    bhh = bhh - 0.01 * hhdelta
    bih = bih - 0.01 * hidelta
  
    error = np.sum(np.mean(errors)**2)
    print(error)


 


# In[ ]:




