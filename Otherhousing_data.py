import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn import preprocessing



# In[143]:
def file_path_input_data(filename):
    """Script which creates the path to the mondrian xml file.
        Taken from the following StackOverflow post.
        https://stackoverflow.com/questions/16503748/navigating-file-locations
        ^This can be deleted eventually^
    """
    cwd = os.path.dirname(os.path.realpath(__file__))  # path to current file
    par_dir = cwd.split('Titanic/')[0] # path to parent directory
    my_file = os.path.join(par_dir,
                           'Data/',
                           filename
                           )
    return my_file

data = pd.read_csv(file_path_input_data('OtherHousing.csv'))

Y = data[["SalePrice"]]
cols_to_drop = ["SalePrice", "Fence", "Misc.Feature", "Misc.Val", "Utilities", "Heating", "Gr.Liv.Area", "Electrical", "Street"]
X = data.drop(cols_to_drop, axis = 1)

# In[144]:
le = LabelEncoder()
X = X.apply(lambda x: x.astype('category').cat.codes if x.dtype == 'object' else x, axis = 0)

imp = Imputer(missing_values=float("NaN"), strategy="median")
X = imp.fit_transform(X)

Xtr, Xt, Ytr, Yt = train_test_split(X, Y, test_size = 0.1, random_state=52)
#
Y = Ytr
X = Xtr
#
# # In[151]:
#
#
Xnorm = preprocessing.scale(Xtr)
Ynorm = preprocessing.scale(Ytr)
Xtnorm = preprocessing.scale(Xt)
Ytnorm = preprocessing.scale(Yt)
Y = Ynorm
X = Xnorm
Xt = Xtnorm
Yt = Ytnorm
print(X)

# w = weights b = biase
# ih is input to hidden
# hh is hidden to hidden
# ho is hidden to output

inputs = X.shape[1]
print(inputs)
print(X)
hidden = 10
hidden2 = 9
output = 1
#
wih = np.random.randn(inputs, hidden)
print(wih)
whh = np.random.randn(hidden, hidden2)
who = np.random.randn(hidden2, output)
#
bih = np.zeros((1, hidden))
bhh = np.zeros((1, hidden2))
bho = np.zeros((1, output))
#
for i in range(10000):

    if i > 1:
        bih = bihup
        bhh = bhhup
        bho = bhoup
        wih = wihup
        whh = whhup
        who = whoup

    pre1 = np.dot(X, wih) + bih #X.dot(wih) + bih
    act1 = 1/(1+np.exp(pre1))
    pre2 = np.dot(act1, whh) + bhh #Xact1.dot(whh) + bhh
    act2 = 1/(1+np.exp(pre2))
    pre3 = np.dot(act2, who) + bho #Xact2.dot(who) + bho
    act3 = 1/(1+np.exp(pre3))


    errors = Y - act3
    derrors = errors * (1/(1+np.exp(act3)) *  (1-(1/(1+np.exp(act3)))))
    hherror = np.dot(derrors, who.T)
    hhdelta = hherror * (1/(1+np.exp(act2)) * (1-(1/(1+np.exp(act2)))))
    hierror = np.dot(hhdelta, whh.T)
    hidelta = hierror * (1/(1+np.exp(act1)) * (1-(1/(1+np.exp(act1)))))



    whoup = who - 0.05 * np.dot(act2.T, derrors) #act2.T.dot(derrors)
    whhup = whh - 0.05 * np.dot(act1.T, hhdelta) #act1.T.dot(hhdelta)
    wihup = wih - 0.05 * np.dot(X.T, hidelta) #X.T.dot(hidelta)
    bhoup = bho - 0.05 * derrors
    bhhup = bhh - 0.05 * hhdelta
    bihup = bih - 0.05 * hidelta

    error = np.sum((errors)**2)*0.5
    print(error)
