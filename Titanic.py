import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import numpy as np
import pandas as pd
# from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import os
# Read the data
class Titanic:
    def __init__(self):
        cwd = os.path.dirname(os.path.realpath(__file__))  # path to current file
        par_dir = cwd.split('Titanic/')[0] # path to parent directory
        report_list = os.listdir(par_dir + '/Data/')
        self.files = report_list
        
    def train(self):
        return pd.read_csv('~/Titanic/Data/train.csv')

    def test(self):
        return pd.read_csv('~/Titanic/Data/test.csv')

    def gender(self):
        return pd.read_sv('~/Titanic/Data/gender_submission.csv')

print(Titanic().files)

X = Titanic().train().drop(['Survived','Name','Ticket','Embarked','Cabin'],axis=1)
y = Titanic().train()['Survived']


# clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
#                                max_depth=3, min_samples_leaf=5)
#
# clf_gini.fit(X,y)
