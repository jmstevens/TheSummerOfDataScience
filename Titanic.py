# !/usr/bin/env python -W ignore::DeprecationWarning
import numpy as np
import pandas as pd
from ggplot import *
from fabric.colors import green, red, yellow, blue, magenta, cyan
#from sklearn.cross_validation import train_test_split
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
#from sklearn import tree
# from sklearn.cross_validation import train_test_split
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
#from sklearn import tree
import os
# Read the data
class Titanic(object):
    """Data Dictionary
    Variable	Definition	Key
    survival	Survival	0 = No, 1 = Yes
    pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
    sex	Sex
    Age	Age in years
    sibsp	# of siblings / spouses aboard the Titanic
    parch	# of parents / children aboard the Titanic
    ticket	Ticket number
    fare	Passenger fare
    cabin	Cabin number
    embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
    Variable Notes
    pclass: A proxy for socio-economic status (SES)
    1st = Upper
    2nd = Middle
    3rd = Lower

    age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

    sibsp: The dataset defines family relations in this way...
    Sibling = brother, sister, stepbrother, stepsister
    Spouse = husband, wife (mistresses and fiancés were ignored)

    parch: The dataset defines family relations in this way...
    Parent = mother, father
    Child = daughter, son, stepdaughter, stepson
    Some children travelled only with a nanny, therefore parch=0 for them,
    """
    def __init__(self):
        cwd = os.path.dirname(os.path.realpath(__file__))  # path to current file
        par_dir = cwd.split('Titanic/')[0] # path to parent directory
        report_list = os.listdir(par_dir + '/Data/')
        self.files = report_list

    def edit_data(self, df, dropna=True):
        """Function to clean up Titanic datasets.
        Procedure is as follows
        1.) Drop NA's
        2.) Set PassengerId as Index
        3.) Change Data Categories
        4.) Strip Cabin into two columns
            a.) Cabin Letter
            b.) Cabin Number
        """
        # Drop NA's
        if dropna:
            df = df.dropna()
        # Set Index
        # df['id'] = df['id'].astype('category')
        df = df.set_index(['PassengerId'])
        df.index.names = ['id']
        # Edit data types
        # df[df['Sex'] == 'male'] = 0
        # df[df['Sex'] == 'female'] = 1
        df['Survived'] = df.Survived.astype('category')
        df['Pclass'] = df.Pclass.astype('category')
        df['Sex'] = df.Sex.astype('category')
        df['Embarked'] = df.Embarked.astype('category')

        # Add Titles
        df['Family_Size']=df['SibSp']+df['Parch']
        df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)

        return df

    def train(self):
        df = pd.read_csv('~/Titanic/Data/train.csv')
        return self.edit_data(df)

    def test(self):
        return pd.read_csv('~/Titanic/Data/test.csv')

    def gender(self):
        return pd.read_sv('~/Titanic/Data/gender_submission.csv')

print(yellow(Titanic().__doc__))
train = Titanic().train()

print(yellow(train.head()))
for i in Titanic().train().columns.values.tolist():
    print(cyan(i))
    print(green(Titanic().train()[i].describe()))
# [print(red(i)) for i in train['SibSp'].tolist()]
# Find group by survivability
print(cyan(train.groupby(['Survived','Sex']).mean()))
print(cyan(train.groupby(['Survived','Sex','Pclass']).count()))

# ggplot(data=train, x='Age', y='')
# X_train = Titanic().train().dropna().drop(['Survived','Name'],axis=1).dropna()
# y_train = Titanic().train().draopna()['Survived']

print(cyan("Begin data profiling"))
# X_train.drop([''])


# clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
#                                max_depth=3, min_samples_leaf=5)
#
# clf_gini.fit(X,y)
