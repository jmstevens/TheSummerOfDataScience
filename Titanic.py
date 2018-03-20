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
        df[df['Sex'] == 'male'] = 0
        df[df['Sex'] == 'female'] = 1
        # df['Survived'] = df.Survived.astype('category')
        # df['Pclass'] = df.Pclass.astype('category')
        # df['Sex'] = df.Sex.astype('category')
        # df['Embarked'] = df.Embarked.astype('category')

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
        return pd.read_csv('~/Titanic/Data/gender_submission.csv')

print(yellow(Titanic().__doc__))
train = Titanic().train()

# print(yellow(train.head()))
# for i in Titanic().train().columns.values.tolist():
#     print(cyan(i))
#     print(green(Titanic().train()[i].astype('category').describe()))
#
# print(cyan(train.groupby(['Survived','Sex']).mean()))
# print(cyan(train.groupby(['Survived','Sex','Pclass']).count()))

#matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import ProbPlot
plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)

model_f = 'Survived ~ Sex + \
                 Pclass + \
                 Age + \
                 SibSp + \
                 Parch + \
                 Ticket + \
                 Fare + \
                 Family_Size + \
                 Fare_Per_Person'
model = smf.ols(formula=model_f, data=train)
model_fit = model.fit()

# fitted values (need a constant term for intercept)
model_fitted_y = model_fit.fittedvalues
# model residuals
model_residuals = model_fit.resid
# print(model_residuals)
# normalized residuals
model_norm_residuals = model_fit.get_influence().resid_studentized_internal
#absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
# absolute residuals
model_abs_resid = np.abs(model_residuals)
# leverage, from statsmodels internals
model_leverage = model_fit.get_influence().hat_matrix_diag
# cook's distance, from statsmodels internals
model_cooks = model_fit.get_influence().cooks_distance[0]
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)
plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'Survived', data=train,
                          lowess=True,
                          scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')


# annotations
abs_resid = model_abs_resid.sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]
for i in abs_resid_top_3.index:
    plot_lm_1.axes[0].annotate(i,
                               xy=(model_fitted_y[i],
                                   model_residuals[i]));
plt.show()
