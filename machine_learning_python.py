## Machine Learning with Python Cookbook
# Chapter 2: Loading Data
# Replacing values
# Column is a feature
import pandas as pd
url = 'https://tinyurl.com/titanic-csv'

dataframe = pd.read_csv(url)

dataframe['Sex'].replace("female","Woman").head(2)

dataframe.replace(r"1st","First", regex=True).head(2)

# Rename columns
dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2)

# Change the name of multiple columns as once
dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2)

# 4.7 Finding the minimum, maximum, sum, average, and count
# Calculate statistics
print('Maximum:', dataframe['Age'].max())
print('Maximum:', dataframe['Age'].min())
print('Maximum:', dataframe['Age'].mean())
print('Maximum:', dataframe['Age'].sum())
print('Maximum:', dataframe['Age'].count())

# Delete column
df.drop('Age', axis=1).head(2)

# Delete multiple columns
df.drop(['Age','Sex'], axis=1).head(2)

# Delete rows, show first two rows of output
df[df['Sex'] != 'male'].head(2)

# Delete row, show first two rows of output
df[df['Name'] != 'Allison, Miss Helen Loraine'].head(2)

# Delete row, show first two rows of output
df[df.index != 0].head(2)

# Dropping duplicate rows (wrong way)
df.drop_duplicates().head(2)

# Drop duplicates correct way
df.drop_duplicates(subset=['Sex'])

# Drop duplicates
df.drop_duplicates(sub=['Sex'], keep='last')

# Groupby is one of the most powerful features in pandas
df.groupby('Sex').mean()

# Group rows, count rows
df.groupby('Survived')['Name'].count()

# Group rows, calculate mean
df.groupby(['Sex', 'Survived'])['Age'].mean()

## Grouping rows by time
time_index = pd.date_range('06/06/2017', periodes=100000, freq='30S')

# Create DataFrame
df = pd.DataFrame(index=time_index)

# Create column of random values
df['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Group rows by week, calculate sum per week
df.resample('W').sum()

# Group by two weeks, calculate mean
df.resample('2W').mean()

# Group by month, count rows
df.resample('M').count()

# You might notice that in the two outputs the datetime index is a date despite
# the fact that we are grouping by weeks and months, respectively.
# The reason is because by default resample returns the label of the right “edge”
# (the last label) of the time group. We can control this behavior using the label parameter:

# Group by month, count rows
# http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
df.resample('M', label='left').count()

# 3.15 Looping Over a Column
# Print first two names uppercased
for name in df['Name'][0:2]:
    print(name.upper())

# List comprehension version of the Looping
[name.upper() for name in df['Name'][0:2]]

# Create function
def uppercase(x):
    return x.upper()

# Apply function show two rows
df['Name'].apply(uppercase)[0:2]

# apply is a great way to do data cleaning and wrangling.
# It is common to write a function to perform some useful operation
# (separate first and last names, convert strings to floats, etc.)
# and then map that function to every element in a column.

# Group rows using groupby and want to apply a function to each groups
df.groupby('Sex').apply(lambda x: x.count())

# I mentioned apply.
# apply is particularly useful when you want to apply a function to groups.
# By combining groupby and apply we can calculate custom statistics or
# apply any function to each group separately.

# 3.18 Concatenating DataFrames
data_a = {'id': ['1', '2', '3'],
          'first': ['Alex', 'Amy', 'Allen'],
          'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns=['id', 'first', 'last'])

# Create DataFrame
data_b = {'id': ['4', '5', '6'],
          'first': ['Billy', 'Brian', 'Bran'],
          'last': ['Bonder', 'Black', 'Balwner']}

dataframe_b = pd.DataFrame(data_b, columns=['id','first','last'])

# Concatenate DataFrames by rows
pd.concat([dataframe_a, dataframe_b], axis=0)

# Concatenate DataFrames by columns
pd.concat([dataframe_a, dataframe_b], axis=1)

# Create row
row = pd.Series([10, 'Chris', 'Chillon'], index=['id','first','last'])

# Append row
dataframe_a.append(row, ignore_index=True)

## 3.19 Merging DataFrames
employee_data = {'employee_id': ['1', '2', '3', '4'],
                 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                 'Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id',
                                                              'name'])
# Create DataFrame
sales_data = {'employee_id': ['3', '4', '5', '6'],
              'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id',
                                                      'total_sales'])
# Merge DataFrames outer
pd.merge(dataframe_employees, dataframe_sales, on='employee_id')

# Merge DataFrames left or right
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left')

# Merge DataFrames
pd.merge(dataframe_employees,
         dataframe_sales,
         left_on='employee_id',
         right_on='employee_id')

# Oftentimes, the data we need to use is complex;
# it doesn’t always come in one piece.
# Instead in the real world, we’re usually faced with disparate datasets,
# from multiple database queries or files. To get all that data into one place,
# we can load each data query or data file into pandas as individual DataFrames and
# then merge them together into a single DataFrame.
#
# This process might be familiar to anyone who has used SQL, a popular language
# for doing merging operations (called joins).
# While the exact parameters used by pandas will be different, they follow the same general patterns used by other software languages and tools.
#
# There are three aspects to specify with any merge operation.
# First, we have to specify the two DataFrames we want to merge together.
# In the solution we named them dataframe_employees and dataframe_sales.
# Second, we have to specify the name(s) of the columns to merge on—that is,
# the columns whose values are shared between the two DataFrames.
# For example, in our solution both DataFrames have a column named employee_id.
# To merge the two DataFrames we will match up the values in each DataFrame’s employee_id
# column with each other. If these two columns use the same name, we can use the on parameter.
# However, if they have different names we can use left_on and right_on.
#
# What is the left and right DataFrame? The simple answer is that the left DataFrame is
# the first one we specified in merge and the right DataFrame is the second one.
# This language comes up again in the next sets of parameters we will need.
#
# The last aspect, and most difficult for some people to grasp,
# is the type of merge operation we want to conduct.
# This is specified by the how parameter. merge supports the four main types of joins:
#
# Inner
# Return only the rows that match in both DataFrames (e.g., return any row with
# an employee_id value appearing in both dataframe_employees and dataframe_sales).
#
# Outer
# Return all rows in both DataFrames. If a row exists in one DataFrame but not in
# the other DataFrame, fill NaN values for the missing values
# (e.g., return all rows in both employee_id and dataframe_sales).
#
# Left
# Return all rows from the left DataFrame but only rows from the right DataFrame
# that matched with the left DataFrame. Fill NaN values for the missing values
# (e.g., return all rows from dataframe_employees but only rows from dataframe_sales
# that have a value for employee_id that appears in dataframe_employees).
#
# Right
# Return all rows from the right DataFrame but only rows from the left DataFrame
# that matched with the right DataFrame. Fill NaN values for the missing values
# (e.g., return all rows from dataframe_sales but only rows
#  from dataframe_employees that have a value for employee_id that appears in dataframe_sales).
#
# If you did not understand all of that right now,
# I encourage you to play around with the how parameter in your code
# and see how it affects what merge returns.

# Visual explanation of joins
# https://blog.codinghorror.com/a-visual-explanation-of-sql-joins/

### 4: Handling Numeric Data
# 4.1 Rescaling a Feature
import numpy as np
from skilearn import preprocessing

# Create feature
feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])
# Create scaler
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))

# Scale feature
scaled_feature = minmax_scale.fit_transform(feature)

# Rescaling is a common preprocessing task in machine learning.
# Many of the algorithms described later in this book will assume
# all features are on the same scale,
# typically 0 to 1 or –1 to 1.
# There are a number of rescaling techniques, but one of the simplest
# is called min-max scaling. Min-max scaling uses the minimum and maximum values
# of a feature to rescale values to within a range. Specifically, min-max calculates:

# where x is the feature vector, x’i is
# an individual element of feature x, and x’i is
# the rescaled element. In our example, we can see from the outputted
# array that the feature has been successfully rescaled to between 0 and 1:

# scikit-learn’s MinMaxScaler offers two options to rescale a feature.
# One option is to use fit to calculate the minimum and maximum values of the feature,
# then use transform to rescale the feature. The second option is to use fit_transform
# to do both operations at once. There is no mathematical difference between the two options,
# but there is sometimes a practical benefit to keeping the operations separate because
# it allows us to apply the same transformation to different sets of the data.

# 4.2 Standardizing a feature
# Problem: you want to transform a feature to have a mean of 0 and a std of 1
# Solution: scikit-learn's StandardScaler performs both transformations
import numpy as np
from sklearn import preprocessing

x = np.array([[-1000.1],
              [-200.2],
              [500.5],
              [600.6],
              [9000.6]])
# Create scaler
scaler = preprocessing.StandardScaler()
# Transform the feature
standardized = scaler.fit_transform(x)

# Show feature
standardized

# Notes
# Standarizing is a common go-to scaling method for machine learning
# preprocessing (used more than min-max scaling).
# Depending on the learning algorithm though, principle component analysis
# often works better using standardization, while neural networks work
# well with min-max scaler

# Default to standardization unless you have a specific reason

# See the effect of standardization by printing mean and standard deviation
print("Mean:", round(standardized.mean()))
print("Standard deviation:", standardized.std())

# If there's significant outliers, it can negatively impact our standardization
# by affecting the feature's mean and variance.

# If this is th case, it is often helpful to rescale the feature using the median
# and quartile range.

# In scikit-learn use the RobustScaler method

# Create scaler
robust_scaler = preprocessing.RobustScaler()

# Transform feature
robust_scaler.fit_transform(x)

## 4.3 Normalizing Observations
# Problem: You want to rescale the feature values of observations to have unit norm (a total length of 1)
# Solution: Use Normalizer with a norm argument

# L2 is the Euclidean norm
# How far the crow flies
# L1 Manhatten Norm
# Load libraries
import numpy as np
from sklearn.preprocessing import Normalizer

# Create feature matrix
features = np.array([[0.5, 0.5],
                     [1.1, 3.4],
                     [1.5, 20.2],
                     [1.63, 34.4],
                     [10.9, 3.3]])
# Create normalizer
normalizer = Normalizer(norm="12")

# Transform feature matrix
normalizer.transform(features)

# Notes: Many rescaling methods (min-max and standardization) operate on features;
# however we can also rescale across individual operations. Normalizer rescales
# the values on an individual observations to have unit norm (the sum of their lengths is 1).
# This type of rescaling is often used when we have many equivalent features
# (e.g., text classification when every word or n-word group is a feature)

# Transform feature matrix
features_12_norm = Normalizer(norm="l2").transform(features)

# Show feature matrix
features_l2_norm

# Print sum
print("Sum of the first observation\'s values:",
      features_l1_norm[0,0] + features_l1_norm[0,1])

## 4.4 Generating Polynomial and Interaction Features
# Load libraries
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create feature matrix
features = np.array([[3, 3],
                     [2, 3],
                     [2, 3]])

# Create PolynomialFeatures object
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)

# Create polynomial features
polynomial_interaction.fit_transform(features)

# Chapter 11: Model Evaluation











# Chapter 13
# Linear Regression

# Load libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load data with only two features
boston = load_boston()
features = boston.data[:,0:2]
target = boston.target

# Create linear regression
regression = LinearRegression()

# Fit the linear regession
model = regression.fit(features, target)

# LinearRegression regression assumes that the relationship between the features and the target vector is
# approximently linear. That is, the effect (also called coefficient, weight, or parameter) fo the features
# on the target vecor is constant. In our solution, for the sake of explanation we have trained out model
# using only two features

# View the intercept
model.intercept_

# View the feature
model.coef_

# First value in the target vector multiplied by 1000
target[0] * 1000

import numpy
