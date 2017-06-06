import pandas as pd

# Read the data
train = pd.read_csv('~/Titanic/Data/train.csv')
test = pd.read_csv('~/Titanic/Data/test.csv')
gender = pd.read_csv('~/Titanic/Data/gender_submission.csv')

print('training set is loaded')
print(train)
