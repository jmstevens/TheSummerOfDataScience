import pandas as pd

# Read the data
class Titanic:
    def train(self):
        return pd.read_csv('~/Titanic/Data/train.csv')
    def test(self):
        return pd.read_csv('~/Titanic/Data/test.csv')
    def gender(self):
        return pd.read_csv('~/Titanic/Data/gender_submission.csv')




