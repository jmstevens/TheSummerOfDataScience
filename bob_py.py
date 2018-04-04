import pandas as pd
import statsmodels.api as sm


df = pd.read_csv(filepath_or_buffer='/Users/jrzemien/Titanic/Data/train_bob_2.csv')
res = sm.formula.glm("Survived~Pclass+Sex+Age+SibSp+Parch",  family=sm.families.Binomial(),
                     data=df).fit()
sum = res.summary()
print(sum)
