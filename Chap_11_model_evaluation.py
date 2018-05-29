# 11.1 Cross-Validating Models
# Problem: You want to evaluate how well your model will work in the real world
# Solution: Create a pipeline which preprocesses the data, trains model, and evaluates
# using cross validation.

# Notes:
# Our goal is not to evaluate how well the model does on the training set but how well it does on
# data it has never seen before.

# Validation approach
    # features and targets --> training set and test set
        # Split data into two sets
        # Train model then evaluate with the test set
    # Two major weaknesses
        # 1.) Performance is highly dependent on the data in the two splits
        # 2.) Model isnt being trained with all available data
# Better approach
    # k-fold cross-validation (KFCV)
        # Split data into k parts called "folds"
        # Model is trained using k-1 folds
        # Combined into one training set
        # Last fold is used as the test set
        # Repeat k times, using a different fold of the test dataset
        # Performance of each k iteration is then averaged to produce an overall measurement

from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# load digits dataset
digits = datasets.load_digits()

# create features matrix
features = digits.data

# create target vector
target = digits.target

# create standardizer
standardizer = StandardScaler()

# create logistic regression
logit = LogisticRegression()

# Create a pipeline that standardizes, then runs logistic regression
pipeline = make_pipeline(standardizer, logit)

# Create k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Conduct k-fold cross-validation
cv_results = cross_val_score(pipeline,
                             features,
                             target,
                             cv=kf,
                             scoring="accuracy",
                             n_jobs=-1)

# Important things to consider for KFCV
# 1.) Assumes IID (independent identically distributed)
    # If it is, good idea to shuffle observations, i.e. shuffle=True
# 2.) Beneficial to have folds contain roughly the same percentage of observations
#       # I.e. target vector has 80% male, 20% female
    # We do this by using KFold class with StratifiedKFold instead of KFold
# 3.) Its important to preprocess the data, based on the training set and apply those transformations
    # using transform to both training and test sets

# Import Library
from sklearn.model_selection import train_test_split

# Create training and test sets
features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target,
                                                                            test_size=0.1,
                                                                            random_state=1)
# Fit standardizer to training set
standardizer.fit(features_train)

# Apply to both training and test sets
features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)

# The reason we do this is because we are pretending that the test set is unknown data
# If we fit both our preprocessers using observations from both training and test sets,
# some info leaks from the test set into the training set.

# This applies to any preprocessing step such as feature selection
## TODO : Try to understand that the above

# scikit-learn's pipeline package makes this easy!!!!
# First create a pipeline that preprocesses the data (e.g. standardizer) and then trains a model (LogisticRegression, logit)

# Create a pipeline
pipeline = make_pipeline(standardizer, logit)

# Do KFCV using the pipeline and scikit does all the work for us!
# Do k-fold cross-validation
cv_results = cross_val_score(pipeline,
                             features,
                             target,
                             cv=kf,
                             scoring="accuracy",
                             n_jobs=1)
# cv determines our cross-validation technique
# scoring parameter defines our metric of success
# n_jobs = -1 tells scikit-learn to use every core.
