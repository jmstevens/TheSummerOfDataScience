## TODO: Implement Cross Validation on Page 83

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')
print(mnist)

X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

some_digit = X[36000]
print(some_digit)
some_digit_image = some_digit.reshape(28, 28)

# Print image
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
#                interpolation="nearest")
# plt.axis("off")
# plt.show()

print(y[36000])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
#Let’s also shuffle the training set
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Training a Binary Classifier
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits. y_test_5 = (y_test == 5)
# Lets use Gradient Descent
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))

# Performance Measures
# Evaluating a classifier is often significantly trickier than evaluating a regressor, so we will spend a large part of this chapter on this topic.

# Measuring Accuracy Using Cross-Validation
print(X_train)
print(y_train_5)
cross_score = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(cross_score)

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_score = cross_val_score(never_5_clf, X_train, y_train, cv=3, scoring="accuracy")
print(cross_score)

# This demonstrates why accuracy is
# generally not the preferred performance measure for classifiers,
# especially when you are dealing with skewed datasets (i.e., when some
# classes are much more frequent than others).

## Confusion Matrix --> Much better way to evaluate classifiers (Counts time, A are classfied as B)
# Need a set of predictions to compare to actual targets
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5) # Instead of returning evaluation scores, it returns predictions
print(y_train_pred)
print(y_train_5)
_confusion = confusion_matrix(y_train_5, y_train_pred)
print(_confusion)

# Precision = TP/TP + FP
# Recall = TP/TP + FN

print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))
print(f1_score(y_train_5, y_train_pred))

y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

threshold = 200000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                                 method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()

# You can see that precision really starts to fall sharply around 80% recall.
# You will probably want to select a precision/recall tradeoff just before that
# drop—for example, at around 60% recall. But of course the choice depends on your project.

## The ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores) # false positive rate, true positive rate

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

# plot_roc_curve(fpr, tpr)
# plt.show()

## One way to compare classifiers is to measure the area under the curve (AUC)
### Perfect classifier will have ROC AUC equal to 1, purely random will have an AUC of 0.5
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

# GENERAL RULE:
    # USE THE PR CRUVE WHENEVER THE POSITIVE CLASS IS RARE OR WHEN YOU CARE MORE ABOUT FALSE POSITIVES THAN FALSE NEGATIVES!!!!

# Lets train a RandomForestClassifier
    # Compare its ROC and ROC AUC score to the SGDClassifier
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                            method="predict_proba")
print(y_probas_forest.shape)
print(y_probas_forest[:, 1])
# For ROC we need scores not probabilities, simple solution is to use the postive class's probability as the score
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)

print(fpr_forest)
print(tpr_forest)
print(threshold_forest)

# plt.plot(fpr, tpr, "b:", label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.legend(loc="lower left")
# plt.show()


print(roc_auc_score(y_train_5, y_scores_forest))


precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores_forest)
print(precisions)
print(recalls)
print(thresholds)
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)



# Multiclass Classification
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
print(sgd_clf.predict([some_digit]))
some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)

from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
print(ovo_clf.classes_)
print(len(ovo_clf.estimators_))


forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
forest_clf.predict_proba([some_digit])
print(forest_clf.predict_proba([some_digit]))

print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))


## Error Analysis
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()

# Focus on errors, need to calculate error rates
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# Fill diagonal with zeros to keep only errors
np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()


# cl_a, cl_b = 3, 5
# X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
# X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
# X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
# X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
# plt.figure(figsize=(8,8))
# plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
# plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
# plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
# plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
# plt.show()
## Multilabel Classification
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])
print(knn_clf.predict([some_digit]))
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
print(f1_score(y_train, y_train_knn_pred, average="macro"))



# Multioutput Clasification
noise = rnd.randint(0, 100, (len(X_train), 784))
noise = rnd.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
