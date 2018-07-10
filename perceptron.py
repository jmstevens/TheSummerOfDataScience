import numpy as np
import random
from sklearn.datasets import load_boston
"""Based on code written by Sidd Karamcheti, https://github.com/siddk/multiclass_perceptron/blob/master/mc_perceptron.py"""
BIAS = 1
TRAIN_TEST_RATIO = .90
ITERATIONS = 10000

class Perceptron():
    def __init__(self, features, target, train_test_ratio=TRAIN_TEST_RATIO, iterations=ITERATIONS):
        dataset = np.column_stack((features, target))
        self.train_test_ratio = train_test_ratio
        np.random.shuffle(dataset)
        train = dataset[:int(len(dataset) * train_test_ratio)]
        test = dataset[int(len(dataset) * train_test_ratio):]
        self.test_set = test
        self.train_set = train
        self.iterations = iterations

        training_transform = self.transform(self.train_set)
        target_transformed = training_transform[:, len(training_transform[0])-1]

        classes = np.unique(target_transformed).tolist()
        classes = [int(i) for i in classes]
        self.classes = classes
        # Create weights
        self.weights = {c: np.array([0 for _ in range(len(train[0]) - 1)]) for c in classes}

    def normalization(self, x):
        """Scales a numpy array by min-max."""
        return ((x - x.min()) / (x.max() - x.min()))

    def transform(self, x):
        features_scaled = np.apply_along_axis(self.normalization, 0,
                                              x[:, :len(x[0])-1])
        target_binned = x[:, len(x[0])-1]
        target_binned = np.digitize(np.ceil(target_binned), [20, 30, 50])
        return np.column_stack((features_scaled, target_binned))

    def train(self, bias=1):
        target_feature_i = len(self.train_set[0]) - 1
        x_y = self.transform(self.train_set)
        x = x_y[:, :target_feature_i]
        y = x_y[:, target_feature_i]
        for _ in range(self.iterations):
            errors = 0
            for xi, target in zip(x, y):
                # Initialize arg_max
                arg_max, predicted_class = 0, self.classes[0]

                # MultiClass Rule
                for c in self.classes:
                    current_activation = np.dot(xi, self.weights[c]) + bias
                    if current_activation >= arg_max:
                        arg_max, predicted_class = current_activation, c
                    # print(current_activation)
                # Update Rule:
                if not (int(target) == int(predicted_class)):
                    errors += 1
                    self.weights[target] = self.weights[target] + xi
                    self.weights[predicted_class] = self.weights[predicted_class] - xi

    def predict(self, X, bias=1):
        xi = X
        arg_max, predicted_class = 0, self.classes[0]

        # MultiClass Rule
        for c in self.classes:
            current_activation = np.dot(xi, self.weights[c]) + bias
            if current_activation >= arg_max:
                arg_max, predicted_class = current_activation, c
        return predicted_class

if __name__ == '__main__':
    boston = load_boston()
    features = boston.data
    target = boston.target
    _p = Perceptron(features, target)
    print(_p.weights[0].shape)
    _p.train()
    xy_test = _p.transform(_p.test_set)
    # print(len(xy_test[0]) - 1)
    x = xy_test[:, :len(xy_test[0]) - 1]
    # print(x.shape)
    y = xy_test[:, len(xy_test[0]) - 1]
    # y_predict = _p.predict(x)
    error=0
    itor=0
    for xi, y in zip(x, y):
        predicted_class = _p.predict(xi)
        actual_class = y
        itor += 1
        print(itor)
        if predicted_class != actual_class:
            error += 1
            print("Errors found={}".format(error))
            print(predicted_class)
            print(actual_class)
    print("{}%".format(int(100*(1-(error/len(xy_test))))))
