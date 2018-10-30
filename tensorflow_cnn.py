import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
# %matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28,1)
x_test = x_test.reshape(-1, 28, 28,1)

y_train = keras.utils.to_categorical(y_train, num_classes=10)
random_seed = 2

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = random_seed)

plt.imshow(x_train[0][:,:,0])
plt.show()


# Sizes and Model Parameters
img_size = x_train.shape[1] + x_train.shape[1]
img_flat = x_train.shape[1] * x_train.shape[2]
img_shape = x_train.shape[1:3]
num_channels = 1
num_classes = y_train.shape[1]
class CNN:
    def __init__(self, img_size)
