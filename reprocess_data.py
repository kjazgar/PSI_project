import random

from read_data import read_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random as rn
import numpy as np

def reprocess_data1():
    X, labels = read_data()

    label_dict = {'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}

    y = [label_dict[labels[i]] for i in range(len(labels))]

    X_train_full, X_test, y_train_full, y_test = train_test_split(X,y, test_size = 0.15, random_state = 42)

    X_valid, X_train = X_train_full[:1837] / 255., X_train_full[1837:] / 255.
    y_valid, y_train = y_train_full[:1837], y_train_full[1837:]
    X_test = X_test / 255.

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def reprocess_data2(n_samples):
    X, labels = read_data()
    indexes = random.sample(range(1, len(X)), n_samples)
    X = [x for idx, x in enumerate(X) if idx in indexes]
    X = np.array(X)
    labels = [label for idx, label in enumerate(labels) if idx in indexes]

    label_dict = {'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}

    y = [label_dict[labels[i]] for i in range(len(labels))]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    X_train = X_train / 255.
    X_test = X_test / 255.

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = X_train.reshape(X_train.shape[0], 128 * 128 * 3)
    X_test = X_test.reshape(X_test.shape[0], 128 * 128 * 3)

    return X_train, X_test, y_train, y_test