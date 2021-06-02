from read_data import read_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random as rn


X, labels = read_data()

def reprocess_data():

    label_dict = {'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}

    y = [label_dict[labels[i]] for i in range(len(labels))]

    X_train_full, X_test, y_train_full, y_test = train_test_split(X,y, test_size = 0.15, random_state = 42)

    X_valid, X_train = X_train_full[:1837] / 255., X_train_full[1837:] / 255.
    y_valid, y_train = y_train_full[:1837], y_train_full[1837:]
    X_test = X_test / 255.

    return X_train, X_test, X_valid, y_train, y_test, y_valid
