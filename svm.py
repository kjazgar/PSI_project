from random import uniform

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from reprocess_data import reprocess_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC


def svm(X_train, X_test, X_valid, y_train, y_test, y_valid):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = X_train.reshape(1837,128*128*3)
    X_test = X_test.reshape(649,128*128*3)

    kfold = StratifiedKFold(n_splits=5, shuffle=False)

    param_grid = {
        'C': [0.001, 0.01, 0.1]
    }

    grid = GridSearchCV(SVC(kernel='linear'), param_grid, cv=kfold, error_score='raise')

    grid.fit(X_train, y_train)
    print(grid.best_params_)

    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(accuracy)

    return grid.best_estimator_



X_train, X_test, X_valid, y_train, y_test, y_valid = reprocess_data()
svm(X_train, X_test, X_valid, y_train, y_test, y_valid)
#0.4560862865947612
