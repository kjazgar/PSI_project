from random import uniform

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from reprocess_data import reprocess_data2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC


def svm_linear(X_train, X_test, y_train, y_test):
    kfold = StratifiedKFold(n_splits=5, shuffle=False)

    pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC(kernel='linear'))])

    param_grid = {
        'preprocessing': [MinMaxScaler(), StandardScaler(), None],
        'classifier__C': [0.001],
        'classifier__gamma': [0.01]
    }

    grid = GridSearchCV(pipe, param_grid, cv=kfold, error_score='raise')

    grid.fit(X_train, y_train)
    print(grid.best_params_)

    grid.fit(X_train, y_train)


#     y_pred = grid.predict(X_test)
#     accuracy = metrics.accuracy_score(y_test, y_pred)
#     print(accuracy)


    return grid



# X_train, X_test, y_train, y_test = reprocess_data2()
# svm_linear(X_train, X_test, y_train, y_test)
#0.4560862865947612
