from random import uniform

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reprocess_data import reprocess_data2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import linear_model


def logistic(X_train, X_test, y_train, y_test):

    kfold = StratifiedKFold(n_splits=5, shuffle=False)

    pipe = Pipeline([
        ('preprocessing', StandardScaler()),
        ('classifier', linear_model.LogisticRegression(C=1, solver='newton-cg'))])

    param_grid = {
        'classifier__C': [0.001, 0.01]
    }

    grid = GridSearchCV(pipe, param_grid, cv=kfold, error_score='raise')

    grid.fit(X_train, y_train)
    print(grid.best_params_)

    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(accuracy)

    return grid

# X_train, X_test, y_train, y_test = reprocess_data2()

# logistic(X_train, X_test, X_valid, y_train, y_test, y_valid)
#0.43605546995377503
