from random import uniform

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from reprocess_data import reprocess_data2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC



def extra_tree(X_train, X_test, y_train, y_test):

    kfold = StratifiedKFold(n_splits=5, shuffle=False)

    pipe = Pipeline([('preprocessing', StandardScaler()),
                       ('classifier', ExtraTreesClassifier(n_jobs=-1, random_state=42))])

    param_grid = {'preprocessing': [MinMaxScaler(), StandardScaler(), None],
    'classifier__n_estimators': [50, 100, 500, 1000],
    'classifier__max_leaf_nodes': [4, 8, 16, 64]}


    grid = GridSearchCV(pipe, param_grid, cv=kfold, error_score='raise')

    grid.fit(X_train, y_train)
    print(grid.best_params_)

#     y_pred = grid.predict(X_test)
#     accuracy = metrics.accuracy_score(y_test, y_pred)
#     print(accuracy)

    return grid


# X_train, X_test, y_train, y_test = reprocess_data2()
# svm_linear(X_train, X_test, y_train, y_test)
#0.4560862865947612
