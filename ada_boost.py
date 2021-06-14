from random import uniform

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from reprocess_data import reprocess_data2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC



def ada(X_train, X_test, y_train, y_test):

    kfold = StratifiedKFold(n_splits=5, shuffle=False)

    pipe = Pipeline([('preprocessing', StandardScaler()),
                   ('classifier', AdaBoostClassifier(
                       DecisionTreeClassifier(random_state=42),
                       n_estimators=500,
                       learning_rate=0.5))])

    param_grid = {'classifier__n_estimators': [10,50,100],
              'classifier__learning_rate': [0.1, 0.2, 0.3,0.5,0.6,0.7,0.8,0.9]}


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
