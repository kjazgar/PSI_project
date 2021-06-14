from random import uniform

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from reprocess_data import reprocess_data2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



def random_forest(X_train, X_test, X_valid, y_train, y_test, y_valid):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # X_train = X_train.reshape(1837,128*128*3)
    # X_test = X_test.reshape(649,128*128*3)

    # rf1 = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)

    kfold = StratifiedKFold(n_splits=5, shuffle=False)

    param_grid = {
        'n_estimators': [5, 50, 500],
        'max_leaf_nodes': [2, 16]

    }

    rf1 = GridSearchCV(RandomForestClassifier(), param_grid, cv=kfold, error_score='raise')

    rf1.fit(X_train, y_train)


#     y_pred = rf1.predict(X_test)
#     accuracy = metrics.accuracy_score(y_test, y_pred)
#     print(accuracy)


    return rf1



# X_train, X_test,  y_train, y_test = reprocess_data2()
# rf(X_train, X_test,  y_train, y_test)
#0.4884437596302003

