from sklearn import  metrics
from reprocess_data import reprocess_data2
from logistic import logistic
from svm_linear import svm_linear
from svm_poly import svm_poly
from svm_rbf import svm_rbf
from random_forest import random_forest
from knn import knn
X_train, X_test, y_train, y_test = reprocess_data2(100)
log = logistic(X_train, X_test, y_train, y_test)
svm_l = svm_linear(X_train, X_test, y_train, y_test)
svm_p = svm_poly(X_train, X_test, y_train, y_test)
svm_r = svm_rbf(X_train, X_test, y_train, y_test)
rf = random_forest(X_train, X_test, y_train, y_test)
knn = knn(X_train, X_test, y_train, y_test)

models = []
models.append(('LR', log.best_estimator_))
models.append(('SVC linear', svm_l.best_estimator_))
models.append(('SVC poly', svm_p.best_estimator_))
models.append(('SVC rbf', svm_r.best_estimator_))
models.append(('RandomForest', rf.best_estimator_))
# models.append(('AdaBoost', grid_6.best_estimator_))
# models.append(('GradientBoosting', grid_7.best_estimator_))
models.append(('KNeighbours', knn.best_estimator_))
# models.append(('DecisionTree', grid_9.best_estimator_))
# models.append(('ExtraTrees', grid_10.best_estimator_))

precision_score = []
recall_score = []
f1_score = []
accuracy_score = []
for name, model in models:
  y_pred = model.predict(X_test)
  print(name)
  print("precision_score: {}".format(metrics.precision_score(y_test, y_pred, average='micro')))
  print("recall_score: {}".format(metrics.recall_score(y_test, y_pred, average='micro')))
  print("f1_score: {}".format(metrics.f1_score(y_test, y_pred, average='micro')))
  print("accuracy_score: {}".format(metrics.accuracy_score(y_test, y_pred)))
  precision_score.append(metrics.precision_score(y_test, y_pred, average='micro'))
  recall_score.append(metrics.recall_score(y_test, y_pred, average='micro'))
  f1_score.append(metrics.f1_score(y_test, y_pred, average='micro'))
  accuracy_score.append(metrics.accuracy_score(y_test, y_pred))



import pandas as pd
d = {'precision_score': precision_score,
     'recall_score': recall_score,
     'f1_score': f1_score,
     'accuracy_score' : accuracy_score}
df = pd.DataFrame(data=d)
df.insert(loc=0, column='Method', value=['LR', 'SVC linear', 'SVC poly', 'SVC rbf', 'Random Forest', 'KNeighbours'])
print(df)
