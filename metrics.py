from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

from simple_models.ada_boost import ada
from simple_models.dec_tree import dec_tree
from simple_models.extra_tree import extra_tree
from reprocess_data import reprocess_data2
from simple_models.logistic import logistic
from simple_models.svm_linear import svm_linear
from simple_models.svm_poly import svm_poly
from simple_models.svm_rbf import svm_rbf
from simple_models.random_forest import random_forest
from simple_models.knn import knn
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = reprocess_data2()
log = logistic(X_train, X_test, y_train, y_test)
svm_l = svm_linear(X_train, X_test, y_train, y_test)
svm_p = svm_poly(X_train, X_test, y_train, y_test)
svm_r = svm_rbf(X_train, X_test, y_train, y_test)
rf = random_forest(X_train, X_test, y_train, y_test)
knn = knn(X_train, X_test, y_train, y_test)
dec = dec_tree(X_train, X_test, y_train, y_test)
extr = extra_tree(X_train, X_test, y_train, y_test)
ad = ada(X_train, X_test, y_train, y_test)

models = []
models.append(('LR', log.best_estimator_))
models.append(('SVC linear', svm_l.best_estimator_))
models.append(('SVC poly', svm_p.best_estimator_))
models.append(('SVC rbf', svm_r.best_estimator_))
models.append(('RandomForest', rf.best_estimator_))
models.append(('KNeighbours', knn.best_estimator_))
models.append(('DecisionTree', dec.best_estimator_))
models.append(('ExtraTrees', extr.best_estimator_))
models.append(('AdaBoost', ad.best_estimator_))


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
    print(confusion_matrix(y_test, y_pred))
    precision_score.append(metrics.precision_score(y_test, y_pred, average='micro'))
    recall_score.append(metrics.recall_score(y_test, y_pred, average='micro'))
    f1_score.append(metrics.f1_score(y_test, y_pred, average='micro'))
    accuracy_score.append(metrics.accuracy_score(y_test, y_pred))




d = {'precision_score': precision_score,
     'recall_score': recall_score,
     'f1_score': f1_score,
     'accuracy_score' : accuracy_score}
df = pd.DataFrame(data=d)
df.insert(loc=0, column='Method', value=['LR', 'SVC linear', 'SVC poly', 'SVC rbf', 'Random Forest', 'KNeighbours', 'DecisionTree', 'ExtraTrees', 'AdaBoost'])
print(df)

num = 1
for mod in (log, svm_l):
    print(num)
    y_score = mod.fit(X_train, y_train).decision_function(X_test)
    n_classes = 5
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    num = num + 1