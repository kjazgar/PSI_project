import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from model1 import model1
from reprocess_data import reprocess_data

from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd

X_train, X_test, X_valid, y_train, y_test, y_valid = reprocess_data()

model1 = model1(X_train, X_test, X_valid, y_train, y_test, y_valid)

models = []
models.append(('model1', model1))

# precision_score = []
# recall_score = []
# f1_score = []
accuracy_score = []
# roc_auc_score = []
for name, model in models:
    y_pred = model.predict_classes(X_test)
    print(name)
    # print("precision_score: {}".format(metrics.precision_score(y_test, y_pred1)))
    # print("recall_score: {}".format(metrics.recall_score(y_test, y_pred1)))
    # print("f1_score: {}".format(metrics.f1_score(y_test, y_pred1)))
    print("accuracy_score: {}".format(metrics.accuracy_score(y_test, model.predict_classes(X_test))))

    # precision_score.append(metrics.precision_score(y_test, y_pred1))
    # recall_score.append(metrics.recall_score(y_test, y_pred1))
    # f1_score.append(metrics.f1_score(y_test, y_pred1))
    accuracy_score.append(metrics.accuracy_score(y_test, model.predict_classes(X_test)))

    from sklearn.metrics import confusion_matrix

    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)

    df_cm = pd.DataFrame(conf_mat)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, cmap="OrRd")
    plt.show()


import pandas as pd
d = {'accuracy_score' : accuracy_score  }
df = pd.DataFrame(data=d)
df.insert(loc=0, column='Method', value=['model1'])
print(df)
