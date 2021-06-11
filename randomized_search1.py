from keras.layers import MaxPool2D

from reprocess_data import reprocess_data

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import History
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import  metrics
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Flatten


def rnd(X_train, X_test, X_valid, y_train, y_test, y_valid):

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_valid = np.array(y_valid)


    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    def build_model(n_hidden=1, filters=128, dropout =0.3, learning_rate=3e-3):
        model = keras.models.Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu", input_shape=(128, 128, 3)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        for layer in range(n_hidden):
            model.add(Conv2D(filters, kernel_size=(3, 3), padding="Same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(5, activation="softmax"))
        optimizer = keras.optimizers.SGD(lr=learning_rate)
        model.compile(model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']))
        return model

    keras_class = tf.keras.wrappers.scikit_learn.KerasClassifier(build_model)

    from scipy.stats import reciprocal
    from sklearn.model_selection import RandomizedSearchCV

    param_distribs = {
        "n_hidden": [2, 3, 4],
        "dropout": [0.2, 0.3, 0.5]

    }

    rnd_search_cv = RandomizedSearchCV(keras_class, param_distribs, n_iter=10, cv=3, verbose=2)
    rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)])

    print(rnd_search_cv.best_params_)


    # pd.DataFrame(keras_class.history).plot(figsize=(8, 5))
    # plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    # plt.show()

    y_pred = rnd_search_cv.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(accuracy)

    return rnd_search_cv


X_train, X_test, X_valid, y_train, y_test, y_valid = reprocess_data()
rnd(X_train, X_test, X_valid, y_train, y_test, y_valid)
#0.7226502311248074