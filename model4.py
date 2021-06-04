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


def model4(X_train, X_test, X_valid, y_train, y_test, y_valid):

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_valid = np.array(y_valid)


    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (128,128,3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(16, (2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(5,activation="softmax"))
    model.summary()

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="Adam",
                  metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid))

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    y_pred = model.predict_classes(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(accuracy)

    return model


X_train, X_test, X_valid, y_train, y_test, y_valid = reprocess_data()
model4(X_train, X_test, X_valid, y_train, y_test, y_valid)
#0.5408320493066255