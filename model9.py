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


def model9(X_train, X_test, X_valid, y_train, y_test, y_valid):

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_valid = np.array(y_valid)


    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu", input_shape=(128, 128, 3)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    #3rd Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # 1st Fully Connected Layer
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    # Add output layer
    model.add(Dense(5, activation="softmax"))

    model.summary()  # print summary my model

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # compile model

    # model.compile(loss="sparse_categorical_crossentropy",
    #               optimizer="Adam",
    #               metrics=["accuracy"])

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
model9(X_train, X_test, X_valid, y_train, y_test, y_valid)
#0.6733436055469953