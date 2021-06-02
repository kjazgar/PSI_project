# Ignore  the warnings
import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns


# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# preprocess.
from keras.preprocessing.image import ImageDataGenerator

# dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
# from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2
import numpy as np
from tqdm import tqdm
import os
from random import shuffle
from zipfile import ZipFile
from PIL import Image


###################################
def read_data():
    X = list()
    y = list()
    IMG_SIZE = 128
    for i in os.listdir("./flowers/flowers/daisy"):
        try:
            path = "./flowers/flowers/daisy/"+i
            img = plt.imread(path)
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            X.append(img)
            y.append("daisy")
        except:
            None
    for i in os.listdir("./flowers/flowers/dandelion"):
        try:
            path = "./flowers/flowers/dandelion/"+i
            img = plt.imread(path)
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            X.append(img)
            y.append("dandelion")
        except:
            None
    for i in os.listdir("./flowers/flowers/rose"):
        try:
            path = "./flowers/flowers/rose/"+i
            img = plt.imread(path)
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            X.append(img)
            y.append("rose")
        except:
            None
    for i in os.listdir("./flowers/flowers/sunflower"):
        try:
            path = "./flowers/flowers/sunflower/"+i
            img = plt.imread(path)
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            X.append(img)
            y.append("sunflower")
        except:
            None
    for i in os.listdir("./flowers/flowers/tulip"):
        try:
            path = "./flowers/flowers/tulip/"+i
            img = plt.imread(path)
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            X.append(img)
            y.append("tulip")
        except:
            None

    X = np.array(X)

    fig, ax = plt.subplots(5, 2)
    fig.set_size_inches(15, 15)
    for i in range(5):
        for j in range(2):
            l = rn.randint(0, len(y))
            ax[i, j].imshow(X[l])
            ax[i, j].set_title('Flower: ' + y[l])

    plt.tight_layout()
    plt.show()

    return X, y

read_data()

