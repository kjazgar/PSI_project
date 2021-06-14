import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import linear_model
from read_data import read_data
import random as rn


def plot_random(X, y):
    fig, ax = plt.subplots(5, 2)
    fig.set_size_inches(15, 15)
    for i in range(5):
        for j in range(2):
            l = rn.randint(0, len(y))
            ax[i, j].imshow(X[l])
            ax[i, j].set_title('Flower: ' + y[l])

    plt.tight_layout()
    plt.show()


X, y = read_data()
plot_random(X, y)

def plot_kind(X, y):
    for amount in range(3):
        plt.figure(figsize=(20, 20))
        for i in range(5):
            img = X[950 * i + amount]
            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(y[950 * i])


plot_kind(X, y)