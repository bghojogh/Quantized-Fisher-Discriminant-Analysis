from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib import offsetbox
import pandas as pd
import scipy.io
import csv
import scipy.misc
import os
import math
from sklearn.model_selection import train_test_split   #--> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def main():
    dataset = "Fashion_MNIST"  #--> ATT_glasses, Fashion_MNIST
    plot_error_MNIST_in_RDA(dataset=dataset, show_legends=False)

def plot_error_MNIST_in_RDA(dataset="RDA", show_legends=True):
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.arange(0, 63+1, 1)
    path = './values/' + dataset + '/'
    levels = load_variable(name_of_variable="best_number_of_levels", path=path)
    plt.bar(x=x, height=levels)
    plt.grid()
    plt.xlabel("Frequency", fontsize=13)
    plt.ylabel("Optimum # quantization levels", fontsize=13)
    # if show_legends is not None:
    #     ax.legend()
    plt.show()

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

if __name__ == '__main__':
    main()