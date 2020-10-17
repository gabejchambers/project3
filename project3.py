import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt


def normalize(arr):
    max_a = np.amax(arr)
    min_a = np.amin(arr)
    len_a = len(arr)
    normed = np.zeros(len_a)
    for i in range(0, len_a):
        normed[i] = (arr[i] - min_a) / (max_a - min_a)
    return normed[np.newaxis].T


def addBias(matrix):
    bs = np.ones((matrix.shape[0], 1))
    matrix = np.concatenate([matrix[:, 0][np.newaxis].T, bs, matrix[:, 1][np.newaxis].T], axis=1)
    return matrix


def preprocess(matrix):
    matrix = addBias(matrix)
    # matrix = np.concatenate((normalize(matrix[:, 0]), matrix[:, 1][np.newaxis].T, normalize(matrix[:, 2])), axis=1) #if need normalize time
    matrix = np.concatenate((matrix[:, :2], normalize(matrix[:, 2])), axis=1)
    return matrix


data_1 = pd.read_csv('Project3_data/train_data_1.txt', sep=",", header=None).to_numpy()
data_2 = pd.read_csv('Project3_data/train_data_2.txt', sep=",", header=None).to_numpy()
data_3 = pd.read_csv('Project3_data/train_data_3.txt', sep=",", header=None).to_numpy()
data_test = pd.read_csv('Project3_data/test_data_4.txt', sep=",", header=None).to_numpy()

# helpful numpy syntax:
# print(npA)
# print(npA[2]) # row
# print(npA[:,2]) # column 2
# print(npA[2,1]) #value

data_1 = preprocess(data_1)

print(data_1)


