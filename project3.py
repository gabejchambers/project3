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


# activation function - summation
# assuming calculating net IS the activation function
# net = E (wi*xi) <- including bias*w^bias
def activation(net):
    return net


def plotLinear(data, weights, dataName):
    # LOBF:
    x = np.arange(-0.1, 1.1, step=0.01)
    y = weights[0] * x + weights[1]
    title = "Linear Regression " + str(dataName) + " alpha =" + str(alpha) + " #iterations = " + str(cycles)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.plot(x, y)
    # Data:
    x = data[:,0]
    y = data[:,1]
    plt.scatter(x,y)
    plt.show()


def plotQuad(data, weights, dataName):
    # LOBF:
    x = np.arange(-0.1, 1.1, step=0.1)
    y = weights[0] * x**2 + weights[1] * x + weights[-1]
    title = "Quadratic Regression " + str(dataName) + " alpha =" + str(alpha) + " #iterations = " + str(cycles)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.plot(x, y)
    # Data:
    x = data[:,0]
    y = data[:,1]
    plt.scatter(x,y)
    plt.show()

def plotCube(data, weights, dataName):
    # LOBF:
    x = np.arange(-0.1, 1.1, step=0.1)
    y = weights[0] * x**3 + weights[1] * x**2 + weights[2] * x + weights[-1]
    title = "Cubic Regression " + str(dataName) + " alpha =" + str(alpha) + " #iterations = " + str(cycles)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.plot(x, y)
    # Data:
    x = data[:,0]
    y = data[:,1]
    plt.scatter(x,y)
    plt.show()


# use standard alpha
def trainLinear(matrix):
    row_num = matrix.shape[0]
    weights = np.random.uniform(-0.5, 0.5, 2) # 2 weights, 1 for bias 1 for x
    for cycle in range(cycles):
        for row in range(row_num):
            net = weights[0]*matrix[row, 0] + weights[1]
            out = activation(net)
            error = matrix[row, 1] - out # desired output - our output
            learn = alpha * error
            weights[0] += learn * matrix[row, 0]
            weights[-1] += learn
    return weights

# use standard alpha
def trainQuad(matrix):
    row_num = matrix.shape[0]
    weights = np.random.uniform(-0.5, 0.5, 3)  # 3 weights, 1 for bias, 1 for x, 1 for x^2
    for cycle in range(cycles):
        for row in range(row_num):
            net = weights[0] * matrix[row, 0]**2 + weights[1] * matrix[row, 0] + weights[-1]
            out = activation(net)
            error = matrix[row, 1] - out  # desired output - our output
            learn = alpha * error
            weights[0] += learn * matrix[row, 0]**2
            weights[1] += learn * matrix[row, 0]
            weights[-1] += learn
    return weights


# cubic does much much better with a higher alpha, around .8 or .9, along with a high number of cycles
# this is because the true coefficients for the cubic function are so much larger than the starting weights
# that with a low alpha and cycle count, there isnt time to reach the true coefficients
def trainCube(matrix):
    row_num = matrix.shape[0]
    weights = np.random.uniform(-0.5, 0.5, 4)  # 4 weights, 1 for bias, 1 for x, 1 for x^2, 1 for x^3
    # weights = np.array([9.036, -12.0568, 3.9093, 0.1506])
    for cycle in range(cycles):
        for row in range(row_num):
            net = weights[0] * matrix[row, 0]**3 + weights[1] * matrix[row, 0]**2 + weights[2] * matrix[row, 0] + weights[-1]
            out = activation(net)
            error = matrix[row, 1] - out  # desired output - our output
            learn = alpha * error
            weights[0] += learn * matrix[row, 0]**3
            weights[1] += learn * matrix[row, 0]**2
            weights[2] += learn * matrix[row, 0]
            weights[-1] += learn
    return weights


def preprocess(matrix):
    matrix = np.concatenate((normalize(matrix[:, 0]), normalize(matrix[:, 1])), axis=1)
    np.random.shuffle(matrix)
    return matrix


def startDataSet(dataset, dataname):
    dataset = preprocess(dataset)
    # print all rows of data to cross check with regression calculators:
    # for data in dataset[:,0]:
    #     print(data)
    # print()
    # for data in dataset[:,1]:
    #     print(data)
    w_L = trainLinear(dataset)
    print(w_L)
    plotLinear(dataset, w_L, dataname)
    w_Q = trainQuad(dataset)
    print(w_Q)
    plotQuad(dataset, w_Q, dataname)
    w_C = trainCube(dataset)
    print(w_C)
    plotCube(dataset, w_C, dataname)
    return


alpha = .2
cycles = 5000
data_1 = pd.read_csv('Project3_data/train_data_1.txt', sep=",", header=None).to_numpy()
data_2 = pd.read_csv('Project3_data/train_data_2.txt', sep=",", header=None).to_numpy()
data_3 = pd.read_csv('Project3_data/train_data_3.txt', sep=",", header=None).to_numpy()
data_test = pd.read_csv('Project3_data/test_data_4.txt', sep=",", header=None).to_numpy()

# helpful numpy syntax:
# print(npA)
# print(npA[2]) # row
# print(npA[:,2]) # column 2
# print(npA[2,1]) #value

startDataSet(data_1, "data_1")
startDataSet(data_2, "data_2")
startDataSet(data_3, "data_3")


