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


def totalError(matrix):
    # TE = SUM((d-o)^2)
    linTE = 0
    quadTE = 0
    cubeTE = 0
    for row in range(matrix.shape[0]):
        linerr = matrix[row,1] - matrix[row,2]
        linTE += linerr ** 2
        quaderr = matrix[row,1] - matrix[row,3]
        quadTE += quaderr ** 2
        cubeerr = matrix[row,1] - matrix[row,4]
        cubeTE += cubeerr ** 2
    return linTE, quadTE, cubeTE


def trainTotalError(matrix, wts, typ):
    # linwts = wts[0]
    # quadwts = wts[1]
    # cubewts = wts[2]
    TE = 0
    obs = 0
    for row in range(matrix.shape[0]):
        if typ == "lin":
            obs = wts[0] * matrix[row, 0] + wts[1]
        elif typ == "quad":
            obs = wts[0] * matrix[row, 0] ** 2 + wts[1] * matrix[row, 0] + wts[2]
        elif typ == "cube":
            y = wts[0] * matrix[row, 0] ** 3 + wts[1] * matrix[row, 0] ** 2 + wts[2] * matrix[row, 0] + wts[3]
        else:
            obs = "bop" + 123
        err = matrix[row, 1] - obs
        TE += err ** 2
    return TE



def plotLinear(data, wts, dataName):
    # LOBF:
    x = np.arange(-0.1, 1.1, step=0.01)
    y = wts[0] * x + wts[1]
    title = "Linear Regression " + str(dataName) + " alpha=" + str(alpha) + "\n #iterations=" + str(cycles) + " equation=" + str(round(wts[0], 4)) + "x+" + str(round(wts[1], 4)) + "\n Total Error=" + str(round(trainTotalError(data, wts, "lin"), 4))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.plot(x, y)
    # Data:
    x = data[:,0]
    y = data[:,1]
    plt.scatter(x,y)
    plt.show()


def plotQuad(data, wts, dataName):
    # LOBF:
    x = np.arange(-0.1, 1.1, step=0.1)
    y = wts[0] * x ** 2 + wts[1] * x + wts[-1]
    title = "Quadratic Regression " + str(dataName) + " alpha =" + str(alpha) + " #iterations = " + str(cycles) + "\n equation=" + str(round(wts[0], 4)) + "x^2+" + str(round(wts[1], 4)) + "x+" + str(round(wts[2], 4)) + "\n Total Error=" + str(round(trainTotalError(data, wts, "quad"), 4))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.plot(x, y)
    # Data:
    x = data[:,0]
    y = data[:,1]
    plt.scatter(x,y)
    plt.show()


def plotCube(data, wts, dataName):
    # LOBF:
    x = np.arange(-0.1, 1.1, step=0.1)
    y = wts[0] * x ** 3 + wts[1] * x ** 2 + wts[2] * x + wts[-1]
    title = "Cubic Regression " + str(dataName) + " alpha =" + str(alpha) + " #iterations = " + str(cycles) + "\n equation=" + str(round(wts[0], 4)) + "x^3+" + str(round(wts[1], 4)) + "x^2+" + str(round(wts[2], 4)) + "x+" + str(round(wts[3], 4)) + "\n Total Error=" + str(round(trainTotalError(data, wts, "quad"), 4))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.plot(x, y)
    # Data:
    x = data[:,0]
    y = data[:,1]
    plt.scatter(x,y)
    plt.show()


def plotTest(data, wts, tes):
    linwts = wts[0]
    quadwts = wts[1]
    cubewts = wts[2]
    linte = tes[0]
    quadte = tes[1]
    cubete = tes[2]
    xline = np.arange(-0.1, 1.1, step=0.1)

    title = "Linear Prediction \n Total Error=" + str(round(linte, 4))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    y = linwts[0] * xline + linwts[1]
    plt.plot(xline, y, color="red", label="Predicted")
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, color='blue', label="Actual")
    y = data[:, 2]
    plt.scatter(x, y, color='red')
    plt.legend(loc="upper left")
    plt.show()

    title = "Quadratic Prediction \n Total Error=" + str(round(quadte, 4))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    y = quadwts[0] * xline**2 + quadwts[1] * xline + quadwts[2]
    plt.plot(xline, y, color="orange", label="Predicted")
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, color='blue', label="Actual")
    y = data[:, 3]
    plt.scatter(x, y, color='orange')
    plt.legend(loc="upper left")
    plt.show()

    title = "Cubic Prediction \n Total Error=" + str(round(cubete, 4))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    y = cubewts[0] * xline**3 + cubewts[1] * xline**2 + cubewts[2] * xline + cubewts[3]
    plt.plot(xline, y, color="green", label="Predicted")
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, color='blue', label="Actual")
    y = data[:, 4]
    plt.scatter(x, y, color='green')
    plt.legend(loc="upper left")
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


def predict(wts, matrix):
    linwts = wts[0]
    quadwts = wts[1]
    cubewts = wts[2]
    row_num = matrix.shape[0]
    for row in range(row_num):
        matrix[row, 2] = linwts[0] * matrix[row, 0] + linwts[1]
        matrix[row, 3] = quadwts[0] * matrix[row, 0]**2 + quadwts[1] * matrix[row, 0] + quadwts[2]
        matrix[row, 4] = cubewts[0] * matrix[row, 0]**3 + cubewts[1] * matrix[row, 0]**2 + cubewts[2] * matrix[row, 0] + cubewts[3]
    return matrix


def addCols(matrix):
    calcs = np.zeros((matrix.shape[0], 1))
    matrix = np.concatenate([matrix, calcs, calcs, calcs], axis=1)
    return matrix


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
    return w_L, w_Q, w_C


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

# startDataSet(data_1, "data_1")
# startDataSet(data_2, "data_2")
# startDataSet(data_3, "data_3")
weights = startDataSet(np.concatenate((data_1, data_2, data_3), axis=0), "Training Data")

data_test = preprocess(data_test)
print(data_test)
data_test = addCols(data_test)
print(data_test)
data_test = predict(weights, data_test)
print(data_test)
totalerrors = totalError(data_test)
plotTest(data_test, weights, totalerrors)


