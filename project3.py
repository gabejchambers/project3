import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt


def totalError(matrix, weights):
    rows = matrix.shape[0]
    TE = 0
    for row in range(rows):
        d = matrix[row, -1]
        o = activation(weights[0] * matrix[row, 0] + weights[1] * matrix[row, 1] + weights[2] * matrix[row, 2])
        TE += (d - o) ** 2
    return TE


def activation(net):
    out = 0
    if aFunc == 'hard':
        if net >= 0:
            out = 1
        elif net < 0:
            out = 0
    elif aFunc == 'soft':
        out = 1 / (1 + math.exp(-1 * gain * net))
    return out


def plot(matrix, weights=None, title="Car Size Classification"):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Normalized Price")
    ax.set_ylabel("Normalized Weight")

    map_min = 0.0
    map_max = 1.1
    y_res = 0.001
    x_res = 0.001
    ys = np.arange(map_min, map_max, y_res)
    xs = np.arange(map_min, map_max, x_res)
    zs = []
    for cur_y in np.arange(map_min, map_max, y_res):
        for cur_x in np.arange(map_min, map_max, x_res):
            zs.append(round(activation(1 * weights[0] + cur_x * weights[1] + cur_y * weights[2])))
    xs, ys = np.meshgrid(xs, ys)
    zs = np.array(zs)
    zs = zs.reshape(xs.shape)
    cp = plt.contourf(xs, ys, zs, levels=[-1, -0.0001, 0, 1], colors=('b', 'r'), alpha=0.1)

    c1_data = [[], []]
    c0_data = [[], []]
    for i in range(len(matrix)):
        cur_i1 = matrix[i, 1]
        cur_i2 = matrix[i, 2]
        cur_y = matrix[i, -1]
        if cur_y == 1:
            c1_data[0].append(cur_i1)
            c1_data[1].append(cur_i2)
        else:
            c0_data[0].append(cur_i1)
            c0_data[1].append(cur_i2)

    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)

    c0s = plt.scatter(c0_data[0], c0_data[1], s=40.0, c='r', label='Large Car')
    c1s = plt.scatter(c1_data[0], c1_data[1], s=40.0, c='b', label='Small Car')

    plt.legend(fontsize=10, loc=1)
    plt.show()
    return


def normalize(matrix):
    maxes = np.amax(matrix, axis=0)
    mins = np.amin(matrix, axis=0)
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    normed = np.zeros(shape=(rows, cols))
    for i in range(0, rows):
        for j in range(0, cols):
            normed[i, j] = (matrix[i, j] - mins[j]) / (maxes[j] - mins[j])
    return normed


def train(matrix, epsilon):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    weights = np.random.uniform(-0.5, 0.5, (cols - 1))
    TE = totalError(matrix, weights)
    TEmin = (math.inf, -1)
    cycount = 0
    for cycle in range(cycles):
        cycount += 1
        TE = totalError(matrix, weights)
        if TE < TEmin[0]: TEmin = (TE, cycount)
        if TE < epsilon: break
        for row in range(rows):
            net = 0
            for col in range(cols - 1):
                net = net + matrix[row, col] * weights[col]
            out = activation(net)
            error = matrix[row, -1] - out
            learn = alpha * error
            for weight in range(len(weights)):
                weights[weight] += learn * matrix[row][weight]
    # print('final weights', weights)
    # print('number of cycles:', cycount)
    # print('final Total Error:', TE)
    # print('Minimum Total Error Produced:', TEmin[0], '    On cycle:', TEmin[1])
    # print('')
    # print('')
    return weights


def addBias(matrix):
    bs = np.ones((matrix.shape[0], 1))
    matrix = np.concatenate([bs, matrix], axis=1)
    return matrix


def processData(matrix, epsilon, group, split):
    matrix = normalize(matrix)
    matrix = addBias(matrix)
    np.random.shuffle(matrix)
    npGsplit = np.array_split(matrix, 4)
    if split[0] == 75:
        npGtrain = np.concatenate((npGsplit[0], npGsplit[1], npGsplit[2]), axis=0)
        npGtest = npGsplit[3]
        pd.DataFrame(npGtrain).to_csv(group + "_7525_train.csv")
        pd.DataFrame(npGtest).to_csv(group + "_7525_test.csv")
    elif split[0] == 25:
        npGtest = np.concatenate((npGsplit[0], npGsplit[1], npGsplit[2]), axis=0)
        npGtrain = npGsplit[3]
        pd.DataFrame(npGtrain).to_csv(group + "_2575_train.csv")
        pd.DataFrame(npGtest).to_csv(group + "_2575_test.csv")
    # for whole thing train and plot:
    # weightsG = train(matrix, epsilon)
    # plot(matrix, weightsG)
    weightsG = train(npGtrain, epsilon)
    trainTitle = 'Car Size for Group ' + group + ' ' + str(split[0]) + '% Training with ' + aFunc + ' activation function'
    testTitle = 'Car Size for Group ' + group + ' ' + str(split[1]) + '% Testing with ' + aFunc + ' activation function'
    plot(npGtrain, weightsG, title=trainTitle)
    plot(npGtest, weightsG, title=testTitle)
    TEtrain = totalError(npGtrain, weightsG)
    TEtest = totalError(npGtest, weightsG)
    print(str(split[0]) + '% Training Total Error Group', group, ':', TEtrain)
    print(str(split[1]) + '% Testing Total Error Group', group, ':', TEtest)
    print(str(split[0]) + '% Training Percent Error Group', group, ':', round(100*TEtrain/npGtrain.shape[0],2),'%')
    print(str(split[0]) + '% Testing Percent Error Group', group, ':', round(100*TEtest/npGtest.shape[0],2),'%')
    print('')
    print('')
    confusionMatrix(npGtrain, weightsG, group, 'Training Data')
    confusionMatrix(npGtest, weightsG, group, 'Testing Data')
    return


def confusionMatrix(matrix, weights, group, dataset):
    rows = matrix.shape[0]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for row in range(rows):
        d = matrix[row, -1]
        o = round(activation(weights[0] * matrix[row, 0] + weights[1] * matrix[row, 1] + weights[2] * matrix[row, 2]))
        if d == 0:
            if o == 0:
                TP += 1
            elif o == 1:
                FN += 1
        elif d == 1:
            if o == 0:
                FP += 1
            elif o == 1:
                TN += 1
    print('')
    print('                                      Confusion matrix Group', group, dataset, ':')
    print('                          Actual Positive (0)                 Actual Negative (1)')
    print('Calculated Positive (0)', '       ', TP, '                                 ', FP)
    print('Calculated Negative (1)', '       ', FN, '                                 ', TN)
    print('')
    print('')
    return


split = (75, 25)
cycles = 100
epA = 0.00001
epB = 200
epC = 700
epT = 1
aFunc = 'soft'  # 'hard' # 'soft
alpha = 1
gain = .9
dfa = pd.read_csv('Pr2_Data_ALT/GroupA.csv')
dfb = pd.read_csv('Pr2_Data_ALT/GroupB.csv')
dfc = pd.read_csv('Pr2_Data_ALT/GroupC.csv')
npA = dfa.to_numpy()
npB = dfb.to_numpy()
npC = dfc.to_numpy()

# helpful numpy syntax:
# print(npA)
# print(npA[2]) # row
# print(npA[:,2]) # column 2
# print(npA[2,1]) #value



processData(npA, epA*3/4, 'A', split)
processData(npB, epB*3/4, 'B', split)
processData(npC, epC*3/4, 'C', split)