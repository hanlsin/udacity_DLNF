import numpy as np
import pandas as pd

# Setting the random seed
np.random.seed(48)


def step(t):
    if t >= 0:
        return 1
    else:
        return 0


def prediction(X, W, b):
    y = np.matmul(X, W) + b
    return step(y[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.


def perceptronStep(X, y, W, b, learn_rate=0.01):
    for i, x in enumerate(X):
        s = prediction(x, W, b)

        # come closer
        if s != y[i]:
            if y[i] == 1:
                # prediction = 0
                # positive point in negative area
                W[0] += learn_rate * x[0]
                W[1] += learn_rate * x[1]
                b += learn_rate
            elif y[i] == 0:
                # prediction = 1
                # negative point in positive area
                W[0] -= learn_rate * x[0]
                W[1] -= learn_rate * x[1]
                b -= learn_rate
    return W, b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.


def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0] / W[1], -b / W[1]))
    return boundary_lines


filepath = './Part1.NeuralNetworks/Lesson1.IntroToNN/data.csv'
# data = np.genfromtxt(filepath, delimiter=',')
datafile = pd.read_csv(filepath, sep=',', header=None)
data = np.array(datafile.values)
X = data[:, [0, 1]]
Y = data[:, 2]

print(trainPerceptronAlgorithm(X, Y))
