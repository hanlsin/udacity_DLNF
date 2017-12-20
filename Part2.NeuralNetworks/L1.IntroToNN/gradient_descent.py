import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_points(X, y):
    admitted = X[np.argwhere(y == 1)]
    rejected = X[np.argwhere(y == 0)]

    plt.scatter([s[0][0] for s in rejected],
                [s[0][1] for s in rejected],
                s=25, color='blue', edgecolors='k')
    plt.scatter([s[0][0] for s in admitted],
                [s[0][1] for s in admitted],
                s=25, color='red', edgecolors='k')


def display(m, b, color='g--'):
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m * x + b, color)


filepath = './udacity_DLNF/Part1.NeuralNetworks/Lesson1.IntroToNN/gradient-descent/data.csv'
data = pd.read_csv(filepath, header=None)
X = np.array(data[[0, 1]])
print("Number of Records = " + str(X.shape[0]))
print("Number of Features = " + str(X.shape[1]))
y = np.array(data[2])

plot_points(X, y)
# plt.show()

##
# Sigmoid


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


##
# Output(prediction) formula


def output_formula(features, weights, bias):
    return sigmoid(np.matmul(features, weights) + bias)


##
# Error (log-loss) formula
def error_formula(y, output):
    return -y * np.log(output) - (1 - y) * np.log(1 - output)

##
# Gradient descent step


def update_weights(x, y, weights, bias, learnrate):
    # delta error = - (y - prediction)
    delta_error = - y + output_formula(x, weights, bias)
    weights = weights - learnrate * delta_error * x
    bias = bias - learnrate * delta_error
    return weights, bias

##
# Traning Function


np.random.seed(44)

epochs = 100
learnrate = 0.01


def train(features, targets, epochs, learnrate, graph_lines=False):
    errors = []
    n_records, n_features = features.shape
    last_loss = None

    # Step 1. Start with random weights:
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0

    plotcolors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    plotcoloridx = 0

    for e in range(epochs):
        del_w = np.zeros(weights.shape)

        # Step 2. For every point X
        for x, y in zip(features, targets):
            # prediction
            output = output_formula(x, weights, bias)
            print(output)
            # error function = cross entropy
            error = error_formula(y, output)
            print(error)
            # gradient descent
            weights, bias = update_weights(x, y, weights, bias, learnrate)
            print("------")

        # Printing out the log-loss error on the training set
        out = output_formula(features, weights, bias)
        print(output)
        err = error_formula(targets, out)
        print(err)
        loss = np.mean(err)
        print(loss)
        print("++++++++")
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e, "==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 100) == 0:
            display(-weights[0] / weights[1], -bias /
                    weights[1], color=plotcolors[plotcoloridx])
            if plotcoloridx == len(plotcolors) - 1:
                plotcoloridx = 0
            else:
                plotcoloridx += 1

    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0] / weights[1], -bias / weights[1], 'black')

    # Plotting the data
    plot_points(features, targets)
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()


train(X, y, epochs, learnrate, True)
