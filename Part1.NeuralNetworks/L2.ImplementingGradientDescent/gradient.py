import numpy as np

# for the case of only one output unit


def sigmoid(x):
    '''
    # the sigmoid function as the activation function
    '''
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    '''
    # Derivative of the sigmoid function
    '''
    return sigmoid(x) * (1 - sigmoid(x))


learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

# initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

# Calculate one gradient descent step for each weight
# Note: Some steps have been consilated, so there are
#       fewer variable names than in the above sample code

# TODO: Calculate the node's linear combination of inputs and weights
h = np.matmul(x, w)
print(h)

# TODO: Calculate output of neural network
nn_output = sigmoid(h)
print(nn_output)

# TODO: Calculate error of neural network = output error
error = y - nn_output
print(error)

# TODO: Calculate the error term
#       Remember, this requires the output gradient, which we haven't
#       specifically added a variable for.
error_term = (y - nn_output) * sigmoid_prime(h)
print(error_term)

# TODO: Calculate change in weights
del_w = learnrate * error_term * x
print(del_w)

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
