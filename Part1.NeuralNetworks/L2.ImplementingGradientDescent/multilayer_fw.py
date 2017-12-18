import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Network Size
N_input = 4
N_hidden = 3
N_output = 2

# Make some fake data
np.random.seed(42)
X = np.random.randn(4)
print('X =', X)

w_i_h = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
print('Weights for Input to Hidden:', w_i_h.shape, '\n', w_i_h)
w_h_o = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))
print('Weights for Hidden to Output:', w_h_o.shape, '\n', w_h_o)

# TODO: Make a forward pass through the network
hidden_layer_in = np.dot(X, w_i_h)
print('Hidden Layer IN:', hidden_layer_in)
hidden_layer_out = sigmoid(hidden_layer_in)
print('Hidden Layer OUT:', hidden_layer_out)

output_layer_in = np.dot(hidden_layer_out, w_h_o)
print('Output Layer IN:', output_layer_in)
output_layer_out = sigmoid(output_layer_in)
print('Output Layer OUT:', output_layer_out)
