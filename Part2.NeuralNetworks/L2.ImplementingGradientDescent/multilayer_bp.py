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

# TODO: Make a forward pass through the network
print('[HI] Hidden Layer IN:', X)
hidden_layer_in = np.dot(X, w_i_h)
print('Hidden Layer before Activation Func.:', hidden_layer_in)
hidden_layer_out = sigmoid(hidden_layer_in)
print('[HO] Hidden Layer OUT:', hidden_layer_out)

w_h_o = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))
print('Weights for Hidden to Output:', w_h_o.shape, '\n', w_h_o)

print('[OI] Output Layer IN:', hidden_layer_out)
output_layer_in = np.dot(hidden_layer_out, w_h_o)
print('Output Layer before Activation Func.:', output_layer_in)
output_layer_out = sigmoid(output_layer_in)
print('[OO] Output Layer OUT:', output_layer_out)

learnrate = 0.5
print('\nLearning rate =', learnrate)
target = np.random.randn(2)
print('Target =', target)

# output error = target - OO
output_error = (target - output_layer_out)
print('Output Error = ', output_error)

# output error term = (Y - OO) * OO' = (Y - OO) * (OO * (1 - OO))
output_error_term = output_error * output_layer_out * (1 - output_layer_out)
print('[dO] Output Error Term = ', output_error_term)

# need to calculate the error term for the hidden unit with backpropagation
h_error_term = np.dot(w_h_o, output_error_term) * \
    hidden_layer_out * (1 - hidden_layer_out)
print('[dH] Hidden Error Term = ', h_error_term)

del_w_h_o = learnrate * output_error_term * hidden_layer_out[:, None]
print('old weights H to O =\n', w_h_o)
print(del_w_h_o)
w_h_o += del_w_h_o
print('new weights H to O =\n', w_h_o)

del_w_i_h = learnrate * h_error_term * X[:, None]
print('old weights I to H =\n', w_i_h)
print(del_w_i_h)
w_i_h += del_w_i_h
print('new weights I to H =\n', w_i_h)
