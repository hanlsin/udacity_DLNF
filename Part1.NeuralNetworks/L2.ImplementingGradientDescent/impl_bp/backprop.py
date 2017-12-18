import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# number of hidden layer
n_hidden = 2

learnrate = 0.5
epochs = 1000

n_records, n_features = features.shape
print('Number of records =', n_records)
print('Features =', features.columns.values)

# Init weights
w_i_h = np.random.normal(scale=1 / n_features ** .5,
                         size=(n_features, n_hidden))
print('init weights I to H:\n', w_i_h)
w_h_o = np.random.normal(scale=1 / n_features ** .5, size=(n_hidden))
print('init weights H to O:\n', w_h_o)

last_loss = None

for e in range(epochs):
    ########
    # Set the weight steps for each layer to zero
    ####
    # The input to hidden weights
    delta_w_i_h = np.zeros(w_i_h.shape)
    # The hidden to output weights
    delta_w_h_o = np.zeros(w_h_o.shape)

    ########
    # For each record in the training data:
    ####
    for x, y in zip(features.values, targets):
        ########
        # Make a forward pass through the network, calculating the output
        ####
        hidden_input = np.dot(x, w_i_h)
        hidden_output = sigmoid(hidden_input)
        output = sigmoid(np.dot(hidden_output, w_h_o))

        ########
        # Calculate the error gradient in the output unit
        ####
        error = y - output
        output_error_term = error * output * (1 - output)

        ########
        # Propagate the errors to the hidden layer
        ####
        hidden_error = np.dot(w_h_o, output_error_term)
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        # Update the change in weights
        delta_w_i_h += hidden_error_term * x[:, None]
        delta_w_h_o += output_error_term * hidden_output

    ########
    # Update the weight steps
    ####
    w_i_h += learnrate * delta_w_i_h / n_records
    w_h_o += learnrate * delta_w_h_o / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, w_i_h))
        out = sigmoid(np.dot(hidden_output, w_h_o))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, w_i_h))
out = sigmoid(np.dot(hidden, w_h_o))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
