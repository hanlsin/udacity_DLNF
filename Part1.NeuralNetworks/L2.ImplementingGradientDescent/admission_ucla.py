'''
As an example, I'm going to have you use gradient descent
to train a network on graduate school admissions data
found at http://www.ats.ucla.edu/stat/data/binary.csv.
This dataset has three input features: GRE score, GPA,
and the rank of the undergraduate school (numbered 1 through 4).
Institutions with rank 1 have the highest prestige,
those with rank 4 have the lowest.
'''

'''
1. Data Cleanup
'''
import pandas as pd

data = pd.read_csv(
    './udacity_DLNF/Part1.NeuralNetworks/L2.ImplementingGradientDescent/binary.csv')
# print(data.head())
# print(data.tail())

'''
we need to use dummy variables to encode rank,
splitting the data into four new columns encoded with ones or zeros
'''
dummies = pd.get_dummies(data['rank'], prefix='rank')
# print(dummies.head())
data = data.drop(['rank'], axis=1)
# print(data.head())
data = pd.concat([data, dummies], axis=1)
# print(data.head())
# print(data.shape)

'''
We'll also need to standardize the GRE and GPA data,
which means to scale the values
such they have zero mean and a standard deviation of 1.
This is necessary because the sigmoid function squashes
really small and really large inputs.
The gradient of really small and large inputs is zero,
which means that the gradient descent step will go to zero too.

Since the GRE and GPA values are fairly large,
we have to be really careful about how we initialize the weights
or the gradient descent steps will die off
and the network won't train.

Instead, if we standardize the data,
we can initialize the weights easily and everyone is happy.
'''
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:, field] = (data[field] - mean) / std
# print(data.head())

'''
# split off random 10% of the data for testing
'''
import numpy as np

np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data) * 0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

'''
# split into features and targets
'''
features, targets = data.drop('admit', axis=1), data['admit']
test_features, test_targets = test_data.drop(
    'admit', axis=1), test_data['admit']

'''
2. MSE (Mean Square Error)

Here's the general algorithm for updating the weights with gradient descent:

* Set the weight step to zero:
* For each record in the training data:
  * Make a forward pass through the network, calculating the output
  * Calculate the error term for the output unit,
  * Update the weight step
* Update the weights. Here we're averaging the weight steps
  to help reduce any large variations in the training data.
* Repeat for e epochs.
'''


def activation_func(x):
    # sigmoid
    return 1 / (1 + np.exp(-x))


def activation_func_prime(x):
    # derivative of sigmoid
    return activation_func(x) * (1 - activation_func(x))


def calculate_output_error(y, y_hat):
    return (y - y_hat)


def calculate_error_term(h, output_error):
    # δ = output_error * f'(h)
    return output_error * activation_func_prime(h)


def calculate_h(w, x):
    return np.matmul(w, x)


'''
# initial weights
First, you'll need to initialize the weights.
We want these to be small
such that the input to the sigmoid is in the linear region near 0
and not squashed at the high and low ends.
It's also important to initialize them randomly
so that they all have different starting values and diverge, breaking symmetry.
So, we'll initialize the weights from a normal distribution centered at 0.
A good value for the scale is 1/√​n​ where n is the number of input units.
This keeps the input to the sigmoid low for increasing numbers of input units.
'''
n_input_rows, n_input_units = features.shape
# print(np.sqrt(n_input_units))
# print(n_input_units**.5)
weights = np.random.normal(
    scale=(1 / n_input_units**.5), size=n_input_units)
# print(weights)

# find the first prediction with the initial weights
# print(features.shape)
# print(weights.shape)
# predict = activation_func(np.matmul(features, weights))
# print(predict)

# calculate MSE
# print(predict.shape)
# print(targets.shape)
# loss = np.mean((targets - predict) ** 2)
# print(loss)

# print(np.dot(weights, data.values[0]))
# print(calculate_h(weights, data.values[0]))

epochs = 1000
learnrate = 0.5
last_loss = None

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for X, y in zip(features.values, targets):
        # print(X)
        # print(y)
        # print(weights)

        h = np.matmul(weights, X)
        # print(h)

        output = activation_func(h)
        # print(output)

        output_error = calculate_output_error(y, output)
        # print(output_error)

        error_term = calculate_error_term(h, output_error)
        # print(error_term)

        del_w += error_term * X
        # print(del_w)

    weights += learnrate * del_w / n_input_rows
    # print(weights)

    # print MSE on the training set
    if e % (epochs / 5) == 0:
        predict = activation_func(np.matmul(features, weights))
        loss = np.mean((targets - predict) ** 2)
        if last_loss and last_loss < loss:
            print("Train [", e, "] loss: ", loss,
                  "  WARNING - Loss Increasing")
        else:
            print("Train [", e, "] loss: ", loss)
        last_loss = loss

# calculate accuracy on test data
test_predict = activation_func(np.matmul(test_features, weights))
# print(test_predict)
predictions = test_predict > 0.5
# print(predictions)
accuracy = np.mean(predictions == test_targets)
print("Prediction accuracy: {:.3f}".format(accuracy))
