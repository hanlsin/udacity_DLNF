import sys
import numpy as np
import keras

# X has shape (num_rows, num_cols), where the training data are stored
# as row vectors
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
print(X.shape)

# y must have an output vector for each input vector
y = np.array([[0], [0], [0], [1]], dtype=np.float32)
y_binary = keras.utils.to_categorical(y)
print(y)
print(y_binary)

# sys.exit(0)

from keras.models import Sequential
from keras.layers.core import Dense, Activation

# Create the Sequential model
model = Sequential()

# 1st Layer - Add an input layer of 32 nodes with the same input shape as
# the training samples in X
# 입력 2, 출력 32
model.add(Dense(32, input_dim=X.shape[1]))

# Add a softmax activation layer
model.add(Activation('softmax'))


# 2nd Layer - Add a fully connected output layer
# 입력 32, 출력 1
model.add(Dense(1))

# Add a sigmoid activation layer
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(X, y, epochs=1000, verbose=1)

score = model.evaluate()
print(model.metrics_names)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])
