from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D

model = Sequential()
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                    padding='valid', input_shape=(100, 100, 15)))
model.summary()
