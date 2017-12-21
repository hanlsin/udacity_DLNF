from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import prepare_data

model = Sequential()

model.add(Conv2D(223, 223, 16, activation='relu', input_shape=(224, 224, 3)))

model.summary()
