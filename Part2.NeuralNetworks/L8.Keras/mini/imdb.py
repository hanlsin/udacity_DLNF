
'''
#################
Loading the data
#################
'''
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=1000,
                                                      skip_top=2,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

print(x_train.shape)
print(x_train[0])
print(y_train.shape)
print(y_train[0])

'''
#################
Preprocessing the data
#################
'''
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

# One-hot encoding the output into vector mode, each of length 1000
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('X train shape =', x_train.shape)
print(x_train[0])
print('X test shape =', x_test.shape)

# One-hot encoding the output
num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print('Y train shape =', y_train.shape)
print(y_train[0])
print('Y test shape =', y_test.shape)

'''
#################
Building the model architecture
#################
'''
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

sa_model = Sequential()
sa_model.add(Dense(32, input_dim=x_train.shape[1]))
sa_model.add(Activation('relu'))
sa_model.add(Dropout(0.5))
sa_model.add(Dense(2))
sa_model.add(Activation('softmax'))

sa_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

sa_model.summary()

history = sa_model.fit(x_train, y_train, epochs=50, verbose=0)
print(history.history.keys())

score = sa_model.evaluate(x_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy: ', score[1])

'''
import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
