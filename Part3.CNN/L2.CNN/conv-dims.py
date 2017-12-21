from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
'''
the number of filters = filters = f
the height and width of the conv. layer = kernel_size = (kx, ky)
the stride of the convolution = strides = s
depth of previous input = input_shape[-1] = pd
the height and width of the previous layer = (px, py)

Formula: param #
    the number of weights per filter = kx * ky * pd
    total number of weights in the convolutional layer = f * kx * ky * pd
    the number of biases = f
    param # = (f * kx * ky * pd) + f

Formula: shape of conv. layer
    if padding = 'same', no missing:
        the height of the conv. layer = h = ceil(float(px) / float(s))
        the width of the conv. layer = w = ceil(float(py) / float(s))
    if padding = 'valid', missing the last few:
        the height of the conv. layer = h = ceil(float(px - kx + 1) / float(s))
        the width of the conv. layer = w = ceil(float(py - ky + 1) / float(s))
    
    the depth of the conv. layer = d = the number of filters = f
'''

'''
param #:
    f = 16
    kx = ky = 2
    pd = 1
    param # = 80
shape:
    px = py = 200
    kx = ky = 2
    s = 2
    f = 16
    h = w = ceil((200 - 2 + 1) / 2) = ceil(99.5) = 100
    d = f = 16
    shape = (100, 100, 16)
'''
model.add(Conv2D(16, 2, strides=2, padding='valid',
                 activation='relu', input_shape=(200, 200, 1)))
'''
param #:
    f = 1
    kx = ky = 3
    pd = 16
    param # = 145
shape:
    px = py = 100
    s = 1
    f = 1
    h = w = ceil((100) / 1) = ceil(100) = 100
    d = f = 1
    shape = (100, 100, 1)
'''
model.add(Conv2D(1, 3, strides=1, padding='same', activation='relu'))
model.summary()
