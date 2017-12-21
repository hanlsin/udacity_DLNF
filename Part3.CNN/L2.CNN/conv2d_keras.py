from keras.layers import Conv2D
from keras.models import Sequential

model = Sequential()

"""
* filters - The number of filters.
* kernel_size - Number specifying both the height and width of
                the (square) convolution window.

There are some additional, optional arguments that you might like to tune:

* strides - The stride of the convolution.
            If you don't specify anything, strides is set to 1.
* padding - One of 'valid' or 'same'.
            If you don't specify anything, padding is set to 'valid'.
* activation - Typically 'relu'.
                If you don't specify anything,
                no activation is applied.
                You are strongly encouraged
                to add a ReLU activation function
                to every convolutional layer in your networks.
"""

"""
Example #1:
Say I'm constructing a CNN, and my input layer accepts grayscale images
that are 200 by 200 pixels (corresponding to a 3D array
with height 200, width 200, and depth 1). Then, say I'd like
the next layer to be a convolutional layer
with 16 filters, each with a width and height of 2.
When performing the convolution, I'd like the filter
to jump two pixels at a time. I also don't want the filter
to extend outside of the image boundaries; in other words,
I don't want to pad the image with zeros.
Then, to construct this convolutional layer,
I would use the following line of code:
"""
model.add(Conv2D(filters=16, kernel_size=2, strides=2,
                 padding='valid', activation='relu',
                 input_shape=(200, 200, 1))

"""
Example #2:
Say I'd like the next layer in my CNN to be a convolutional layer
that takes the layer constructed in Example 1 as input.
Say I'd like my new layer to have 32 filters,
each with a height and width of 3. When performing the convolution,
I'd like the filter to jump 1 pixel at a time.
I want the convolutional layer to see all regions of the previous layer,
and so I don't mind if the filter hangs over the edge
of the previous layer when it's performing the convolution.
Then, to construct this convolutional layer,
I would use the following line of code:
"""
model.add(Conv2D(filters=32, kernel_size=3, strides=(1, 1),
                 padding='same', activation='relu'))

model.add(Conv2D(64, (2, 2), activation='relu')

model.summary()
