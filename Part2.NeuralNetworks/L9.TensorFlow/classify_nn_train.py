import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('.', one_hot=True, reshape=False)

import numpy as np
train_imgs = mnist.train.images
print(train_imgs.shape)
print(train_imgs[0].shape)
import matplotlib.pyplot as plt
img = (train_imgs[0] * 255).astype("uint8")
plt.imshow(img.reshape([28, 28]))
# plt.show()

'''
Learning Parameters
'''
learning_rate = 0.001
training_epochs = 1
# Decrease batch size if you don't have enough memory
batch_size = 128
display_step = 1

n_input = 784   # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

n_hidden_layer = 256  # layer number of features

# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer]), name='w_h'),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]), name='w_o')
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer]), name='b_h'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='b_o')
}

# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1], name='x')
y = tf.placeholder("float", [None, n_classes], name='y')

x_flat = tf.reshape(x, [-1, n_input])

# probability to keep units
keep_prob = tf.placeholder(tf.float32)

# Hidden layer with RELU activation
layer_1 = tf.add(
    tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)
layer_1 = tf.nn.dropout(layer_1, keep_prob)
# Output layer with linear activation
logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

# Define loss and optimizer
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32), name='accuracy')

# Saving Variables
save_file = './model.ckpt'
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            result = sess.run([optimizer, cost],
                              feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            # print(result)

        # Print status for every 10 epochs
        if epoch % 10 == 0:
            valid_accuracy = sess.run(
                accuracy,
                feed_dict={
                    x: mnist.validation.images,
                    y: mnist.validation.labels,
                    keep_prob: 1.0})
            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                valid_accuracy))

    valid_accuracy = sess.run(
        accuracy,
        feed_dict={
            x: mnist.validation.images,
            y: mnist.validation.labels,
            keep_prob: 1.0})
    print('Validation Accuracy: {}'.format(valid_accuracy))

    print(weights['hidden_layer'].eval()[0])
    print(weights['out'].eval()[0])
    print(biases['hidden_layer'].eval()[0])
    print(biases['out'].eval()[0])
    saver.save(sess, save_file)
