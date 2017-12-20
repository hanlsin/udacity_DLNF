import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('.', one_hot=True, reshape=False)

'''
# Loading Variables
'''
# Remove the previous weights and bias
tf.reset_default_graph()

sess = tf.Session()

# Class used to save and/or restore Tensor Variables
saver = tf.train.import_meta_graph('./model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

# check weights
print(sess.run('w_h:0'))
print(sess.run('w_o:0'))
# check biases
print(sess.run('b_h:0'))
print(sess.run('b_o:0'))

# access and create placeholders
graph = tf.get_default_graph()

# get tf Graph input
x = graph.get_tensor_by_name('x:0')
y = graph.get_tensor_by_name('y:0')

weights = {
    'hidden_layer': graph.get_tensor_by_name('w_h:0'),
    'out': graph.get_tensor_by_name('w_o:0')
}
biases = {
    'hidden_layer': graph.get_tensor_by_name('b_h:0'),
    'out': graph.get_tensor_by_name('b_o:0')
}

accuracy = graph.get_tensor_by_name('accuracy:0')

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print(weights['hidden_layer'].eval()[0])
    print(weights['out'].eval()[0])
    print(biases['hidden_layer'].eval()[0])
    print(biases['out'].eval()[0])

    test_accuracy = session.run(
        accuracy,
        feed_dict={x: mnist.test.images, y: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))
