import tensorflow as tf
import numpy as np


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
    '''Leaky ReLU activation'''
    with tf.variable_scope(name) as scope:
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak*x)


def linear1d(inputlin, inputdim, outputdim, name="linear1d", std=0.02, mn=0.0):

    with tf.variable_scope(name) as scope:

        weight = tf.get_variable("weight",[inputdim, outputdim])
        bias = tf.get_variable("bias",[outputdim], dtype=np.float32, initializer=tf.constant_initializer(0.0))

        out = tf.matmul(inputlin, weight) + bias
        out = tf.contrib.layers.batch_norm(out, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")
        return tf.nn.relu(out)

