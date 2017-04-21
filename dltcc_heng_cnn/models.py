from __future__ import absolute_import

import tensorflow as tf


class DltccHeng(object):
    def __init__(self, ngf=64, image_width=200, image_height=40):
        self.ngf = ngf
        self.image_width = image_width
        self.image_height = image_height

    # Build the models
    def build(self, inputs):
        if inputs is None:
            print("Input should not none!")
        self.x_reshape = tf.reshape(inputs, [-1, self.image_width, self.image_height, 1], name="x_reshape")

        with tf.variable_scope("conv_layers"):
            # conv_1: [batch, 200, 4, 1] => [batch, 100, 20, ngf]
            self.conv1_1 = conv2d(self.x_reshape, self.ngf*2, stride=1, name="conv1_1")
            self.conv1_2 = conv2d(self.conv1_1, self.ngf*2, stride=1, name="conv1_2")
            self.max_pool1 = maxpool2d(self.conv1_2, k=2)

            # [batch, 100, 20, ngf] => [batch, 50, 10, ngf*2]
            self.conv2_1 = conv2d(self.max_pool1, self.ngf * 4, stride=1, name="conv2_1")
            self.conv2_2 = conv2d(self.conv2_1, self.ngf * 4, stride=1, name="conv2_2")
            self.conv2_3 = conv2d(self.conv2_2, self.ngf * 4, stride=1, name="conv2_3")
            self.conv2_4 = conv2d(self.conv2_3, self.ngf * 4, stride=1, name="conv2_4")
            self.max_pool2 = maxpool2d(self.conv2_4, k=2)

            # [batch, 50, 10, ngf*2] => [batch, 25, 5, ngf*4]
            self.conv3_1 = conv2d(self.max_pool2, self.ngf * 8, stride=1, name="conv3_1")
            self.conv3_2 = conv2d(self.conv3_1, self.ngf * 8, stride=1, name="conv3_2")
            self.conv3_3 = conv2d(self.conv3_2, self.ngf * 8, stride=1, name="conv3_3")
            self.conv3_4 = conv2d(self.conv3_3, self.ngf * 8, stride=1, name="conv3_4")
            self.max_pool3 = maxpool2d(self.conv3_4, k=2)

        with tf.variable_scope("deconv_layers"):
            # [batch, 25, 5, ngf*4] => [batch, 50, 10, ngf*2]
            self.deconv3 = deconv(self.max_pool3, self.ngf*2, name="deconv3")
            self.de_batchnorm3 = batchnorm(self.deconv3, name="de_batchnorm3")
            self.de_rectified3 = tf.nn.relu(self.de_batchnorm3, name="de_rectified3")

            # [batch, 50, 10, ngf*2] => [batch, 100, 20, ngf]
            self.deconv2 = deconv(self.de_rectified3, self.ngf, name="deconv2")
            self.de_batchnorm2 = batchnorm(self.deconv2, name="de_batchnorm2")
            self.de_rectified2 = tf.nn.relu(self.de_batchnorm2, name="de_rectified2")

            # [batch, 100, 20, ngf] => [batch, 200, 40, 1]
            self.deconv1 = deconv(self.de_rectified2, out_channels=1, name="deconv1")
            self.de_batchnorm1 = batchnorm(self.deconv1, name="de_batchnorm1")
            self.out = tf.tanh(self.de_batchnorm1, name="out")


def conv2d(batch_input, out_channels, stride, name):
    with tf.variable_scope(name):
        in_channels = batch_input.get_shape()[3]

        filter = tf.get_variable("filter", [3, 3, in_channels, out_channels], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, 0.02))
        b = tf.Variable(tf.random_normal([out_channels]))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d(batch_input, filter, strides=[1, stride, stride, 1], padding="SAME")
        conv = tf.nn.bias_add(conv, b)
        return tf.nn.relu(conv)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")


def lrelu(x, a, name):
    with tf.name_scope(name):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input, name):
    with tf.variable_scope(name):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels, name):
    with tf.variable_scope(name):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [3, 3, out_channels, in_channels], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels],
                                          [1, 2, 2, 1], padding="SAME")
        return conv
