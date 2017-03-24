from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from functools import reduce

SIZE = 100

IMAGE_WIDTH = SIZE
IMAGE_HEIGHT = SIZE

DENSE_1_OUTPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 16
DENSE_2_OUTPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT


class Dltcc(object):
    def __init__(self):
        pass

    # Build the models
    def build(self, inputs):
        if inputs is None:
            print("Input should not none!")

        self.x_reshape = tf.reshape(inputs, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name="x_reshape")

        # Conv 1
        with tf.name_scope("conv1"):
            self.conv1 = conv_layer(input=self.x_reshape, input_channels=1, filter_size=3, output_channels=16, use_pooling=True)

        with tf.name_scope("conv2"):
            self.conv2 = conv_layer(input=self.conv1, input_channels=16, filter_size=3, output_channels=16, use_pooling=True)
            # Conv 3
        with tf.name_scope("conv3"):
            self.conv3 = conv_layer(input=self.conv2, input_channels=16, filter_size=3, output_channels=32, use_pooling=True)

            # Conv 4
        # with tf.name_scope("conv4"):
        #     self.conv4 = conv_layer(input=self.conv3, input_channels=32, filter_size=3, output_channels=64, use_pooling=True)

            # Flatten layer
        with tf.name_scope("flatten1"):
            self.layer_flat, self.num_flat_features = flatten_layer(self.conv3)

        with tf.name_scope("fc_layer"):
            self.layer_fc1 = new_fc_layer(input=self.layer_flat, num_inputs=self.num_flat_features,
                                     num_outputs=IMAGE_WIDTH * IMAGE_HEIGHT * 4, use_sigmoid=True)

            self.layer_fc2 = new_fc_layer(input=self.layer_fc1, num_inputs=IMAGE_WIDTH * IMAGE_HEIGHT * 4,
                                     num_outputs=IMAGE_WIDTH * IMAGE_HEIGHT, use_sigmoid=True)

            # Predict
        with tf.name_scope("probability"):
            # layer_dropped = tf.nn.dropout(layer_fc2, keep_prob=1.0)
            self.y_prob = tf.sigmoid(self.layer_fc2)

            print("Build models end!")


# Create a new Convolution layer
def conv_layer(input,  # The previous layer.
                   input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   output_channels,  # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, input_channels, output_channels]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=output_channels)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer


def avg_pool(inputs, name):
    return tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(inputs, name):
    return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


# flattening a layer
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


# Fully connected layer
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_sigmoid=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_sigmoid:
        # layer = tf.nn.relu(layer)
        layer = tf.nn.sigmoid(layer)

    return layer

# create new variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))




class Dltcc1(object):
    def __init__(self, npy_path=None, trainable=True, dropout=0.5):
        self.dataset = None
        self.trainable = trainable
        self.dropout = dropout

    '''
        Models build
    '''
    def build(self, images):

        if images is None:
            print("Images is none")
            return

        print('image shape:', images.shape)
        images_reshaped = tf.reshape(images, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

        # Conv 1
        with tf.name_scope("Conv_1"):
            self.conv1_1 = tf.layers.conv2d(inputs=images_reshaped, filters=16, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu)
            # self.conv1_2 = tf.layers.conv2d(inputs=self.conv1_1, filters=16, kernel_size=[3, 3], padding='same',
            #                                 activation=tf.nn.relu)
            self.pool1 = tf.layers.average_pooling2d(inputs=self.conv1_1, pool_size=[2, 2], strides=1, padding='same')

        # Conv 2
        # with tf.name_scope("Conv_2"):
        #     self.conv2_1 = tf.layers.conv2d(inputs=self.pool1, filters=32, kernel_size=[3, 3], padding='same',
        #                                     activation=tf.nn.relu)
        #
        #     self.conv2_2 = tf.layers.conv2d(inputs=self.conv2_1, filters=32, kernel_size=[3, 3], padding='same',
        #                                     activation=tf.nn.relu)
        #
        #     self.conv2_3 = tf.layers.conv2d(inputs=self.conv2_2, filters=32, kernel_size=[3, 3], padding='same',
        #                                     activation=tf.nn.relu)
        #
        #     self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2_3, pool_size=[2, 2], strides=1, padding='same')
        #
        #
        #
        # # Conv 3
        # with tf.name_scope("Conv_3"):
        #
        #     self.conv3_1 = tf.layers.conv2d(inputs=self.pool2, filters=64, kernel_size=[3, 3], padding='same',
        #                                     activation=tf.nn.relu)
        #
        #     self.conv3_2 = tf.layers.conv2d(inputs=self.conv3_1, filters=64, kernel_size=[3, 3], padding='same',
        #                                     activation=tf.nn.relu)
        #
        #     self.conv3_3 = tf.layers.conv2d(inputs=self.conv3_2, filters=64, kernel_size=[3, 3], padding='same',
        #                                     activation=tf.nn.relu)
        #
        #     self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3_3, pool_size=[2, 2], strides=1, padding='same')

        # Flat layers
        with tf.name_scope("Flat_1"):
            self.flat1 = tf.reshape(self.pool1, [-1, IMAGE_WIDTH * IMAGE_HEIGHT * 16])

        # Dense Layer
        with tf.name_scope("Dense"):
            self.dense1 = tf.layers.dense(inputs=self.flat1, units=DENSE_1_OUTPUT_SIZE, activation=tf.nn.relu)

            self.dropout1 = tf.layers.dropout(inputs=self.dense1, rate=self.dropout)

            self.dense2 = tf.layers.dense(inputs=self.dropout1, units=DENSE_2_OUTPUT_SIZE, activation=tf.nn.relu)
        # Softmax function
        with tf.name_scope("predict"):
            self.pred = tf.nn.softmax(self.dense2)

        print('Build the models finished!')


# Simple models
class Dltcc2(object):
    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

    def build(self, images):
        if images is None:
            print("Images is none")
            return

        print('image shape:', images.shape)
        images_reshaped = tf.reshape(images, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

        # CNN structures
        self.conv1 = self.conv_layer(images_reshaped, 1, 4, "conv1")
        self.max_pool1 = self.max_pool(self.conv1, "max_pool1")

        # self.fc1 = self.fc_layer(self.max_pool1, 250*250*16, 250*250*4, "fc1")
        self.fc2 = self.fc_layer(self.max_pool1, IMAGE_WIDTH*IMAGE_HEIGHT*10, IMAGE_WIDTH*IMAGE_HEIGHT, "fc1")
        self.pred = tf.nn.softmax(self.fc2)
        print("Build models end!")

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count




