from __future__ import absolute_import

import tensorflow as tf


class DltccHeng(object):
    def __init__(self):
        pass

    # Build the models
    def build(self, inputs, phase_train, width, height):
        if inputs is None:
            print("Input should not none!")

        self.x_reshape = tf.reshape(inputs, [-1, width, height, 1], name="x_reshape")

        # Conv 1
        with tf.name_scope("conv1"):
            self.conv1 = conv_layer(input=self.x_reshape, input_channels=1, filter_size=3, output_channels=8,
                                    use_pooling=False, phase_train=phase_train)

        with tf.name_scope("conv2"):
            self.conv2 = conv_layer(input=self.conv1, input_channels=8, filter_size=3, output_channels=16,
                                    use_pooling=True, phase_train=phase_train)
            # Conv 3
        with tf.name_scope("conv3"):
            self.conv3 = conv_layer(input=self.conv2, input_channels=16, filter_size=3, output_channels=32,
                                    use_pooling=True, phase_train=phase_train)

            # Conv 4
        with tf.name_scope("conv4"):
            self.conv4 = conv_layer(input=self.conv3, input_channels=32, filter_size=3, output_channels=64,
                                    use_pooling=False, phase_train=phase_train)
        # Conv 5
        with tf.name_scope("conv5"):
            self.conv5 = conv_layer(input=self.conv4, input_channels=64, filter_size=3, output_channels=64,
                                    use_pooling=True, phase_train=phase_train)

            # Flatten layer
        with tf.name_scope("flatten1"):
            self.layer_flat, self.num_flat_features = flatten_layer(self.conv5)

        with tf.name_scope("fc_layer"):

            self.layer_fc2 = new_fc_layer(input=self.layer_flat, num_inputs=self.num_flat_features,
                                     num_outputs=width * height, use_sigmoid=True)

            # Predict
        with tf.name_scope("probability"):
            # layer_dropped = tf.nn.dropout(layer_fc2, keep_prob=1.0)
            self.y_prob = tf.sigmoid(self.layer_fc2)


# Create a new Convolution layer
def conv_layer(input,  # The previous layer.
                   input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   output_channels,  # Number of filters.
                   use_pooling=True,
                   phase_train=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, input_channels, output_channels]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=output_channels)

    # Create the TensorFlow operation for convolution.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding="SAME")

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="SAME")

    # batch normalization
    # layer = batch_norm(layer, phase_train=phase_train)

    # Rectified Linear Unit (ReLU).
    layer = tf.nn.relu(layer)

    return layer


def avg_pool(inputs, name):
    return tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)


def max_pool(inputs, name):
    return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)


# flattening a layer
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].

    layer_flat = tf.reshape(layer, [-1, num_features])

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
        layer = tf.nn.sigmoid(layer)

    return layer


# create new variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def batch_norm(input, phase_train, scope="batch_normal"):
    with tf.name_scope(scope):
        out_filters = input.get_shape()[-1]
        beta = tf.Variable(tf.constant(0.0, shape=[out_filters]), name="beta", trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[out_filters]), name="gamma", trainable=True)
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name="moments")
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normaled = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
        return normaled