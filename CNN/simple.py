import sys
sys.path.append("/Users/liupeng/Documents/python/DeepLearning2TCC/DataSet")
import input_data

import tensorflow as tf
import numpy as np
import time

# data set file
train_data_dir = "../DataSetFiles/TrainSet/Kai_train_50_50_200_npy.npy"
train_target_dir = "../DataSetFiles/TrainSet/Qigong_train_50_50_200_npy.npy"

test_data_dir = "../DataSetFiles/TestSet/Kai_test_50_50_40_npy.npy"
test_target_dir = "../DataSetFiles/TestSet/Kai_test_50_50_40_npy.npy"

# validation size
VALIDATION_SIZE = 50

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

# Configuration of Neural Network


# create new variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# Create a new Convolution layer
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

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
    return layer, weights


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
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# conv1
filter_size = 3
num_filters1 = 16

# conv2
num_filters2 = 32

# conv 3
num_filters3 = 64

# conv4
num_filters4 = 128

# conv5
num_filters5 = 256


# Fully connected layer
fc_size1 = 640000
fc_size2 = 2500



# Load data
dataset = input_data.read_data_sets(train_dir, validation_size=40)

print("train data size:", len(dataset.train.data))
print("train target size:", len(dataset.train.target))
print("test data size:", len(dataset.test.data))
print("test target size:", len(dataset.test.target))


x = tf.placeholder(tf.float32, shape=[None, 2500], name="x")
y_true = tf.placeholder(tf.float32, shape=[None, 2500], name="y_true")

x_reshaped = tf.reshape(x, [-1, 50, 50, 1])


# conv 1
layer_conv1, weights_conv1 = new_conv_layer(input=x_reshaped,
                   num_input_channels=1,
                   filter_size=3,
                   num_filters=num_filters1,
                   use_pooling=True)
# conv 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=3,
                                            num_filters=num_filters2,
                                            use_pooling=True)
#conv 3
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
                                            num_input_channels=num_filters2,
                                            filter_size=3,
                                            num_filters=num_filters3,
                                            use_pooling=True)

# conv 4
layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3,
                                            num_input_channels=num_filters3,
                                            filter_size=3,
                                            num_filters=num_filters4,
                                            use_pooling=True)
# conv 5
layer_conv5, weights_conv5 = new_conv_layer(input=layer_conv4,
                                            num_input_channels=num_filters4,
                                            filter_size=3,
                                            num_filters=num_filters5,
                                            use_pooling=True)

# flatten layer
layer_flat, num_features = flatten_layer(layer_conv5)

# fc 1
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=80000,
                         use_relu=True)

# fc 2
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=80000,
                         num_outputs=2500,
                         use_relu=False)

# predict
y_pred = tf.nn.softmax(layer_fc2)

# cost
loss_op = tf.reduce_mean(tf.abs(y_pred - y_true))
optimizer_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss_op)

# accuracy
accuracy_op = 1 - tf.reduce_mean(tf.abs(y_pred - y_true))

init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init_op)

    acc = sess.run(accuracy_op, feed_dict={x: dataset.test.data,
                                   y_true: dataset.test.target})
    print("before accuracy:", acc)

    for step in range(50):

        x_batch, y_batch = dataset.train.next_batch(10)

        opt = sess.run([optimizer_op], feed_dict={x: x_batch,
                                                  y_true: y_batch})
        if step % 5 == 0:
            acc = sess.run(accuracy_op, feed_dict={x: dataset.test.data,
                                                   y_true: dataset.test.target})
            print("Step %d : %d" % (step, acc))

    acc = sess.run(accuracy_op, feed_dict={x: dataset.test.data,
                                           y_true: dataset.test.target})
    # dataset = None
    print("after accuracy:", acc)
    dataset = None
