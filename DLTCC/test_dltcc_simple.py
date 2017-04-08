from __future__ import absolute_import

import tensorflow as tf
import os
import datetime
import numpy as np
import time

import input_data
import ImageDisplay

# 250x250 data set
# train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_250_250_400_train.npy"
# train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_250_250_400_train.npy"
#
# test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_250_250_40_test.npy"
# test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_250_250_40_test.npy"

# 50x50 data set
# train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_50_50_200_train.npy"
# train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_50_50_200_train.npy"
#
# test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_50_50_20_test.npy"
# test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_50_50_20_test.npy"

# 100x100 data set
# train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_100_100_200_train.npy"
# train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_100_100_200_train.npy"
#
# test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_100_100_20_test.npy"
# test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_100_100_20_test.npy"

# 200x200 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_150_150_200_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_150_150_200_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_150_150_20_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_150_150_20_test.npy"

# validation size
VALIDATION_SIZE = 50

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

SIZE = 150
IMAGE_WIDTH = SIZE
IMAGE_HEIGHT = SIZE

# model_path = "../../checkpoints/models_50_50"
# checkpoint_path = "../../checkpoints/checkpoints_50_50"
model_path = "../../checkpoints/models_150_200"
checkpoint_path = "../../checkpoints/checkpoints_150_200"

# threshold
THEROSHOLD = 0.8

# max training epoch
MAX_TRAIN_EPOCH = 10


# Train models
def train():
    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=40)

    print("Start build models")

    with tf.Graph().as_default():
        start_time = time.time()

        # place variable
        x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")
        y_true = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="y_true")
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        dltcc_obj = Dltcc()
        dltcc_obj.build(x, phase_train)

        # Loss
        with tf.device("cpu:0"):
            cost_op = tf.reduce_mean((y_true - dltcc_obj.y_prob) ** 2)
            optimizer_op = tf.train.RMSPropOptimizer(0.01).minimize(cost_op)

        print("Build models end!")

        # initialize variable
        init_op = tf.global_variables_initializer()

        # save the models and checkpoints.
        saver = tf.train.Saver()

        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        now = datetime.datetime.now()
        today = "{}-{}-{}".format(now.year, now.month, now.day)

        # Train models

        with tf.Session() as sess:
            sess.run(init_op)

            # Train the models
            for epoch in range(MAX_TRAIN_EPOCH):
                x_batch, y_batch = data_set.train.next_batch(200)

                _, cost = sess.run([optimizer_op, cost_op], feed_dict={x: x_batch, y_true: y_batch,
                                                                           phase_train: True})

                if epoch % 100 == 0:
                    print("Epoch {0} : {1}".format(epoch, cost))

            duration = time.time() - start_time
            print("Train time:{}".format(duration))

            # Save the trained models.
            # saver.save(sess, os.path.join(model_path, "models-{}".format(today)))
            # print("Training end!")

            prediction = sess.run(dltcc_obj.y_prob, feed_dict={x: data_set.train.data,
                                                               phase_train: False})
            print(prediction[0])



# Evaluate the models
def evaluate():

    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=40)

    # place variable
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")
    y_true = data_set.train.target
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # Build models
    dltcc_obj = dltcc_models.Dltcc()
    dltcc_obj.build(x, phase_train)

    # Saver
    saver = tf.train.Saver()

    # output probability
    with tf.Session() as sess:
        # Reload the well-trained models
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("The checkpoint models found!")
        else:
            print("The checkpoint models not found!")

        # prediction shape: [batch_size, width * height]
        prediction = sess.run(dltcc_obj.y_prob, feed_dict={x: data_set.train.data,
                                                           phase_train: False})

        if prediction is None:
            print("Prediction is none")
            return
        print(prediction)
        assert prediction.shape == y_true.shape

        # average accuracy
        avg_accuracy = 0.0
        accuracy = 0.0
        for x in range(prediction.shape[0]):
            prediction_item = prediction[x]
            y_pred = []
            for y in range(prediction_item.shape[0]):
                if prediction_item[y] > THEROSHOLD:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

            y_pred = np.array(y_pred)
            y_true = np.array(data_set.train.target[x])

            sub_array = np.abs(np.subtract(y_pred, y_true))
            sum = 0.0
            for i in range(len(sub_array)):
                sum += sub_array[i]
            accuracy += sum / len(sub_array)
            print("accuracy:{}".format(1 - sum / len(sub_array)))

        avg_accuracy = 1 - accuracy / prediction.shape[0]

        print("Avg accuracy:{}".format(avg_accuracy))


def inference(input, target):
    if input is None:
        print("Input should not none!")

    # place variable
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # Build models
    dltcc_obj = dltcc_models.Dltcc()
    dltcc_obj.build(x, phase_train)

    # Saver
    saver = tf.train.Saver()

    # output probability
    with tf.Session() as sess:
        # Reload the well-trained models
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("The checkpoint models found!")
        else:
            print("The checkpoint models not found!")

        # prediction shape: [batch_size, width * height]
        prediction = sess.run(dltcc_obj.y_prob, feed_dict={x: input,
                                                           phase_train: False})

        print(prediction.shape)

        # normalize the result of prediction prob
        minPredVal = np.amin(prediction)
        maxPredVal = np.amax(prediction)
        # prediction_normed = normalize_func(prediction, minVal=minPredVal, maxVal=maxPredVal)
        prediction_normed = prediction

        print(prediction_normed)

        if prediction_normed is None:
            return
        img_pred = []
        for index in range(prediction_normed.shape[1]):
            if prediction_normed[0][index] >= THEROSHOLD:
                img_pred.append(1)
            else:
                img_pred.append(0)

        return np.array(img_pred)


def test_inference():
    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=40)

    index = 14

    input = data_set.test.data[index]
    input = np.reshape(input, [-1, IMAGE_WIDTH * IMAGE_HEIGHT])
    target = data_set.test.target[index]
    # Predict
    predict = inference(input, target)

    # Accuracy
    target = np.array(target)

    diff = np.abs(np.subtract(predict, target))
    print(diff.shape)
    sum = np.sum(diff)
    accuracy = 1 - sum / diff.shape[0]
    print("Accuracy:{}".format(accuracy))

    # statistics
    print(np.amin(predict))
    print(np.mean(predict))


    ImageDisplay.show_bitmap(input)
    ImageDisplay.show_bitmap(predict)
    ImageDisplay.show_bitmap(target)


def normalize_func(x, minVal, maxVal, newMinVal=0, newMaxVal=1):
    result = (x-minVal)*newMaxVal/(maxVal-minVal) + newMinVal
    return result


class Dltcc(object):
    def __init__(self):
        pass

    # Build the models
    def build(self, inputs, phase_train):
        if inputs is None:
            print("Input should not none!")

        self.x_reshape = tf.reshape(inputs, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name="x_reshape")

        # Conv 1
        with tf.name_scope("conv1"):
            self.conv1 = conv_layer(input=self.x_reshape, input_channels=1, filter_size=3, output_channels=5,
                                    use_pooling=True, phase_train=phase_train)

        with tf.name_scope("conv2"):
            self.conv2 = conv_layer(input=self.conv1, input_channels=5, filter_size=3, output_channels=10,
                                    use_pooling=True, phase_train=phase_train)
            # Conv 3
        with tf.name_scope("conv3"):
            self.conv3 = conv_layer(input=self.conv2, input_channels=10, filter_size=3, output_channels=20,
                                    use_pooling=True, phase_train=phase_train)

            # Conv 4
        with tf.name_scope("conv4"):
            self.conv4 = conv_layer(input=self.conv3, input_channels=20, filter_size=3, output_channels=25,
                                    use_pooling=True, phase_train=phase_train)

            # Flatten layer
        with tf.name_scope("flatten"):
            self.layer_flat, self.num_flat_features = flatten_layer(self.conv4)

        with tf.name_scope("fc_layer"):

            self.layer_fc1 = new_fc_layer(self.layer_flat, num_inputs=self.num_flat_features,
                                          num_outputs=IMAGE_WIDTH*IMAGE_HEIGHT)
            self.dropput1= tf.nn.dropout(self.layer_fc1, keep_prob=0.6)

        with tf.name_scope("prob"):
            self.y_prob = tf.sigmoid(self.dropput1)


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
    layer = batch_norm(layer, phase_train=phase_train)

    # Rectified Linear Unit (ReLU).
    layer = tf.nn.relu(layer)

    return layer


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
                 num_outputs):  # Num. outputs.

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

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

if __name__ == "__main__":
    train()
    # evaluate()
    # test_inference()