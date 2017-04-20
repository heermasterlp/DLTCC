from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import datetime
import json
import glob
import random
import collections
import math
import time
from skimage.io import imsave

import input_data

EPS = 1e-12
CROP_SIZE = 256
ngf = 8
ndf = 8
gan_weight = 1.0
l1_weight = 100.0

batch_size = 64
image_width = 128
image_height = 128
image_size = image_width * image_height

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, "
                                        "discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")
# model dir
model_dir = "../../checkpoints/models_128_128_4_20"

# 128x128 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/kai_128_128_200_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_128_128_200_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/kai_128_128_20_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_128_128_20_test.npy"

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--mode', dest='mode', default='train', help='train, test')
parser.add_argument('--device', dest='device', default='cpu:0', help='device: cpu or gpu')

args = parser.parse_args()


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [3, 3, in_channels, out_channels], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        print(batch_input.get_shape())
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [3, 3, out_channels, in_channels], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels],
                                          [1, 2, 2, 1], padding="SAME")
        return conv


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        print('input shape:{}'.format(generator_inputs.shape))
        output = conv(generator_inputs, ngf, stride=2)
        layers.append(output)

    layer_specs = [
        ngf * 4,  # encoder_2: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8,  # encoder_3: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8,  # encoder_4: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8,  # encoder_6: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8,  # encoder_7: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)
    print("last encode layer shape:{}".format(layers[-1].get_shape()))

    layer_specs = [
        (ngf * 8, 0.5),   # decoder_7: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_6: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_5: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_4: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_3: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0)   # decoder_2: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)
    print('Generator part finished!')

    return layers[-1]


def create_discriminator(discrim_inputs, discrim_targets):
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets):

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.variable_scope("real_discriminator"):
        # with tf.variable_scope("discriminator"):
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        predict_real = create_discriminator(inputs, targets)

    with tf.variable_scope("fake_discriminator"):
        # with tf.variable_scope("discriminator", reuse=True):
        # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
        predict_fake = create_discriminator(inputs, outputs)

    return outputs, predict_real, predict_fake


def train(args):
    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=2)

    if data_set is None:
        print("data sets are none!")
        return
    else:
        print("data set train data shape:{}".format(data_set.train.data.shape))
        print("data set train target shape:{}".format(data_set.train.target.shape))
        print("data set test data shape:{}".format(data_set.test.data.shape))
        print("data set test target shape:{}".format(data_set.test.target.shape))

    # input placeholder
    X = tf.placeholder(tf.float32, shape=[batch_size, image_width * image_height])
    y = tf.placeholder(tf.float32, shape=[batch_size, image_width * image_height])

    X_reshape = tf.reshape(X, [batch_size, image_width, image_height, 1])
    y_reshape = tf.reshape(y, [batch_size, image_width, image_height, 1])

    # Model
    print('Build the model')
    outputs, predict_real, predict_fake = create_model(X_reshape, y_reshape)
    print('Build the model end!')

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(y_reshape - outputs))
        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight

    with tf.device(args.device):
        with tf.name_scope('discriminator_op'):
            discrim_op = tf.train.RMSPropOptimizer(0.01).minimize(discrim_loss)
        with tf.name_scope('generator_op'):
            gen_op = tf.train.RMSPropOptimizer(0.01).minimize(gen_loss)

    now = datetime.datetime.now()
    today = "{}-{}-{}".format(now.year, now.month, now.day)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(args.epoch):

            x_batch, y_batch = data_set.train.next_batch(batch_size)

            for it in range(int(data_set.train.data.shape[0] / batch_size)):
                _, D_loss = sess.run([discrim_op, discrim_loss], feed_dict={X: x_batch, y: y_batch})
            _, G_loss = sess.run([gen_op, gen_loss], feed_dict={X: x_batch, y: y_batch})

            if epoch % 50 == 0:
                print('epoch:{} D_loss:{} G_loss:{}'.format(epoch, D_loss, G_loss))
                x_out = sess.run(outputs, feed_dict={X: x_batch})
                show_result(x_out, "out/test_result{}.jpg".format(epoch))


        print("Training end!")

        # Save the models
        saver.save(sess, os.path.join(model_dir, today))
        print("Model saved success!")


def test(args):

    chkpt_fname = tf.train.latest_checkpoint(model_dir)

    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=2)

    # input placeholder
    X = tf.placeholder(tf.float32, shape=[batch_size, image_width * image_height])

    X_reshape = tf.reshape(X, [batch_size, image_width, image_height, 1])

    # Model
    print('Build the model')
    prediction = create_generator(X_reshape, 1)
    print('Build the model end!')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, chkpt_fname)

        x_batch, y_batch = data_set.train.next_batch(batch_size)
        pred_val = sess.run(prediction, feed_dict={X: x_batch})

        show_result(pred_val, "test_result.jpg")


def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], image_height, image_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)


def main(_):
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        print("train or test")


if __name__ == "__main__":
    tf.app.run()