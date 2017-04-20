from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import argparse
import datetime
import os
import numpy as np
from skimage.io import imsave
from input_data import *


parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--mode', dest='mode', default='train', help='train, test')
parser.add_argument('--device', dest='device', default='cpu:0', help='device: cpu or gpu')

args = parser.parse_args()

mb_size = 32
batch_size = 64
image_width = 128
image_height = 128
X_dim = image_width * image_height
Z_dim = X_dim

h_dim = 100
h1_dim = 200
h2_dim = 200
h3_dim = 100

EPS = 1e-12
l1_weight = 100.0   # weight on L1 term for generator gradient
gan_weight = 1.0

model_dir = "../../checkpoints/models_128_128"

# 200x40 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/kai_128_128_200_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_128_128_200_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/kai_128_128_20_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_128_128_20_test.npy"

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}


def discriminator(x_data, x_generated, keep_prob):
    x_in = tf.concat([x_data, x_generated], 0)

    w1 = tf.Variable(tf.truncated_normal([X_dim, h2_dim], stddev=0.1), name="d_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h2_dim]), name="d_b1", dtype=tf.float32)

    w2 = tf.Variable(tf.truncated_normal([h2_dim, h1_dim], stddev=0.1), name="d_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h1_dim]), name="d_b2", dtype=tf.float32)

    w3 = tf.Variable(tf.truncated_normal([h1_dim, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)

    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob=keep_prob)
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob=keep_prob)
    h3 = tf.matmul(h2, w3) + b3

    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name="y_data"))
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name="y_generated"))
    d_params = [w1, b1, w2, b2, w3, b3]

    return y_generated, d_params


def generator(z_prior):
    w1 = tf.Variable(tf.truncated_normal([Z_dim, h1_dim], stddev=0.1), name="g_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h1_dim]), name="g_b1", dtype=tf.float32)

    w2 = tf.Variable(tf.truncated_normal([h1_dim, h2_dim], stddev=0.1), name="g_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h2_dim]), name="g_b2", dtype=tf.float32)

    w3 = tf.Variable(tf.truncated_normal([h2_dim, Z_dim], stddev=0.1), name="g_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([Z_dim]), name="g_b3")

    h1 = tf.nn.tanh(tf.matmul(z_prior, w1) + b1)
    h2 = tf.nn.tanh(tf.matmul(h1, w2) + b2)
    h3 = tf.matmul(h2, w3) + b3
    x_generate = tf.nn.tanh(h3)
    g_params = [w1, b1, w2, b2, w3, b3]
    return x_generate, g_params


def train(args):
    # Data set
    data_set = read_data_sets(train_dir, validation_size=2)

    # Inputs
    x_data = tf.placeholder(tf.float32, [batch_size, X_dim], name="x_data")
    y_target = tf.placeholder(tf.float32, [batch_size, X_dim], name="y_target")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # model
    x_generated, g_params = generator(x_data)
    y_real, d_params_real = discriminator(x_data, y_target, keep_prob)
    y_fake, d_params_fake = discriminator(x_data, x_generated, keep_prob)

    d_loss = tf.reduce_mean(-(tf.log(y_real + EPS) + tf.log(1 - y_fake + EPS)))

    g_loss_GAN = tf.reduce_mean(-tf.log(y_fake + EPS))
    g_loss_L1 = tf.reduce_mean(tf.abs(y_target - x_generated))
    g_loss = g_loss_GAN * gan_weight + g_loss_L1 * l1_weight

    with tf.device("cpu:0"):
        D_optim = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(d_loss)
        G_optim = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(g_loss)

    now = datetime.datetime.now()
    today = "{}-{}-{}".format(now.year, now.month, now.day)
    if not os.path.exists(model_dir + today):
        os.mkdir(model_dir + today)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(args.epoch):

            x_batch, y_batch = data_set.train.next_batch(batch_size)

            # Discriminator
            for it in range(10):
                sess.run([D_optim, d_loss], feed_dict={x_data: x_batch, y_target: y_batch,
                                                        keep_prob: np.sum(0.7).astype(np.float32)})

            _, g_loss_GAN_val, g_loss_L1_val = sess.run([G_optim, g_loss_GAN, g_loss_L1], feed_dict={x_data: x_batch, y_target: y_batch,
                                                 keep_prob: np.sum(0.7).astype(np.float32)})
            if epoch % 20 == 0:
                print("epoch: {} g_loss_GAN:{} g_loss_L1:{}".format(epoch, g_loss_GAN_val, g_loss_L1_val))

        print("Train model finished!")
        # save the models
        saver.save(sess, os.path.join(model_dir + today, "models-{}".format(today)))
        print("Save success!")


def test():
    x_input = tf.placeholder(tf.float32, [batch_size, X_dim], name="x_input")
    x_generated, _ = generator(x_input)
    chkpt_fname = tf.train.latest_checkpoint(model_dir + "2017-4-20")

    # Data set
    data_set = read_data_sets(train_dir, validation_size=2)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, chkpt_fname)

        x_batch, y_batch = data_set.train.next_batch(batch_size)
        x_gen_val = sess.run(x_generated, feed_dict={x_input: x_batch})

        show_result(x_gen_val, "test_result.jpg")


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


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
        test()
    else:
        print("train or test?")


if __name__ == "__main__":
    tf.app.run()