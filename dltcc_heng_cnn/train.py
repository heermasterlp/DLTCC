from __future__ import absolute_import

import datetime
import time
import os
import argparse
import tensorflow as tf
from input_data import *
from model256x256 import *
from ops import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='# images in batch')
parser.add_argument('--mode', dest='mode', default='train', help='train or test')
parser.add_argument('--device', dest='device', default='cpu:0', help='cpu or gpu')
args = parser.parse_args()

# 200x40 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_256_256_30_train_heng.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_256_256_30_train_heng.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_256_256_6_test_heng.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_256_256_6_test_heng.npy"

# train data set files---
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

input_width = 256
output_width = 256

generator_dim = 64
batch_size = 20

Ltv_penalty = 100

model_dir = "../../checkpoints/models_256_256_heng_mac_4_28"

# max training epoch
MAX_TRAIN_EPOCH = 1000
DISPLAY_STEP = 100


def train():
    # Data set
    data_set = read_data_sets(train_dir, validation_size=2)

    if data_set is None:
        print("data sets are none!")
        return
    else:
        print("data set train data shape:{}".format(data_set.train.data.shape))
        print("data set train target shape:{}".format(data_set.train.target.shape))
        print("data set test data shape:{}".format(data_set.test.data.shape))
        print("data set test target shape:{}".format(data_set.test.target.shape))

    print("Start build models")
    # place variable
    x = tf.placeholder(tf.float32, shape=[batch_size, input_width * input_width], name="x")
    y_true = tf.placeholder(tf.float32, shape=[batch_size, output_width * output_width], name="y_true")

    # Models
    is_training = False
    if args.mode == "train":
        is_training = True
    models = DltccHeng(batch_size=batch_size, generator_dim=64, input_width=256, output_width=256, input_filters=3,
                       output_filters=3, is_training=is_training)
    models.build_model(x)

    out = models.output
    print("Build models end!")

    # prediction and reshape
    y_pred = tf.reshape(out, shape=y_true.shape)

    with tf.device(args.device):
        l1_loss_op = tf.reduce_mean((y_true - y_pred)**2)

        # tv loss
        width = output_width
        y_pred_reshape = tf.reshape(y_pred, [batch_size, input_width, input_width, 1])
        tv_loss_op = (tf.nn.l2_loss(y_pred_reshape[:, 1:, :, :] - y_pred_reshape[:, :width - 1, :, :]) / width
                   + tf.nn.l2_loss(y_pred_reshape[:, :, 1:, :] - y_pred_reshape[:, :, :width - 1, :]) / width) * Ltv_penalty

        total_loss = l1_loss_op + tv_loss_op
        optimizer_op = tf.train.RMSPropOptimizer(0.01).minimize(total_loss)

    print("Build models end!")

    init_op = tf.global_variables_initializer()

    # Save the models
    saver = tf.train.Saver()
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    now = datetime.datetime.now()
    today = "{}-{}-{}".format(now.year, now.month, now.day)

    with tf.Session() as sess:
        start_time = time.time()
        sess.run(init_op)

        # Train the models
        for epoch in range(args.epoch):
            x_batch, y_batch = data_set.train.next_batch(batch_size)
            _, li_loss, tv_loss = sess.run([optimizer_op, l1_loss_op, tv_loss_op], feed_dict={x: x_batch, y_true: y_batch})

            if epoch % 100 == 0:
                print("Epoch total loss:{} l1_loss: {} tv_loss:{}".format(epoch, li_loss, tv_loss))

        duration = time.time() - start_time

        # Save the trained models
        saver.save(sess, os.path.join(model_dir, "models-{}".format(today)))
        print("Save models success")
        print("Train time:{}".format(duration))


if __name__ == "__main__":
    train()