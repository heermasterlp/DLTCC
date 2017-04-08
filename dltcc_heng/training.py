from __future__ import absolute_import

import datetime
import os
import time

import tensorflow as tf

import input_data
import net_2_hidden

# 200x40 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_heng_200_40_30_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_heng_200_40_30_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_heng_200_40_11_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_heng_200_40_11_test.npy"

# validation size
VALIDATION_SIZE = 50

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 40

model_path = "../../checkpoints/models_200_40_mac_4_8"
checkpoint_path = "../../checkpoints/checkpoints_200_40_mac"

# max training epoch
MAX_TRAIN_EPOCH = 1000000
DISPLAY_STEP = 100

def train():
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

    print("Start build models")
    # place variable
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="y_true")


    # Models
    y_pred = net_2_hidden.net(x, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

    # Loss
    with tf.device("gpu:0"):
        loss_op = tf.reduce_mean((y_true - y_pred) ** 2)
        optimizer_op = tf.train.RMSPropOptimizer(0.01).minimize(loss_op)

    print("Build models end!")

    # initialize variable
    init_op = tf.global_variables_initializer()

    # save the models and checkpoints.
    # the formatting: (models) models-date.ckpt, (checkpoint) checkpoint-date-step.ckpt
    saver = tf.train.Saver()

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    now = datetime.datetime.now()
    today = "{}-{}-{}".format(now.year, now.month, now.day)

    # Train models
    with tf.Session() as sess:
        start_time = time.time()

        sess.run(init_op)

        # Train the models
        for epoch in range(MAX_TRAIN_EPOCH):
            x_batch = data_set.train.data
            y_batch = data_set.train.target

            _, cost = sess.run([optimizer_op, loss_op], feed_dict={x: x_batch,
                                                                    y_true: y_batch})

            if epoch % DISPLAY_STEP == 0:
                print("Epoch {0} : {1}".format(epoch, cost))

        duration = time.time() - start_time

        # Save the trained models.
        saver.save(sess, os.path.join(model_path, "models-{}".format(today)))
        print("Training end!{}".format(duration))



if __name__ == '__main__':
    train()