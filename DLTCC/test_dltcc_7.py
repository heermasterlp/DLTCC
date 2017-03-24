from __future__ import absolute_import

import tensorflow as tf
import os
import datetime
import numpy as np

import input_data
import ImageDisplay

import dltcc_7

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
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_100_100_200_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_100_100_200_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_100_100_20_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_100_100_20_test.npy"

# validation size
VALIDATION_SIZE = 50

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

SIZE = 100
IMAGE_WIDTH = SIZE
IMAGE_HEIGHT = SIZE

model_path = "../../checkpoints/models_dltcc_7_100_100"
checkpoint_path = "../../checkpoints/checkpoints_dltcc_7_100_100"

# threshold
THEROSHOLD = 0.6

# max training epoch
MAX_TRAIN_EPOCH = 100

# Train models
def train():
    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=40)

    print("Start build models")

    # place variable
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="y_true")

    dltcc_obj = dltcc_7.Dltcc7()
    dltcc_obj.build(x)

    # Loss
    with tf.device("gpu:0"):
        cost_op = tf.reduce_mean((y_true - dltcc_obj.y_prob)**2)
        optimizer_op = tf.train.RMSPropOptimizer(0.01).minimize(cost_op)

    print("Build models end!")

    # initialize variable
    # init_op = tf.global_variables_initializer()
    init_op = tf.initialize_all_variables()

    # save the models and checkpoints. the formatting: (models) models-date.ckpt, (checkpoint) checkpoint-date-step.ckpt
    saver = tf.train.Saver()
    model_path = "../../checkpoints/models_100_100"
    checkpoint_path = "../../checkpoints/checkpoints_100_100"

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

            _, cost = sess.run([optimizer_op, cost_op], feed_dict={x: x_batch, y_true: y_batch})

            if epoch % 10 == 0:
                print("Epoch {0} : {1}".format(epoch, cost))

        # Save the trained models.
        saver.save(sess, os.path.join(model_path, "models-{}".format(today)))
        print("Training end!")


if __name__ == "__main__":
    train()