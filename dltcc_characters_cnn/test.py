from __future__ import absolute_import

import datetime
import time
import os
import argparse
import tensorflow as tf
import numpy as np
from input_data import *
from models import *

# 200x40 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_heng_200_40_30_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_heng_200_40_30_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_heng_200_40_11_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_heng_200_40_11_test.npy"

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 40

ngf = 16
batch_size = 28

model_dir = "../../checkpoints/models_200_40_mac_4_21"

# Threshold
threshold = 0.6

def test():
    # Data set
    data_set = read_data_sets(train_dir, validation_size=2)

    if data_set is None:
        print("data set is none!")
        return

    # place variable
    x = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")
    # y_true = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_WIDTH * IMAGE_HEIGHT], name="y_true")

    # Models
    models = DltccHeng(ngf, IMAGE_WIDTH, IMAGE_HEIGHT)
    models.build(x)
    # Get the prediction with shape:[batch_size, image_width, image_height, 1]
    predict_op = tf.reshape(models.out, shape=[batch_size, IMAGE_WIDTH * IMAGE_HEIGHT])
    print("Build models end!")

    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Reload the well-trained models
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("The checkpoint models fount!")
        else:
            print("The checkpoint models not found!")

        # data set
        x_batch, y_true = (data_set.train.data, data_set.train.target)

        # predict
        y_pred = sess.run(predict_op, feed_dict={x: x_batch})

        assert y_pred.shape == y_true.shape
        print("y_pred shape:{} y_true shape:{}".format(y_pred.shape, y_true.shape))

        y_pred_arr = np.array(y_pred)
        y_true_arr = np.array(y_true)
        print(y_true_arr.shape)
        assert y_pred_arr.shape == y_true_arr.shape

        # threshold
        avg_error = 0.0
        for bt in range(y_pred.shape[0]):
            print("bt:{}".format(bt))
            item_error = 0.0
            sum_y_true = 0

            for it in range(y_true.shape[1]):
                if y_true[bt][it] == 1.0:
                    sum_y_true += 1
            print("y_true sum:{}".format(sum_y_true))

            for it in range(y_pred.shape[1]):
                if y_pred[bt][it] >= threshold and y_true[bt][it] != 1.0:
                        item_error += 1
                elif y_pred[bt][it] < threshold and y_true[bt][it] == 1.0:
                        item_error += 1
            item_error = item_error / sum_y_true
            avg_error += item_error
            print("item error:{}".format(item_error))
        print("avg error:{}".format(avg_error / y_pred.shape[0]))


if __name__ == "__main__":
    test()