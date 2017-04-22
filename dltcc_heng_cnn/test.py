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
batch_size = 11

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
    y_true = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_WIDTH * IMAGE_HEIGHT], name="y_true")

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
        x_batch, y_batch = (data_set.test.data, data_set.test.target)

        # predict
        y_pred = sess.run(predict_op, feed_dict={x: x_batch, y_true: y_batch})

        assert y_pred.shape == y_true.shape
        print("y_pred shape:{} y_true shape:{}".format(y_pred.shape, y_true.shape))

        # threshold
        y_pred_arr = np.array(y_pred)
        y_true_arr = np.array(y_true)

        y_pred_list =[]
        for bt in range(y_pred.shape[0]):
            y_pred_item_list = []
            print(y_pred[bt])
            print(np.amax(y_pred[bt]))
            print(np.amin(y_pred[bt]))
            for it in range(y_pred.shape[1]):

                if y_pred[bt][it] >= threshold:
                    y_pred_item_list.append(1)
                else:
                    y_pred_item_list.append(0)

            y_pred_list.append(y_pred_item_list)


        # Calculate the accuracy
        # for bt in range(y_pred.shape[0]):
        #     y_pred_item = y_pred[bt]
        #     y_true_item = y_true[bt]
        #     for it in range(y_pred.shape[1]):
        #         pass



if __name__ == "__main__":
    test()