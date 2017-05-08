from __future__ import absolute_import

import datetime
import time
import os
import argparse
from skimage.io import imsave
import tensorflow as tf
import numpy as np
from input_data import *
from modelautoencoder import *

# 200x40 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_256_256_30_train_heng.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_256_256_30_train_heng.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_256_256_6_test_heng.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_256_256_6_test_heng.npy"

# train data set files---
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

ngf = 20
batch_size = 8

model_dir = "../../checkpoints/models_256_256_heng_mac_4_28"

# Threshold
threshold = 0.8


def test():
    # Data set
    data_set = read_data_sets(train_dir, validation_size=2)
    if data_set is None:
        print("data set is none!")
        return

    batch_size = data_set.train.data.shape[0]
    print("batch_size: {}".format(batch_size))

    # place variable
    x = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")
    # y_true = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_WIDTH * IMAGE_HEIGHT], name="y_true")

    # Models
    models = DltccHeng(batch_size=batch_size, generator_dim=64, input_width=256, output_width=256, input_filters=3,
                       output_filters=3, is_training=False)
    models.build_model(x)
    # Get the prediction with shape:[batch_size, image_width, image_height, 1]
    predict_op = tf.reshape(models.output, shape=[batch_size, IMAGE_WIDTH * IMAGE_HEIGHT])
    print("Build models end!")

    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Reload the well-trained models
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("The checkpoint models found!")
        else:
            print("The checkpoint models not found!")

        # data set
        # x_batch, y_true = (data_set.train.data, data_set.train.target)
        x_batch, y_true = data_set.train.next_batch(batch_size)

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
        accuracy = []
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
            print("item error:{}".format(1 - item_error))
            accuracy.append(1-item_error)
        for bt in range(y_pred.shape[0]):
            for it in range(y_pred.shape[1]):
                if y_pred[bt][it] >= threshold:
                    y_pred[bt][it] = 1.0
                else:
                    y_pred[bt][it] = 0.0
        # assert y_pred.shape == y_true.shape
        # for it in range(y_pred.shape[0]):
            # show_result(y_pred[it], "predict_{}_threshold_{}_accuracy_{}_.jpg".format(it, threshold, accuracy[it]))
            # show_result(y_true[it], "true_{}_threshold_{}_accuracy_{}_.jpg".format(it, threshold, accuracy[it]))
        # show_result(y_pred, "test_result_predict_threshold {}.jpg".format(threshold))
        # show_result(y_true, "test_result_true_threshold {}.jpg".format(threshold))
        print("avg error:{}".format(1 - avg_error / y_pred.shape[0]))
        save_result_samples(y_pred, y_true, "train_id_1_result_threshold{}.jpg".format(threshold), grid_size=(batch_size, 2))


def save_result_samples(batch_pred, batch_true, fname, grid_size=(6, 2), grid_pad=5):
    batch_pred = batch_pred.reshape((batch_pred.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT))
    batch_true = batch_true.reshape((batch_true.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT))

    img_h, img_w = batch_pred.shape[1], batch_pred.shape[2]

    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_h * grid_size[1] + grid_pad * (grid_size[1] - 1)

    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for i in range(batch_pred.shape[0]):
        print(i)
        res_pred = batch_pred[i]
        res_true = batch_true[i]

        img_pred = (res_pred) * 255
        img_true = (res_true) * 255

        img_pred = img_pred.astype(np.uint8)
        img_true = img_true.astype(np.uint8)

        row = i * (img_h + grid_pad)
        col_pred = 0
        col_true = img_w + grid_pad
        img_grid[row:row+img_h, col_pred:col_pred+img_w] = img_pred
        img_grid[row:row+img_h, col_true:col_true+img_w] = img_true
    imsave(fname, img_grid)


def show_result(batch_res, fname, grid_size=(4, 4), grid_pad=5):
    batch_res = batch_res.reshape((batch_res.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH))
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

if __name__ == "__main__":
    test()