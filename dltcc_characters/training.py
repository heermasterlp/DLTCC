import tensorflow as tf
import os
import datetime
import numpy as np
import time

import input_data
import ImageDisplay

import models
from utils import construct_network
from utils import block
from utils import max_pool
from utils import total_variation_loss


# 150x150 data set
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
model_path = "../../checkpoints/models_150_200_4_4"
checkpoint_path = "../../checkpoints/checkpoints_150_200"

# threshold
THEROSHOLD = 0.8

# max training epoch
MAX_TRAIN_EPOCH = 100000


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

        # dltcc_obj = models.Dltcc()
        # dltcc_obj.build(x, phase_train, IMAGE_WIDTH, IMAGE_HEIGHT)

        x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='x_image')
        y_image = tf.reshape(y_true, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='y_image')

        with tf.name_scope('input'):
            # the laywers:
            conv64 = construct_network(x_image, size=64, infilters=1, outfilters=8, laywers=2,
                                       phase_train=phase_train, scope='conv64')
            conv32 = construct_network(conv64, size=32, infilters=8, outfilters=32, laywers=2,
                                       phase_train=phase_train, scope='conv32');
            conv16 = construct_network(conv32, size=16, infilters=32, outfilters=64, laywers=2,
                                       phase_train=phase_train, scope='conv16');
            conv7 = construct_network(conv16, size=7, infilters=64, outfilters=128, laywers=2,
                                      phase_train=phase_train, scope='conv7');
        with tf.name_scope('conv3'):
            conv3_1 = block(conv7, [3, 3, 128, 128], phase_train=phase_train, scope='conv3_1');
            conv3_2 = block(conv3_1, [3, 3, 128, 1], phase_train=phase_train, scope='conv3_2');
        pooled = max_pool(conv3_2);
        with tf.name_scope('normalization'):
            dropped = tf.nn.dropout(pooled, keep_prob=0.9);
            y_pre_img = tf.sigmoid(dropped);

            weight_matrix = tf.random_uniform(shape=(IMAGE_WIDTH, IMAGE_HEIGHT), minval=0.0, maxval=0.5)

        with tf.device('gpu:0'), tf.name_scope('train'):
            with tf.name_scope('loss'):
                pixel_loss = tf.reduce_meam(tf.abs(y_image, y_pre_img))
                tv_loss = 0.002 * total_variation_loss(y_pre_img, IMAGE_WIDTH)
                combined_loss = pixel_loss + tv_loss
            optimizer_op = tf.train.RMSPropOptimizer(0.01).minimize(combined_loss)

        #
        #
        # # Loss
        # with tf.device("gpu:0"):
        #     cost_op = tf.reduce_mean((y_true - dltcc_obj.y_prob) ** 2)
        #     # cost_op = tf.reduce_mean(tf.abs(y_true * tf.log(dltcc_obj.y_prob) + (1 - y_true) * tf.log(1 - dltcc_obj.y_prob)))
        #     optimizer_op = tf.train.RMSPropOptimizer(0.01).minimize(cost_op)
        #
        # print("Build models end!")

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

                _, cost = sess.run([optimizer_op, combined_loss], feed_dict={x: x_batch, y_true: y_batch,
                                                                           phase_train: True})

                if epoch % 100 == 0:
                    print("Epoch {0} : {1}".format(epoch, cost))

            duration = time.time() - start_time
            print("Train time:{}".format(duration))

            # Save the trained models.
            saver.save(sess, os.path.join(model_path, "models-{}".format(today)))
            print("Training end!")

if __name__ == "__main__":
    train()