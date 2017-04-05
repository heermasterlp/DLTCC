import tensorflow as tf
import os
import datetime
import time

import input_data
import models


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

        dltcc_obj = models.Dltcc()
        dltcc_obj.build(x, phase_train, IMAGE_WIDTH, IMAGE_HEIGHT)

        # Loss
        with tf.device("gpu:0"):
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
            saver.save(sess, os.path.join(model_path, "models-{}".format(today)))
            print("Training end!")

if __name__ == "__main__":
    train()