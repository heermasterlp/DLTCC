from __future__ import absolute_import

import tensorflow as tf
import os
import datetime
import numpy as np

import input_data
# import ImageDisplay

import dltcc_models

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
# train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_100_100_200_train.npy"
# train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_100_100_200_train.npy"
#
# test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_100_100_20_test.npy"
# test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_100_100_20_test.npy"

# 200x200 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_200_200_200_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_200_200_200_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_200_200_20_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_200_200_20_test.npy"

# validation size
VALIDATION_SIZE = 50

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

SIZE = 200
IMAGE_WIDTH = SIZE
IMAGE_HEIGHT = SIZE

# model_path = "../../checkpoints/models_50_50"
# checkpoint_path = "../../checkpoints/checkpoints_50_50"
model_path = "../../checkpoints/models_200_200"
checkpoint_path = "../../checkpoints/checkpoints_200_200"

# threshold
THEROSHOLD = 0.7

# max training epoch
MAX_TRAIN_EPOCH = 1000


# Train models
def train():
    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=40)

    print("Start build models")

    # place variable
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")
    y_true = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="y_true")

    dltcc_obj = dltcc_models.Dltcc()
    dltcc_obj.build(x)

    # Loss
    with tf.device("gpu:0"):
        cost_op = tf.reduce_mean((y_true - dltcc_obj.y_prob)**2)
        optimizer_op = tf.train.RMSPropOptimizer(0.01).minimize(cost_op)

    print("Build models end!")

    # initialize variable
    # init_op = tf.global_variables_initializer()
    init_op = tf.global_variables_initializer()

    # save the models and checkpoints. the formatting: (models) models-date.ckpt, (checkpoint) checkpoint-date-step.ckpt
    saver = tf.train.Saver()

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    now = datetime.datetime.now()
    today = "{}-{}-{}".format(now.year, now.month, now.day)

    # Train models
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(init_op)

        # Train the models
        for epoch in range(MAX_TRAIN_EPOCH):
            x_batch, y_batch = data_set.train.next_batch(200)

            _, cost = sess.run([optimizer_op, cost_op], feed_dict={x: x_batch, y_true: y_batch})

            if epoch % 100 == 0:
                print("Epoch {0} : {1}".format(epoch, cost))

        # Save the trained models.
        saver.save(sess, os.path.join(model_path, "models-{}".format(today)))
        print("Training end!")


# Evaluate the models
def evaluate():

    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=40)

    # place variable
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")
    y_true = data_set.train.target

    # Build models
    dltcc_obj = dltcc_models.Dltcc()
    dltcc_obj.build(x)

    # Saver
    saver = tf.train.Saver()

    # output probability
    with tf.Session() as sess:
        # Reload the well-trained models
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("The checkpoint models found!")
        else:
            print("The checkpoint models not found!")

        # prediction shape: [batch_size, width * height]
        prediction = sess.run(dltcc_obj.y_prob, feed_dict={x: data_set.train.data})

        if prediction is None:
            print("Prediction is none")
            return
        print(prediction.shape)
        assert prediction.shape == y_true.shape

        # average accuracy
        avg_accuracy = 0.0
        accuracy = 0.0
        for x in range(prediction.shape[0]):
            prediction_item = prediction[x]
            y_pred = []
            for y in range(prediction_item.shape[0]):
                if prediction_item[y] > THEROSHOLD:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

            y_pred = np.array(y_pred)
            y_true = np.array(data_set.train.target[x])

            sub_array = np.abs(np.subtract(y_pred, y_true))
            sum = 0.0
            for i in range(len(sub_array)):
                sum += sub_array[i]
            accuracy += sum / len(sub_array)
            print("accuracy:{}".format(1 - sum / len(sub_array)))

        avg_accuracy = 1 - accuracy / prediction.shape[0]

        print("Avg accuracy:{}".format(avg_accuracy))


def inference(input, target):
    if input is None:
        print("Input should not none!")

    # place variable
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")

    # Build models
    dltcc_obj = dltcc_models.Dltcc()
    dltcc_obj.build(x)

    # Saver
    saver = tf.train.Saver()

    # output probability
    with tf.Session() as sess:
        # Reload the well-trained models
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("The checkpoint models found!")
        else:
            print("The checkpoint models not found!")

        # prediction shape: [batch_size, width * height]
        prediction = sess.run(dltcc_obj.y_prob, feed_dict={x: input})

        print(prediction.shape)

        if prediction is None:
            return
        img_pred = []
        for index in range(prediction.shape[1]):
            if prediction[0][index] >= THEROSHOLD:
                img_pred.append(1)
            else:
                img_pred.append(0)

        ImageDisplay.show_bitmap(img_pred)
        ImageDisplay.show_bitmap(target)

        return np.array(img_pred)


def test_inference():
    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=40)

    index = 14

    input = data_set.test.data[index]
    input = np.reshape(input, [-1, IMAGE_WIDTH * IMAGE_HEIGHT])
    target = data_set.test.target[index]
    # Predict
    predict = inference(input, target)

    # Accuracy
    target = np.array(target)

    diff = np.abs(np.subtract(predict, target))
    print(diff.shape)
    sum = np.sum(diff)
    accuracy = 1 - sum / diff.shape[0]
    print("Accuracy:{}".format(accuracy))






if __name__ == "__main__":
    train()
    # evaluate()
    # test_inference()