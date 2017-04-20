from __future__ import absolute_import

import tensorflow as tf
import numpy as np

import input_data
import net_2_hidden

import models
import ImageDisplay


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

# threshold
THEROSHOLD = 0.8


def inference(input):
    if input is None:
        print("Input should not none!")

    # place variable
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")

    # Build models
    y_pred = net_2_hidden.net(x, IMAGE_WIDTH, IMAGE_HEIGHT)

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
        prediction = sess.run(y_pred, feed_dict={x: input})

        print(prediction)

        if prediction is None:
            return
        img_pred = []
        for index in range(prediction.shape[1]):
            if prediction[0][index] >= THEROSHOLD:
                img_pred.append(1)
            else:
                img_pred.append(0)

        return np.array(img_pred)


def test_inference():
    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=2)

    index = 2

    input = data_set.train.data[index]
    input = np.reshape(input, [-1, IMAGE_WIDTH * IMAGE_HEIGHT])

    target = data_set.train.target[index]
    # Predict
    predict = inference(input)

    # Accuracy
    target = np.array(target)

    diff = np.abs(np.subtract(predict, target))
    print(diff.shape)
    sum = np.sum(diff)
    accuracy = 1 - sum / diff.shape[0]
    print("Accuracy:{}".format(accuracy))

    ImageDisplay.show_bitmap(input)
    ImageDisplay.show_bitmap(predict)
    ImageDisplay.show_bitmap(target)


if __name__ == "__main__":
    test_inference()