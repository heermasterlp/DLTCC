from __future__ import absolute_import

import tensorflow as tf
import numpy as np

import input_data
import net_2_hidden

import utils


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
THEROSHOLD = 0.57


def test():
    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=2)

    if data_set is None:
        print("data set is none!")
        return

    # place variable
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")
    y_true = data_set.train.target

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
        prediction = sess.run(y_pred, feed_dict={x: data_set.train.data})

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
            print(prediction_item)
            pred_arr = np.array(prediction_item)
            print(min(pred_arr))
            print(max(pred_arr))

            minPredVal = np.amin(pred_arr)
            maxPredVal = np.amax(pred_arr)
            # prediction_normed = utils._normalize_func(pred_arr, minVal=minPredVal, maxVal=maxPredVal)
            prediction_normed = pred_arr

            y_pred = []
            for y in range(prediction_normed.shape[0]):
                if prediction_normed[y] > THEROSHOLD:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

            y_pred = np.array(y_pred)
            y_true = np.array(data_set.train.target[x])

            sum = 0.0
            assert y_pred.shape == y_true.shape

            for i in range(y_pred.shape[0]):
                if y_pred[i] == 1.0 and y_true[i] != 1.0:
                    sum += 1
            accuracy = 1 - sum / y_pred.shape[0]

            print("accuracy:{}".format(accuracy))

        avg_accuracy = 1 - accuracy / prediction.shape[0]

        print("Avg accuracy:{}".format(avg_accuracy))


if __name__ == "__main__":
    test()