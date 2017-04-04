import tensorflow as tf
import numpy as np

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

# model path
model_path = "../../checkpoints/models_150_200_4_1"
checkpoint_path = "../../checkpoints/checkpoints_150_200"

# threshold
THEROSHOLD = 0.6


# Evaluate the models
def test():

    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=40)

    # place variable
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")
    y_true = data_set.test.target
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # Build models
    dltcc_obj = models.Dltcc()
    dltcc_obj.build(x, phase_train, IMAGE_WIDTH, IMAGE_HEIGHT)

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
        prediction = sess.run(dltcc_obj.y_prob, feed_dict={x: data_set.test.data,
                                                           phase_train: False})

        if prediction is None:
            print("Prediction is none")
            return
        print(prediction)
        assert prediction.shape == y_true.shape

        # average accuracy
        accuracy = 0.0
        for x in range(prediction.shape[0]):
            prediction_item = prediction[x]
            print(prediction_item)
            pred_arr = np.array(prediction_item)
            print(np.amin(pred_arr))
            print(np.amax(pred_arr))
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

if __name__ == "__main__":
    test()