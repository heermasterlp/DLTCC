import tensorflow as tf
import numpy as np

import models
import input_data
import ImageDisplay

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


def inference(input, target):
    if input is None:
        print("Input should not none!")

    # place variable
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name="x")
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
        prediction = sess.run(dltcc_obj.y_prob, feed_dict={x: input,
                                                           phase_train: False})

        print(prediction.shape)

        # normalize the result of prediction prob
        minPredVal = np.amin(prediction)
        maxPredVal = np.amax(prediction)
        # prediction_normed = normalize_func(prediction, minVal=minPredVal, maxVal=maxPredVal)
        prediction_normed = prediction

        print(prediction_normed)

        if prediction_normed is None:
            return
        img_pred = []
        for index in range(prediction_normed.shape[1]):
            if prediction_normed[0][index] >= THEROSHOLD:
                img_pred.append(1)
            else:
                img_pred.append(0)

        return np.array(img_pred)


def test_inference():
    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=40)

    index = 100

    input = data_set.train.data[index]
    input = np.reshape(input, [-1, IMAGE_WIDTH * IMAGE_HEIGHT])
    target = data_set.train.target[index]
    # Predict
    predict = inference(input, target)

    # Accuracy
    target = np.array(target)

    diff = np.abs(np.subtract(predict, target))
    print(diff.shape)
    sum = np.sum(diff)
    accuracy = 1 - sum / diff.shape[0]
    print("Accuracy:{}".format(accuracy))

    # statistics
    print(np.amin(predict))
    print(np.mean(predict))

    ImageDisplay.show_bitmap(input)
    ImageDisplay.show_bitmap(predict)
    ImageDisplay.show_bitmap(target)

if __name__ == "__main__":
    test_inference()