import argparse
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from ImageDisplay import *
from input_data import *
from wgan_net import *


parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='# of epoch')
parser.add_argument('--mode', dest='mode', help='train or test')

args = parser.parse_args()

# 200x200 data set
# train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_200_200_200_train.npy"
# train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_200_200_200_train.npy"
#
# test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_200_200_20_test.npy"
# test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_200_200_20_test.npy"

# 200x40 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/kai_128_128_200_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_128_128_200_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/kai_128_128_20_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_128_128_20_test.npy"

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

# Data set
data_set = read_data_sets(train_dir, validation_size=2)

mb_size = 32
X_dim = data_set.train.data.shape[1]    # width * height
Z_dim = X_dim                           # width * height

threshold = 0.4

image_width = 128
image_height = 128


def train(args):
    # model saver
    model_dir = "../../checkpoints/models_200_200_4_17_heng"

    X = tf.placeholder(tf.float32, shape=[None, X_dim])
    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

    # Model net
    wgan_model = wgan_multilayer_perceptron_net(X, Z, X_dim)
    wgan_model.build()


    with tf.device("cpu:0"):

        D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-wgan_model.D_loss, var_list=wgan_model.theta_D))
        G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(wgan_model.G_loss, var_list=wgan_model.theta_G))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # save the models and checkpoints.
        saver = tf.train.Saver()
        now = datetime.datetime.now()
        today = "{}-{}-{}".format(now.year, now.month, now.day)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        for it in range(args.epoch):

            for _ in range(5):
                X_mb, _ = data_set.train.next_batch(mb_size)
                _, D_loss_curr= sess.run([D_solver, wgan_model.D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
            _, G_loss_curr = sess.run([G_solver, wgan_model.G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

            if it % 100 == 0:
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr, G_loss_curr))

        # Save the trained models.
        saver.save(sess, os.path.join(model_dir, "models-cgan-{}".format(today)))
        print("Training end!")


def test():

    # Saver
    saver = tf.train.Saver()

    # output probability
    with tf.Session() as sess:
        # Reload the well-trained models
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("The checkpoint models found!")
        else:
            print("The checkpoint models not found!")

        # prediction shape: [batch_size, width * height]
        prediction = sess.run(wgan_model.G_sample, feed_dict={Z: data_set.test.data})

        for it in range(prediction.shape[0]):
            y_true = data_set.test.target[it]
            y_pred = prediction[it]

            print(y_pred)
            y_pred_arr = np.array(y_pred)
            y_pred = filter(y_pred_arr, threshold)
            print(y_pred_arr.shape[0])
            maxV = np.amax(y_pred_arr)
            minV = np.amin(y_pred_arr)
            print('max:{} min:{}'.format(maxV, minV))

            acc_sum = 0
            for i in range(prediction.shape[1]):
                if y_true[i] != 1.0 and y_pred[i] == 1.0:
                    acc_sum += 1
            acc = acc_sum / prediction.shape[1]
            print('error:{}'.format(acc))

            # if it == 10:
            #     show_bitmap(data_set.test.data[5], image_width, image_height)
            #     show_bitmap(data_set.test.target[5], image_width, image_height)
            #     show_bitmap(y_pred, image_width, image_height)


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def filter(ay, threshold):
    result = []
    for it in range(ay.shape[0]):
        if ay[it] >= threshold:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)


def main(_):
    # train(args)
    # test()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test()
    else:
        print('Train or test')


if __name__ == '__main__':
    tf.app.run()