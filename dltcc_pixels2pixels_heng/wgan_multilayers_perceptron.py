import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from ImageDisplay import *
from input_data import *
import datetime

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='# of epoch')

# 200x200 data set
# train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_200_200_200_train.npy"
# train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_200_200_200_train.npy"
#
# test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_200_200_20_test.npy"
# test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_200_200_20_test.npy"

# 200x40 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_heng_200_40_30_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_heng_200_40_30_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_heng_200_40_11_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_heng_200_40_11_test.npy"

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

# Data set
data_set = read_data_sets(train_dir, validation_size=2)

mb_size = 32
X_dim = data_set.train.data.shape[1]    # width * height
y_dim = data_set.train.target.shape[1]  # width * height
z_dim = X_dim                           # width * height
d_h1_dim = 1000
d_h2_dim = 1000
d_h3_dim = 1000
d_h4_dim = 1000

g_h1_dim = 1000
g_h2_dim = 1000
g_h3_dim = 1000
g_h4_dim = 1000

threshold = 0.4

image_width = 40
image_height = 200

# model saver
model_dir = "../../checkpoints/models_200_200_4_17_heng"


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim, d_h1_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[d_h1_dim]))

D_W2 = tf.Variable(xavier_init([d_h1_dim, d_h2_dim]))
D_b2 = tf.Variable(tf.zeros(shape=[d_h2_dim]))

D_W3 = tf.Variable(xavier_init([d_h2_dim, d_h3_dim]))
D_b3 = tf.Variable(tf.zeros(shape=[d_h3_dim]))

D_W4 = tf.Variable(xavier_init([d_h3_dim, d_h4_dim]))
D_b4 = tf.Variable(tf.zeros(shape=[d_h4_dim]))

D_W5 = tf.Variable(xavier_init([d_h4_dim, 1]))
D_b5 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2,D_W3, D_W4, D_W5, D_b1, D_b2, D_b3, D_b4, D_b5]

""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, g_h1_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[g_h1_dim]))

G_W2 = tf.Variable(xavier_init([g_h1_dim, g_h2_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[g_h2_dim]))

G_W3 = tf.Variable(xavier_init([g_h2_dim, g_h3_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[g_h3_dim]))

G_W4 = tf.Variable(xavier_init([g_h3_dim, g_h4_dim]))
G_b4 = tf.Variable(tf.zeros(shape=[g_h4_dim]))

G_W5 = tf.Variable(xavier_init([g_h4_dim, X_dim]))
G_b5 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5, G_b1, G_b2, G_b3, G_b4, G_b5]


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
    D_h4 = tf.nn.relu(tf.matmul(D_h3, D_W4) + D_b4)
    out = tf.matmul(D_h4, D_W5) + D_b5
    return out


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)

    G_log_prob = tf.matmul(G_h4, G_W5) + G_b5
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

G_sample = generator(Z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

args = parser.parse_args()


def train(args):

    with tf.device("gpu:0"):
        D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
        G_loss = -tf.reduce_mean(D_fake)

        D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-D_loss, var_list=theta_D))
        G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # save the models and checkpoints.
        saver = tf.train.Saver()
        now = datetime.datetime.now()
        today = "{}-{}-{}".format(now.year, now.month, now.day)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists('out_cgan/'):
            os.makedirs('out_cgan/')

        i = 0

        for it in range(args.epoch):

            for _ in range(5):
                X_mb, _ = data_set.train.next_batch(mb_size)
                _, D_loss_curr, _ = sess.run( [D_solver, D_loss, clip_D], feed_dict={X: X_mb, Z: sample_Z(mb_size, z_dim)})
            _, G_loss_curr = sess.run( [G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, z_dim)})

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
        prediction = sess.run(G_sample, feed_dict={Z: data_set.test.data})

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

            if it == 10:
                show_bitmap(data_set.test.data[5], image_width, image_height)
                show_bitmap(data_set.test.target[5], image_width, image_height)
                show_bitmap(y_pred, image_width, image_height)


def filter(ay, threshold):
    result = []
    for it in range(ay.shape[0]):
        if ay[it] >= threshold:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)


def main(_):
    train(args)
    # test()

if __name__ == '__main__':
    tf.app.run()