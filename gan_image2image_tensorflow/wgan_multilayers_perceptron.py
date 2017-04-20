from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import argparse

from utils import input_data


parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--mode', dest='mode', default='train', help='train, test')

args = parser.parse_args()

mb_size = 32
image_width = 128
image_height = 128
X_dim = image_width * image_height
Z_dim = X_dim

h_dim = 1000

# 200x40 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/kai_128_128_200_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_128_128_200_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/kai_128_128_20_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_128_128_20_test.npy"

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}


def net(X, Z):
    D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D = [D_W1, D_W2, D_b1, D_b2]

    G_W1 = tf.Variable(xavier_init([Z_dim, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

    theta_G = [G_W1, G_W2, G_b1, G_b2]

    def generator(z):
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)

        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob

    def discriminator(x):
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)

        out = tf.matmul(D_h1, D_W2) + D_b2
        return out

    # Generator net
    G_sample = generator(z=Z)
    # Discriminator net
    D_real = discriminator(X)
    D_fake = discriminator(G_sample)

    return D_real, D_fake, theta_D, theta_G


def train(args):
    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=2)

    # Inputs
    X = tf.placeholder(tf.float32, [None, X_dim], name="X")
    Z = tf.placeholder(tf.float32, [None, Z_dim], name="Z")

    # model
    D_real, D_fake, theta_D, theta_G = net(X, Z)

    with tf.name_scope("loss"):
        D_loss = tf.reduce_mean(D_real) + tf.reduce_mean(D_fake)
        G_loss = -tf.reduce_mean(D_fake)

        D_optim = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-D_loss,
                                                                         var_list=theta_D)
        G_optim = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss,
                                                                         var_list=theta_G)
        clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(args.epoch):
            # Discriminator
            for _ in range(5):
                X_mb, _ = data_set.train.next_batch(mb_size)
                _, D_loss_curr, _ = sess.run([D_optim, D_loss, clip_D],
                                             feed_dict={X:X_mb, Z: sample_z(mb_size, Z_dim)})

            # Generator
            _, G_loss_curr = sess.run([G_optim, G_loss], feed_dict={Z: sample_z()})

            if epoch % 50 == 0:
                print("D_loss:{} G_loss:{}".format(D_loss_curr, G_loss_curr))


def test():
    pass


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def main(_):
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test()
    else:
        print("train or test?")


if __name__ == "__main__":
    tf.app.run()