import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import input_data

# 200x200 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_200_200_200_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_200_200_200_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_200_200_20_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_200_200_20_test.npy"

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
data_set = input_data.read_data_sets(train_dir, validation_size=2)

mb_size = 64
Z_dim = 200*200
X_dim = data_set.train.data.shape[1]
y_dim = data_set.train.target.shape[1]
h1_dim = 200
h2_dim = 1000
h3_dim = 1000
h4_dim = 1000


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, 200*200])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h1_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h1_dim]))

# D_W2 = tf.Variable(xavier_init([h1_dim, h2_dim]))
# D_b2 = tf.Variable(tf.zeros(shape=[h2_dim]))

# D_W3 = tf.Variable(xavier_init([h2_dim, h3_dim]))
# D_b3 = tf.Variable(tf.zeros([h3_dim]))

# D_W4 = tf.Variable(xavier_init([h3_dim, h4_dim]))
# D_b4 = tf.Variable(tf.zeros([h4_dim]))

D_W5 = tf.Variable(xavier_init([h1_dim, 1]))
D_b5 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W5, D_b1, D_b5]


def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    # D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    # D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
    # D_h4 = tf.nn.relu(tf.matmul(D_h3, D_W4) + D_b4)
    D_logit = tf.matmul(D_h1, D_W5) + D_b5
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h1_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h1_dim]))

# G_W2 = tf.Variable(xavier_init([h1_dim, h2_dim]))
# G_b2 = tf.Variable(tf.zeros(shape=[h2_dim]))

# G_W3 = tf.Variable(xavier_init([h2_dim, h3_dim]))
# G_b3 = tf.Variable(tf.zeros(shape=[h3_dim]))

# G_W4 = tf.Variable(xavier_init([h3_dim, h4_dim]))
# G_b4 = tf.Variable(tf.zeros(shape=[h4_dim]))

G_W5 = tf.Variable(xavier_init([h1_dim, X_dim]))
G_b5 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W5, G_b1, G_b5]


def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    # G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    # G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    # G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)
    G_log_prob = tf.matmul(G_h1, G_W5) + G_b5
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(200, 200), cmap='Greys_r')

    return fig


G_sample = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

with tf.device("gpu:0"):
    # D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    # D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    # D_loss = D_loss_real + D_loss_fake
    # G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    #
    # D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    # G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
    G_loss = -tf.reduce_mean(D_fake)

    D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(-D_loss, var_list=theta_D))
    G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G))

    # clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0

    for it in range(10000):
        if it % 500 == 0:
            n_sample = 16

            Z_sample = sample_Z(n_sample, Z_dim)
            y_sample = np.zeros(shape=[n_sample, y_dim])
            y_sample[:, 7] = 1

            samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_sample})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_mb, y_mb = data_set.train.next_batch(mb_size)

        Z_sample = sample_Z(mb_size, Z_dim)
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y: y_mb})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y: y_mb})

        if it % 50 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()
