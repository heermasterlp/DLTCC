import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.python.framework import ops
import math

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
h_dim = 200


df_dim = 64
gf_dim = 64
batch_size = 1
output_size = 256

input_c_dim = 1
output_c_dim = 1


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


class batch_norm(object):

    # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name


    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            scale=True, scope=self.name)


def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                                (1. - targets) * tf.log(1. - preds + eps)))


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator1(x, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:

        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(x, df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(batch_norm(conv2d(h0, df_dim * 2, name='d_h1_conv')))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(batch_norm(conv2d(h1, df_dim * 4, name='d_h2_conv')))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(batch_norm(conv2d(h2, df_dim * 8, d_h=1, d_w=1, name='d_h3_conv')))
        # h3 is (16 x 16 x self.df_dim*8)
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4


def generator1(z, y=None):
    with tf.variable_scope("generator") as scope:
        s = output_size
        s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
            s / 64), int(s / 128)

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(z, gf_dim, name='g_e1_conv')
        # e1 is (128 x 128 x self.gf_dim)
        e2 = batch_norm(conv2d(lrelu(e1), gf_dim * 2, name='g_e2_conv'))
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = batch_norm(conv2d(lrelu(e2), gf_dim * 4, name='g_e3_conv'))
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = batch_norm(conv2d(lrelu(e3), gf_dim * 8, name='g_e4_conv'))
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = batch_norm(conv2d(lrelu(e4), gf_dim * 8, name='g_e5_conv'))
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = batch_norm(conv2d(lrelu(e5), gf_dim * 8, name='g_e6_conv'))
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = batch_norm(conv2d(lrelu(e6), gf_dim * 8, name='g_e7_conv'))
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = batch_norm(conv2d(lrelu(e7), gf_dim * 8, name='g_e8_conv'))
        # e8 is (1 x 1 x self.gf_dim*8)

        d1, d1_w, d1_b = deconv2d(tf.nn.relu(e8), [batch_size, s128, s128, gf_dim * 8], name='g_d1', with_w=True)
        d1 = tf.nn.dropout(batch_norm(d1), 0.5)
        d1 = tf.concat([d1, e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2, d2_w, d2_b = deconv2d(tf.nn.relu(d1), [batch_size, s64, s64, gf_dim * 8], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(batch_norm(d2), 0.5)
        d2 = tf.concat([d2, e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3, d3_w, d3_b = deconv2d(tf.nn.relu(d2),
                                                 [batch_size, s32, s32, gf_dim * 8], name='g_d3', with_w=True)
        d3 = tf.nn.dropout(batch_norm(d3), 0.5)
        d3 = tf.concat([d3, e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4, d4_w, d4_b = deconv2d(tf.nn.relu(d3),[batch_size, s16, s16, gf_dim * 8], name='g_d4', with_w=True)
        d4 = batch_norm(d4)
        d4 = tf.concat([d4, e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5, d5_w, d5_b = deconv2d(tf.nn.relu(d4), [batch_size, s8, s8, gf_dim * 4], name='g_d5', with_w=True)
        d5 = batch_norm(d5)
        d5 = tf.concat([d5, e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6, d6_w, d6_b = deconv2d(tf.nn.relu(d5), [batch_size, s4, s4, gf_dim * 2], name='g_d6', with_w=True)
        d6 = batch_norm(d6)
        d6 = tf.concat([d6, e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7, d7_w, d7_b = deconv2d(tf.nn.relu(d6), [batch_size, s2, s2, gf_dim], name='g_d7', with_w=True)
        d7 = batch_norm(d7)
        d7 = tf.concat([d7, e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8, d8_w, d8_b = deconv2d(tf.nn.relu(d7), [batch_size, s, s, output_c_dim], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)





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


G_sample = generator1(Z, y)
D_real, D_logit_real = discriminator1(X, y)
D_fake, D_logit_fake = discriminator1(G_sample, y)

with tf.device("cpu:0"):

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

    for it in range(1000):
        if it % 10 == 0:
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

        if it % 10 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()








