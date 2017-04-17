import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from ImageDisplay import *
from input_data import *
import datetime


# 200x200 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_200_200_200_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_200_200_200_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_200_200_20_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_200_200_20_test.npy"

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

# Data set
data_set = read_data_sets(train_dir, validation_size=2)

mb_size = 32
X_dim = data_set.train.data.shape[1]    # width * height
y_dim = data_set.train.target.shape[1]  # width * height
z_dim = X_dim                           # width * height
h_dim = 1000

# model saver
model_dir = "../../checkpoints/models_200_200_4_17"


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
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


G_sample = generator(Z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]


def train():

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

        for it in range(1000):

            for _ in range(5):
                X_mb, _ = data_set.train.next_batch(mb_size)

                _, D_loss_curr, _ = sess.run( [D_solver, D_loss, clip_D], feed_dict={X: X_mb, Z: sample_Z(mb_size, z_dim)})

            _, G_loss_curr = sess.run( [G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, z_dim)})

            if it % 100 == 0:
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr, G_loss_curr))

                # if it % 100 == 0:
                #     samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, z_dim)})

                    # fig = plot(samples)
                    # plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    # i += 1
                    # plt.close(fig)

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
        prediction = sess.run(G_sample, feed_dict={Z: data_set.train.data, y: data_set.train.target})

        print(prediction.shape)
        for it in range(prediction.shape[0]):
            y_true = data_set.train.target[it]
            y_pred = prediction[it]

            acc_sum = 0
            for i in range(prediction.shape[1]):
                if y_true[i] != 1.0 and y_pred[i] == 1.0:
                    acc_sum += 1
            acc = acc_sum / prediction.shape[1]
            print('error:{}'.format(acc))

        show_bitmap(data_set.train.data[0])
        show_bitmap(prediction[0])
        show_bitmap(data_set.train.target[0])


if __name__ == '__main__':
    train()
    # test()