from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import datetime
import os
import time

import input_data

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

model_path = "../../checkpoints/models_150_200_4_1"
checkpoint_path = "../../checkpoints/checkpoints_150_200"

# threshold
THEROSHOLD = 0.8

# max training epoch
MAX_TRAIN_EPOCH = 100000


src_path='./src/train_50_bk_1000.npy';
tar_path='./target/test_50_qg_1000.npy';
out_dir='./out/CNNADV50/';
default_gif_name='cnnadv.gif';
summary_dir='./out/summary50/'
# checkpoints_dir='./out/CNNADV50/checkpoints/'
# checkpoints_dir='/home/william/ML/GPU/50_50000_b/CNNADV50/checkpoints'
checkpoints_dir='/home/william/ML/GPU/small/CNNADV50/checkpoints'

# bitmap_dir='./out/CNNADV50_b/infer/';
bitmap_dir='/home/william/ML/GPU/CNNADV50_b/CNNADV50/infer_eval/';
validation_src='./src_b/eval_50_bk_10.npy'
validation_target='./target_b/eval_50_qg_10.npy'
SRC_IMG=100;
TAR_IMG=50;
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('display_step',10,'Nuber of steps to print');
flags.DEFINE_string('src_path',src_path,'The font source path');
flags.DEFINE_string('tar_path',tar_path,'The target font path');
flags.DEFINE_integer('batch_size',10,'The number of batch size');
flags.DEFINE_float('learning_rate',0.01,'The Initial learning rate');
flags.DEFINE_integer('validation_words',20,'The number of words to be trained');
flags.DEFINE_integer('total_words',1000,'The total number of words');
flags.DEFINE_integer('max_steps',3000,'The number os steps to train');
flags.DEFINE_boolean('unit_scale',True,'If use,use unit scale data');
flags.DEFINE_float('keep_probability',0.9,'The probability be dropped out');
flags.DEFINE_float('alpha',-1.0,'alpha slope for leaky relu if non-negative, otherwise use relu');
flags.DEFINE_integer('laywers',2,'the number of hidden laywers');
flags.DEFINE_float('tv',0.0002,'weight for tv loss, use to force smooth output');
flags.DEFINE_integer('num_ckpt',5,'number of model checkpoints to keep')
flags.DEFINE_string('summary_dir',summary_dir,'the dir to store summary');
flags.DEFINE_integer('checkpoint_steps',50,'number of steps between two checkpoints')
flags.DEFINE_string('model','infer','The phase of train or infer');


def conv2d_block(X, shape, strides, padding, scope='conv2d'):
    with tf.name_scope(scope):
        if not strides:
            strides = [1, 1, 1, 1]
        outfilters = shape[-1]
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.01), name='weights')
        b = tf.Variable(tf.constant(0.1, shape=[outfilters]), name='biases')
        cnn_layer = tf.nn.conv2d(X, W, strides, padding) + b
        return cnn_layer


def max_pool(X, scope='max_pool'):
    with tf.name_scope(scope):
        return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], padding='SAME')


def batch_norm(X, phase_train, scope='batch_normal'):
    with tf.name_scope(scope):
        outFilters = X.get_shape()[-1]
        beta = tf.Variable(tf.constant(0.0, shape=[outFilters]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[outFilters]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(X, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity([batch_var])
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normaled = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)
        return normaled


def Leak_relu(X, alpha):
    return tf.maximum(X, alpha*X)


def block(X, shape, phase_train, strides=None, padding='SAME', scope='_block'):
    with tf.name_scope(scope):
        cnn = conv2d_block(X, shape, strides, padding)
        cnn_bn = batch_norm(cnn, phase_train)
        if FLAGS.alpha < 0:
            _relu = tf.nn.relu(cnn_bn)
        else:
            _relu = Leak_relu(cnn_bn, FLAGS.alpha)
        return _relu


def construct_network(X, size, infilters, outfilters, laywers, phase_train,
                      strides=None, scope='construct_network'):
    with tf.name_scope(scope):
        # The first laywer:
        conv1 = block(X, [size, size, infilters, outfilters], phase_train, strides,
                      scope="conv1_%dx%d" % (outfilters, outfilters))
        cur_conv = conv1
        for i in range(laywers - 1):
            the_next_laywer = block(cur_conv, [size, size, outfilters, outfilters], phase_train, strides,
                                    scope='conv%d_%dx%d' % (i + 2, outfilters, outfilters))
            cur_conv = the_next_laywer
        return cur_conv


def total_variation_loss(x, side):
    """
        Total variation loss for regularization of image smoothness
        """
    loss = tf.nn.l2_loss(x[:, 1:, :, :] - x[:, :side - 1, :, :]) / side + \
           tf.nn.l2_loss(x[:, :, 1:, :] - x[:, :, :side - 1, :]) / side
    return loss


def train():
    keep_prob = 0.9
    learning_rate = 0.01

    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=2)

    if data_set is None:
        print("data sets are none!")
        return
    else:
        print("data set train data shape:{}".format(data_set.train.data.shape))
        print("data set train target shape:{}".format(data_set.train.target.shape))
        print("data set test data shape:{}".format(data_set.test.data.shape))
        print("data set test target shape:{}".format(data_set.test.target.shape))

    # save the models and checkpoints.
    # the formatting: (models) models-date.ckpt, (checkpoint) checkpoint-date-step.ckpt
    saver = tf.train.Saver()
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    now = datetime.datetime.now()
    today = "{}-{}-{}".format(now.year, now.month, now.day)

    # Create the CNN models
    with tf.name_scope('models'):
        # place variable
        x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name='x')
        y_true = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT], name='y_true')
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='X')
        y_image = tf.reshape(y_true, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='Y')

        # the laywers:
        conv64 = construct_network(x_image, size=64, infilters=1, outfilters=8, laywers=2,
                                   phase_train=phase_train, scope='conv64')
        conv32 = construct_network(conv64, size=32, infilters=8, outfilters=32, laywers=2,
                                   phase_train=phase_train, scope='conv32')
        conv16 = construct_network(conv32, size=16, infilters=32, outfilters=64, laywers=2,
                                   phase_train=phase_train, scope='conv16')
        conv7 = construct_network(conv16, size=7, infilters=64, outfilters=128, laywers=2,
                                  phase_train=phase_train, scope='conv7')
    with tf.name_scope('conv3'):
        conv3_1 = block(conv7, [3, 3, 128, 128], phase_train=phase_train, scope='conv3_1');
        conv3_2 = block(conv3_1, [3, 3, 128, 1], phase_train=phase_train, scope='conv3_2');
    pooled = max_pool(conv3_2);
    with tf.name_scope('normalization'):
        dropped = tf.nn.dropout(pooled, keep_prob=keep_prob)
        y_pre_img = tf.sigmoid(dropped)

        weight_matrix = tf.random_uniform(shape=(IMAGE_WIDTH, IMAGE_HEIGHT), minval=0.0, maxval=0.5)
    with tf.name_scope('train'):
        with tf.name_scope('loss'):
            pixel_loss = tf.reduce_mean(tf.abs(y_image, y_pre_img))
            tv_loss = 0.0002 * total_variation_loss(y_pre_img, IMAGE_WIDTH)
            combined_loss = pixel_loss + tv_loss

        optimizer_op = tf.train.RMSPropOptimizer(learning_rate).minimize(combined_loss)

    # initi
    init = tf.global_variables_initializer()
    with tf.Graph.as_default():
        with tf.Session() as sess:
            start_time = time.time()
            sess.run(init)

            for epoch in range(MAX_TRAIN_EPOCH):
                batch_x, batch_y = data_set.next_batch(200)

                _, loss = sess.run([optimizer_op, combined_loss], feed_dict={
                    x: batch_x, y_true: batch_y
                })
                if epoch % 100 == 0:
                    print("{0} : {1}".format(epoch, loss))

        duration = time.time() - start_time

        # Save the trained models.
        saver.save(sess, os.path.join(model_path, "models-{}".format(today)))
        print("Training end!{}".format(duration))


if __name__ == '__main__':
    train()
