from __future__ import absolute_import

import tensorflow as tf

IMAGE_WIDTH = 250
IMAGE_HEIGHT = 250

DENSE_1_OUTPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 2
DENSE_2_OUTPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

class Dltcc(object):
    def __init__(self, npy_path=None, trainable=True, dropout=0.5):
        self.dataset = None
        self.trainable = trainable
        self.dropout = dropout

    '''
        Models build
    '''
    def build(self, images):

        if images is None:
            print("Images is none")
            return

        print('image shape:', images.shape)
        images_reshaped = tf.reshape(images, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

        # Conv 1
        with tf.name_scope("Conv_1"):
            self.conv1_1 = tf.layers.conv2d(inputs=images_reshaped, filters=16, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu)
            self.conv1_2 = tf.layers.conv2d(inputs=self.conv1_1, filters=16, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu)
            self.pool1 = tf.layers.average_pooling2d(inputs=self.conv1_2, pool_size=[2, 2], strides=1, padding='same')

        # Conv 2
        with tf.name_scope("Conv_2"):
            self.conv2_1 = tf.layers.conv2d(inputs=self.pool1, filters=32, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu)

            self.conv2_2 = tf.layers.conv2d(inputs=self.conv2_1, filters=32, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu)

            self.conv2_3 = tf.layers.conv2d(inputs=self.conv2_2, filters=32, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu)

            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2_3, pool_size=[2, 2], strides=1, padding='same')



        # Conv 3
        with tf.name_scope("Conv_3"):

            self.conv3_1 = tf.layers.conv2d(inputs=self.pool2, filters=64, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu)

            self.conv3_2 = tf.layers.conv2d(inputs=self.conv3_1, filters=64, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu)

            self.conv3_3 = tf.layers.conv2d(inputs=self.conv3_2, filters=64, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu)

            self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3_3, pool_size=[2, 2], strides=1, padding='same')

        # Flat layers
        with tf.name_scope("Flat_1"):
            self.flat1 = tf.reshape(self.pool3, [-1, IMAGE_WIDTH * IMAGE_HEIGHT * 64])

        # Dense Layer
        with tf.name_scope("Dense"):
            self.dense1 = tf.layers.dense(inputs=self.flat1, units=DENSE_1_OUTPUT_SIZE, activation=tf.nn.relu)

            self.dropout1 = tf.layers.dropout(inputs=self.dense1, rate=self.dropout)

            self.dense2 = tf.layers.dense(inputs=self.dropout1, units=DENSE_2_OUTPUT_SIZE, activation=tf.nn.relu)
        # Softmax function
        with tf.name_scope("predict"):
            self.pred = tf.nn.softmax(self.dense2)

        print('Build the models finished!')
